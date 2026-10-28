"""End-to-end retrieval benchmark runner and dataset loading utilities.

Implements benchmark dataset loading, corpus indexing, latency measurement,
retrieval evaluation across set-based and positional metrics, failure diagnostics,
and aggregate EvaluationReport compilation using Python standard library only, per ADR-005.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from reliability_lab.core.interfaces import BaseMetric, BaseRetriever
from reliability_lab.core.models import Chunk, EvaluationReport, Query
from reliability_lab.evaluation.retrieval_metrics import (
    HitRateAtK,
    MRRAtK,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
)


def load_corpus(path: str | Path) -> list[Chunk]:
    """Load and validate a collection of citation-aware Chunks from a JSONL file.

    Args:
        path: Path to the JSONL corpus file.

    Returns:
        List of deserialized Chunk domain entities.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        ValueError: If the file is empty, contains invalid JSON, or contains duplicate chunk IDs.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Corpus file does not exist: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Corpus file is empty: {file_path}")

    chunks: list[Chunk] = []
    seen_ids: set[str] = set()

    for line_num, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON on line {line_num} of corpus {file_path.name}: {exc}"
            ) from exc

        if not isinstance(data, dict):
            raise ValueError(f"Line {line_num} of {file_path.name} must be a JSON object.")

        try:
            chunk = Chunk.from_dict(data)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid chunk record on line {line_num} of {file_path.name}: {exc}"
            ) from exc

        if chunk.chunk_id in seen_ids:
            raise ValueError(
                f"Duplicate chunk_id '{chunk.chunk_id}' on line {line_num} of {file_path.name}."
            )
        seen_ids.add(chunk.chunk_id)
        chunks.append(chunk)

    if not chunks:
        raise ValueError(f"Corpus file contains no valid chunk records: {file_path}")

    return chunks


def load_benchmark_dataset(
    path: str | Path,
    corpus_chunks: Sequence[Chunk] | None = None,
) -> list[Query]:
    """Load, validate, and deserialize benchmark query records from a JSONL file.

    Supports schema variations:
    - Query text under either 'query' or 'text'.
    - Graded relevance judgments under 'relevance_judgments' or 'relevance_grades'.
    - Required citation identifiers or structured citation dictionaries.

    Args:
        path: Path to the JSONL benchmark query file.
        corpus_chunks: Optional collection of corpus chunks for cross-referencing validity.

    Returns:
        List of validated Query domain entities.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If file is empty, contains malformed JSON, duplicate query IDs,
            or unknown chunk references when corpus_chunks is provided.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Benchmark file does not exist: {file_path}")

    text = file_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Benchmark file is empty: {file_path}")

    valid_corpus_ids: set[str] | None = (
        {c.chunk_id for c in corpus_chunks} if corpus_chunks is not None else None
    )

    queries: list[Query] = []
    seen_ids: set[str] = set()

    for line_num, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed JSON on line {line_num} of benchmark {file_path.name}: {exc}"
            ) from exc

        if not isinstance(record, dict):
            raise ValueError(f"Line {line_num} of {file_path.name} must be a JSON object.")

        query_id = record.get("query_id")
        if not isinstance(query_id, str) or not query_id.strip():
            raise ValueError(f"Missing or invalid query_id on line {line_num} of {file_path.name}.")

        if query_id in seen_ids:
            raise ValueError(
                f"Duplicate query_id '{query_id}' on line {line_num} of {file_path.name}."
            )
        seen_ids.add(query_id)

        # Extract query text (support 'query' and 'text')
        query_text = record.get("query") or record.get("text")
        if not isinstance(query_text, str) or not query_text.strip():
            raise ValueError(f"Query text missing or empty for query '{query_id}'.")

        # Parse relevance judgments
        relevance_grades: dict[str, int] = {}
        relevant_chunk_ids: list[str] = []

        if "relevance_judgments" in record:
            judgments = record["relevance_judgments"]
            if not isinstance(judgments, list):
                raise ValueError(f"'relevance_judgments' must be a list in query '{query_id}'.")
            for j in judgments:
                if not isinstance(j, dict) or "chunk_id" not in j or "grade" not in j:
                    raise ValueError(f"Malformed relevance judgment in query '{query_id}'.")
                cid = str(j["chunk_id"])
                grade = j["grade"]
                if isinstance(grade, bool) or not isinstance(grade, int) or grade < 0:
                    raise ValueError(
                        f"Relevance grade must be a non-negative int for chunk '{cid}' "
                        f"in query '{query_id}'."
                    )
                if valid_corpus_ids is not None and cid not in valid_corpus_ids:
                    raise ValueError(f"Query '{query_id}' references unknown corpus chunk '{cid}'.")
                relevance_grades[cid] = grade
                if grade > 0 and cid not in relevant_chunk_ids:
                    relevant_chunk_ids.append(cid)
        elif "relevance_grades" in record:
            raw_grades = record["relevance_grades"]
            if not isinstance(raw_grades, dict):
                raise ValueError(f"'relevance_grades' must be a dict in query '{query_id}'.")
            for cid_raw, grade_raw in raw_grades.items():
                cid = str(cid_raw)
                if isinstance(grade_raw, bool) or not isinstance(grade_raw, int) or grade_raw < 0:
                    raise ValueError(
                        f"Grade for '{cid}' must be non-negative int in query '{query_id}'."
                    )
                if valid_corpus_ids is not None and cid not in valid_corpus_ids:
                    raise ValueError(f"Query '{query_id}' references unknown corpus chunk '{cid}'.")
                relevance_grades[cid] = grade_raw
                if grade_raw > 0 and cid not in relevant_chunk_ids:
                    relevant_chunk_ids.append(cid)

        # Merge explicit relevant_chunk_ids if provided
        if "relevant_chunk_ids" in record and isinstance(record["relevant_chunk_ids"], list):
            for cid in record["relevant_chunk_ids"]:
                cid_str = str(cid)
                if valid_corpus_ids is not None and cid_str not in valid_corpus_ids:
                    raise ValueError(
                        f"Query '{query_id}' references unknown corpus chunk '{cid_str}'."
                    )
                if cid_str not in relevant_chunk_ids and relevance_grades.get(cid_str, 1) > 0:
                    relevant_chunk_ids.append(cid_str)
                    if cid_str not in relevance_grades:
                        relevance_grades[cid_str] = 1

        # Extract citations
        required_citations: list[str] = []
        citation_records: list[dict[str, Any]] = []
        citation_to_chunk_id: dict[str, str] = {}
        if "required_citations" in record and isinstance(record["required_citations"], list):
            for item in record["required_citations"]:
                if isinstance(item, dict):
                    cit_id = str(item.get("citation_id") or item.get("chunk_id") or "").strip()
                    if cit_id:
                        required_citations.append(cit_id)
                        citation_records.append(dict(item))
                        chunk_id = str(item.get("chunk_id", "")).strip()
                        if chunk_id:
                            existing_chunk = citation_to_chunk_id.get(cit_id)
                            if existing_chunk is not None and existing_chunk != chunk_id:
                                raise ValueError(
                                    f"Query '{query_id}' has conflicting chunk_id mapping for "
                                    f"citation '{cit_id}': '{existing_chunk}' vs '{chunk_id}'"
                                )
                            citation_to_chunk_id[cit_id] = chunk_id
                elif isinstance(item, str) and item.strip():
                    required_citations.append(item.strip())

        query_metadata = dict(record.get("metadata", {}))
        if citation_records:
            query_metadata["citation_records"] = citation_records
        if citation_to_chunk_id:
            query_metadata["citation_to_chunk_id"] = citation_to_chunk_id

        query = Query(
            query_id=query_id,
            text=query_text,
            relevant_chunk_ids=relevant_chunk_ids,
            relevance_grades=relevance_grades,
            ground_truth_answer=record.get("ground_truth_answer"),
            required_citations=required_citations,
            metadata=query_metadata,
        )
        queries.append(query)

    if not queries:
        raise ValueError(f"Benchmark file contains no valid query records: {file_path}")

    return queries


def calculate_latency_percentiles(latencies: Sequence[float]) -> dict[str, float]:
    """Compute summary statistics and percentiles for a sequence of latency measurements.

    Args:
        latencies: Sequence of non-negative latency measurements in milliseconds.

    Returns:
        Dictionary mapping statistic keys to float values:
        - 'mean_latency_ms'
        - 'latency_p50_ms' (median)
        - 'latency_p90_ms'
        - 'latency_p95_ms'
        - 'latency_p99_ms'
        - 'min_latency_ms'
        - 'max_latency_ms'
    """
    if not latencies:
        return {
            "mean_latency_ms": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p90_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }

    sorted_vals = sorted(float(v) for v in latencies)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / float(n)

    def _calc_p(p: float) -> float:
        if n == 1:
            return sorted_vals[0]
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        d0 = sorted_vals[int(f)] * (c - k)
        d1 = sorted_vals[int(c)] * (k - f)
        return float(d0 + d1)

    return {
        "mean_latency_ms": float(mean_val),
        "latency_p50_ms": _calc_p(50.0),
        "latency_p90_ms": _calc_p(90.0),
        "latency_p95_ms": _calc_p(95.0),
        "latency_p99_ms": _calc_p(99.0),
        "min_latency_ms": float(sorted_vals[0]),
        "max_latency_ms": float(sorted_vals[-1]),
    }


class RetrievalBenchmarkRunner:
    """End-to-end benchmark runner for evaluating retrieval algorithms.

    Executes a retriever against a golden benchmark dataset, measures query latencies,
    evaluates set-based and positional retrieval metrics, extracts query failure traces,
    and returns a canonical EvaluationReport.
    """

    def __init__(
        self,
        retriever: BaseRetriever,
        metrics: Sequence[BaseMetric] | None = None,
        top_k: int = 5,
        experiment_name: str = "retrieval_benchmark",
    ) -> None:
        """Initialize RetrievalBenchmarkRunner.

        Args:
            retriever: An instance implementing BaseRetriever.
            metrics: Optional sequence of BaseMetric instances. If None, defaults to
                Recall@K, Precision@K, HitRate@K, MRR@K, and NDCG@K at cutoff top_k.
            top_k: Top-K ranking cutoff for retrieval and evaluation (default 5).
            experiment_name: Label identifying this benchmark experiment run.

        Raises:
            TypeError: If retriever or metrics are not valid interface instances.
            ValueError: If top_k is not an integer >= 1 or experiment_name is empty.
        """
        if not isinstance(retriever, BaseRetriever):
            raise TypeError(
                f"retriever must implement BaseRetriever, got: {type(retriever).__name__}"
            )
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got: {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be a positive integer >= 1, got: {top_k}")
        if not isinstance(experiment_name, str) or not experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string.")

        self._retriever = retriever
        self._top_k = top_k
        self._experiment_name = experiment_name.strip()

        if metrics is None:
            self._metrics: list[BaseMetric] = [
                RecallAtK(k=top_k),
                PrecisionAtK(k=top_k),
                HitRateAtK(k=top_k),
                MRRAtK(k=top_k),
                NDCGAtK(k=top_k),
            ]
        else:
            if not isinstance(metrics, Sequence) or not metrics:
                raise ValueError("metrics must be a non-empty Sequence of BaseMetric instances.")
            for m in metrics:
                if not isinstance(m, BaseMetric):
                    raise TypeError(
                        f"All metrics must implement BaseMetric, got: {type(m).__name__}"
                    )
            self._metrics = list(metrics)

    @property
    def retriever(self) -> BaseRetriever:
        """Return the underlying retriever instance."""
        return self._retriever

    @property
    def top_k(self) -> int:
        """Return the evaluation cutoff top_k."""
        return self._top_k

    @property
    def metrics(self) -> list[BaseMetric]:
        """Return the configured evaluation metrics."""
        return list(self._metrics)

    @property
    def experiment_name(self) -> str:
        """Return the experiment name."""
        return self._experiment_name

    def run(
        self,
        corpus: Sequence[Chunk] | str | Path,
        benchmark: Sequence[Query] | str | Path,
        auto_index: bool = True,
    ) -> EvaluationReport:
        """Execute the end-to-end benchmark evaluation run.

        Args:
            corpus: Sequence of Chunk objects or a file path to a corpus JSONL.
            benchmark: Sequence of Query objects or a file path to a benchmark JSONL.
            auto_index: If True, calls retriever.index(corpus) before query execution.

        Returns:
            A populated, valid EvaluationReport containing aggregate metrics,
            latency percentiles, per-query diagnostics, and run configuration.

        Raises:
            ValueError: If corpus or benchmark query lists are empty.
        """
        # Resolve corpus
        corpus_chunks: list[Chunk]
        if isinstance(corpus, (str, Path)):
            corpus_chunks = load_corpus(corpus)
        elif isinstance(corpus, Sequence):
            if len(corpus) == 0:
                raise ValueError("Corpus chunk sequence must not be empty.")
            corpus_chunks = []
            for c in corpus:
                if not isinstance(c, Chunk):
                    raise TypeError(f"All corpus items must be Chunk instances, got {type(c)}")
                corpus_chunks.append(c)
        else:
            raise TypeError("corpus must be a Sequence[Chunk] or a file path string / Path.")

        # Resolve benchmark queries
        benchmark_queries: list[Query]
        if isinstance(benchmark, (str, Path)):
            benchmark_queries = load_benchmark_dataset(benchmark, corpus_chunks=corpus_chunks)
        elif isinstance(benchmark, Sequence):
            if len(benchmark) == 0:
                raise ValueError("Benchmark query sequence must not be empty.")
            benchmark_queries = []
            for q in benchmark:
                if not isinstance(q, Query):
                    raise TypeError(f"All benchmark items must be Query instances, got {type(q)}")
                benchmark_queries.append(q)
        else:
            raise TypeError("benchmark must be a Sequence[Query] or a file path string / Path.")

        # Index corpus if requested
        if auto_index:
            self._retriever.index(corpus_chunks)

        per_query_results: list[dict[str, Any]] = []
        metric_accumulators: dict[str, list[float]] = {m.name: [] for m in self._metrics}
        latencies_ms: list[float] = []
        zero_hits = 0

        # Execute benchmark queries
        for query in benchmark_queries:
            t0 = time.perf_counter()
            retrieved = self._retriever.retrieve(query.text, top_k=self._top_k)
            t1 = time.perf_counter()

            latency_ms = (t1 - t0) * 1000.0
            latencies_ms.append(latency_ms)

            # Compute metrics for this query
            query_metrics: dict[str, float] = {}
            for metric in self._metrics:
                score = metric.compute(retrieved, query)
                query_metrics[metric.name] = float(score)
                metric_accumulators[metric.name].append(score)

            # Analyze hits and failure trace
            retrieved_chunk_ids = [res.chunk.chunk_id for res in retrieved]
            relevant_ids = set(query.relevant_chunk_ids) | {
                cid for cid, g in query.relevance_grades.items() if g > 0
            }
            critical_ids = {cid for cid, g in query.relevance_grades.items() if g >= 2}

            retrieved_hits = [cid for cid in retrieved_chunk_ids if cid in relevant_ids]
            if not retrieved_hits:
                zero_hits += 1

            first_relevant_rank: int | None = None
            for rank_idx, res in enumerate(retrieved, start=1):
                if res.chunk.chunk_id in relevant_ids:
                    first_relevant_rank = rank_idx
                    break

            missed_relevant = [cid for cid in relevant_ids if cid not in retrieved_chunk_ids]
            missed_critical = [cid for cid in critical_ids if cid not in retrieved_chunk_ids]

            has_failure = bool(missed_critical or len(retrieved_hits) == 0)

            failure_trace: dict[str, Any] = {
                "has_failure": has_failure,
                "zero_hit": len(retrieved_hits) == 0,
                "first_relevant_rank": first_relevant_rank,
                "missed_relevant_chunk_ids": missed_relevant,
                "missed_critical_chunk_ids": missed_critical,
                "retrieved_hit_count": len(retrieved_hits),
                "total_relevant_count": len(relevant_ids),
            }

            per_query_results.append(
                {
                    "query_id": query.query_id,
                    "text": query.text,
                    "latency_ms": float(latency_ms),
                    "metrics": query_metrics,
                    "retrieved": [
                        {
                            "chunk_id": res.chunk.chunk_id,
                            "doc_id": res.chunk.doc_id,
                            "rank": res.rank,
                            "score": float(res.score),
                            "grade": query.get_grade(res.chunk.chunk_id),
                        }
                        for res in retrieved
                    ],
                    "failure_trace": failure_trace,
                }
            )

        # Calculate aggregate metrics
        total_queries = len(benchmark_queries)
        aggregate_metrics: dict[str, float] = {}

        for metric_name, scores in metric_accumulators.items():
            mean_score = sum(scores) / float(total_queries)
            # Provide canonical name (e.g. recall@5) and descriptive prefix (mean_recall@5)
            aggregate_metrics[metric_name] = float(mean_score)
            aggregate_metrics[f"mean_{metric_name}"] = float(mean_score)

        # Add latency percentiles
        latency_stats = calculate_latency_percentiles(latencies_ms)
        for stat_name, stat_val in latency_stats.items():
            aggregate_metrics[stat_name] = float(stat_val)

        aggregate_metrics["zero_hit_rate"] = float(zero_hits) / float(total_queries)
        aggregate_metrics["total_queries"] = float(total_queries)

        config: dict[str, Any] = {
            "retriever_name": self._retriever.name,
            "top_k": self._top_k,
            "corpus_size": len(corpus_chunks),
            "query_count": total_queries,
            "metrics": [m.name for m in self._metrics],
        }

        return EvaluationReport(
            experiment_name=self._experiment_name,
            metrics=aggregate_metrics,
            per_query_results=per_query_results,
            config=config,
        )


BenchmarkRunner = RetrievalBenchmarkRunner
