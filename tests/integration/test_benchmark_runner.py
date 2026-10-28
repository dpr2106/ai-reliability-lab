"""Integration tests for the retrieval benchmark runner (EVAL-003).

Validates end-to-end execution of BM25 retrieval over the golden benchmark dataset,
latency measurement and percentile aggregation, failure diagnostics,
corpus/benchmark loading, schema flexibility, and EvaluationReport serialization.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, EvaluationReport, Query, RetrievalResult
from reliability_lab.evaluation.benchmark_runner import (
    BenchmarkRunner,
    RetrievalBenchmarkRunner,
    calculate_latency_percentiles,
    load_benchmark_dataset,
    load_corpus,
)
from reliability_lab.evaluation.retrieval_metrics import (
    HitRateAtK,
    NDCGAtK,
    RecallAtK,
)
from reliability_lab.indexing.bm25_index import BM25Retriever

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"
CORPUS_PATH = DATA_DIR / "sample_corpus.jsonl"
BENCHMARK_PATH = DATA_DIR / "sample_benchmark.jsonl"


class TestDatasetLoaders:
    """Tests for load_corpus and load_benchmark_dataset utilities."""

    def test_load_corpus_success(self) -> None:
        chunks = load_corpus(CORPUS_PATH)
        assert len(chunks) == 25
        assert all(isinstance(c, Chunk) for c in chunks)
        # Verify first chunk
        first = chunks[0]
        assert first.chunk_id == "CHK-CRAN-001"
        assert first.doc_id == "doc-cranfield-ir"
        assert "Cranfield" in first.content

    def test_load_corpus_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_corpus(tmp_path / "nonexistent.jsonl")

    def test_load_corpus_empty_file(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_corpus(empty_file)

    def test_load_corpus_malformed_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text(
            '{"chunk_id": "C1", "doc_id": "D1", "content": "ok"}\n{not json}\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_corpus(bad_file)

    def test_load_corpus_invalid_chunk_record(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "missing_doc.jsonl"
        bad_file.write_text('{"chunk_id": "C1", "content": "missing doc id"}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid chunk record"):
            load_corpus(bad_file)

    def test_load_corpus_duplicate_chunk_id(self, tmp_path: Path) -> None:
        dup_file = tmp_path / "dup.jsonl"
        dup_file.write_text(
            '{"chunk_id": "C1", "doc_id": "D1", "content": "text 1"}\n'
            '{"chunk_id": "C1", "doc_id": "D1", "content": "text 2"}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate chunk_id 'C1'"):
            load_corpus(dup_file)

    def test_load_benchmark_dataset_success(self) -> None:
        corpus = load_corpus(CORPUS_PATH)
        queries = load_benchmark_dataset(BENCHMARK_PATH, corpus_chunks=corpus)
        assert len(queries) == 16
        assert all(isinstance(q, Query) for q in queries)

        first_q = queries[0]
        assert first_q.query_id == "Q-001"
        assert "BM25" in first_q.text
        assert "CHK-HYB-001" in first_q.relevant_chunk_ids
        assert first_q.get_grade("CHK-HYB-001") == 2
        assert first_q.get_grade("CHK-LAT-001") == 0

    def test_load_benchmark_dataset_file_not_found(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            load_benchmark_dataset(tmp_path / "nonexistent.jsonl")

    def test_load_benchmark_dataset_empty_file(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("   \n", encoding="utf-8")
        with pytest.raises(ValueError, match="empty"):
            load_benchmark_dataset(empty_file)

    def test_load_benchmark_dataset_malformed_json(self, tmp_path: Path) -> None:
        bad_file = tmp_path / "bad.jsonl"
        bad_file.write_text('{"query_id": "Q1", "query": "test"}\n{bad json}\n', encoding="utf-8")
        with pytest.raises(ValueError, match="Malformed JSON"):
            load_benchmark_dataset(bad_file)

    def test_load_benchmark_dataset_duplicate_query_id(self, tmp_path: Path) -> None:
        dup_file = tmp_path / "dup_q.jsonl"
        dup_file.write_text(
            '{"query_id": "Q1", "query": "test 1"}\n{"query_id": "Q1", "query": "test 2"}\n',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Duplicate query_id 'Q1'"):
            load_benchmark_dataset(dup_file)

    def test_load_benchmark_dataset_unknown_corpus_chunk(self, tmp_path: Path) -> None:
        corpus = [
            Chunk(chunk_id="CHK-1", doc_id="D1", content="Corpus chunk 1"),
        ]
        q_file = tmp_path / "q_unknown.jsonl"
        q_file.write_text(
            json.dumps(
                {
                    "query_id": "Q-01",
                    "query": "Where is ghost chunk?",
                    "relevance_judgments": [{"chunk_id": "CHK-UNKNOWN", "grade": 2}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="references unknown corpus chunk 'CHK-UNKNOWN'"):
            load_benchmark_dataset(q_file, corpus_chunks=corpus)

    def test_load_benchmark_dataset_alternative_schema_fields(self, tmp_path: Path) -> None:
        """Verify loader handles 'text' instead of 'query' and 'relevance_grades' dict."""
        q_file = tmp_path / "alt_schema.jsonl"
        q_file.write_text(
            json.dumps(
                {
                    "query_id": "Q-ALT-1",
                    "text": "What is the alternative format query?",
                    "relevance_grades": {"CHK-1": 2, "CHK-2": 1, "CHK-3": 0},
                    "required_citations": ["CIT-01", "CIT-02"],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        queries = load_benchmark_dataset(q_file)
        assert len(queries) == 1
        q = queries[0]
        assert q.query_id == "Q-ALT-1"
        assert q.text == "What is the alternative format query?"
        assert set(q.relevant_chunk_ids) == {"CHK-1", "CHK-2"}
        assert q.get_grade("CHK-1") == 2
        assert q.get_grade("CHK-2") == 1
        assert q.get_grade("CHK-3") == 0


class TestLatencyPercentiles:
    """Unit tests for calculate_latency_percentiles."""

    def test_empty_latencies(self) -> None:
        stats = calculate_latency_percentiles([])
        assert stats["mean_latency_ms"] == 0.0
        assert stats["latency_p50_ms"] == 0.0
        assert stats["latency_p90_ms"] == 0.0
        assert stats["min_latency_ms"] == 0.0
        assert stats["max_latency_ms"] == 0.0

    def test_single_latency(self) -> None:
        stats = calculate_latency_percentiles([42.5])
        assert stats["mean_latency_ms"] == 42.5
        assert stats["latency_p50_ms"] == 42.5
        assert stats["latency_p90_ms"] == 42.5
        assert stats["latency_p95_ms"] == 42.5
        assert stats["latency_p99_ms"] == 42.5
        assert stats["min_latency_ms"] == 42.5
        assert stats["max_latency_ms"] == 42.5

    def test_multi_latency_distribution(self) -> None:
        latencies = [10.0, 20.0, 30.0, 40.0, 50.0]
        stats = calculate_latency_percentiles(latencies)
        assert stats["mean_latency_ms"] == 30.0
        assert stats["latency_p50_ms"] == 30.0
        assert stats["min_latency_ms"] == 10.0
        assert stats["max_latency_ms"] == 50.0
        assert stats["latency_p90_ms"] == 46.0


class TestRetrievalBenchmarkRunnerIntegration:
    """End-to-end integration tests using BM25Retriever over the golden benchmark."""

    def test_end_to_end_bm25_benchmark_run_with_file_paths(self) -> None:
        retriever = BM25Retriever(k1=1.5, b=0.75)
        runner = RetrievalBenchmarkRunner(
            retriever=retriever,
            top_k=5,
            experiment_name="bm25_golden_benchmark",
        )

        report = runner.run(corpus=CORPUS_PATH, benchmark=BENCHMARK_PATH)

        assert isinstance(report, EvaluationReport)
        assert report.experiment_name == "bm25_golden_benchmark"
        assert report.report_id
        assert report.timestamp

        # Check required metrics
        metrics = report.metrics
        for req_metric in [
            "recall@5",
            "mean_recall@5",
            "precision@5",
            "mean_precision@5",
            "hit_rate@5",
            "mean_hit_rate@5",
            "mrr@5",
            "mean_mrr@5",
            "ndcg@5",
            "mean_ndcg@5",
            "mean_latency_ms",
            "latency_p50_ms",
            "latency_p90_ms",
            "latency_p95_ms",
            "latency_p99_ms",
            "zero_hit_rate",
            "total_queries",
        ]:
            assert req_metric in metrics, f"Missing required aggregate metric '{req_metric}'"

        assert metrics["total_queries"] == 16.0
        assert 0.0 <= metrics["recall@5"] <= 1.0
        assert 0.0 <= metrics["precision@5"] <= 1.0
        assert 0.0 <= metrics["hit_rate@5"] <= 1.0
        assert 0.0 <= metrics["mrr@5"] <= 1.0
        assert 0.0 <= metrics["ndcg@5"] <= 1.0
        assert metrics["mean_latency_ms"] >= 0.0
        assert metrics["latency_p50_ms"] >= 0.0

        # BM25 baseline sanity bounds on this dataset:
        # At top_k=5 on 16 queries, recall and MRR must be positive non-trivial numbers
        assert metrics["hit_rate@5"] >= 0.5
        assert metrics["recall@5"] >= 0.3
        assert metrics["mrr@5"] >= 0.4
        assert metrics["ndcg@5"] >= 0.3

        # Check per-query results
        assert len(report.per_query_results) == 16
        for query_result in report.per_query_results:
            assert "query_id" in query_result
            assert "text" in query_result
            assert "latency_ms" in query_result
            assert "metrics" in query_result
            assert "retrieved" in query_result
            assert "failure_trace" in query_result

            q_metrics = query_result["metrics"]
            assert "recall@5" in q_metrics
            assert "mrr@5" in q_metrics
            assert "ndcg@5" in q_metrics

            retrieved = query_result["retrieved"]
            assert len(retrieved) <= 5
            for rank_idx, item in enumerate(retrieved, start=1):
                assert item["rank"] == rank_idx
                assert "chunk_id" in item
                assert "score" in item
                assert "grade" in item

            trace = query_result["failure_trace"]
            assert "has_failure" in trace
            assert "zero_hit" in trace
            assert "first_relevant_rank" in trace
            assert "missed_relevant_chunk_ids" in trace
            assert "missed_critical_chunk_ids" in trace

    def test_run_with_preloaded_chunks_and_queries(self) -> None:
        corpus = load_corpus(CORPUS_PATH)
        benchmark = load_benchmark_dataset(BENCHMARK_PATH, corpus_chunks=corpus)

        retriever = BM25Retriever()
        runner = BenchmarkRunner(retriever=retriever, top_k=5)

        report = runner.run(corpus=corpus, benchmark=benchmark, auto_index=True)
        assert len(report.per_query_results) == 16
        assert report.config["corpus_size"] == 25
        assert report.config["query_count"] == 16
        assert report.config["top_k"] == 5

    def test_run_with_pre_indexed_retriever(self) -> None:
        corpus = load_corpus(CORPUS_PATH)
        benchmark = load_benchmark_dataset(BENCHMARK_PATH, corpus_chunks=corpus)

        retriever = BM25Retriever()
        retriever.index(corpus)

        runner = RetrievalBenchmarkRunner(retriever=retriever, top_k=5)
        # auto_index=False skips re-indexing
        report = runner.run(corpus=corpus, benchmark=benchmark, auto_index=False)
        assert report.metrics["total_queries"] == 16.0
        assert report.metrics["hit_rate@5"] > 0.0

    def test_custom_metrics_and_top_k(self) -> None:
        retriever = BM25Retriever()
        custom_metrics = [RecallAtK(k=3), NDCGAtK(k=3), HitRateAtK(k=3)]
        runner = RetrievalBenchmarkRunner(
            retriever=retriever,
            metrics=custom_metrics,
            top_k=3,
            experiment_name="custom_eval_top3",
        )

        assert runner.top_k == 3
        assert len(runner.metrics) == 3
        assert runner.experiment_name == "custom_eval_top3"

        report = runner.run(corpus=CORPUS_PATH, benchmark=BENCHMARK_PATH)
        assert "recall@3" in report.metrics
        assert "ndcg@3" in report.metrics
        assert "hit_rate@3" in report.metrics
        assert "mrr@5" not in report.metrics

    def test_evaluation_report_serialization_round_trip(self) -> None:
        retriever = BM25Retriever()
        runner = RetrievalBenchmarkRunner(retriever=retriever, top_k=5)
        report = runner.run(corpus=CORPUS_PATH, benchmark=BENCHMARK_PATH)

        # To dict & from dict
        rep_dict = report.to_dict()
        reconstructed = EvaluationReport.from_dict(rep_dict)
        assert reconstructed.report_id == report.report_id
        assert reconstructed.experiment_name == report.experiment_name
        assert reconstructed.metrics["recall@5"] == pytest.approx(report.metrics["recall@5"])
        assert len(reconstructed.per_query_results) == len(report.per_query_results)

        # To JSON & from JSON
        rep_json = report.to_json()
        reconstructed_json = EvaluationReport.from_json(rep_json)
        assert reconstructed_json.report_id == report.report_id
        assert reconstructed_json.experiment_name == report.experiment_name
        assert reconstructed_json.metrics["ndcg@5"] == pytest.approx(report.metrics["ndcg@5"])


class MockZeroRetriever(BaseRetriever):
    """Retriever mock returning empty results for testing zero-hit failure diagnostics."""

    @property
    def name(self) -> str:
        return "mock_zero_retriever"

    def index(self, chunks: list[Chunk]) -> None:
        pass

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return []


class TestRunnerDiagnosticsAndFailureModes:
    """Tests for edge cases, failure traces, and input validations."""

    def test_zero_hit_failure_trace_diagnostics(self) -> None:
        zero_retriever = MockZeroRetriever()
        runner = RetrievalBenchmarkRunner(retriever=zero_retriever, top_k=5)

        corpus = [Chunk(chunk_id="C1", doc_id="D1", content="Text")]
        benchmark = [
            Query(
                query_id="Q1",
                text="Find something",
                relevant_chunk_ids=["C1"],
                relevance_grades={"C1": 2},
            )
        ]

        report = runner.run(corpus=corpus, benchmark=benchmark)
        assert report.metrics["zero_hit_rate"] == 1.0
        assert report.metrics["recall@5"] == 0.0
        assert report.metrics["mrr@5"] == 0.0

        q_res = report.per_query_results[0]
        trace = q_res["failure_trace"]
        assert trace["has_failure"] is True
        assert trace["zero_hit"] is True
        assert trace["first_relevant_rank"] is None
        assert trace["missed_relevant_chunk_ids"] == ["C1"]
        assert trace["missed_critical_chunk_ids"] == ["C1"]

    def test_runner_validation_errors(self) -> None:
        retriever = BM25Retriever()

        # Invalid top_k
        with pytest.raises(ValueError, match="positive integer"):
            RetrievalBenchmarkRunner(retriever=retriever, top_k=0)
        with pytest.raises(TypeError, match="integer"):
            RetrievalBenchmarkRunner(retriever=retriever, top_k=True)  # type: ignore[arg-type]

        # Invalid experiment_name
        with pytest.raises(ValueError, match="non-empty string"):
            RetrievalBenchmarkRunner(retriever=retriever, experiment_name="   ")

        # Invalid retriever
        with pytest.raises(TypeError, match="BaseRetriever"):
            RetrievalBenchmarkRunner(retriever="not_a_retriever")  # type: ignore[arg-type]

        # Empty corpus
        runner = RetrievalBenchmarkRunner(retriever=retriever)
        with pytest.raises(ValueError, match="Corpus chunk sequence must not be empty"):
            runner.run(corpus=[], benchmark=BENCHMARK_PATH)

        # Empty benchmark
        with pytest.raises(ValueError, match="Benchmark query sequence must not be empty"):
            runner.run(corpus=CORPUS_PATH, benchmark=[])
