"""End-to-end RAG answer generation, multi-stage evaluation, and experiment persistence runner.

Coordinates retrieval, LLM answer generation, lexical groundedness, citation validation,
local NLI entailment, structured LLM judging, statistical correlation, and SQLite persistence.
Implemented using Python standard library interfaces per ADR-005 and ADR-011.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from reliability_lab.core.interfaces import BaseMetric, BaseRetriever
from reliability_lab.core.models import Chunk, EvaluationReport, Query, RetrievalResult
from reliability_lab.evaluation.benchmark_runner import (
    calculate_latency_percentiles,
    load_benchmark_dataset,
    load_corpus,
)
from reliability_lab.evaluation.citation_metrics import CitationEvaluator, CitationValidationResult
from reliability_lab.evaluation.groundedness import GroundednessEvaluator, GroundednessResult
from reliability_lab.evaluation.nli_evaluator import NLIEntailmentEvaluator, NLIResult
from reliability_lab.evaluation.retrieval_metrics import (
    HitRateAtK,
    MRRAtK,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
)
from reliability_lab.llm.client import BaseLLMClient
from reliability_lab.llm.judge import (
    CorrelationReport,
    JudgeConfig,
    JudgeResult,
    LLMJudge,
    calculate_llm_nli_correlation,
)
from reliability_lab.llm.models import LLMGenerationConfig, LLMMessage
from reliability_lab.pipeline.models import (
    GenerationProvenance,
    PerQueryRAGResult,
    RAGPipelineConfig,
    RAGPipelineExecutionResult,
)
from reliability_lab.pipeline.prompt import build_rag_prompt
from reliability_lab.storage.repository import ExperimentRepository

logger = logging.getLogger(__name__)


class RAGEvaluationRunner:
    """End-to-end orchestrator for RAG answer generation and multi-faceted evaluation."""

    def __init__(
        self,
        retriever: BaseRetriever,
        generator: BaseLLMClient,
        judge: LLMJudge | None = None,
        groundedness_evaluator: GroundednessEvaluator | None = None,
        citation_evaluator: CitationEvaluator | None = None,
        nli_evaluator: NLIEntailmentEvaluator | None = None,
        config: RAGPipelineConfig | None = None,
        metrics: Sequence[BaseMetric] | None = None,
    ) -> None:
        """Initialize the RAG evaluation runner with components and configuration.

        Args:
            retriever: Component implementing BaseRetriever (BM25, FAISS, Hybrid, etc.).
            generator: LLM client implementing BaseLLMClient for answer generation.
            judge: Optional LLMJudge instance for factual judging.
            groundedness_evaluator: Optional GroundednessEvaluator instance.
            citation_evaluator: Optional CitationEvaluator instance.
            nli_evaluator: Optional NLIEntailmentEvaluator instance.
            config: Optional RAGPipelineConfig tuning hyperparameters and stage flags.
            metrics: Optional sequence of BaseMetric instances for retrieval evaluation.

        Raises:
            TypeError: If retriever or generator do not conform to expected interfaces.
        """
        if not isinstance(retriever, BaseRetriever):
            raise TypeError(
                f"retriever must implement BaseRetriever, got: {type(retriever).__name__}"
            )
        if not isinstance(generator, BaseLLMClient):
            raise TypeError(
                f"generator must implement BaseLLMClient, got: {type(generator).__name__}"
            )

        self._retriever = retriever
        self._generator = generator
        self._config = config or RAGPipelineConfig()

        # Initialize Groundedness Evaluator
        if self._config.enable_groundedness:
            if groundedness_evaluator is not None:
                if not isinstance(groundedness_evaluator, GroundednessEvaluator):
                    raise TypeError("groundedness_evaluator must be a GroundednessEvaluator.")
                self._groundedness_evaluator: GroundednessEvaluator | None = groundedness_evaluator
            else:
                self._groundedness_evaluator = GroundednessEvaluator(
                    containment_threshold=self._config.groundedness_containment_threshold,
                    lcs_threshold=self._config.groundedness_lcs_threshold,
                )
        else:
            self._groundedness_evaluator = None

        # Initialize Citation Evaluator
        if self._config.enable_citations:
            if citation_evaluator is not None:
                if not isinstance(citation_evaluator, CitationEvaluator):
                    raise TypeError("citation_evaluator must be a CitationEvaluator.")
                self._citation_evaluator: CitationEvaluator | None = citation_evaluator
            else:
                self._citation_evaluator = CitationEvaluator()
        else:
            self._citation_evaluator = None

        # Initialize LLM Judge
        if self._config.enable_judge:
            if judge is not None:
                if not isinstance(judge, LLMJudge):
                    raise TypeError("judge must be an LLMJudge instance.")
                self._judge: LLMJudge | None = judge
            else:
                judge_cfg = JudgeConfig(
                    model=self._generator.model_name,
                    prompt_version=self._config.prompt_version,
                )
                self._judge = LLMJudge(client=self._generator, config=judge_cfg)
        else:
            self._judge = None

        # Initialize NLI Evaluator
        if self._config.enable_nli:
            if nli_evaluator is not None:
                if not isinstance(nli_evaluator, NLIEntailmentEvaluator):
                    raise TypeError("nli_evaluator must be an NLIEntailmentEvaluator.")
                self._nli_evaluator: NLIEntailmentEvaluator | None = nli_evaluator
            else:
                try:
                    import torch  # noqa: F401
                    import transformers  # noqa: F401

                    self._nli_evaluator = NLIEntailmentEvaluator()
                except ImportError as exc:
                    logger.warning(
                        "NLI dependencies not installed (%s); disabling NLI evaluation stage.",
                        exc,
                    )
                    self._nli_evaluator = None
        else:
            self._nli_evaluator = None

        # Configure retrieval metrics
        top_k = self._config.top_k
        if metrics is None:
            self._retrieval_metrics: list[BaseMetric] = [
                RecallAtK(k=top_k),
                PrecisionAtK(k=top_k),
                HitRateAtK(k=top_k),
                MRRAtK(k=top_k),
                NDCGAtK(k=top_k),
            ]
        else:
            self._retrieval_metrics = list(metrics)

    @property
    def retriever(self) -> BaseRetriever:
        """Return the underlying retriever instance."""
        return self._retriever

    @property
    def generator(self) -> BaseLLMClient:
        """Return the underlying answer generation LLM client."""
        return self._generator

    @property
    def judge(self) -> LLMJudge | None:
        """Return the LLM judge instance if enabled."""
        return self._judge

    @property
    def groundedness_evaluator(self) -> GroundednessEvaluator | None:
        """Return the groundedness evaluator instance if enabled."""
        return self._groundedness_evaluator

    @property
    def citation_evaluator(self) -> CitationEvaluator | None:
        """Return the citation evaluator instance if enabled."""
        return self._citation_evaluator

    @property
    def nli_evaluator(self) -> NLIEntailmentEvaluator | None:
        """Return the NLI evaluator instance if enabled."""
        return self._nli_evaluator

    @property
    def config(self) -> RAGPipelineConfig:
        """Return the pipeline configuration."""
        return self._config

    @property
    def is_mock(self) -> bool:
        """Indicate whether any generative stage in the pipeline uses a mock provider."""
        generator_mock = bool(getattr(self._generator, "is_mock", False))
        judge_mock = bool(self._judge.is_mock) if self._judge is not None else False
        return generator_mock or judge_mock

    def evaluate_query(
        self,
        query: Query,
        retrieved: Sequence[RetrievalResult] | None = None,
    ) -> PerQueryRAGResult:
        """Execute the full RAG vertical slice for a single benchmark query.

        Args:
            query: Validated Query domain model instance.
            retrieved: Optional pre-retrieved candidates. If None, retrieved via self.retriever.

        Returns:
            Populated PerQueryRAGResult containing all intermediate and final outputs.
        """
        if not isinstance(query, Query):
            raise TypeError(f"query must be a Query instance, got: {type(query).__name__}")

        t_start = time.perf_counter()

        # 1. Stage A: Retrieval
        if retrieved is None:
            retrieved_candidates = self._retriever.retrieve(query.text, top_k=self._config.top_k)
        else:
            retrieved_candidates = list(retrieved)

        # Compute retrieval metrics
        retrieval_metrics: dict[str, float] = {}
        for m in self._retrieval_metrics:
            score = m.compute(retrieved_candidates, query)
            retrieval_metrics[m.name] = float(score)

        # 2. Stage B: Answer Generation
        sys_prompt, user_prompt = build_rag_prompt(
            query_text=query.text,
            evidence=retrieved_candidates,
            prompt_version=self._config.prompt_version,
        )

        messages = [
            LLMMessage(role="system", content=sys_prompt),
            LLMMessage(role="user", content=user_prompt),
        ]
        gen_cfg = LLMGenerationConfig(
            temperature=self._config.temperature,
            max_output_tokens=self._config.max_output_tokens,
        )

        t_gen_start = time.perf_counter()
        resp = self._generator.generate(prompt=messages, config=gen_cfg)
        gen_latency_ms = (time.perf_counter() - t_gen_start) * 1000.0

        generated_answer = resp.text
        prompt_tokens = resp.usage.prompt_tokens if resp.usage is not None else 0
        completion_tokens = resp.usage.completion_tokens if resp.usage is not None else 0
        total_tokens = resp.usage.total_tokens if resp.usage is not None else 0

        gen_provider = getattr(
            self._generator, "provider", "mock" if self._generator.is_mock else "live"
        )
        provenance = GenerationProvenance(
            provider=gen_provider,
            model=self._generator.model_name,
            is_mock=self._generator.is_mock,
            prompt_version=self._config.prompt_version,
            latency_ms=gen_latency_ms,
            token_usage={
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        )

        # Query metrics dictionary to collect all scalar metrics
        query_metrics: dict[str, float] = dict(retrieval_metrics)

        # 3. Stage C: Groundedness Evaluation
        groundedness_dict: dict[str, Any] | None = None
        if self._groundedness_evaluator is not None:
            ground_res: GroundednessResult = self._groundedness_evaluator.evaluate(
                answer_text=generated_answer,
                retrieved=retrieved_candidates,
            )
            groundedness_dict = ground_res.to_dict()
            query_metrics["lexical_groundedness"] = float(ground_res.aggregate_score)

        # 4. Stage C: Citation Evaluation
        citation_dict: dict[str, Any] | None = None
        if self._citation_evaluator is not None:
            cit_res: CitationValidationResult = self._citation_evaluator.evaluate(
                answer_text=generated_answer,
                retrieved=retrieved_candidates,
                query=query,
            )
            citation_dict = cit_res.to_dict()
            query_metrics["citation_precision"] = float(cit_res.citation_precision)
            query_metrics["citation_recall"] = float(cit_res.citation_recall)

        # 5. Stage C: Local NLI Entailment
        nli_dict: dict[str, Any] | None = None
        if self._nli_evaluator is not None:
            try:
                nli_res: NLIResult = self._nli_evaluator.evaluate(
                    answer_text=generated_answer,
                    retrieved=retrieved_candidates,
                )
                nli_dict = nli_res.to_dict()
                query_metrics["nli_hallucination_score"] = float(nli_res.hallucination_score)
            except Exception as exc:
                logger.warning(
                    "Local NLI evaluation failed (%s); continuing without NLI metrics for query.",
                    exc,
                )

        # 6. Stage C: Structured LLM Judge
        judge_dict: dict[str, Any] | None = None
        if self._judge is not None:
            j_res: JudgeResult = self._judge.evaluate(
                query=query,
                answer=generated_answer,
                evidence=retrieved_candidates,
            )
            judge_dict = j_res.to_dict()
            query_metrics["llm_faithfulness"] = float(j_res.overall_faithfulness_score)
            query_metrics["llm_unsupported_risk"] = float(j_res.overall_unsupported_risk)

        total_query_latency_ms = (time.perf_counter() - t_start) * 1000.0

        # Format retrieved candidates for query payload
        retrieved_payload = [
            {
                "chunk_id": res.chunk.chunk_id,
                "doc_id": res.chunk.doc_id,
                "rank": res.rank,
                "score": float(res.score),
                "grade": query.get_grade(res.chunk.chunk_id),
            }
            for res in retrieved_candidates
        ]

        return PerQueryRAGResult(
            query_id=query.query_id,
            text=query.text,
            retrieved=retrieved_payload,
            retrieval_metrics=retrieval_metrics,
            generated_answer=generated_answer,
            generation_provenance=provenance,
            groundedness=groundedness_dict,
            citation_validation=citation_dict,
            nli_result=nli_dict,
            judge_result=judge_dict,
            latency_ms=total_query_latency_ms,
            metrics=query_metrics,
        )

    def run_pipeline(
        self,
        corpus: Sequence[Chunk] | str | Path,
        benchmark: Sequence[Query] | str | Path,
        auto_index: bool = True,
        repository: ExperimentRepository | None = None,
        dataset_reference: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> RAGPipelineExecutionResult:
        """Execute the end-to-end RAG pipeline returning full execution details."""
        # 1. Resolve corpus
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

        # 2. Resolve benchmark queries
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

        # 3. Index corpus if requested
        if auto_index:
            self._retriever.index(corpus_chunks)

        # 4. Evaluate queries
        per_query_results: list[PerQueryRAGResult] = []
        metric_accumulators: dict[str, list[float]] = {}
        total_latencies: list[float] = []
        gen_latencies: list[float] = []
        prompt_tokens_total = 0
        completion_tokens_total = 0

        judge_results_list: list[JudgeResult] = []
        nli_results_list: list[NLIResult] = []

        for query in benchmark_queries:
            q_res = self.evaluate_query(query)
            per_query_results.append(q_res)

            total_latencies.append(q_res.latency_ms)
            gen_latencies.append(q_res.generation_provenance.latency_ms)
            prompt_tokens_total += q_res.generation_provenance.token_usage.get("prompt_tokens", 0)
            completion_tokens_total += q_res.generation_provenance.token_usage.get(
                "completion_tokens", 0
            )

            # Accumulate metrics
            for m_name, score in q_res.metrics.items():
                if m_name not in metric_accumulators:
                    metric_accumulators[m_name] = []
                metric_accumulators[m_name].append(score)

            # Collect judge and nli results for correlation if present
            if q_res.judge_result is not None:
                # Reconstruct minimum JudgeResult container for correlation
                claims_from_dict = []
                for c in q_res.judge_result.get("claims", []):
                    from reliability_lab.llm.judge import ClaimJudgeResult, JudgeVerdict

                    raw_span = c.get("span", [0, 0])
                    span_tuple = (int(raw_span[0]), int(raw_span[1]))
                    claims_from_dict.append(
                        ClaimJudgeResult(
                            claim_id=c.get("claim_id", 0),
                            claim_text=c.get("claim_text", ""),
                            span=span_tuple,
                            verdict=JudgeVerdict(c.get("verdict", "UNSUPPORTED")),
                            faithfulness_score=float(c.get("faithfulness_score", 0.0)),
                            unsupported_risk=float(c.get("unsupported_risk", 1.0)),
                            cited_chunk_ids=list(c.get("cited_chunk_ids", [])),
                            reasoning=str(c.get("reasoning", "")),
                        )
                    )
                judge_results_list.append(
                    JudgeResult(
                        query=q_res.text,
                        answer=q_res.generated_answer,
                        overall_faithfulness_score=float(
                            q_res.judge_result.get("overall_faithfulness_score", 0.0)
                        ),
                        overall_unsupported_risk=float(
                            q_res.judge_result.get("overall_unsupported_risk", 1.0)
                        ),
                        total_claims=len(claims_from_dict),
                        claims=claims_from_dict,
                        verdict_counts=dict(q_res.judge_result.get("verdict_counts", {})),
                        is_mock=bool(q_res.judge_result.get("is_mock", False)),
                        model_name=str(q_res.judge_result.get("model_name", "")),
                        prompt_version=str(q_res.judge_result.get("prompt_version", "")),
                    )
                )

            if q_res.nli_result is not None:
                claims_nli = []
                for c in q_res.nli_result.get("claims", []):
                    from reliability_lab.evaluation.nli_evaluator import ClaimNLIResult

                    raw_span_nli = c.get("span", [0, 0])
                    span_tuple_nli = (int(raw_span_nli[0]), int(raw_span_nli[1]))
                    claims_nli.append(
                        ClaimNLIResult(
                            claim_text=c.get("claim_text", ""),
                            span=span_tuple_nli,
                            best_chunk_id=c.get("best_chunk_id"),
                            entailment_prob=float(c.get("entailment_prob", 0.0)),
                            neutral_prob=float(c.get("neutral_prob", 0.0)),
                            contradiction_prob=float(c.get("contradiction_prob", 0.0)),
                            unsupported_risk=float(c.get("unsupported_risk", 1.0)),
                            predicted_label=None,
                            has_evidence=bool(c.get("has_evidence", False)),
                            candidate_scores={},
                            diagnostics={},
                        )
                    )
                nli_results_list.append(
                    NLIResult(
                        hallucination_score=float(q_res.nli_result.get("hallucination_score", 0.0)),
                        total_claims=len(claims_nli),
                        has_claims=bool(q_res.nli_result.get("has_claims", False)),
                        claims=claims_nli,
                        entailed_claims_count=int(q_res.nli_result.get("entailed_claims_count", 0)),
                        neutral_claims_count=int(q_res.nli_result.get("neutral_claims_count", 0)),
                        contradicted_claims_count=int(
                            q_res.nli_result.get("contradicted_claims_count", 0)
                        ),
                        no_evidence_claims_count=int(
                            q_res.nli_result.get("no_evidence_claims_count", 0)
                        ),
                        retrieved_chunk_ids=list(q_res.nli_result.get("retrieved_chunk_ids", [])),
                        model_name=str(q_res.nli_result.get("model_name", "")),
                        diagnostics={},
                    )
                )

        # 5. Compute aggregate metrics
        total_queries = len(benchmark_queries)
        aggregate_metrics: dict[str, float] = {}

        for m_name, scores in metric_accumulators.items():
            mean_score = sum(scores) / float(len(scores)) if scores else 0.0
            aggregate_metrics[m_name] = float(mean_score)
            aggregate_metrics[f"mean_{m_name}"] = float(mean_score)

        # Add latency percentiles
        lat_stats = calculate_latency_percentiles(total_latencies)
        for k, v in lat_stats.items():
            aggregate_metrics[k] = float(v)

        if gen_latencies:
            aggregate_metrics["mean_generation_latency_ms"] = float(
                sum(gen_latencies) / len(gen_latencies)
            )

        aggregate_metrics["total_prompt_tokens"] = float(prompt_tokens_total)
        aggregate_metrics["total_completion_tokens"] = float(completion_tokens_total)
        aggregate_metrics["total_tokens"] = float(prompt_tokens_total + completion_tokens_total)
        aggregate_metrics["total_queries"] = float(total_queries)

        # 6. Statistical correlation between LLM judge and NLI
        correlation_report: CorrelationReport | None = None
        if judge_results_list and nli_results_list:
            correlation_report = calculate_llm_nli_correlation(judge_results_list, nli_results_list)
            if (
                correlation_report.is_statistically_valid
                and correlation_report.pearson_r is not None
            ):
                aggregate_metrics["pearson_correlation_llm_nli"] = float(
                    correlation_report.pearson_r
                )
            if (
                correlation_report.is_statistically_valid
                and correlation_report.spearman_rho is not None
            ):
                aggregate_metrics["spearman_correlation_llm_nli"] = float(
                    correlation_report.spearman_rho
                )

        # 7. Construct final config and metadata
        pipeline_config: dict[str, Any] = {
            "pipeline": "RAGEvaluationRunner",
            "is_mock": self.is_mock,
            "retriever": self._retriever.name,
            "top_k": self._config.top_k,
            "generator_provider": getattr(
                self._generator, "provider", "mock" if self._generator.is_mock else "live"
            ),
            "generator_model": self._generator.model_name,
            "judge_provider": (
                getattr(
                    self._judge.client,
                    "provider",
                    "mock" if (self._judge and self._judge.is_mock) else "live",
                )
                if self._judge
                else None
            ),
            "judge_model": self._judge.client.model_name if self._judge else None,
            "judge_is_mock": self._judge.is_mock if self._judge else None,
            "nli_enabled": self._nli_evaluator is not None,
            "judge_enabled": self._judge is not None,
            "groundedness_enabled": self._groundedness_evaluator is not None,
            "citations_enabled": self._citation_evaluator is not None,
            "prompt_version": self._config.prompt_version,
            "corpus_size": len(corpus_chunks),
            "query_count": total_queries,
        }
        if correlation_report is not None:
            pipeline_config["correlation_valid"] = correlation_report.is_statistically_valid
            if correlation_report.warning:
                pipeline_config["correlation_warning"] = correlation_report.warning

        report_per_query = [q.to_dict() for q in per_query_results]

        report = EvaluationReport(
            experiment_name=self._config.experiment_name,
            metrics=aggregate_metrics,
            per_query_results=report_per_query,
            config=pipeline_config,
        )

        # 8. Persist if repository supplied
        if repository is not None:
            run_metadata = dict(metadata or {})
            run_metadata["is_mock"] = self.is_mock
            run_metadata["llm_provider"] = "mock" if self._generator.is_mock else "live"
            run_metadata["generator_model"] = self._generator.model_name
            if self._judge:
                run_metadata["judge_model"] = self._judge.client.model_name

            repository.save_report(
                report=report,
                index_config=pipeline_config,
                dataset_reference=str(dataset_reference) if dataset_reference else None,
                metadata=run_metadata,
            )

        return RAGPipelineExecutionResult(
            report=report,
            correlation=correlation_report,
            is_mock=self.is_mock,
            total_queries=total_queries,
            queries_evaluated=per_query_results,
            metrics=aggregate_metrics,
            config=pipeline_config,
        )

    def run(
        self,
        corpus: Sequence[Chunk] | str | Path,
        benchmark: Sequence[Query] | str | Path,
        auto_index: bool = True,
        repository: ExperimentRepository | None = None,
        dataset_reference: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> EvaluationReport:
        """Execute the end-to-end RAG benchmark evaluation run.

        Conforms to the standard BenchmarkRunner interface returning EvaluationReport.
        """
        result = self.run_pipeline(
            corpus=corpus,
            benchmark=benchmark,
            auto_index=auto_index,
            repository=repository,
            dataset_reference=dataset_reference,
            metadata=metadata,
        )
        return result.report
