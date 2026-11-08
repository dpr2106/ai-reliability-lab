"""Evaluation Execution Service coordinating pipeline configuration and runner execution."""

from __future__ import annotations

import concurrent.futures
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
from reliability_lab.core.models import Chunk, EvaluationReport
from reliability_lab.evaluation.citation_metrics import CitationEvaluator
from reliability_lab.evaluation.groundedness import GroundednessEvaluator
from reliability_lab.evaluation.nli_evaluator import NLIEntailmentEvaluator
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.indexing.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from reliability_lab.indexing.faiss_index import FaissVectorRetriever
from reliability_lab.indexing.hybrid import HybridRetriever
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import (
    BaseLLMClient,
    MockLLMClient,
    OllamaLLMClient,
    get_llm_client,
)
from reliability_lab.llm.judge import JudgeConfig, LLMJudge
from reliability_lab.pipeline.models import RAGPipelineConfig
from reliability_lab.pipeline.runner import RAGEvaluationRunner
from reliability_lab.storage.repository import ExperimentRepository

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentExecutionConfig:
    """Validated configuration for launching an evaluation run."""

    experiment_name: str
    dataset_id: str
    document_ids: list[str] = field(default_factory=list)
    retriever_type: str = "hybrid"  # "bm25", "faiss", "hybrid"
    retriever_params: dict[str, Any] = field(default_factory=dict)
    top_k: int = 5
    provider: str = "ollama"  # "ollama", "mock", "gemini"
    model_name: str | None = None
    temperature: float = 0.0
    max_output_tokens: int = 1024
    prompt_version: str = "v1"
    enable_groundedness: bool = True
    enable_citations: bool = True
    enable_nli: bool = True
    enable_judge: bool = True
    judge_provider: str | None = None
    judge_model: str | None = None


@dataclass
class EvaluationJob:
    """Asynchronous evaluation job state and progress tracker."""

    job_id: str
    experiment_name: str
    dataset_id: str
    status: str = "pending"  # "pending", "running", "completed", "failed"
    queries_completed: int = 0
    total_queries: int = 0
    run_id: str | None = None
    error_message: str | None = None
    created_at: str = field(default_factory=_utc_now_iso)
    completed_at: str | None = None

    @property
    def progress_percent(self) -> float:
        if self.total_queries <= 0:
            return 0.0 if self.status != "completed" else 100.0
        return round((self.queries_completed / self.total_queries) * 100.0, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "experiment_name": self.experiment_name,
            "dataset_id": self.dataset_id,
            "status": self.status,
            "queries_completed": self.queries_completed,
            "total_queries": self.total_queries,
            "progress_percent": self.progress_percent,
            "run_id": self.run_id,
            "error_message": self.error_message,
            "created_at": self.created_at,
            "completed_at": self.completed_at,
        }


class EvaluationExecutionService:
    """Coordinates corpus resolution, pipeline construction, and evaluation execution."""

    def __init__(
        self,
        experiment_repository: ExperimentRepository,
        document_repository: DocumentRepository,
        dataset_repository: BenchmarkDatasetRepository,
        ingestion_service: DocumentIngestionService | None = None,
        embedder: BaseEmbedder | None = None,
    ) -> None:
        self.experiment_repository = experiment_repository
        self.document_repository = document_repository
        self.dataset_repository = dataset_repository
        self.ingestion_service = ingestion_service
        self._custom_embedder = embedder

        self._jobs: dict[str, EvaluationJob] = {}
        self._jobs_lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=2)

    def _resolve_corpus(self, document_ids: list[str]) -> list[Chunk]:
        """Fetch all chunks for target document IDs or entire document repository."""
        chunks: list[Chunk] = []

        target_ids = list(document_ids)
        if not target_ids:
            # Query all documents in repository
            all_docs = self.document_repository.list_documents(limit=500)
            target_ids = [d.doc_id for d in all_docs]

        for doc_id in target_ids:
            doc_chunks = self.document_repository.get_chunks(doc_id)
            chunks.extend(doc_chunks)

        if not chunks:
            raise ValueError(
                "Selected document collection contains zero chunks. Please ingest documents."
            )
        return chunks

    def _resolve_embedder(self) -> BaseEmbedder:
        """Resolve sentence transformer or deterministic embedder."""
        if self._custom_embedder is not None:
            return self._custom_embedder
        if self.ingestion_service and hasattr(self.ingestion_service, "_embedder"):
            emb = getattr(self.ingestion_service, "_embedder", None)
            if isinstance(emb, BaseEmbedder):
                return emb
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            logger.warning("Falling back to mock embedder due to: %s", exc)
            from tests.unit.test_investigation_hardening import DeterministicEmbedder

            return DeterministicEmbedder()

    def _build_retriever(self, config: ExperimentExecutionConfig) -> Any:
        """Construct the requested retriever implementation."""
        rtype = config.retriever_type.lower().strip()
        params = config.retriever_params

        if rtype == "bm25":
            k1 = float(params.get("k1", 1.5))
            b = float(params.get("b", 0.75))
            return BM25Retriever(k1=k1, b=b)

        embedder = self._resolve_embedder()

        if rtype in ("faiss", "dense"):
            index_type = params.get("index_type", "flat")
            return FaissVectorRetriever(embedder=embedder, index_type=index_type)

        if rtype in ("hybrid", "hybrid_rrf"):
            k1 = float(params.get("k1", 1.5))
            b = float(params.get("b", 0.75))
            rrf_k = int(params.get("rrf_k", 60))
            sparse = BM25Retriever(k1=k1, b=b)
            dense = FaissVectorRetriever(embedder=embedder, index_type="flat")
            return HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=rrf_k)

        msg = (
            f"Unsupported retriever_type '{config.retriever_type}'."
            " Must be 'bm25', 'faiss', or 'hybrid'."
        )
        raise ValueError(msg)

    def _build_generator(self, config: ExperimentExecutionConfig) -> BaseLLMClient:
        """Construct LLM generator with clear provenance classification."""
        provider = config.provider.lower().strip()
        model_name = config.model_name

        if provider == "ollama":
            resolved_model = model_name or OllamaLLMClient.DEFAULT_MODEL
            client = OllamaLLMClient(model=resolved_model)
            try:
                resp = client._client.get(f"{client.base_url}/api/tags", timeout=2.0)
                if resp.status_code >= 500:
                    logger.warning(
                        "Ollama server returned status %d at %s", resp.status_code, client.base_url
                    )
            except Exception as exc:
                logger.info("Ollama reachability probe note: %s", exc)
            return client

        if provider == "mock":
            return MockLLMClient(model_name=model_name or "mock-model")

        if provider == "gemini":
            return get_llm_client(provider="gemini", model=model_name)

        raise ValueError(f"Unsupported LLM provider '{config.provider}'.")

    def execute_evaluation(
        self,
        config: ExperimentExecutionConfig,
        progress_callback: Any = None,
    ) -> EvaluationReport:
        """Synchronously execute evaluation pipeline with full reproducibility tracking."""
        # 1. Resolve dataset
        dataset = self.dataset_repository.get_dataset(config.dataset_id)
        if not dataset:
            raise ValueError(f"Benchmark dataset '{config.dataset_id}' not found.")
        if not dataset.queries:
            raise ValueError(f"Benchmark dataset '{config.dataset_id}' contains no queries.")

        # 2. Resolve document corpus
        target_doc_ids = config.document_ids or dataset.record.document_scope
        corpus_chunks = self._resolve_corpus(target_doc_ids)

        # 3. Construct retriever
        retriever = self._build_retriever(config)

        # 4. Construct generator and judge
        generator = self._build_generator(config)

        judge: LLMJudge | None = None
        if config.enable_judge:
            j_provider = config.judge_provider or config.provider
            j_model = config.judge_model or config.model_name
            try:
                j_client = get_llm_client(provider=j_provider, model=j_model)
                judge = LLMJudge(client=j_client, config=JudgeConfig())
            except Exception as exc:
                logger.warning("Could not initialize LLM judge: %s. Continuing without judge.", exc)
                judge = None

        # 5. Diagnostic evaluators
        groundedness = GroundednessEvaluator() if config.enable_groundedness else None
        citations = CitationEvaluator() if config.enable_citations else None

        nli: NLIEntailmentEvaluator | None = None
        if config.enable_nli:
            try:
                nli = NLIEntailmentEvaluator()
            except Exception as exc:
                logger.warning(
                    "NLI Evaluator unavailable (%s); continuing with lexical diagnostics.", exc
                )
                nli = None

        # 6. Pipeline configuration
        pipe_cfg = RAGPipelineConfig(
            top_k=config.top_k,
            retriever_name=config.retriever_type,
            prompt_version=config.prompt_version,
            temperature=config.temperature,
            max_output_tokens=config.max_output_tokens,
            enable_groundedness=config.enable_groundedness and groundedness is not None,
            enable_citations=config.enable_citations and citations is not None,
            enable_nli=config.enable_nli and nli is not None,
            enable_judge=config.enable_judge and judge is not None,
            experiment_name=config.experiment_name,
        )

        runner = RAGEvaluationRunner(
            retriever=retriever,
            generator=generator,
            judge=judge,
            groundedness_evaluator=groundedness,
            citation_evaluator=citations,
            nli_evaluator=nli,
            config=pipe_cfg,
        )

        # Determine provenance
        prov_type = (
            "mock"
            if isinstance(generator, MockLLMClient)
            else ("local" if isinstance(generator, OllamaLLMClient) else "cloud")
        )

        metadata = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "dataset_is_synthetic": dataset.record.is_synthetic,
            "document_ids": target_doc_ids,
            "corpus_chunk_count": len(corpus_chunks),
            "retriever_type": config.retriever_type,
            "retriever_params": config.retriever_params,
            "provider": config.provider,
            "model_name": config.model_name or getattr(generator, "model_name", "unknown"),
            "provenance_type": prov_type,
            "is_mock": prov_type == "mock",
            "mock_warning": prov_type == "mock",
            "prompt_version": config.prompt_version,
            "execution_timestamp": _utc_now_iso(),
        }

        # 7. Execute runner
        report = runner.run(
            corpus=corpus_chunks,
            benchmark=dataset.queries,
            auto_index=True,
            repository=self.experiment_repository,
            dataset_reference=dataset.dataset_id,
            metadata=metadata,
        )

        logger.info(
            "Evaluation experiment '%s' completed successfully (run_id: %s).",
            config.experiment_name,
            report.report_id,
        )
        return report

    def start_evaluation_job(self, config: ExperimentExecutionConfig) -> EvaluationJob:
        """Launch background evaluation job returning tracker."""
        job_id = f"job_{uuid4().hex[:12]}"
        dataset = self.dataset_repository.get_dataset(config.dataset_id)
        total_q = dataset.query_count if dataset else 0

        job = EvaluationJob(
            job_id=job_id,
            experiment_name=config.experiment_name,
            dataset_id=config.dataset_id,
            status="pending",
            total_queries=total_q,
        )

        with self._jobs_lock:
            self._jobs[job_id] = job

        def _worker() -> None:
            with self._jobs_lock:
                job.status = "running"
            try:
                report = self.execute_evaluation(config)
                with self._jobs_lock:
                    job.status = "completed"
                    job.run_id = report.report_id
                    job.queries_completed = job.total_queries
                    job.completed_at = _utc_now_iso()
            except Exception as exc:
                logger.exception("Evaluation job '%s' failed: %s", job_id, exc)
                with self._jobs_lock:
                    job.status = "failed"
                    job.error_message = str(exc)
                    job.completed_at = _utc_now_iso()

        self._executor.submit(_worker)
        return job

    def get_job(self, job_id: str) -> EvaluationJob | None:
        with self._jobs_lock:
            return self._jobs.get(job_id)

    def list_jobs(self) -> list[EvaluationJob]:
        with self._jobs_lock:
            return list(reversed(list(self._jobs.values())))
