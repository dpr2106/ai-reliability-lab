"""Evaluation Execution Service coordinating pipeline configuration and runner execution."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field, replace
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


def _dedup_list(items: list[str]) -> list[str]:
    seen: set[str] = set()
    res: list[str] = []
    for item in items:
        if item not in seen:
            seen.add(item)
            res.append(item)
    return res


def compute_dataset_content_hash(dataset: Any) -> str:
    """Compute deterministic SHA-256 digest over dataset queries and ground truth annotations."""
    records = []
    for q in sorted(dataset.queries, key=lambda x: str(x.query_id)):
        sorted_grades = sorted(
            [(str(k), int(v)) for k, v in q.relevance_grades.items()],
            key=lambda x: x[0],
        )
        sorted_chunks = sorted([str(cid) for cid in q.relevant_chunk_ids])
        sorted_cits = sorted([str(c) for c in q.required_citations])
        clean_meta = {
            k: v
            for k, v in sorted(q.metadata.items())
            if not k.startswith("_") and k not in ("created_at", "updated_at")
        }
        records.append(
            {
                "query_id": str(q.query_id),
                "text": str(q.text),
                "relevant_chunk_ids": sorted_chunks,
                "relevance_grades": sorted_grades,
                "ground_truth_answer": str(q.ground_truth_answer or ""),
                "required_citations": sorted_cits,
                "metadata": clean_meta,
            }
        )
    canonical_json = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def compute_corpus_fingerprints(
    corpus_chunks: list[Chunk],
    doc_repository: DocumentRepository,
    target_doc_ids: list[str],
) -> dict[str, dict[str, Any]]:
    """Compute deterministic fingerprints for corpus documents and their constituent chunks."""
    fingerprints: dict[str, dict[str, Any]] = {}
    for doc_id in sorted(target_doc_ids):
        doc = doc_repository.get_document(doc_id)
        doc_chunks = [
            c
            for c in corpus_chunks
            if getattr(c, "doc_id", None) == doc_id
            or getattr(c, "document_id", None) == doc_id
            or c.chunk_id.startswith(f"{doc_id}_")
        ]
        if not doc_chunks:
            doc_chunks = doc_repository.get_chunks(doc_id)

        chunk_digests: list[str] = []
        for c in sorted(doc_chunks, key=lambda x: str(x.chunk_id)):
            c_text = getattr(c, "content", None) or getattr(c, "text", "")
            c_str = f"{c.chunk_id}:{c_text}:{getattr(c, 'page_number', None)}"
            chunk_digests.append(c_str)
        chunks_hash = hashlib.sha256("\n".join(chunk_digests).encode("utf-8")).hexdigest()

        fingerprints[doc_id] = {
            "content_hash": getattr(doc, "content_hash", "") if doc else "",
            "chunk_count": len(doc_chunks),
            "chunks_hash": chunks_hash,
        }
    return fingerprints


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
        if self.status == "completed":
            return 100.0
        if self.total_queries <= 0:
            return 0.0
        pct = (self.queries_completed / self.total_queries) * 100.0
        return round(max(0.0, min(100.0, pct)), 1)

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
        """Fetch and strictly validate all chunks for target document IDs."""
        if not document_ids:
            raise ValueError("Document ID list cannot be empty for corpus resolution.")

        target_ids = _dedup_list(document_ids)

        chunks: list[Chunk] = []
        for doc_id in target_ids:
            doc = self.document_repository.get_document(doc_id)
            if doc is None:
                raise ValueError(f"Requested document '{doc_id}' not found in repository.")
            status_val = getattr(doc, "status", None)
            status_str = (
                status_val.value
                if (status_val is not None and hasattr(status_val, "value"))
                else str(status_val or "")
            )
            if status_str.upper() != "READY":
                raise ValueError(
                    f"Requested document '{doc_id}' is not in ready state "
                    f"(current status: '{status_str}')."
                )
            doc_chunks = self.document_repository.get_chunks(doc_id)
            chunks.extend(doc_chunks)

        if not chunks:
            raise ValueError(
                "Selected document collection contains zero chunks. Please ingest documents."
            )
        return chunks

    def _resolve_embedder(self) -> BaseEmbedder:
        """Resolve sentence transformer embedder or fail closed with actionable error."""
        if self._custom_embedder is not None:
            return self._custom_embedder
        if self.ingestion_service and hasattr(self.ingestion_service, "_embedder"):
            emb = getattr(self.ingestion_service, "_embedder", None)
            if isinstance(emb, BaseEmbedder):
                return emb
        try:
            return SentenceTransformerEmbedder()
        except Exception as exc:
            raise RuntimeError(
                f"Dense vector embedding engine is unavailable: {exc}. "
                "Install sentence-transformers and PyTorch, or inject an explicit "
                "BaseEmbedder instance via dependency injection."
            ) from exc

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

        # 2. Resolve document corpus & enforce scope divergence
        dataset_scope = list(dataset.record.document_scope or [])
        deduped_dataset_scope = _dedup_list(dataset_scope)

        if config.document_ids:
            requested_doc_ids = _dedup_list(config.document_ids)
            if deduped_dataset_scope:
                out_of_scope = [d for d in requested_doc_ids if d not in set(deduped_dataset_scope)]
                if out_of_scope:
                    raise ValueError(
                        "Experiment document selection contains documents outside dataset scope: "
                        f"{out_of_scope}."
                    )
            target_doc_ids = requested_doc_ids
        else:
            if not deduped_dataset_scope:
                raise ValueError(
                    "Evaluation requires an explicit document scope. Neither experiment "
                    "configuration nor benchmark dataset specifies document IDs."
                )
            target_doc_ids = deduped_dataset_scope

        corpus_chunks = self._resolve_corpus(target_doc_ids)

        # 3. Validate benchmark chunk references exist in selected corpus
        corpus_chunk_ids = {c.chunk_id for c in corpus_chunks}
        for q in dataset.queries:
            for cid in q.relevant_chunk_ids:
                if cid not in corpus_chunk_ids:
                    raise ValueError(
                        f"Benchmark query '{q.query_id}' references chunk '{cid}' "
                        f"which is missing from selected corpus."
                    )
            for cid in q.relevance_grades:
                if cid not in corpus_chunk_ids:
                    raise ValueError(
                        f"Benchmark query '{q.query_id}' has relevance grade for chunk '{cid}' "
                        f"which is missing from selected corpus."
                    )
            for cit in q.required_citations:
                mapped_cid = q.metadata.get("citation_to_chunk_id", {}).get(cit, cit)
                if mapped_cid not in corpus_chunk_ids and cit not in corpus_chunk_ids:
                    if cit in q.metadata.get("citation_to_chunk_id", {}):
                        raise ValueError(
                            f"Benchmark query '{q.query_id}' requires citation '{cit}' targeting "
                            f"chunk '{mapped_cid}' which is missing from selected corpus."
                        )

        # 4. Construct retriever
        retriever = self._build_retriever(config)

        # 5. Construct generator and judge
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

        # 6. Diagnostic evaluators
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

        # 7. Pipeline configuration
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

        ds_hash = compute_dataset_content_hash(dataset)
        corpus_fps = compute_corpus_fingerprints(
            corpus_chunks, self.document_repository, target_doc_ids
        )

        metadata = {
            "dataset_id": dataset.dataset_id,
            "dataset_name": dataset.name,
            "dataset_is_synthetic": dataset.record.is_synthetic,
            "dataset_content_hash": ds_hash,
            "corpus_fingerprints": corpus_fps,
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

        # 8. Execute runner
        report = runner.run(
            corpus=corpus_chunks,
            benchmark=dataset.queries,
            auto_index=True,
            repository=self.experiment_repository,
            dataset_reference=dataset.dataset_id,
            metadata=metadata,
            progress_callback=progress_callback,
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

        def _on_progress(completed: int, total: int) -> None:
            with self._jobs_lock:
                job.queries_completed = completed
                if total > 0:
                    job.total_queries = total

        def _worker() -> None:
            with self._jobs_lock:
                job.status = "running"
            try:
                report = self.execute_evaluation(config, progress_callback=_on_progress)
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
            job = self._jobs.get(job_id)
            if job is None:
                return None
            return replace(job)

    def list_jobs(self) -> list[EvaluationJob]:
        with self._jobs_lock:
            return [replace(j) for j in reversed(list(self._jobs.values()))]
