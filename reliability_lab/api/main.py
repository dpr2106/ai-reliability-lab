"""FastAPI backend adapter exposing experiment storage, regression, and diagnostics.

Reuses ExperimentRepository, RegressionDetector, and CitationEvaluator without
duplicating SQLite persistence or metric calculations.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Annotated, Any

import anyio
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Request,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from reliability_lab.api.middleware import RequestBodyLimitMiddleware
from reliability_lab.api.privacy import sanitize_path_message
from reliability_lab.api.schemas import (
    BenchmarkQuerySchema,
    ChunkSchema,
    CitationAuditQueryItem,
    CitationAuditResponse,
    CompareRequest,
    CreateDatasetRequest,
    CreateInvestigationRequest,
    CreateQueryRequest,
    CreateSessionRequest,
    DatasetDetailResponse,
    DatasetListResponse,
    DatasetSummarySchema,
    DocumentChunkListResponse,
    DocumentDeleteResponse,
    DocumentDetailResponse,
    DocumentListResponse,
    DocumentSummarySchema,
    DocumentUploadResponse,
    EvaluationJobListResponse,
    EvaluationJobResponse,
    HealthResponse,
    ImportDatasetRequest,
    InvestigationDeleteResponse,
    InvestigationDetailResponse,
    InvestigationListResponse,
    InvestigationReportSchema,
    InvestigationSummarySchema,
    LaunchEvaluationRequest,
    MalformedCitationSchema,
    MessageResponse,
    MetricDefinitionSchema,
    MetricsMetaResponse,
    QueryRequest,
    QueryResponse,
    ResolvedCitationSchema,
    RunDetailResponse,
    RunListResponse,
    RunReproducibilityResponse,
    RunSummarySchema,
    SessionDeleteResponse,
    SessionDetailResponse,
    SessionListResponse,
    SessionQueryRequest,
    SessionQueryResponse,
    SessionSummarySchema,
    UnresolvedCitationSchema,
    UpdateDatasetRequest,
    UpdateQueryRequest,
)
from reliability_lab.benchmark import (
    BenchmarkDatasetRepository,
    BenchmarkDatasetService,
    DatasetValidationError,
)
from reliability_lab.core.models import Chunk, RetrievalResult
from reliability_lab.core.models import Query as BenchmarkQuery
from reliability_lab.evaluation.citation_metrics import CitationEvaluator, CitationParser
from reliability_lab.evaluation.execution_service import (
    EvaluationExecutionService,
    ExperimentExecutionConfig,
)
from reliability_lab.evaluation.metrics_meta import (
    METRIC_DEFINITIONS,
    classify_query_failure,
)
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import (
    DocumentDeletionError,
    DocumentIngestionService,
    EmptyFileError,
    IngestionError,
    InvalidDocumentFormatError,
    UploadSizeLimitExceededError,
)
from reliability_lab.investigation import (
    InvestigationRepository,
    InvestigationService,
)
from reliability_lab.query import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentQueryService,
    QueryStreamEvent,
    UnprocessableDocumentError,
)
from reliability_lab.session import (
    SessionNotFoundError,
    SessionRepository,
    SessionService,
)
from reliability_lab.storage.regression import (
    IncomparableRunsError,
    InvalidToleranceError,
    RegressionDetector,
)
from reliability_lab.storage.repository import (
    ExperimentRepository,
    StorageError,
    StoredRun,
)

logger = logging.getLogger(__name__)


def get_db_path() -> Path:
    """Return configured database path from environment or default."""
    raw = os.environ.get("RELIABILITY_LAB_DB", ".reliability-lab/experiments.sqlite3")
    return Path(raw).resolve()


def get_repository() -> ExperimentRepository:
    """FastAPI dependency providing an active ExperimentRepository."""
    db_path = get_db_path()
    return ExperimentRepository(db_path)


def get_document_repository() -> DocumentRepository:
    """FastAPI dependency providing an active DocumentRepository."""
    db_path = get_db_path()
    return DocumentRepository(db_path)


def get_ingestion_service() -> DocumentIngestionService:
    """FastAPI dependency providing an active DocumentIngestionService."""
    repo = get_document_repository()
    return DocumentIngestionService(repository=repo)


def get_query_service(
    repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> DocumentQueryService:
    """FastAPI dependency providing DocumentQueryService."""
    return DocumentQueryService(repository=repo, ingestion_service=ingestion)


def get_session_repository() -> SessionRepository:
    """FastAPI dependency providing an active SessionRepository."""
    db_path = get_db_path()
    return SessionRepository(db_path)


def get_session_service(
    session_repo: Annotated[SessionRepository, Depends(get_session_repository)],
    query_service: Annotated[DocumentQueryService, Depends(get_query_service)],
) -> SessionService:
    """FastAPI dependency providing an active SessionService."""
    return SessionService(session_repository=session_repo, query_service=query_service)


RepositoryDep = Annotated[ExperimentRepository, Depends(get_repository)]
DocRepoDep = Annotated[DocumentRepository, Depends(get_document_repository)]
IngestionServiceDep = Annotated[DocumentIngestionService, Depends(get_ingestion_service)]
QueryServiceDep = Annotated[DocumentQueryService, Depends(get_query_service)]
SessionRepoDep = Annotated[SessionRepository, Depends(get_session_repository)]
SessionServiceDep = Annotated[SessionService, Depends(get_session_service)]


def get_investigation_repository() -> InvestigationRepository:
    """FastAPI dependency providing an active InvestigationRepository."""
    db_path = get_db_path()
    return InvestigationRepository(db_path)


def get_investigation_service(
    investigation_repo: Annotated[InvestigationRepository, Depends(get_investigation_repository)],
    doc_repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    query_service: Annotated[DocumentQueryService, Depends(get_query_service)],
) -> InvestigationService:
    """FastAPI dependency providing an active InvestigationService."""
    return InvestigationService(
        investigation_repository=investigation_repo,
        document_repository=doc_repo,
        query_service=query_service,
    )


InvestigationRepoDep = Annotated[InvestigationRepository, Depends(get_investigation_repository)]
InvestigationServiceDep = Annotated[InvestigationService, Depends(get_investigation_service)]

_shared_eval_service: EvaluationExecutionService | None = None
_eval_service_lock = threading.Lock()


def get_dataset_repository() -> BenchmarkDatasetRepository:
    """FastAPI dependency providing an active BenchmarkDatasetRepository."""
    db_path = get_db_path()
    return BenchmarkDatasetRepository(db_path)


def get_dataset_service(
    dataset_repo: Annotated[BenchmarkDatasetRepository, Depends(get_dataset_repository)],
    doc_repo: Annotated[DocumentRepository, Depends(get_document_repository)],
) -> BenchmarkDatasetService:
    """FastAPI dependency providing an active BenchmarkDatasetService."""
    return BenchmarkDatasetService(
        dataset_repository=dataset_repo,
        document_repository=doc_repo,
    )


def get_evaluation_service(
    repo: Annotated[ExperimentRepository, Depends(get_repository)],
    doc_repo: Annotated[DocumentRepository, Depends(get_document_repository)],
    dataset_repo: Annotated[BenchmarkDatasetRepository, Depends(get_dataset_repository)],
    ingestion: Annotated[DocumentIngestionService, Depends(get_ingestion_service)],
) -> EvaluationExecutionService:
    """FastAPI dependency providing EvaluationExecutionService."""
    global _shared_eval_service
    with _eval_service_lock:
        if _shared_eval_service is None:
            _shared_eval_service = EvaluationExecutionService(
                experiment_repository=repo,
                document_repository=doc_repo,
                dataset_repository=dataset_repo,
                ingestion_service=ingestion,
            )
        else:
            _shared_eval_service.experiment_repository = repo
            _shared_eval_service.document_repository = doc_repo
            _shared_eval_service.dataset_repository = dataset_repo
            _shared_eval_service.ingestion_service = ingestion
        return _shared_eval_service


DatasetRepoDep = Annotated[BenchmarkDatasetRepository, Depends(get_dataset_repository)]
DatasetServiceDep = Annotated[BenchmarkDatasetService, Depends(get_dataset_service)]
EvaluationServiceDep = Annotated[EvaluationExecutionService, Depends(get_evaluation_service)]


def _count_runs(repo: ExperimentRepository, experiment_name: str | None = None) -> int:
    """Count runs in repository safely via database connection."""
    with repo.database.get_connection() as conn:
        if experiment_name:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM experiment_runs WHERE experiment_name = ?",
                (experiment_name,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as cnt FROM experiment_runs").fetchone()
        return int(row["cnt"]) if row else 0


def format_sse_event(event: str, data: dict[str, Any] | str) -> str:
    """Format a typed Server-Sent Event frame conforming to W3C SSE standard."""
    if isinstance(data, dict):
        payload = json.dumps(data, ensure_ascii=False)
    else:
        payload = str(data)
    return f"event: {event}\ndata: {payload}\n\n"


def serialize_stream_complete_payload(data: dict[str, Any]) -> dict[str, Any]:
    """Serialize the completed query/trace payload matching the REST schema."""
    trace = data["trace"]
    trace_dict = trace.to_dict() if hasattr(trace, "to_dict") else trace
    payload: dict[str, Any] = {
        "answer": data["answer"],
        "query_text": trace.query_text,
        "selected_doc_ids": list(trace.selected_doc_ids),
        "strategy": trace.requested_strategy,
        "effective_strategy": trace.effective_strategy,
        "fallback_reason": trace.fallback_reason,
        "abstained": trace.abstained,
        "retrieval_abstained": trace.retrieval_abstained,
        "generation_abstained": trace.generation_abstained,
        "abstention_reason": (
            trace.retrieval_abstention_reason or trace.generation_abstention_reason
        ),
        "resolved_citations": [c.to_dict() for c in trace.resolved_citations],
        "unresolved_citations": [c.to_dict() for c in trace.unresolved_citations],
        "malformed_citations": [c.to_dict() for c in trace.malformed_citations],
        "trace": trace_dict,
        "latency_ms": trace.total_latency_ms,
    }
    if "session_id" in data:
        payload["session_id"] = data["session_id"]
    if "user_message_id" in data:
        payload["user_message_id"] = data["user_message_id"]
    if "assistant_message_id" in data:
        payload["assistant_message_id"] = data["assistant_message_id"]
    return payload


_STREAM_DONE = object()


async def sse_stream_generator(
    stream_iter: Iterator[QueryStreamEvent], raw_request: Request
) -> AsyncIterator[str]:
    """Asynchronously yield Server-Sent Events with disconnect checks and sanitized errors."""
    try:
        while True:
            if await raw_request.is_disconnected():
                logger.info("Client disconnected; aborting SSE stream")
                break

            event = await anyio.to_thread.run_sync(next, stream_iter, _STREAM_DONE)
            if event is _STREAM_DONE or not isinstance(event, QueryStreamEvent):
                break

            if await raw_request.is_disconnected():
                logger.info("Client disconnected; aborting SSE stream")
                break

            if event.event == "complete":
                complete_payload = serialize_stream_complete_payload(event.data)
                yield format_sse_event("complete", complete_payload)
            else:
                yield format_sse_event(event.event, event.data)
    except Exception as exc:
        logger.error("Error during streaming SSE response: %s", exc, exc_info=True)
        if not await raw_request.is_disconnected():
            err_msg = sanitize_path_message(str(exc))
            yield format_sse_event("error", {"error": err_msg})
    finally:
        if hasattr(stream_iter, "close"):
            try:
                await anyio.to_thread.run_sync(stream_iter.close)
            except Exception:
                pass


def create_app(max_request_body_size: int | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="AI Reliability Lab Diagnostic API",
        description="Local REST API for the AI Reliability Lab Svelte Diagnostic Studio",
        version="0.1.0",
    )

    # Enforce ASGI-level request body limits across all incoming HTTP requests
    app.add_middleware(
        RequestBodyLimitMiddleware,
        max_body_size=max_request_body_size,
    )

    # Browser development CORS narrowly scoped to local frontend dev servers
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:4173",
            "http://127.0.0.1:4173",
        ],
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        """Report server and database readiness."""
        db_path = get_db_path()
        db_exists = db_path.is_file()
        total_runs = 0

        if db_exists:
            try:
                with ExperimentRepository(db_path) as repo:
                    total_runs = _count_runs(repo)
            except StorageError:
                return HealthResponse(
                    status="degraded",
                    database=db_path.name,
                    database_exists=True,
                    total_runs=0,
                )

        return HealthResponse(
            status="ok",
            database=db_path.name,
            database_exists=db_exists,
            total_runs=total_runs,
        )

    @app.get("/api/runs", response_model=RunListResponse)
    def list_runs(
        repo: RepositoryDep,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        experiment_name: str | None = None,
    ) -> RunListResponse:
        """Return paginated list of persisted experiment run summaries."""
        try:
            with repo:
                total = _count_runs(repo, experiment_name=experiment_name)
                summaries = repo.list_runs(
                    limit=limit,
                    offset=offset,
                    experiment_name=experiment_name,
                )
                items = [
                    RunSummarySchema(
                        run_id=s.run_id,
                        experiment_name=s.experiment_name,
                        run_timestamp=s.run_timestamp,
                        config_hash=s.config_hash,
                        dataset_reference=s.dataset_reference,
                        created_at=s.created_at,
                        metric_count=s.metric_count,
                        query_count=s.query_count,
                        metrics=s.metrics,
                    )
                    for s in summaries
                ]
                return RunListResponse(
                    total=total,
                    limit=limit,
                    offset=offset,
                    runs=items,
                )
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage error reading runs: {exc}",
            ) from exc

    @app.get("/api/runs/{run_id}", response_model=RunDetailResponse)
    def get_run(
        run_id: str,
        repo: RepositoryDep,
    ) -> RunDetailResponse:
        """Return complete detail for an individual experiment run."""
        try:
            with repo:
                run: StoredRun | None = repo.get_run(run_id)
                if run is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Experiment run '{run_id}' not found.",
                    )
                return RunDetailResponse(
                    run_id=run.run_id,
                    experiment_name=run.experiment_name,
                    run_timestamp=run.run_timestamp,
                    config_hash=run.config_hash,
                    config=run.config,
                    dataset_reference=run.dataset_reference,
                    metadata=run.metadata,
                    created_at=run.created_at,
                    metrics=run.metrics,
                    query_count=len(run.per_query_results),
                    per_query_results=run.per_query_results,
                )
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage error reading run: {exc}",
            ) from exc

    @app.get("/api/runs/{run_id}/queries")
    def get_run_queries(
        run_id: str,
        repo: RepositoryDep,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        query_id: str | None = None,
        failed_only: bool = Query(False),
        failure_type: str | None = None,
        sort_by: str = Query("default"),
    ) -> dict[str, Any]:
        """Return paginated query diagnostic records with failure classification."""
        try:
            with repo:
                run = repo.get_run(run_id)
                if run is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Experiment run '{run_id}' not found.",
                    )

                queries = run.per_query_results
                annotated_queries: list[dict[str, Any]] = []
                for q in queries:
                    q_copy = dict(q)
                    diag = classify_query_failure(q_copy)
                    q_copy["failure_diagnostic"] = diag
                    if not q_copy.get("failure_trace") and diag["has_failure"]:
                        q_copy["failure_trace"] = diag["explanation"]
                    annotated_queries.append(q_copy)

                if query_id:
                    annotated_queries = [
                        q for q in annotated_queries if q.get("query_id") == query_id
                    ]

                if failed_only:
                    annotated_queries = [
                        q
                        for q in annotated_queries
                        if q.get("failure_diagnostic", {}).get("has_failure")
                    ]

                if failure_type:
                    annotated_queries = [
                        q
                        for q in annotated_queries
                        if q.get("failure_diagnostic", {}).get("classification") == failure_type
                    ]

                if sort_by == "worst_hitrate":
                    annotated_queries.sort(
                        key=lambda q: float(q.get("metrics", {}).get("HitRate@K", 1.0))
                    )
                elif sort_by == "worst_recall":
                    annotated_queries.sort(
                        key=lambda q: float(q.get("metrics", {}).get("Recall@K", 1.0))
                    )
                elif sort_by == "latency":
                    annotated_queries.sort(
                        key=lambda q: float(q.get("latency_ms", 0.0) or 0.0), reverse=True
                    )

                total = len(annotated_queries)
                paged = annotated_queries[offset : offset + limit]

                return {
                    "run_id": run.run_id,
                    "total": total,
                    "limit": limit,
                    "offset": offset,
                    "queries": paged,
                }
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage error reading queries: {exc}",
            ) from exc

    @app.post("/api/runs/compare")
    def compare_runs(
        req: CompareRequest,
        repo: RepositoryDep,
    ) -> dict[str, Any]:
        """Execute regression comparison between a baseline and candidate run."""
        try:
            with repo:
                detector = RegressionDetector(repo)
                result = detector.compare_runs(
                    baseline_run_id=req.baseline_run_id,
                    candidate_run_id=req.candidate_run_id,
                    tolerance=req.tolerance,
                    metric_tolerances=req.metric_tolerances,
                )
                return result.to_dict()
        except IncomparableRunsError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=str(exc),
            ) from exc
        except InvalidToleranceError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage error comparing runs: {exc}",
            ) from exc

    @app.get("/api/runs/{run_id}/citation-audit", response_model=CitationAuditResponse)
    def audit_citations(
        run_id: str,
        repo: RepositoryDep,
    ) -> CitationAuditResponse:
        """Audit citation validity and reference linkages across queries in a run."""
        try:
            with repo:
                run = repo.get_run(run_id)
                if run is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Experiment run '{run_id}' not found.",
                    )

                evaluator = CitationEvaluator()
                parser = CitationParser()
                audit_items: list[CitationAuditQueryItem] = []
                precisions: list[float] = []
                recalls: list[float] = []

                for q in run.per_query_results:
                    qid = str(q.get("query_id", "unknown"))
                    text = q.get("text")
                    answer = q.get("answer") or q.get("generated_answer") or q.get("answer_text")

                    # Extract retrieved chunk IDs
                    retrieved_raw = q.get("retrieved") or q.get("candidates") or []
                    retrieved_objs: list[RetrievalResult] = []
                    retrieved_chunk_ids: list[str] = []

                    for idx, c in enumerate(retrieved_raw):
                        cid = str(c.get("chunk_id") if isinstance(c, dict) else c)
                        content_str = (
                            str(c.get("content", "")) if isinstance(c, dict) else ""
                        ) or "Retrieved evidence chunk"
                        if cid:
                            retrieved_chunk_ids.append(cid)
                            score_val = float(c.get("score", 1.0)) if isinstance(c, dict) else 1.0
                            retrieved_objs.append(
                                RetrievalResult(
                                    chunk=Chunk(
                                        chunk_id=cid,
                                        doc_id="audit-doc",
                                        content=content_str,
                                        page_number=1,
                                        char_start=0,
                                        char_end=len(content_str),
                                    ),
                                    score=score_val,
                                    rank=idx + 1,
                                    retriever_name="retriever",
                                )
                            )

                    # If an answer with citations is present, evaluate it
                    if isinstance(answer, str) and answer.strip():
                        val_res = evaluator.evaluate(
                            answer_text=answer,
                            retrieved=retrieved_objs,
                            citation_mapping=q.get("citation_mapping"),
                            required_citations=q.get("required_citations"),
                        )
                        precisions.append(val_res.citation_precision)
                        recalls.append(val_res.citation_recall)
                        audit_items.append(
                            CitationAuditQueryItem(
                                query_id=qid,
                                text=text,
                                has_answer=True,
                                answer_text=answer,
                                citation_precision=val_res.citation_precision,
                                citation_recall=val_res.citation_recall,
                                valid_citations=val_res.valid_citations,
                                invalid_citations=val_res.invalid_citations,
                                unresolved_citations=val_res.unresolved_citations,
                                missing_evidence_citations=val_res.missing_evidence_citations,
                                duplicate_citations=val_res.duplicate_citations,
                                unreferenced_required_citations=val_res.unreferenced_required_citations,
                                malformed_citations=val_res.malformed_citations,
                                retrieved_chunk_ids=retrieved_chunk_ids,
                                citation_to_chunk_map=val_res.citation_to_chunk_map,
                            )
                        )
                    else:
                        # Retrieval-only query (check for required citation coverage)
                        req_cits = [str(c) for c in q.get("required_citations", [])]
                        malformed = parser.find_malformed(text or "")
                        audit_items.append(
                            CitationAuditQueryItem(
                                query_id=qid,
                                text=text,
                                has_answer=False,
                                answer_text=None,
                                citation_precision=1.0 if not req_cits else 0.0,
                                citation_recall=1.0 if not req_cits else 0.0,
                                valid_citations=[],
                                invalid_citations=[],
                                unresolved_citations=[],
                                missing_evidence_citations=[],
                                duplicate_citations=[],
                                unreferenced_required_citations=req_cits,
                                malformed_citations=malformed,
                                retrieved_chunk_ids=retrieved_chunk_ids,
                                citation_to_chunk_map={},
                            )
                        )

                audited_count = len(precisions)
                mean_p = sum(precisions) / audited_count if audited_count > 0 else 0.0
                mean_r = sum(recalls) / audited_count if audited_count > 0 else 0.0

                return CitationAuditResponse(
                    run_id=run.run_id,
                    experiment_name=run.experiment_name,
                    total_queries=len(run.per_query_results),
                    audited_queries_count=audited_count,
                    mean_citation_precision=mean_p,
                    mean_citation_recall=mean_r,
                    queries=audit_items,
                    limitations_note=(
                        "Citation validity measures whether cited markers resolve to retrieved "
                        "evidence chunks. High citation precision/recall does not prove factual "
                        "correctness or semantic entailment."
                    ),
                )
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Storage error auditing citations: {exc}",
            ) from exc

    # -------------------------------------------------------------------------
    # Document Ingestion & Management Endpoints (V2-002)
    # -------------------------------------------------------------------------

    @app.post(
        "/api/documents/upload",
        response_model=DocumentUploadResponse,
        status_code=status.HTTP_201_CREATED,
    )
    async def upload_document(
        background_tasks: BackgroundTasks,
        service: IngestionServiceDep,
        file: Annotated[UploadFile, File()],
        request: Request,
    ) -> DocumentUploadResponse:
        """Upload a PDF, TXT, or Markdown document and schedule local ingestion."""
        # Service-level early check against Content-Length header
        content_length = request.headers.get("content-length")
        if content_length and content_length.isdigit():
            if int(content_length) > service.max_upload_size:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Upload size limit exceeded ({content_length} bytes > "
                        f"{service.max_upload_size} bytes limit)."
                    ),
                )

        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded file must have a valid filename.",
            )

        try:
            record = service.save_upload_stream(
                filename=file.filename,
                file_stream=file.file,
            )
        except EmptyFileError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Empty file rejected: {exc}",
            ) from exc
        except InvalidDocumentFormatError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid document format: {exc}",
            ) from exc
        except UploadSizeLimitExceededError as exc:
            raise HTTPException(
                status_code=413,
                detail=f"Upload size limit exceeded: {exc}",
            ) from exc
        except Exception as exc:
            logger.error("Failed to save uploaded file: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to save uploaded file due to an unexpected storage error.",
            ) from exc
        finally:
            await file.close()

        # Schedule background extraction, chunking, and indexing
        background_tasks.add_task(service.process_document, record.doc_id)

        return DocumentUploadResponse(
            doc_id=record.doc_id,
            filename=record.filename,
            file_type=record.file_type,
            file_size=record.file_size,
            content_hash=record.content_hash,
            status=(record.status.value if hasattr(record.status, "value") else str(record.status)),
            indexing_status=(
                record.indexing_status.value
                if hasattr(record.indexing_status, "value")
                else str(record.indexing_status)
            ),
            created_at=record.created_at,
            message="Document uploaded successfully and queued for ingestion.",
        )

    @app.get("/api/documents", response_model=DocumentListResponse)
    def list_documents(
        repo: DocRepoDep,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
        status_filter: str | None = Query(None, alias="status"),
    ) -> DocumentListResponse:
        """Return a paginated list of ingested documents."""
        total = repo.count_documents(status=status_filter)
        records = repo.list_documents(limit=limit, offset=offset, status=status_filter)

        items = [
            DocumentSummarySchema(
                doc_id=r.doc_id,
                filename=r.filename,
                file_type=r.file_type,
                file_size=r.file_size,
                content_hash=r.content_hash,
                status=r.status.value if hasattr(r.status, "value") else str(r.status),
                page_count=r.page_count,
                chunk_count=r.chunk_count,
                indexing_status=(
                    r.indexing_status.value
                    if hasattr(r.indexing_status, "value")
                    else str(r.indexing_status)
                ),
                error_message=sanitize_path_message(r.error_message),
                created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in records
        ]

        return DocumentListResponse(
            total=total,
            limit=limit,
            offset=offset,
            documents=items,
        )

    @app.get("/api/documents/{document_id}", response_model=DocumentDetailResponse)
    def get_document(
        document_id: str,
        repo: DocRepoDep,
    ) -> DocumentDetailResponse:
        """Return complete metadata, processing state, and chunk count for a document."""
        record = repo.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        return DocumentDetailResponse(
            doc_id=record.doc_id,
            filename=record.filename,
            file_type=record.file_type,
            file_size=record.file_size,
            content_hash=record.content_hash,
            status=(record.status.value if hasattr(record.status, "value") else str(record.status)),
            page_count=record.page_count,
            chunk_count=record.chunk_count,
            indexing_status=(
                record.indexing_status.value
                if hasattr(record.indexing_status, "value")
                else str(record.indexing_status)
            ),
            error_message=sanitize_path_message(record.error_message),
            metadata=record.metadata,
            created_at=record.created_at,
            updated_at=record.updated_at,
        )

    @app.get("/api/documents/{document_id}/chunks", response_model=DocumentChunkListResponse)
    def get_document_chunks(
        document_id: str,
        repo: DocRepoDep,
        limit: int = Query(50, ge=1, le=500),
        offset: int = Query(0, ge=0),
    ) -> DocumentChunkListResponse:
        """Return paginated chunk records with coordinates and source provenance."""
        record = repo.get_document(document_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )

        # Enforce document chunk freshness
        try:
            DocumentIngestionService.validate_document_freshness(
                record, resource_name="Document chunks"
            )
        except IngestionError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=sanitize_path_message(str(exc)),
            ) from exc

        total = repo.count_chunks(document_id)
        chunk_items = repo.list_chunks(document_id, limit=limit, offset=offset)

        chunks = [
            ChunkSchema(
                chunk_id=c.chunk_id,
                doc_id=c.doc_id,
                chunk_index=idx,
                content=c.content,
                page_number=c.page_number,
                char_start=c.char_start,
                char_end=c.char_end,
                metadata=c.metadata,
            )
            for idx, c in chunk_items
        ]

        return DocumentChunkListResponse(
            total=total,
            limit=limit,
            offset=offset,
            chunks=chunks,
        )

    @app.delete("/api/documents/{document_id}", response_model=DocumentDeleteResponse)
    def delete_document(
        document_id: str,
        service: IngestionServiceDep,
    ) -> DocumentDeleteResponse:
        """Safely delete original document file, generated indexes, and metadata records."""
        try:
            deleted = service.delete_document(document_id)
        except DocumentDeletionError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=sanitize_path_message(str(exc)),
            ) from exc

        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Document '{document_id}' not found.",
            )
        return DocumentDeleteResponse(status="deleted", doc_id=document_id)

    @app.post("/api/query", response_model=QueryResponse)
    def execute_query(
        request: QueryRequest,
        query_service: QueryServiceDep,
    ) -> QueryResponse:
        """Execute evidence-first document query and return answer with immutable trace."""
        try:
            answer, trace = query_service.query(
                query_text=request.query,
                selected_doc_ids=request.document_ids,
                strategy=request.strategy,
                top_k=request.top_k,
                candidate_depth=request.candidate_depth,
                min_dense_score=request.min_dense_score,
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except UnprocessableDocumentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except DocumentConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error executing query: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal server error occurred while executing the query.",
            ) from exc

        return QueryResponse(
            answer=answer,
            query_text=trace.query_text,
            selected_doc_ids=list(trace.selected_doc_ids),
            strategy=trace.requested_strategy,
            effective_strategy=trace.effective_strategy,
            fallback_reason=trace.fallback_reason,
            abstained=trace.abstained,
            retrieval_abstained=trace.retrieval_abstained,
            generation_abstained=trace.generation_abstained,
            abstention_reason=trace.retrieval_abstention_reason
            or trace.generation_abstention_reason,
            resolved_citations=[
                ResolvedCitationSchema(**c.to_dict()) for c in trace.resolved_citations
            ],
            unresolved_citations=[
                UnresolvedCitationSchema(**c.to_dict()) for c in trace.unresolved_citations
            ],
            malformed_citations=[
                MalformedCitationSchema(**c.to_dict()) for c in trace.malformed_citations
            ],
            trace=trace.to_dict(),
            latency_ms=trace.total_latency_ms,
        )

    @app.post("/api/query/stream")
    def stream_query(
        request: QueryRequest,
        query_service: QueryServiceDep,
        raw_request: Request,
    ) -> StreamingResponse:
        """Stream an evidence-grounded query response via Server-Sent Events (SSE)."""
        try:
            query_service._validate_lifecycle_and_get_documents(
                request.document_ids, request.strategy
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except UnprocessableDocumentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except DocumentConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=sanitize_path_message(str(exc)),
            ) from exc

        stream_iter = query_service.query_stream(
            query_text=request.query,
            selected_doc_ids=request.document_ids,
            strategy=request.strategy,
            top_k=request.top_k,
            candidate_depth=request.candidate_depth,
            min_dense_score=request.min_dense_score,
        )

        return StreamingResponse(
            sse_stream_generator(stream_iter, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --------------------------------------------------------------------------
    # Session Persistence & Conversational Query Endpoints (V2-003 Phase B)
    # --------------------------------------------------------------------------

    @app.post(
        "/api/sessions",
        response_model=SessionDetailResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_session(
        payload: CreateSessionRequest,
        service: SessionServiceDep,
    ) -> SessionDetailResponse:
        """Create a new conversational session."""
        try:
            record = service.create_session(
                title=payload.title,
                selected_doc_id=payload.selected_doc_id,
                metadata=payload.metadata,
            )
            return SessionDetailResponse(
                session_id=record.session_id,
                title=record.title,
                created_at=record.created_at,
                last_activity_at=record.last_activity_at,
                selected_doc_id=record.selected_doc_id,
                metadata=record.metadata,
                message_count=0,
                messages=[],
            )
        except Exception as exc:
            logger.error("Failed to create session: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal server error occurred while creating session.",
            ) from exc

    @app.get("/api/sessions", response_model=SessionListResponse)
    def list_sessions(
        service: SessionServiceDep,
        limit: int = Query(50, ge=1, le=500, description="Page limit"),
        offset: int = Query(0, ge=0, description="Page offset"),
    ) -> SessionListResponse:
        """List persistent conversation sessions ordered by last activity descending."""
        sessions, total = service.list_sessions(limit=limit, offset=offset)
        return SessionListResponse(
            total=total,
            limit=limit,
            offset=offset,
            sessions=[
                SessionSummarySchema(
                    session_id=s.session_id,
                    title=s.title,
                    created_at=s.created_at,
                    last_activity_at=s.last_activity_at,
                    selected_doc_id=s.selected_doc_id,
                    message_count=s.message_count,
                    metadata=s.metadata,
                )
                for s in sessions
            ],
        )

    @app.get("/api/sessions/{session_id}", response_model=SessionDetailResponse)
    def get_session(
        session_id: str,
        service: SessionServiceDep,
    ) -> SessionDetailResponse:
        """Retrieve a single session by ID and its complete message history."""
        try:
            record = service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc

        return SessionDetailResponse(
            session_id=record.session_id,
            title=record.title,
            created_at=record.created_at,
            last_activity_at=record.last_activity_at,
            selected_doc_id=record.selected_doc_id,
            metadata=record.metadata,
            message_count=record.message_count,
            messages=[
                MessageResponse(
                    message_id=m.message_id,
                    session_id=m.session_id,
                    role=m.role,
                    content=m.content,
                    created_at=m.created_at,
                    query_config=m.query_config,
                    resolved_citations=[ResolvedCitationSchema(**c) for c in m.resolved_citations],
                    unresolved_citations=[
                        UnresolvedCitationSchema(**c) for c in m.unresolved_citations
                    ],
                    malformed_citations=[
                        MalformedCitationSchema(**c) for c in m.malformed_citations
                    ],
                    query_trace=m.query_trace,
                    provider=m.provider,
                    model_name=m.model_name,
                    prompt_tokens=m.prompt_tokens,
                    completion_tokens=m.completion_tokens,
                    retrieval_latency_ms=m.retrieval_latency_ms,
                    generation_latency_ms=m.generation_latency_ms,
                    total_latency_ms=m.total_latency_ms,
                )
                for m in record.messages
            ],
        )

    @app.delete("/api/sessions/{session_id}", response_model=SessionDeleteResponse)
    def delete_session(
        session_id: str,
        service: SessionServiceDep,
    ) -> SessionDeleteResponse:
        """Delete a conversation session and all its message history."""
        try:
            service.delete_session(session_id)
            return SessionDeleteResponse(deleted=True, session_id=session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc

    @app.post(
        "/api/sessions/{session_id}/query",
        response_model=SessionQueryResponse,
    )
    def query_session(
        session_id: str,
        request: SessionQueryRequest,
        service: SessionServiceDep,
    ) -> SessionQueryResponse:
        """Execute a conversational query turn within a persistent session."""
        try:
            answer, trace, user_msg, asst_msg = service.execute_session_query(
                session_id=session_id,
                query_text=request.query,
                document_ids=request.document_ids,
                strategy=request.strategy,
                top_k=request.top_k,
                candidate_depth=request.candidate_depth,
                min_dense_score=request.min_dense_score,
                max_history_messages=request.max_history_messages,
            )
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except DocumentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except UnprocessableDocumentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except DocumentConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except Exception as exc:
            logger.error("Unexpected error in session query: %s", exc, exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="An internal server error occurred while executing the session query.",
            ) from exc

        return SessionQueryResponse(
            session_id=session_id,
            user_message_id=user_msg.message_id,
            assistant_message_id=asst_msg.message_id,
            answer=answer,
            query_text=trace.query_text,
            selected_doc_ids=list(trace.selected_doc_ids),
            strategy=trace.requested_strategy,
            effective_strategy=trace.effective_strategy,
            fallback_reason=trace.fallback_reason,
            abstained=trace.abstained,
            retrieval_abstained=trace.retrieval_abstained,
            generation_abstained=trace.generation_abstained,
            abstention_reason=trace.retrieval_abstention_reason
            or trace.generation_abstention_reason,
            resolved_citations=[
                ResolvedCitationSchema(**c.to_dict()) for c in trace.resolved_citations
            ],
            unresolved_citations=[
                UnresolvedCitationSchema(**c.to_dict()) for c in trace.unresolved_citations
            ],
            malformed_citations=[
                MalformedCitationSchema(**c.to_dict()) for c in trace.malformed_citations
            ],
            trace=trace.to_dict(),
            latency_ms=trace.total_latency_ms,
        )

    @app.post("/api/sessions/{session_id}/query/stream")
    def stream_session_query(
        session_id: str,
        request: SessionQueryRequest,
        service: SessionServiceDep,
        raw_request: Request,
    ) -> StreamingResponse:
        """Stream a conversational query turn within a persistent session via SSE."""
        try:
            session = service.get_session(session_id)
        except SessionNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc

        effective_doc_ids = request.document_ids or (
            [session.selected_doc_id] if session.selected_doc_id else []
        )
        if not effective_doc_ids:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    f"No document selected for query and session '{session_id}' "
                    "has no default document scope."
                ),
            )

        try:
            service.query_service._validate_lifecycle_and_get_documents(
                effective_doc_ids, request.strategy
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except UnprocessableDocumentError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except DocumentConflictError as exc:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc

        stream_iter = service.execute_session_query_stream(
            session_id=session_id,
            query_text=request.query,
            document_ids=request.document_ids,
            strategy=request.strategy,
            top_k=request.top_k,
            candidate_depth=request.candidate_depth,
            min_dense_score=request.min_dense_score,
            max_history_messages=request.max_history_messages,
        )

        return StreamingResponse(
            sse_stream_generator(stream_iter, raw_request),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

    # --------------------------------------------------------------------------
    # Evidence Intelligence Investigation Endpoints (STING V2-004)
    # --------------------------------------------------------------------------

    @app.post(
        "/api/investigations",
        response_model=InvestigationDetailResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_investigation(
        payload: CreateInvestigationRequest,
        service: InvestigationServiceDep,
        background_tasks: BackgroundTasks,
    ) -> InvestigationDetailResponse:
        """Create and optionally execute a cross-document evidence investigation."""
        try:
            record = service.create_investigation(
                query=payload.query,
                document_ids=payload.document_ids,
                title=payload.title,
                comparison_aspects=payload.comparison_aspects or payload.aspects,
            )
        except DocumentNotFoundError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except ValueError as exc:
            if "not found" in str(exc).lower():
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=sanitize_path_message(str(exc)),
                ) from exc
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=sanitize_path_message(str(exc)),
            ) from exc
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Failed to create investigation: {sanitize_path_message(str(exc))}",
            ) from exc

        if payload.run_immediately:
            try:
                report = service.execute_investigation(record.investigation_id)
                record.report = report
                record.status = "completed"
                record.completed_at = report.completed_at
            except Exception as exc:
                logger.error("Immediate execution of investigation failed: %s", exc)
                updated = service.investigation_repository.get_investigation(
                    record.investigation_id
                )
                if updated:
                    record = updated
        else:
            background_tasks.add_task(service.execute_investigation, record.investigation_id)

        rep_dict = record.report.to_dict() if record.report else None

        return InvestigationDetailResponse(
            investigation_id=record.investigation_id,
            title=record.title,
            query=record.query,
            comparison_aspects=record.comparison_aspects,
            document_ids=record.document_ids,
            status=record.status,
            created_at=record.created_at,
            completed_at=record.completed_at,
            error_message=record.error_message,
            report=InvestigationReportSchema(**rep_dict) if rep_dict else None,
        )

    @app.get(
        "/api/investigations",
        response_model=InvestigationListResponse,
    )
    def list_investigations(
        service: InvestigationServiceDep,
        limit: int = Query(20, ge=1, le=100),
        offset: int = Query(0, ge=0),
    ) -> InvestigationListResponse:
        """List past cross-document investigations ordered by creation date."""
        records, total = service.investigation_repository.list_investigations(
            limit=limit, offset=offset
        )
        summaries: list[InvestigationSummarySchema] = []
        for r in records:
            total_findings = r.report.total_findings if r.report else 0
            summaries.append(
                InvestigationSummarySchema(
                    investigation_id=r.investigation_id,
                    title=r.title,
                    query=r.query,
                    comparison_aspects=r.comparison_aspects,
                    document_ids=r.document_ids,
                    status=r.status,
                    created_at=r.created_at,
                    completed_at=r.completed_at,
                    total_findings=total_findings,
                    error_message=r.error_message,
                )
            )
        return InvestigationListResponse(
            investigations=summaries,
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get(
        "/api/investigations/{investigation_id}",
        response_model=InvestigationDetailResponse,
    )
    def get_investigation(
        investigation_id: str,
        service: InvestigationServiceDep,
    ) -> InvestigationDetailResponse:
        """Retrieve full details, matrix, and findings for an investigation."""
        record = service.investigation_repository.get_investigation(investigation_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation '{investigation_id}' not found.",
            )
        rep_dict = record.report.to_dict() if record.report else None
        return InvestigationDetailResponse(
            investigation_id=record.investigation_id,
            title=record.title,
            query=record.query,
            comparison_aspects=record.comparison_aspects,
            document_ids=record.document_ids,
            status=record.status,
            created_at=record.created_at,
            completed_at=record.completed_at,
            error_message=record.error_message,
            report=InvestigationReportSchema(**rep_dict) if rep_dict else None,
        )

    @app.post(
        "/api/investigations/{investigation_id}/run",
        response_model=InvestigationDetailResponse,
    )
    def run_investigation(
        investigation_id: str,
        service: InvestigationServiceDep,
        background_tasks: BackgroundTasks,
        wait: bool = Query(True, description="Wait for execution to finish before returning"),
    ) -> InvestigationDetailResponse:
        """Execute or re-run an investigation pipeline."""
        record = service.investigation_repository.get_investigation(investigation_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation '{investigation_id}' not found.",
            )

        if wait:
            service.execute_investigation(investigation_id)
            record = service.investigation_repository.get_investigation(investigation_id) or record
        else:
            background_tasks.add_task(service.execute_investigation, investigation_id)
            record.status = "running"

        rep_dict = record.report.to_dict() if record.report else None
        return InvestigationDetailResponse(
            investigation_id=record.investigation_id,
            title=record.title,
            query=record.query,
            comparison_aspects=record.comparison_aspects,
            document_ids=record.document_ids,
            status=record.status,
            created_at=record.created_at,
            completed_at=record.completed_at,
            error_message=record.error_message,
            report=InvestigationReportSchema(**rep_dict) if rep_dict else None,
        )

    @app.delete(
        "/api/investigations/{investigation_id}",
        response_model=InvestigationDeleteResponse,
    )
    def delete_investigation(
        investigation_id: str,
        service: InvestigationServiceDep,
    ) -> InvestigationDeleteResponse:
        """Delete an investigation record and its associated findings."""
        record = service.investigation_repository.get_investigation(investigation_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation '{investigation_id}' not found.",
            )
        service.investigation_repository.delete_investigation(investigation_id)
        return InvestigationDeleteResponse(deleted=True, investigation_id=investigation_id)

    @app.get(
        "/api/investigations/{investigation_id}/export",
        response_class=Response,
    )
    def export_investigation_markdown(
        investigation_id: str,
        service: InvestigationServiceDep,
    ) -> Response:
        """Export an investigation's findings and comparison matrix as a Markdown document."""
        record = service.investigation_repository.get_investigation(investigation_id)
        if record is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Investigation '{investigation_id}' not found.",
            )
        if not record.report:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Investigation has not completed yet or has no report to export.",
            )
        markdown_text = record.report.to_markdown()
        filename = f"investigation_{investigation_id}.md"
        return Response(
            content=markdown_text,
            media_type="text/markdown",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
            },
        )

    # ========================================================================
    # V2-005: Evaluation Datasets Studio Endpoints
    # ========================================================================

    @app.post(
        "/api/datasets",
        response_model=DatasetDetailResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def create_dataset_endpoint(
        req: CreateDatasetRequest,
        service: DatasetServiceDep,
    ) -> DatasetDetailResponse:
        """Create a new golden evaluation dataset."""
        try:
            queries = [
                BenchmarkQuery(
                    query_id=q.query_id,
                    text=q.text,
                    relevant_chunk_ids=q.relevant_chunk_ids,
                    relevance_grades=q.relevance_grades,
                    ground_truth_answer=q.ground_truth_answer,
                    required_citations=q.required_citations,
                    metadata=q.metadata,
                )
                for q in req.queries
            ]
            ds = service.create_dataset(
                name=req.name,
                description=req.description,
                document_scope=req.document_scope,
                is_synthetic=req.is_synthetic,
                metadata=req.metadata,
                queries=queries,
            )
            return DatasetDetailResponse(**ds.to_dict())
        except (DatasetValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.get("/api/datasets", response_model=DatasetListResponse)
    def list_datasets_endpoint(
        service: DatasetServiceDep,
        limit: int = Query(50, ge=1, le=200),
        offset: int = Query(0, ge=0),
    ) -> DatasetListResponse:
        """List persisted benchmark datasets with pagination."""
        records = service.list_datasets(limit=limit, offset=offset)
        total = service.dataset_repository.count_datasets()
        return DatasetListResponse(
            datasets=[DatasetSummarySchema(**r.to_dict()) for r in records],
            total=total,
            limit=limit,
            offset=offset,
        )

    @app.get("/api/datasets/{dataset_id}", response_model=DatasetDetailResponse)
    def get_dataset_endpoint(
        dataset_id: str,
        service: DatasetServiceDep,
    ) -> DatasetDetailResponse:
        """Retrieve dataset details and all constituent benchmark queries."""
        try:
            ds = service.get_dataset(dataset_id)
            return DatasetDetailResponse(**ds.to_dict())
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    @app.put("/api/datasets/{dataset_id}", response_model=DatasetSummarySchema)
    def update_dataset_endpoint(
        dataset_id: str,
        req: UpdateDatasetRequest,
        service: DatasetServiceDep,
    ) -> DatasetSummarySchema:
        """Update dataset title, description, or document scope."""
        try:
            updated = service.update_dataset(
                dataset_id=dataset_id,
                name=req.name,
                description=req.description,
                document_scope=req.document_scope,
                metadata=req.metadata,
            )
            return DatasetSummarySchema(**updated.to_dict())
        except (DatasetValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.delete("/api/datasets/{dataset_id}")
    def delete_dataset_endpoint(
        dataset_id: str,
        service: DatasetServiceDep,
    ) -> dict[str, Any]:
        """Delete an evaluation dataset and all its queries."""
        deleted = service.delete_dataset(dataset_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Dataset '{dataset_id}' not found.",
            )
        return {"deleted": True, "dataset_id": dataset_id}

    @app.post("/api/datasets/{dataset_id}/queries", response_model=BenchmarkQuerySchema)
    def add_query_endpoint(
        dataset_id: str,
        req: CreateQueryRequest,
        service: DatasetServiceDep,
    ) -> BenchmarkQuerySchema:
        """Add a single benchmark query to a dataset."""
        try:
            query = BenchmarkQuery(
                query_id=req.query_id,
                text=req.text,
                relevant_chunk_ids=req.relevant_chunk_ids,
                relevance_grades=req.relevance_grades,
                ground_truth_answer=req.ground_truth_answer,
                required_citations=req.required_citations,
                metadata=req.metadata,
            )
            added = service.add_query(dataset_id, query)
            return BenchmarkQuerySchema(**added.to_dict())
        except (DatasetValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.put(
        "/api/datasets/{dataset_id}/queries/{query_id}",
        response_model=BenchmarkQuerySchema,
    )
    def update_query_endpoint(
        dataset_id: str,
        query_id: str,
        req: UpdateQueryRequest,
        service: DatasetServiceDep,
    ) -> BenchmarkQuerySchema:
        """Update an existing benchmark query in a dataset."""
        try:
            query = BenchmarkQuery(
                query_id=query_id,
                text=req.text,
                relevant_chunk_ids=req.relevant_chunk_ids,
                relevance_grades=req.relevance_grades,
                ground_truth_answer=req.ground_truth_answer,
                required_citations=req.required_citations,
                metadata=req.metadata,
            )
            updated = service.update_query(dataset_id, query)
            return BenchmarkQuerySchema(**updated.to_dict())
        except (DatasetValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.delete("/api/datasets/{dataset_id}/queries/{query_id}")
    def delete_query_endpoint(
        dataset_id: str,
        query_id: str,
        service: DatasetServiceDep,
    ) -> dict[str, Any]:
        """Delete an evaluation query from a dataset."""
        deleted = service.delete_query(dataset_id, query_id)
        if not deleted:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Query '{query_id}' not found in dataset '{dataset_id}'.",
            )
        return {"deleted": True, "dataset_id": dataset_id, "query_id": query_id}

    @app.post(
        "/api/datasets/import",
        response_model=DatasetDetailResponse,
        status_code=status.HTTP_201_CREATED,
    )
    def import_dataset_endpoint(
        req: ImportDatasetRequest,
        service: DatasetServiceDep,
    ) -> DatasetDetailResponse:
        """Import an evaluation dataset from raw JSONL text."""
        try:
            ds = service.import_jsonl(
                name=req.name,
                jsonl_content=req.jsonl_content,
                description=req.description,
                document_scope=req.document_scope,
                is_synthetic=req.is_synthetic,
            )
            return DatasetDetailResponse(**ds.to_dict())
        except (DatasetValidationError, ValueError) as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.get("/api/datasets/{dataset_id}/export")
    def export_dataset_endpoint(
        dataset_id: str,
        service: DatasetServiceDep,
    ) -> Response:
        """Export dataset queries to standard benchmark JSONL format."""
        try:
            jsonl_str = service.export_jsonl(dataset_id)
            filename = f"dataset_{dataset_id}.jsonl"
            return Response(
                content=jsonl_str,
                media_type="application/x-jsonlines",
                headers={"Content-Disposition": f'attachment; filename="{filename}"'},
            )
        except DatasetValidationError as exc:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=str(exc),
            ) from exc

    # ========================================================================
    # V2-005: Experiment Execution & Regression Studio Endpoints
    # ========================================================================

    @app.post("/api/evaluation/run", response_model=EvaluationJobResponse)
    def launch_evaluation_endpoint(
        req: LaunchEvaluationRequest,
        eval_service: EvaluationServiceDep,
    ) -> EvaluationJobResponse:
        """Configure and launch a local RAG evaluation experiment asynchronously."""
        try:
            config = ExperimentExecutionConfig(
                experiment_name=req.experiment_name,
                dataset_id=req.dataset_id,
                document_ids=req.document_ids,
                retriever_type=req.retriever_type,
                retriever_params=req.retriever_params,
                top_k=req.top_k,
                provider=req.provider,
                model_name=req.model_name,
                temperature=req.temperature,
                max_output_tokens=req.max_output_tokens,
                prompt_version=req.prompt_version,
                enable_groundedness=req.enable_groundedness,
                enable_citations=req.enable_citations,
                enable_nli=req.enable_nli,
                enable_judge=req.enable_judge,
                judge_provider=req.judge_provider,
                judge_model=req.judge_model,
            )
            job = eval_service.start_evaluation_job(config)
            return EvaluationJobResponse(**job.to_dict())
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=str(exc),
            ) from exc

    @app.get("/api/evaluation/jobs", response_model=EvaluationJobListResponse)
    def list_evaluation_jobs_endpoint(
        eval_service: EvaluationServiceDep,
    ) -> EvaluationJobListResponse:
        """List active and recent evaluation jobs."""
        jobs = eval_service.list_jobs()
        return EvaluationJobListResponse(jobs=[EvaluationJobResponse(**j.to_dict()) for j in jobs])

    @app.get("/api/evaluation/jobs/{job_id}", response_model=EvaluationJobResponse)
    def get_evaluation_job_endpoint(
        job_id: str,
        eval_service: EvaluationServiceDep,
    ) -> EvaluationJobResponse:
        """Get status, progress percentage, and resulting run ID of an evaluation job."""
        job = eval_service.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Evaluation job '{job_id}' not found.",
            )
        return EvaluationJobResponse(**job.to_dict())

    @app.get("/api/evaluation/metrics-meta", response_model=MetricsMetaResponse)
    def get_metrics_meta_endpoint() -> MetricsMetaResponse:
        """Return comprehensive metadata, definitions, directions, and limitations of metrics."""
        return MetricsMetaResponse(
            metrics={k: MetricDefinitionSchema(**v) for k, v in METRIC_DEFINITIONS.items()}
        )

    @app.get("/api/runs/{run_id}/reproducibility", response_model=RunReproducibilityResponse)
    def get_run_reproducibility_endpoint(
        run_id: str,
        repo: RepositoryDep,
        dataset_repo: DatasetRepoDep,
        doc_repo: DocRepoDep,
    ) -> RunReproducibilityResponse:
        """Inspect reproducibility manifest and source dependency existence for a historical run."""
        try:
            with repo:
                run = repo.get_run(run_id)
                if not run:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Run '{run_id}' not found.",
                    )

                meta = run.metadata or {}
                dataset_id = run.dataset_reference or meta.get("dataset_id")
                dataset_exists = (
                    dataset_repo.get_dataset_record(dataset_id) is not None if dataset_id else False
                )

                corpus_doc_ids = meta.get("document_ids", [])
                corpus_exists = True
                if corpus_doc_ids:
                    for d in corpus_doc_ids:
                        if doc_repo.get_document(d) is None:
                            corpus_exists = False
                            break

                can_reproduce = bool(dataset_exists and corpus_exists)

                return RunReproducibilityResponse(
                    run_id=run.run_id,
                    experiment_name=run.experiment_name,
                    config_hash=run.config_hash,
                    dataset_id=dataset_id,
                    dataset_exists=dataset_exists,
                    corpus_document_ids=corpus_doc_ids,
                    corpus_documents_exist=corpus_exists,
                    can_reproduce=can_reproduce,
                    metadata=meta,
                    raw_config=run.config,
                )
        except StorageError as exc:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=str(exc),
            ) from exc

    @app.post("/api/runs/{run_id}/set-baseline")
    def set_run_baseline_endpoint(
        run_id: str,
        repo: RepositoryDep,
        is_baseline: bool = Query(True),
    ) -> dict[str, Any]:
        """Tag an experiment run as a designated baseline reference."""
        with repo.database.get_connection() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM experiment_runs WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Run '{run_id}' not found.",
                )
            meta = json.loads(row["metadata_json"] or "{}")
            meta["is_baseline"] = is_baseline
            conn.execute(
                "UPDATE experiment_runs SET metadata_json = ? WHERE run_id = ?",
                (json.dumps(meta), run_id),
            )
        return {"run_id": run_id, "is_baseline": is_baseline}

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("reliability_lab.api.main:app", host="127.0.0.1", port=8000, reload=True)
