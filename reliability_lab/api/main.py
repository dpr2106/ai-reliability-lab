"""FastAPI backend adapter exposing experiment storage, regression, and diagnostics.

Reuses ExperimentRepository, RegressionDetector, and CitationEvaluator without
duplicating SQLite persistence or metric calculations.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from reliability_lab.api.schemas import (
    CitationAuditQueryItem,
    CitationAuditResponse,
    CompareRequest,
    HealthResponse,
    RunDetailResponse,
    RunListResponse,
    RunSummarySchema,
)
from reliability_lab.core.models import Chunk, RetrievalResult
from reliability_lab.evaluation.citation_metrics import CitationEvaluator, CitationParser
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


def get_db_path() -> Path:
    """Return configured database path from environment or default."""
    raw = os.environ.get("RELIABILITY_LAB_DB", ".reliability-lab/experiments.sqlite3")
    return Path(raw).resolve()


def get_repository() -> ExperimentRepository:
    """FastAPI dependency providing an active ExperimentRepository."""
    db_path = get_db_path()
    return ExperimentRepository(db_path)


RepositoryDep = Annotated[ExperimentRepository, Depends(get_repository)]


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


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(
        title="AI Reliability Lab Diagnostic API",
        description="Local REST API for the AI Reliability Lab Svelte Diagnostic Studio",
        version="0.1.0",
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
        allow_methods=["GET", "POST", "OPTIONS"],
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
                    database=str(db_path),
                    database_exists=True,
                    total_runs=0,
                )

        return HealthResponse(
            status="ok",
            database=str(db_path),
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
    ) -> dict[str, Any]:
        """Return paginated query diagnostic records for a specific run."""
        try:
            with repo:
                run = repo.get_run(run_id)
                if run is None:
                    raise HTTPException(
                        status_code=status.HTTP_404_NOT_FOUND,
                        detail=f"Experiment run '{run_id}' not found.",
                    )

                queries = run.per_query_results
                if query_id:
                    queries = [q for q in queries if q.get("query_id") == query_id]

                total = len(queries)
                paged = queries[offset : offset + limit]

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

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("reliability_lab.api.main:app", host="127.0.0.1", port=8000, reload=True)
