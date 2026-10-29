"""Pydantic data schemas for the AI Reliability Lab FastAPI adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health and storage readiness response."""

    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    database: str = Field(..., description="Path to active SQLite experiment database")
    database_exists: bool = Field(..., description="True if database file exists on disk")
    total_runs: int = Field(0, description="Total number of persisted experiment runs")


class RunSummarySchema(BaseModel):
    """Summary representation of a persisted experiment run."""

    run_id: str
    experiment_name: str
    run_timestamp: str
    config_hash: str
    dataset_reference: str | None = None
    created_at: str
    metric_count: int
    query_count: int
    metrics: dict[str, float]


class RunListResponse(BaseModel):
    """Paginated listing of experiment runs."""

    total: int
    limit: int
    offset: int
    runs: list[RunSummarySchema]


class RunDetailResponse(BaseModel):
    """Full detail of a single experiment run including configuration and queries."""

    run_id: str
    experiment_name: str
    run_timestamp: str
    config_hash: str
    config: dict[str, Any]
    dataset_reference: str | None = None
    metadata: dict[str, Any]
    created_at: str
    metrics: dict[str, float]
    query_count: int
    per_query_results: list[dict[str, Any]]


class CompareRequest(BaseModel):
    """Request payload for comparing two experiment runs."""

    baseline_run_id: str = Field(..., min_length=1, description="Historical baseline run ID")
    candidate_run_id: str = Field(..., min_length=1, description="Candidate evaluation run ID")
    tolerance: float = Field(0.02, ge=0.0, description="Default absolute degradation tolerance")
    metric_tolerances: dict[str, float] | None = Field(
        None, description="Optional per-metric tolerance overrides"
    )


class CitationAuditQueryItem(BaseModel):
    """Individual query audit record within a citation evaluation."""

    query_id: str
    text: str | None = None
    has_answer: bool
    answer_text: str | None = None
    citation_precision: float
    citation_recall: float
    valid_citations: list[str]
    invalid_citations: list[str]
    unresolved_citations: list[str]
    missing_evidence_citations: list[str]
    duplicate_citations: list[str]
    unreferenced_required_citations: list[str]
    malformed_citations: list[str]
    retrieved_chunk_ids: list[str]
    citation_to_chunk_map: dict[str, str]


class CitationAuditResponse(BaseModel):
    """Aggregated citation evaluation audit for an experiment run."""

    run_id: str
    experiment_name: str
    total_queries: int
    audited_queries_count: int
    mean_citation_precision: float
    mean_citation_recall: float
    queries: list[CitationAuditQueryItem]
    limitations_note: str
