"""Pydantic data schemas for the AI Reliability Lab FastAPI adapter."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Health and storage readiness response."""

    status: str = Field(..., description="Service status ('ok' or 'degraded')")
    database: str = Field(..., description="Filename of active SQLite experiment database")
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


class DocumentUploadResponse(BaseModel):
    """Response returned upon successful file upload and queuing."""

    doc_id: str = Field(..., description="Unique generated document identifier")
    filename: str = Field(..., description="Original user filename")
    file_type: str = Field(..., description="Document type ('pdf', 'txt', 'md')")
    file_size: int = Field(..., description="File size in bytes")
    content_hash: str = Field(..., description="SHA-256 hash of file content")
    status: str = Field(..., description="Initial ingestion state (typically 'UPLOADED')")
    indexing_status: str = Field(..., description="Index state (typically 'PENDING')")
    created_at: str = Field(..., description="Upload timestamp in ISO 8601")
    message: str = Field("Document uploaded successfully and queued for ingestion.")


class DocumentSummarySchema(BaseModel):
    """Summary of an ingested document for listing endpoints."""

    doc_id: str
    filename: str
    file_type: str
    file_size: int
    content_hash: str
    status: str
    page_count: int | None = None
    chunk_count: int
    indexing_status: str
    error_message: str | None = None
    created_at: str
    updated_at: str


class DocumentListResponse(BaseModel):
    """Paginated list of uploaded documents."""

    total: int
    limit: int
    offset: int
    documents: list[DocumentSummarySchema]


class DocumentDetailResponse(BaseModel):
    """Full detail of an ingested document record."""

    doc_id: str
    filename: str
    file_type: str
    file_size: int
    content_hash: str
    status: str
    page_count: int | None = None
    chunk_count: int
    indexing_status: str
    error_message: str | None = None
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class ChunkSchema(BaseModel):
    """Individual chunk record with source provenance coordinates."""

    chunk_id: str
    doc_id: str
    chunk_index: int
    content: str
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DocumentChunkListResponse(BaseModel):
    """Paginated listing of extracted chunks for a document."""

    total: int
    limit: int
    offset: int
    chunks: list[ChunkSchema]


class DocumentDeleteResponse(BaseModel):
    """Response payload confirming document deletion."""

    status: str = "deleted"
    doc_id: str


class QueryRequest(BaseModel):
    """Request payload for document-grounded evidence queries."""

    query: str = Field(..., min_length=1, description="User question to answer")
    document_ids: list[str] = Field(
        ..., min_length=1, description="Non-empty list of document IDs to query"
    )
    strategy: str = Field("hybrid", description="Retrieval strategy: 'bm25', 'dense', or 'hybrid'")
    top_k: int = Field(5, ge=1, le=50, description="Number of evidence chunks to present")
    candidate_depth: int = Field(
        60, ge=1, le=500, description="Intermediate retrieval candidate fetch depth"
    )
    min_dense_score: float | None = Field(
        None, description="Optional dense retrieval abstention threshold"
    )


class ResolvedCitationSchema(BaseModel):
    """Citation verified against retrieved evidence chunks."""

    marker: str
    alias: str
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    snippet: str
    is_valid: bool = True


class UnresolvedCitationSchema(BaseModel):
    """Citation marker whose alias was not provided in generation evidence."""

    marker: str
    alias: str
    is_valid: bool = False
    reason: str


class MalformedCitationSchema(BaseModel):
    """Detected citation-like expression failing standard syntax."""

    raw_text: str
    span: list[int]
    is_valid: bool = False
    reason: str


class QueryResponse(BaseModel):
    """Structured response container for query execution, citations, and trace."""

    answer: str
    query_text: str
    selected_doc_ids: list[str]
    strategy: str
    effective_strategy: str
    fallback_reason: str | None = None
    abstained: bool
    retrieval_abstained: bool
    generation_abstained: bool
    abstention_reason: str | None = None
    resolved_citations: list[ResolvedCitationSchema]
    unresolved_citations: list[UnresolvedCitationSchema]
    malformed_citations: list[MalformedCitationSchema]
    trace: dict[str, Any]
    latency_ms: float
