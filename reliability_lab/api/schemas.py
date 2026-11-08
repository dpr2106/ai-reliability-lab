"""Pydantic data schemas for the AI Reliability Lab FastAPI adapter."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, Field, field_validator


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

    @field_validator("top_k", "candidate_depth", mode="before")
    @classmethod
    def validate_int_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Integer field cannot be a boolean.")
        return v

    @field_validator("min_dense_score", mode="before")
    @classmethod
    def validate_float_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("min_dense_score cannot be a boolean.")
        return v

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if isinstance(v, bool):
            raise ValueError("query cannot be a boolean.")
        s = v.strip()
        if not s:
            raise ValueError("query must be a non-empty string.")
        return s

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("document_ids must contain at least one document ID.")
        cleaned: list[str] = []
        for d in v:
            if not isinstance(d, str) or isinstance(d, bool) or not d.strip():
                raise ValueError("document_ids must not contain empty or whitespace-only strings.")
            cleaned.append(d.strip())
        return list(dict.fromkeys(cleaned))

    @field_validator("min_dense_score")
    @classmethod
    def validate_min_dense_score(cls, v: float | None) -> float | None:
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("min_dense_score cannot be a boolean.")
            if not math.isfinite(v):
                raise ValueError("min_dense_score must be a finite float (not NaN or Inf).")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"Invalid strategy '{v}'. Must be 'bm25', 'dense', or 'hybrid'.")
        return s


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


class CreateSessionRequest(BaseModel):
    """Payload to create a new conversational session."""

    title: str | None = Field(None, description="Optional title or label for the session")
    selected_doc_id: str | None = Field(
        None, description="Optional default document scope ID for queries in this session"
    )
    metadata: dict[str, Any] | None = Field(
        None, description="Optional user-provided session metadata"
    )

    @field_validator("title", "selected_doc_id")
    @classmethod
    def validate_clean_strings(cls, v: str | None) -> str | None:
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("String field cannot be a boolean.")
            s = v.strip()
            return s if s else None
        return None


class SessionSummarySchema(BaseModel):
    """Summary of a conversational session."""

    session_id: str
    title: str | None = None
    created_at: str
    last_activity_at: str
    selected_doc_id: str | None = None
    message_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    """Paginated list of conversation sessions."""

    total: int
    limit: int
    offset: int
    sessions: list[SessionSummarySchema]


class MessageResponse(BaseModel):
    """Full detail of a persisted conversational message turn."""

    message_id: str
    session_id: str
    role: str
    content: str
    created_at: str
    query_config: dict[str, Any] | None = None
    resolved_citations: list[ResolvedCitationSchema] = Field(default_factory=list)
    unresolved_citations: list[UnresolvedCitationSchema] = Field(default_factory=list)
    malformed_citations: list[MalformedCitationSchema] = Field(default_factory=list)
    query_trace: dict[str, Any] | None = None
    provider: str | None = None
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None


class SessionDetailResponse(BaseModel):
    """Detailed view of a conversation session including full message history."""

    session_id: str
    title: str | None = None
    created_at: str
    last_activity_at: str
    selected_doc_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    message_count: int = 0
    messages: list[MessageResponse] = Field(default_factory=list)


class SessionDeleteResponse(BaseModel):
    """Response returned upon successful session deletion."""

    deleted: bool = True
    session_id: str


class SessionQueryRequest(BaseModel):
    """Request payload for querying within a persistent conversation session."""

    query: str = Field(..., min_length=1, description="User question or follow-up prompt")
    document_ids: list[str] | None = Field(
        None, description="Optional document IDs to scope query to (defaults to session doc scope)"
    )
    strategy: str = Field("hybrid", description="Retrieval strategy: 'bm25', 'dense', or 'hybrid'")
    top_k: int = Field(5, ge=1, le=50, description="Final evidence chunk limit for generation")
    candidate_depth: int = Field(
        60, ge=1, le=500, description="Intermediate retrieval candidate fetch depth"
    )
    min_dense_score: float | None = Field(
        None, description="Optional dense retrieval abstention threshold"
    )
    max_history_messages: int = Field(
        10,
        ge=0,
        le=100,
        description="Maximum number of previous messages to include as bounded context",
    )

    @field_validator("top_k", "candidate_depth", "max_history_messages", mode="before")
    @classmethod
    def validate_int_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("Integer field cannot be a boolean.")
        return v

    @field_validator("min_dense_score", mode="before")
    @classmethod
    def validate_float_not_bool(cls, v: Any) -> Any:
        if isinstance(v, bool):
            raise ValueError("min_dense_score cannot be a boolean.")
        return v

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        if isinstance(v, bool):
            raise ValueError("query cannot be a boolean.")
        s = v.strip()
        if not s:
            raise ValueError("query must be a non-empty string.")
        return s

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned: list[str] = []
        for d in v:
            if not isinstance(d, str) or isinstance(d, bool) or not d.strip():
                raise ValueError("document_ids must not contain empty or whitespace-only strings.")
            cleaned.append(d.strip())
        return list(dict.fromkeys(cleaned))

    @field_validator("min_dense_score")
    @classmethod
    def validate_min_dense_score(cls, v: float | None) -> float | None:
        if v is not None:
            if isinstance(v, bool):
                raise ValueError("min_dense_score cannot be a boolean.")
            if not math.isfinite(v):
                raise ValueError("min_dense_score must be a finite float (not NaN or Inf).")
        return v

    @field_validator("strategy")
    @classmethod
    def validate_strategy(cls, v: str) -> str:
        s = v.strip().lower()
        if s not in ("bm25", "dense", "hybrid"):
            raise ValueError(f"Invalid strategy '{v}'. Must be 'bm25', 'dense', or 'hybrid'.")
        return s


class SessionQueryResponse(BaseModel):
    """Structured response container for a conversational query turn within a session."""

    session_id: str
    user_message_id: str
    assistant_message_id: str
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


# --------------------------------------------------------------------------
# Evidence Intelligence Investigation Schemas (STING V2-004)
# --------------------------------------------------------------------------


class CreateInvestigationRequest(BaseModel):
    """Payload to initiate a cross-document evidence investigation."""

    query: str = Field(..., min_length=1, description="Investigation topic or research question")
    document_ids: list[str] = Field(
        ..., min_length=1, description="List of document IDs to investigate"
    )
    title: str | None = Field(None, description="Optional custom title for the investigation")
    comparison_aspects: list[str] | None = Field(
        None, description="Optional comparison dimensions/aspects to evaluate across sources"
    )
    aspects: list[str] | None = Field(None, description="Convenience alias for comparison_aspects")
    run_immediately: bool = Field(
        True, description="Whether to execute the investigation immediately before returning"
    )

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("query must be a non-empty string.")
        return s

    @field_validator("document_ids")
    @classmethod
    def validate_document_ids(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("document_ids must contain at least one document ID.")
        cleaned: list[str] = []
        for d in v:
            if not isinstance(d, str) or isinstance(d, bool) or not d.strip():
                raise ValueError("document_ids must not contain empty or whitespace-only strings.")
            cleaned.append(d.strip())
        unique = list(dict.fromkeys(cleaned))
        if not unique:
            raise ValueError("document_ids must contain at least one valid document ID.")
        return unique

    @field_validator("comparison_aspects")
    @classmethod
    def validate_comparison_aspects(cls, v: list[str] | None) -> list[str] | None:
        if v is None:
            return None
        cleaned: list[str] = []
        for a in v:
            if isinstance(a, str) and a.strip():
                cleaned.append(a.strip())
        return list(dict.fromkeys(cleaned)) if cleaned else None


class EvidenceFindingSchema(BaseModel):
    """A claim or empirical finding extracted from a specific document chunk."""

    finding_id: str
    topic_id: str
    doc_id: str
    doc_title: str
    chunk_id: str
    page_number: int | None = None
    claim: str
    supporting_passage: str
    relationship: str
    confidence: float
    methodology_notes: str | None = None
    created_at: str


class ComparisonTopicSchema(BaseModel):
    """A thematic aspect or research dimension compared across documents."""

    topic_id: str
    topic_name: str
    description: str
    findings: list[EvidenceFindingSchema]
    agreements_summary: str
    disagreements_summary: str
    evidence_gaps: list[str]


class InvestigationReportSchema(BaseModel):
    """Full cross-document comparison report with findings matrix and synthesis."""

    investigation_id: str
    title: str
    query: str
    comparison_aspects: list[str]
    document_ids: list[str]
    document_summaries: list[dict[str, Any]]
    topics: list[ComparisonTopicSchema]
    total_findings: int
    relationship_counts: dict[str, int]
    executive_summary: str
    methodology_synthesis: str
    created_at: str
    completed_at: str | None = None
    status: str
    error_message: str | None = None


class InvestigationSummarySchema(BaseModel):
    """High-level summary of an investigation entity."""

    investigation_id: str
    title: str
    query: str
    comparison_aspects: list[str]
    document_ids: list[str]
    status: str
    created_at: str
    completed_at: str | None = None
    total_findings: int = 0
    error_message: str | None = None


class InvestigationListResponse(BaseModel):
    """Paginated collection of investigations."""

    investigations: list[InvestigationSummarySchema]
    total: int
    limit: int
    offset: int


class InvestigationDetailResponse(BaseModel):
    """Complete detail response of an investigation and its associated report."""

    investigation_id: str
    title: str
    query: str
    comparison_aspects: list[str]
    document_ids: list[str]
    status: str
    created_at: str
    completed_at: str | None = None
    error_message: str | None = None
    report: InvestigationReportSchema | None = None


class InvestigationDeleteResponse(BaseModel):
    """Response returned upon successful investigation deletion."""

    deleted: bool = True
    investigation_id: str


# ============================================================================
# V2-005: RAG Evaluation, Comparison & Regression Studio Schemas
# ============================================================================


class BenchmarkQuerySchema(BaseModel):
    """Schema for an individual evaluation query with ground truth annotations."""

    query_id: str = Field(..., min_length=1, description="Stable query identifier")
    text: str = Field(..., min_length=1, description="Query text")
    relevant_chunk_ids: list[str] = Field(default_factory=list, description="Relevant chunk IDs")
    relevance_grades: dict[str, int] = Field(
        default_factory=dict, description="Graded relevance map (chunk_id -> non-negative int)"
    )
    ground_truth_answer: str | None = Field(None, description="Optional reference answer")
    required_citations: list[str] = Field(
        default_factory=list, description="Required citation identifiers"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary evaluation metadata"
    )


class DatasetSummarySchema(BaseModel):
    """Summary representation of a benchmark evaluation dataset."""

    dataset_id: str
    name: str
    description: str | None = None
    document_scope: list[str] = Field(default_factory=list)
    query_count: int = 0
    is_synthetic: bool = False
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class DatasetDetailResponse(BaseModel):
    """Full detail of a benchmark dataset including all constituent queries."""

    dataset_id: str
    name: str
    description: str | None = None
    document_scope: list[str] = Field(default_factory=list)
    query_count: int = 0
    is_synthetic: bool = False
    created_at: str
    updated_at: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    queries: list[BenchmarkQuerySchema] = Field(default_factory=list)


class DatasetListResponse(BaseModel):
    """Paginated collection of benchmark dataset summaries."""

    datasets: list[DatasetSummarySchema]
    total: int
    limit: int
    offset: int


class CreateDatasetRequest(BaseModel):
    """Request payload to create a new evaluation dataset."""

    name: str = Field(..., min_length=1, max_length=200, description="Dataset title")
    description: str | None = Field(None, max_length=2000, description="Optional description")
    document_scope: list[str] = Field(
        default_factory=list, description="Optional scoped document IDs"
    )
    is_synthetic: bool = Field(False, description="Flag indicating synthetic demonstration data")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Dataset metadata")
    queries: list[BenchmarkQuerySchema] = Field(
        default_factory=list, description="Initial benchmark queries"
    )


class UpdateDatasetRequest(BaseModel):
    """Request payload to update dataset header metadata."""

    name: str | None = Field(None, min_length=1, max_length=200)
    description: str | None = Field(None, max_length=2000)
    document_scope: list[str] | None = None
    metadata: dict[str, Any] | None = None


class CreateQueryRequest(BaseModel):
    """Request payload to add a single query to a dataset."""

    query_id: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    ground_truth_answer: str | None = None
    required_citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateQueryRequest(BaseModel):
    """Request payload to update an existing query in a dataset."""

    text: str = Field(..., min_length=1)
    relevant_chunk_ids: list[str] = Field(default_factory=list)
    relevance_grades: dict[str, int] = Field(default_factory=dict)
    ground_truth_answer: str | None = None
    required_citations: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ImportDatasetRequest(BaseModel):
    """Request payload to import benchmark queries from raw JSONL text."""

    name: str = Field(..., min_length=1, max_length=200)
    jsonl_content: str = Field(..., min_length=1, description="Raw JSONL string")
    description: str | None = Field(None, max_length=2000)
    document_scope: list[str] = Field(default_factory=list)
    is_synthetic: bool = Field(False)


class LaunchEvaluationRequest(BaseModel):
    """Request payload to configure and launch an evaluation experiment."""

    experiment_name: str = Field(..., min_length=1, max_length=200)
    dataset_id: str = Field(..., min_length=1)
    document_ids: list[str] = Field(default_factory=list)
    retriever_type: str = Field("hybrid", description="'bm25', 'faiss', or 'hybrid'")
    retriever_params: dict[str, Any] = Field(default_factory=dict)
    top_k: int = Field(5, ge=1, le=100)
    provider: str = Field("ollama", description="'ollama', 'mock', or 'gemini'")
    model_name: str | None = None
    temperature: float = Field(0.0, ge=0.0, le=2.0)
    max_output_tokens: int = Field(1024, ge=1, le=8192)
    prompt_version: str = Field("v1")
    enable_groundedness: bool = True
    enable_citations: bool = True
    enable_nli: bool = True
    enable_judge: bool = True
    judge_provider: str | None = None
    judge_model: str | None = None


class EvaluationJobResponse(BaseModel):
    """Job status and progress tracking response."""

    job_id: str
    experiment_name: str
    dataset_id: str
    status: str
    queries_completed: int
    total_queries: int
    progress_percent: float
    run_id: str | None = None
    error_message: str | None = None
    created_at: str
    completed_at: str | None = None


class EvaluationJobListResponse(BaseModel):
    """List of recent evaluation jobs."""

    jobs: list[EvaluationJobResponse]


class MetricDefinitionSchema(BaseModel):
    """Definition, direction, and measurement limitations of a metric."""

    name: str
    category: str
    direction: str
    range: str
    source: str
    method: str
    description: str
    limitations: str


class MetricsMetaResponse(BaseModel):
    """Catalog of supported evaluation metrics and diagnostic metadata."""

    metrics: dict[str, MetricDefinitionSchema]


class RunReproducibilityResponse(BaseModel):
    """Reproducibility manifest for an experiment run."""

    run_id: str
    experiment_name: str
    config_hash: str
    dataset_id: str | None = None
    dataset_exists: bool
    corpus_document_ids: list[str] = Field(default_factory=list)
    corpus_documents_exist: bool
    can_reproduce: bool
    metadata: dict[str, Any] = Field(default_factory=dict)
    raw_config: dict[str, Any] = Field(default_factory=dict)
