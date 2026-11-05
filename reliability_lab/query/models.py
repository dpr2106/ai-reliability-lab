"""Immutable domain models for query execution, candidate traces, and citation resolution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class ResolvedCitation:
    """Citation verified against retrieved evidence chunks."""

    marker: str  # "[CIT-001]"
    alias: str  # "CIT-001"
    chunk_id: str  # "doc_123:recursive:0:450"
    doc_id: str
    filename: str
    page_number: int | None
    char_start: int | None
    char_end: int | None
    snippet: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "marker": self.marker,
            "alias": self.alias,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "snippet": self.snippet,
            "is_valid": True,
        }


@dataclass(frozen=True)
class UnresolvedCitation:
    """Citation marker whose alias was not provided in generation evidence."""

    marker: str  # e.g., "[CIT-999]"
    alias: str  # "CIT-999"
    reason: str  # "unknown_alias_not_in_evidence"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "marker": self.marker,
            "alias": self.alias,
            "is_valid": False,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class MalformedCitation:
    """Detected citation-like expression failing standard syntax."""

    raw_text: str  # e.g., "[CIT-ABC]" or "[citation: 1]"
    span: tuple[int, int]
    reason: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "raw_text": self.raw_text,
            "span": list(self.span),
            "is_valid": False,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class SparseCandidateTrace:
    """Ranked candidate retrieved via lexical BM25 retrieval."""

    chunk_id: str
    doc_id: str
    content: str
    page_number: int | None
    char_start: int | None
    char_end: int | None
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class DenseCandidateTrace:
    """Ranked candidate retrieved via dense FAISS vector retrieval."""

    chunk_id: str
    doc_id: str
    content: str
    page_number: int | None
    char_start: int | None
    char_end: int | None
    score: float
    rank: int

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "score": self.score,
            "rank": self.rank,
        }


@dataclass(frozen=True)
class FusedCandidateTrace:
    """Ranked candidate produced by Reciprocal Rank Fusion (RRF)."""

    chunk_id: str
    doc_id: str
    content: str
    page_number: int | None
    char_start: int | None
    char_end: int | None
    fused_score: float
    final_rank: int
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    rrf_contributions: tuple[tuple[str, float], ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "fused_score": self.fused_score,
            "final_rank": self.final_rank,
            "sparse_rank": self.sparse_rank,
            "sparse_score": self.sparse_score,
            "dense_rank": self.dense_rank,
            "dense_score": self.dense_score,
            "rrf_contributions": dict(self.rrf_contributions),
        }


@dataclass(frozen=True)
class EvidenceChunkTrace:
    """Aliased chunk provided as prompt evidence to LLM generation."""

    alias: str  # "CIT-001"
    chunk_id: str
    doc_id: str
    filename: str
    page_number: int | None
    char_start: int | None
    char_end: int | None
    content: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "alias": self.alias,
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "filename": self.filename,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "content": self.content,
        }


@dataclass(frozen=True)
class QueryTrace:
    """Deeply immutable trace recording full provenance and metrics for a query execution."""

    trace_id: str
    query_text: str
    normalized_query: str
    selected_doc_ids: tuple[str, ...]
    requested_strategy: str
    effective_strategy: str
    fallback_reason: str | None
    sparse_candidates: tuple[SparseCandidateTrace, ...]
    dense_candidates: tuple[DenseCandidateTrace, ...]
    fused_candidates: tuple[FusedCandidateTrace, ...]
    evidence_chunks: tuple[EvidenceChunkTrace, ...]
    resolved_citations: tuple[ResolvedCitation, ...]
    unresolved_citations: tuple[UnresolvedCitation, ...]
    malformed_citations: tuple[MalformedCitation, ...]
    abstained: bool
    retrieval_abstained: bool
    retrieval_abstention_reason: str | None
    generation_abstained: bool
    generation_abstention_reason: str | None
    provider: str
    model_name: str
    prompt_tokens: int | None
    completion_tokens: int | None
    retrieval_latency_ms: float
    generation_latency_ms: float
    total_latency_ms: float
    dense_search_mode: str = "flat_exact"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert trace to JSON-serializable dictionary."""
        return {
            "trace_id": self.trace_id,
            "query_text": self.query_text,
            "normalized_query": self.normalized_query,
            "selected_doc_ids": list(self.selected_doc_ids),
            "requested_strategy": self.requested_strategy,
            "effective_strategy": self.effective_strategy,
            "fallback_reason": self.fallback_reason,
            "dense_search_mode": self.dense_search_mode,
            "sparse_candidates": [c.to_dict() for c in self.sparse_candidates],
            "dense_candidates": [c.to_dict() for c in self.dense_candidates],
            "fused_candidates": [c.to_dict() for c in self.fused_candidates],
            "evidence_chunks": [e.to_dict() for e in self.evidence_chunks],
            "resolved_citations": [c.to_dict() for c in self.resolved_citations],
            "unresolved_citations": [c.to_dict() for c in self.unresolved_citations],
            "malformed_citations": [c.to_dict() for c in self.malformed_citations],
            "abstained": self.abstained,
            "retrieval_abstained": self.retrieval_abstained,
            "retrieval_abstention_reason": self.retrieval_abstention_reason,
            "generation_abstained": self.generation_abstained,
            "generation_abstention_reason": self.generation_abstention_reason,
            "provider": self.provider,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
        }

