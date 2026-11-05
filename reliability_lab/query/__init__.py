"""STING Query & Trace Orchestration package.

Provides evidence-first, multi-document retrieval orchestration, citation aliasing
and resolution, strategy-specific retrieval and generation abstention, and immutable
structured query tracing.
"""

from __future__ import annotations

from reliability_lab.query.citation import (
    CitationAliasMap,
    CitationResolver,
)
from reliability_lab.query.models import (
    DenseCandidateTrace,
    EvidenceChunkTrace,
    FusedCandidateTrace,
    MalformedCitation,
    QueryTrace,
    ResolvedCitation,
    SparseCandidateTrace,
    UnresolvedCitation,
)
from reliability_lab.query.orchestrator import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentQueryService,
    UnprocessableDocumentError,
)

__all__ = [
    "CitationAliasMap",
    "CitationResolver",
    "DenseCandidateTrace",
    "DocumentConflictError",
    "DocumentNotFoundError",
    "DocumentQueryService",
    "EvidenceChunkTrace",
    "FusedCandidateTrace",
    "MalformedCitation",
    "QueryTrace",
    "ResolvedCitation",
    "SparseCandidateTrace",
    "UnprocessableDocumentError",
    "UnresolvedCitation",
]

