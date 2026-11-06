"""Domain models for persistent conversation sessions and message turns (STING V2-003 Phase B)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class MessageRecord:
    """Immutable domain representation of a persistent conversation message."""

    message_id: str
    session_id: str
    role: str  # 'user' | 'assistant'
    content: str
    created_at: str
    query_config: dict[str, Any] = field(default_factory=dict)
    resolved_citations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    unresolved_citations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    malformed_citations: tuple[dict[str, Any], ...] = field(default_factory=tuple)
    query_trace: dict[str, Any] | None = None
    provider: str | None = None
    model_name: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert message record to JSON-serializable dictionary."""
        return {
            "message_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "created_at": self.created_at,
            "query_config": self.query_config,
            "resolved_citations": list(self.resolved_citations),
            "unresolved_citations": list(self.unresolved_citations),
            "malformed_citations": list(self.malformed_citations),
            "query_trace": self.query_trace,
            "provider": self.provider,
            "model_name": self.model_name,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "retrieval_latency_ms": self.retrieval_latency_ms,
            "generation_latency_ms": self.generation_latency_ms,
            "total_latency_ms": self.total_latency_ms,
        }


@dataclass(frozen=True)
class SessionRecord:
    """Immutable domain representation of a conversation session header and history."""

    session_id: str
    title: str | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_activity_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    selected_doc_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    message_count: int = 0
    messages: tuple[MessageRecord, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        """Convert session record to JSON-serializable dictionary."""
        return {
            "session_id": self.session_id,
            "title": self.title,
            "created_at": self.created_at,
            "last_activity_at": self.last_activity_at,
            "selected_doc_id": self.selected_doc_id,
            "metadata": self.metadata,
            "message_count": self.message_count,
            "messages": [m.to_dict() for m in self.messages],
        }
