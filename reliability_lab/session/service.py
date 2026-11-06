"""Session management and session-aware conversational query service (STING V2-003 Phase B)."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Literal
from uuid import uuid4

from reliability_lab.llm.models import LLMMessage
from reliability_lab.query.models import QueryTrace
from reliability_lab.query.orchestrator import DocumentQueryService
from reliability_lab.session.models import MessageRecord, SessionRecord
from reliability_lab.session.repository import SessionNotFoundError, SessionRepository

logger = logging.getLogger(__name__)

DEFAULT_MAX_HISTORY_MESSAGES: int = 10


class SessionService:
    """Orchestrates persistent session lifecycle, bounded context retrieval, and query execution."""

    def __init__(
        self,
        session_repository: SessionRepository,
        query_service: DocumentQueryService,
        default_max_history: int = DEFAULT_MAX_HISTORY_MESSAGES,
    ) -> None:
        self.session_repository = session_repository
        self.query_service = query_service
        self.default_max_history = max(1, default_max_history)

    def create_session(
        self,
        title: str | None = None,
        selected_doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """Create and persist a new conversation session."""
        return self.session_repository.create_session(
            title=title,
            selected_doc_id=selected_doc_id,
            metadata=metadata,
        )

    def get_session(self, session_id: str) -> SessionRecord:
        """Retrieve a session with full message history or raise SessionNotFoundError."""
        session = self.session_repository.get_session(session_id, include_messages=True)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return session

    def list_sessions(self, limit: int = 50, offset: int = 0) -> tuple[list[SessionRecord], int]:
        """List sessions ordered by last_activity_at descending with pagination."""
        return self.session_repository.list_sessions(limit=limit, offset=offset)

    def delete_session(self, session_id: str) -> bool:
        """Delete a session by ID or raise SessionNotFoundError."""
        deleted = self.session_repository.delete_session(session_id)
        if not deleted:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")
        return True

    def execute_session_query(
        self,
        session_id: str,
        query_text: str,
        document_ids: Sequence[str] | None = None,
        strategy: Literal["bm25", "dense", "hybrid"] | str = "hybrid",
        top_k: int = 5,
        candidate_depth: int = 60,
        min_dense_score: float | None = None,
        max_history_messages: int | None = None,
        prompt_version: str = "v1",
    ) -> tuple[str, QueryTrace, MessageRecord, MessageRecord]:
        """Execute a conversational query within a session using bounded history.

        Guarantees:
        1. Validates session existence.
        2. Resolves target document IDs: falls back to session's selected_doc_id if omitted.
        3. Strictly bounds conversational context to the requested session.
        4. Keeps previous turns as conversational context only; citations resolve
           strictly against current-query document evidence chunks.
        5. Atomically persists user and assistant message records upon query completion.

        Returns:
            Tuple of (answer_text, query_trace, user_message, assistant_message).
        """
        # 1. Validate session existence
        session = self.session_repository.get_session(session_id, include_messages=False)
        if session is None:
            raise SessionNotFoundError(f"Session '{session_id}' not found.")

        # 2. Resolve document IDs
        effective_doc_ids: list[str] = []
        if document_ids:
            effective_doc_ids = [
                d.strip() for d in document_ids if isinstance(d, str) and d.strip()
            ]
        elif session.selected_doc_id:
            effective_doc_ids = [session.selected_doc_id]

        if not effective_doc_ids:
            raise ValueError(
                f"No document selected for query and session '{session_id}' "
                "has no default document scope."
            )

        # 3. Retrieve bounded conversation context strictly from this session
        history_bound = (
            max(0, max_history_messages)
            if max_history_messages is not None
            else self.default_max_history
        )
        bounded_history = (
            self.session_repository.get_bounded_history(session_id, max_messages=history_bound)
            if history_bound > 0
            else []
        )

        conversation_context: list[LLMMessage] = []
        for m in bounded_history:
            if m.role == "user":
                conversation_context.append(LLMMessage(role="user", content=m.content))
            elif m.role == "assistant":
                conversation_context.append(LLMMessage(role="assistant", content=m.content))

        # 4. Execute evidence-first query with conversational context
        query_config = {
            "strategy": strategy,
            "top_k": top_k,
            "candidate_depth": candidate_depth,
            "min_dense_score": min_dense_score,
            "prompt_version": prompt_version,
            "document_ids": effective_doc_ids,
            "max_history_messages": history_bound,
        }

        answer, trace = self.query_service.query(
            query_text=query_text,
            selected_doc_ids=effective_doc_ids,
            strategy=strategy,
            top_k=top_k,
            candidate_depth=candidate_depth,
            min_dense_score=min_dense_score,
            prompt_version=prompt_version,
            conversation_context=conversation_context,
        )

        # 5. Build message records
        now_user = datetime.now(timezone.utc).isoformat()
        now_asst = datetime.now(timezone.utc).isoformat()

        user_message = MessageRecord(
            message_id=f"msg_{uuid4().hex[:16]}",
            session_id=session_id,
            role="user",
            content=query_text.strip(),
            created_at=now_user,
            query_config=query_config,
        )

        assistant_message = MessageRecord(
            message_id=f"msg_{uuid4().hex[:16]}",
            session_id=session_id,
            role="assistant",
            content=answer,
            created_at=now_asst,
            query_config=query_config,
            resolved_citations=tuple(c.to_dict() for c in trace.resolved_citations),
            unresolved_citations=tuple(c.to_dict() for c in trace.unresolved_citations),
            malformed_citations=tuple(c.to_dict() for c in trace.malformed_citations),
            query_trace=trace.to_dict(),
            provider=trace.provider,
            model_name=trace.model_name,
            prompt_tokens=trace.prompt_tokens,
            completion_tokens=trace.completion_tokens,
            retrieval_latency_ms=trace.retrieval_latency_ms,
            generation_latency_ms=trace.generation_latency_ms,
            total_latency_ms=trace.total_latency_ms,
        )

        # 6. Atomically persist conversation turn
        self.session_repository.add_turn(
            session_id=session_id,
            user_message=user_message,
            assistant_message=assistant_message,
        )

        return answer, trace, user_message, assistant_message
