"""SQLite repository for persistent conversation sessions and message history."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from reliability_lab.session.models import MessageRecord, SessionRecord
from reliability_lab.storage.database import Database

logger = logging.getLogger(__name__)

CREATE_SESSION_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS conversation_sessions (
    session_id TEXT PRIMARY KEY,
    title TEXT,
    created_at TEXT NOT NULL,
    last_activity_at TEXT NOT NULL,
    selected_doc_id TEXT,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS conversation_messages (
    message_id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    created_at TEXT NOT NULL,
    query_config_json TEXT,
    citations_json TEXT,
    query_trace_json TEXT,
    provider TEXT,
    model_name TEXT,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    retrieval_latency_ms REAL,
    generation_latency_ms REAL,
    total_latency_ms REAL,
    FOREIGN KEY (session_id) REFERENCES conversation_sessions(session_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sessions_last_activity
    ON conversation_sessions(last_activity_at DESC);
CREATE INDEX IF NOT EXISTS idx_messages_session_created
    ON conversation_messages(session_id, created_at ASC);
"""


class SessionStorageError(Exception):
    """Base exception for conversation session storage errors."""


class SessionNotFoundError(SessionStorageError):
    """Raised when a requested conversation session does not exist."""


class SessionRepository:
    """SQLite-backed persistent repository for conversation sessions and message turns."""

    def __init__(self, database: Database | str | Path = ":memory:") -> None:
        if isinstance(database, Database):
            self._db = database
            self._owns_db = False
        else:
            self._db = Database(database)
            self._owns_db = True
        self._init_tables()

    @property
    def database(self) -> Database:
        """Return the underlying Database instance."""
        return self._db

    def _init_tables(self) -> None:
        """Idempotently create session and message tables."""
        with self._db.get_connection() as conn:
            conn.executescript(CREATE_SESSION_TABLES_SQL)

    def close(self) -> None:
        """Close the database connection if owned."""
        if self._owns_db:
            self._db.close()

    def __enter__(self) -> SessionRepository:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def create_session(
        self,
        session_id: str | None = None,
        title: str | None = None,
        selected_doc_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> SessionRecord:
        """Create and persist a new conversation session."""
        sid = session_id.strip() if session_id and session_id.strip() else f"ses_{uuid4().hex[:16]}"
        now = datetime.now(timezone.utc).isoformat()
        clean_title = title.strip() if title and title.strip() else None
        clean_doc_id = (
            selected_doc_id.strip() if selected_doc_id and selected_doc_id.strip() else None
        )
        clean_meta = dict(metadata) if metadata else {}

        sql = """
        INSERT INTO conversation_sessions (
            session_id, title, created_at, last_activity_at, selected_doc_id, metadata_json
        ) VALUES (?, ?, ?, ?, ?, ?);
        """
        with self._db.transaction() as conn:
            conn.execute(
                sql,
                (
                    sid,
                    clean_title,
                    now,
                    now,
                    clean_doc_id,
                    json.dumps(clean_meta, ensure_ascii=False),
                ),
            )

        return SessionRecord(
            session_id=sid,
            title=clean_title,
            created_at=now,
            last_activity_at=now,
            selected_doc_id=clean_doc_id,
            metadata=clean_meta,
            message_count=0,
            messages=(),
        )

    def get_session(self, session_id: str, include_messages: bool = True) -> SessionRecord | None:
        """Retrieve a session by its ID, optionally with full message history."""
        if not session_id or not session_id.strip():
            return None

        clean_sid = session_id.strip()
        sql = """
        SELECT session_id, title, created_at, last_activity_at, selected_doc_id, metadata_json,
               (SELECT COUNT(*) FROM conversation_messages WHERE session_id = ?) as msg_count
        FROM conversation_sessions
        WHERE session_id = ?;
        """
        rows = self._db.execute_query(sql, (clean_sid, clean_sid))
        if not rows:
            return None

        row = rows[0]
        try:
            meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
        except Exception:
            meta = {}

        messages: tuple[MessageRecord, ...] = ()
        if include_messages:
            messages = tuple(self.get_messages(clean_sid))

        return SessionRecord(
            session_id=row["session_id"],
            title=row["title"],
            created_at=row["created_at"],
            last_activity_at=row["last_activity_at"],
            selected_doc_id=row["selected_doc_id"],
            metadata=meta,
            message_count=int(row["msg_count"]),
            messages=messages,
        )

    def list_sessions(self, limit: int = 50, offset: int = 0) -> tuple[list[SessionRecord], int]:
        """List sessions ordered by last_activity_at descending with pagination."""
        safe_limit = max(1, min(limit, 500))
        safe_offset = max(0, offset)

        count_sql = "SELECT COUNT(*) as total FROM conversation_sessions;"
        count_rows = self._db.execute_query(count_sql)
        total = int(count_rows[0]["total"]) if count_rows else 0

        sql = """
        SELECT s.session_id, s.title, s.created_at, s.last_activity_at,
               s.selected_doc_id, s.metadata_json,
               COUNT(m.message_id) as msg_count
        FROM conversation_sessions s
        LEFT JOIN conversation_messages m ON s.session_id = m.session_id
        GROUP BY s.session_id
        ORDER BY s.last_activity_at DESC
        LIMIT ? OFFSET ?;
        """
        rows = self._db.execute_query(sql, (safe_limit, safe_offset))

        sessions: list[SessionRecord] = []
        for r in rows:
            try:
                meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
            except Exception:
                meta = {}
            sessions.append(
                SessionRecord(
                    session_id=r["session_id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    last_activity_at=r["last_activity_at"],
                    selected_doc_id=r["selected_doc_id"],
                    metadata=meta,
                    message_count=int(r["msg_count"]),
                    messages=(),
                )
            )

        return sessions, total

    def delete_session(self, session_id: str) -> bool:
        """Delete a session and all its messages (via ON DELETE CASCADE)."""
        if not session_id or not session_id.strip():
            return False

        clean_sid = session_id.strip()
        sql = "DELETE FROM conversation_sessions WHERE session_id = ?;"
        with self._db.transaction() as conn:
            cursor = conn.execute(sql, (clean_sid,))
            return cursor.rowcount > 0

    def _row_to_message(self, row: Any) -> MessageRecord:
        """Helper to convert a sqlite3.Row to MessageRecord."""
        try:
            q_config = json.loads(row["query_config_json"]) if row["query_config_json"] else {}
        except Exception:
            q_config = {}

        try:
            cits = json.loads(row["citations_json"]) if row["citations_json"] else {}
        except Exception:
            cits = {}

        try:
            q_trace = json.loads(row["query_trace_json"]) if row["query_trace_json"] else None
        except Exception:
            q_trace = None

        return MessageRecord(
            message_id=row["message_id"],
            session_id=row["session_id"],
            role=row["role"],
            content=row["content"],
            created_at=row["created_at"],
            query_config=q_config,
            resolved_citations=tuple(cits.get("resolved", ())),
            unresolved_citations=tuple(cits.get("unresolved", ())),
            malformed_citations=tuple(cits.get("malformed", ())),
            query_trace=q_trace,
            provider=row["provider"],
            model_name=row["model_name"],
            prompt_tokens=row["prompt_tokens"],
            completion_tokens=row["completion_tokens"],
            retrieval_latency_ms=row["retrieval_latency_ms"],
            generation_latency_ms=row["generation_latency_ms"],
            total_latency_ms=row["total_latency_ms"],
        )

    def get_messages(
        self, session_id: str, limit: int | None = None, offset: int = 0
    ) -> list[MessageRecord]:
        """Retrieve messages for a session in chronological order."""
        clean_sid = session_id.strip()
        if limit is not None:
            sql = """
            SELECT * FROM conversation_messages
            WHERE session_id = ?
            ORDER BY created_at ASC
            LIMIT ? OFFSET ?;
            """
            rows = self._db.execute_query(sql, (clean_sid, max(1, limit), max(0, offset)))
        else:
            sql = """
            SELECT * FROM conversation_messages
            WHERE session_id = ?
            ORDER BY created_at ASC;
            """
            rows = self._db.execute_query(sql, (clean_sid,))

        return [self._row_to_message(r) for r in rows]

    def get_bounded_history(self, session_id: str, max_messages: int = 10) -> list[MessageRecord]:
        """Retrieve latest messages for a session in chronological order, bounded by max_messages.

        Strictly enforces session isolation: only messages matching exact session_id are retrieved.
        """
        clean_sid = session_id.strip()
        safe_max = max(1, min(max_messages, 100))

        # Fetch latest N messages ordered by created_at DESC, then reverse to chronological order
        sql = """
        SELECT * FROM conversation_messages
        WHERE session_id = ?
        ORDER BY created_at DESC
        LIMIT ?;
        """
        rows = self._db.execute_query(sql, (clean_sid, safe_max))
        return [self._row_to_message(r) for r in reversed(rows)]

    def add_message(self, message: MessageRecord) -> MessageRecord:
        """Add a single message to a session and update the session's last_activity_at timestamp."""
        citations_data = {
            "resolved": list(message.resolved_citations),
            "unresolved": list(message.unresolved_citations),
            "malformed": list(message.malformed_citations),
        }
        sql_msg = """
        INSERT INTO conversation_messages (
            message_id, session_id, role, content, created_at, query_config_json,
            citations_json, query_trace_json, provider, model_name,
            prompt_tokens, completion_tokens, retrieval_latency_ms,
            generation_latency_ms, total_latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        sql_session_touch = """
        UPDATE conversation_sessions
        SET last_activity_at = ?
        WHERE session_id = ?;
        """

        with self._db.transaction() as conn:
            conn.execute(
                sql_msg,
                (
                    message.message_id,
                    message.session_id,
                    message.role,
                    message.content,
                    message.created_at,
                    json.dumps(message.query_config, ensure_ascii=False)
                    if message.query_config
                    else None,
                    json.dumps(citations_data, ensure_ascii=False),
                    json.dumps(message.query_trace, ensure_ascii=False)
                    if message.query_trace
                    else None,
                    message.provider,
                    message.model_name,
                    message.prompt_tokens,
                    message.completion_tokens,
                    message.retrieval_latency_ms,
                    message.generation_latency_ms,
                    message.total_latency_ms,
                ),
            )
            cursor = conn.execute(sql_session_touch, (message.created_at, message.session_id))
            if cursor.rowcount == 0:
                raise SessionNotFoundError(
                    f"Session '{message.session_id}' not found when appending message."
                )

        return message

    def add_turn(
        self,
        session_id: str,
        user_message: MessageRecord,
        assistant_message: MessageRecord,
    ) -> None:
        """Atomically persist a user turn and assistant response within a single transaction.

        Guarantees that both messages succeed together and session last_activity_at is updated,
        or the transaction completely rolls back.
        """
        clean_sid = session_id.strip()
        sql_msg = """
        INSERT INTO conversation_messages (
            message_id, session_id, role, content, created_at, query_config_json,
            citations_json, query_trace_json, provider, model_name,
            prompt_tokens, completion_tokens, retrieval_latency_ms,
            generation_latency_ms, total_latency_ms
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """
        sql_session_touch = """
        UPDATE conversation_sessions
        SET last_activity_at = ?
        WHERE session_id = ?;
        """

        with self._db.transaction() as conn:
            # 1. Verify session exists
            cur = conn.execute(
                "SELECT session_id FROM conversation_sessions WHERE session_id = ?;",
                (clean_sid,),
            )
            if not cur.fetchone():
                raise SessionNotFoundError(f"Session '{clean_sid}' not found when appending turn.")

            # 2. Insert user message
            user_cits = {
                "resolved": list(user_message.resolved_citations),
                "unresolved": list(user_message.unresolved_citations),
                "malformed": list(user_message.malformed_citations),
            }
            conn.execute(
                sql_msg,
                (
                    user_message.message_id,
                    clean_sid,
                    user_message.role,
                    user_message.content,
                    user_message.created_at,
                    json.dumps(user_message.query_config, ensure_ascii=False)
                    if user_message.query_config
                    else None,
                    json.dumps(user_cits, ensure_ascii=False),
                    json.dumps(user_message.query_trace, ensure_ascii=False)
                    if user_message.query_trace
                    else None,
                    user_message.provider,
                    user_message.model_name,
                    user_message.prompt_tokens,
                    user_message.completion_tokens,
                    user_message.retrieval_latency_ms,
                    user_message.generation_latency_ms,
                    user_message.total_latency_ms,
                ),
            )

            # 3. Insert assistant message
            asst_cits = {
                "resolved": list(assistant_message.resolved_citations),
                "unresolved": list(assistant_message.unresolved_citations),
                "malformed": list(assistant_message.malformed_citations),
            }
            conn.execute(
                sql_msg,
                (
                    assistant_message.message_id,
                    clean_sid,
                    assistant_message.role,
                    assistant_message.content,
                    assistant_message.created_at,
                    json.dumps(assistant_message.query_config, ensure_ascii=False)
                    if assistant_message.query_config
                    else None,
                    json.dumps(asst_cits, ensure_ascii=False),
                    json.dumps(assistant_message.query_trace, ensure_ascii=False)
                    if assistant_message.query_trace
                    else None,
                    assistant_message.provider,
                    assistant_message.model_name,
                    assistant_message.prompt_tokens,
                    assistant_message.completion_tokens,
                    assistant_message.retrieval_latency_ms,
                    assistant_message.generation_latency_ms,
                    assistant_message.total_latency_ms,
                ),
            )

            # 4. Touch session last activity
            conn.execute(sql_session_touch, (assistant_message.created_at, clean_sid))
