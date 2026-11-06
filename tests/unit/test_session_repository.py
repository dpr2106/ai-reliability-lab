"""Unit tests for SessionRepository (STING V2-003 Phase B)."""

from __future__ import annotations

from pathlib import Path

import pytest

from reliability_lab.session.models import MessageRecord
from reliability_lab.session.repository import (
    SessionNotFoundError,
    SessionRepository,
)
from reliability_lab.storage.database import Database


class TestSessionRepository:
    """Test suite verifying session and message persistence in SQLite."""

    def test_create_and_get_session(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "sessions.db"))
        repo = SessionRepository(db)

        session = repo.create_session(
            title="Analysis Session",
            selected_doc_id="doc_123",
            metadata={"user": "tester", "env": "unit"},
        )

        assert session.session_id.startswith("ses_")
        assert session.title == "Analysis Session"
        assert session.selected_doc_id == "doc_123"
        assert session.metadata == {"user": "tester", "env": "unit"}
        assert session.message_count == 0
        assert session.messages == ()

        # Retrieve session
        fetched = repo.get_session(session.session_id)
        assert fetched is not None
        assert fetched.session_id == session.session_id
        assert fetched.title == "Analysis Session"
        assert fetched.selected_doc_id == "doc_123"
        assert fetched.metadata == {"user": "tester", "env": "unit"}
        assert fetched.message_count == 0

    def test_get_session_nonexistent_returns_none(self, tmp_path: Path) -> None:
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))
        assert repo.get_session("nonexistent_session") is None
        assert repo.get_session("") is None

    def test_list_sessions_ordering_and_pagination(self, tmp_path: Path) -> None:
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))

        s1 = repo.create_session(title="First Session")
        s2 = repo.create_session(title="Second Session")
        s3 = repo.create_session(title="Third Session")
        assert s2.session_id and s3.session_id

        # Touch s1 so it has newest activity
        msg = MessageRecord(
            message_id="msg_touch",
            session_id=s1.session_id,
            role="user",
            content="Hello",
            created_at="2026-11-06T23:59:59Z",
        )
        repo.add_message(msg)

        sessions, total = repo.list_sessions(limit=10, offset=0)
        assert total == 3
        assert len(sessions) == 3
        # s1 has the most recent activity
        assert sessions[0].session_id == s1.session_id
        assert sessions[0].message_count == 1

        # Pagination
        page1, _ = repo.list_sessions(limit=2, offset=0)
        page2, _ = repo.list_sessions(limit=2, offset=2)
        assert len(page1) == 2
        assert len(page2) == 1
        assert page1[0].session_id != page2[0].session_id

    def test_delete_session_cascades_messages(self, tmp_path: Path) -> None:
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))
        session = repo.create_session(title="To Delete")

        user_msg = MessageRecord(
            message_id="msg_user",
            session_id=session.session_id,
            role="user",
            content="Question",
            created_at="2026-11-06T12:00:00Z",
        )
        asst_msg = MessageRecord(
            message_id="msg_asst",
            session_id=session.session_id,
            role="assistant",
            content="Answer",
            created_at="2026-11-06T12:00:01Z",
        )
        repo.add_turn(session.session_id, user_msg, asst_msg)

        assert len(repo.get_messages(session.session_id)) == 2

        # Delete session
        deleted = repo.delete_session(session.session_id)
        assert deleted is True

        assert repo.get_session(session.session_id) is None
        assert repo.get_messages(session.session_id) == []

        # Deleting nonexistent returns False
        assert repo.delete_session("nonexistent") is False

    def test_message_round_tripping_with_full_metadata(self, tmp_path: Path) -> None:
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))
        session = repo.create_session(title="Round Trip Session")

        user_msg = MessageRecord(
            message_id="msg_u1",
            session_id=session.session_id,
            role="user",
            content="What is STING?",
            created_at="2026-11-06T10:00:00Z",
            query_config={"strategy": "hybrid", "top_k": 3},
        )
        asst_msg = MessageRecord(
            message_id="msg_a1",
            session_id=session.session_id,
            role="assistant",
            content="STING is a reliability workbench [CIT-001].",
            created_at="2026-11-06T10:00:02Z",
            query_config={"strategy": "hybrid", "top_k": 3},
            resolved_citations=(
                {
                    "marker": "[CIT-001]",
                    "alias": "CIT-001",
                    "chunk_id": "doc_1:chunk_0",
                    "doc_id": "doc_1",
                    "filename": "sting.pdf",
                    "snippet": "STING is...",
                    "is_valid": True,
                },
            ),
            unresolved_citations=(),
            malformed_citations=(),
            query_trace={"trace_id": "trc_123", "abstained": False},
            provider="mock-llm",
            model_name="mock-model",
            prompt_tokens=150,
            completion_tokens=40,
            retrieval_latency_ms=12.5,
            generation_latency_ms=45.2,
            total_latency_ms=57.7,
        )

        repo.add_turn(session.session_id, user_msg, asst_msg)

        msgs = repo.get_messages(session.session_id)
        assert len(msgs) == 2

        m0 = msgs[0]
        assert m0.message_id == "msg_u1"
        assert m0.role == "user"
        assert m0.content == "What is STING?"
        assert m0.query_config == {"strategy": "hybrid", "top_k": 3}

        m1 = msgs[1]
        assert m1.message_id == "msg_a1"
        assert m1.role == "assistant"
        assert len(m1.resolved_citations) == 1
        assert m1.resolved_citations[0]["alias"] == "CIT-001"
        assert m1.query_trace == {"trace_id": "trc_123", "abstained": False}
        assert m1.provider == "mock-llm"
        assert m1.model_name == "mock-model"
        assert m1.prompt_tokens == 150
        assert m1.completion_tokens == 40
        assert m1.retrieval_latency_ms == 12.5
        assert m1.generation_latency_ms == 45.2
        assert m1.total_latency_ms == 57.7

    def test_session_isolation_strict(self, tmp_path: Path) -> None:
        """Messages from session A must never appear in session B."""
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))
        session_a = repo.create_session(title="Session A")
        session_b = repo.create_session(title="Session B")

        msg_a = MessageRecord(
            message_id="msg_a1",
            session_id=session_a.session_id,
            role="user",
            content="A content",
            created_at="2026-11-06T10:00:00Z",
        )
        msg_b = MessageRecord(
            message_id="msg_b1",
            session_id=session_b.session_id,
            role="user",
            content="B content",
            created_at="2026-11-06T10:01:00Z",
        )

        repo.add_message(msg_a)
        repo.add_message(msg_b)

        history_a = repo.get_bounded_history(session_a.session_id, max_messages=10)
        history_b = repo.get_bounded_history(session_b.session_id, max_messages=10)

        assert len(history_a) == 1
        assert history_a[0].message_id == "msg_a1"
        assert history_a[0].content == "A content"

        assert len(history_b) == 1
        assert history_b[0].message_id == "msg_b1"
        assert history_b[0].content == "B content"

    def test_bounded_history_chronological_ordering(self, tmp_path: Path) -> None:
        """Retrieving bounded history returns the latest N messages in chronological order."""
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))
        session = repo.create_session(title="Bounded History Test")

        # Create 10 messages
        for i in range(10):
            repo.add_message(
                MessageRecord(
                    message_id=f"msg_{i}",
                    session_id=session.session_id,
                    role="user" if i % 2 == 0 else "assistant",
                    content=f"Message {i}",
                    created_at=f"2026-11-06T10:{i:02d}:00Z",
                )
            )

        # Request bound of 4: should get msg_6, msg_7, msg_8, msg_9 in chronological order
        bounded = repo.get_bounded_history(session.session_id, max_messages=4)
        assert len(bounded) == 4
        assert [m.message_id for m in bounded] == ["msg_6", "msg_7", "msg_8", "msg_9"]
        assert [m.content for m in bounded] == [
            "Message 6",
            "Message 7",
            "Message 8",
            "Message 9",
        ]

    def test_persistence_across_repository_reconstruction(self, tmp_path: Path) -> None:
        """Data persists cleanly when repository is reconstructed on same database file."""
        db_path = tmp_path / "persistent_sessions.sqlite3"

        repo1 = SessionRepository(Database(str(db_path)))
        session = repo1.create_session(title="Persistent Session", selected_doc_id="doc_x")
        repo1.add_message(
            MessageRecord(
                message_id="msg_p1",
                session_id=session.session_id,
                role="user",
                content="Persistent test query",
                created_at="2026-11-06T12:00:00Z",
            )
        )
        repo1.close()

        # Reconstruct repo on the same file
        repo2 = SessionRepository(Database(str(db_path)))
        fetched = repo2.get_session(session.session_id)
        assert fetched is not None
        assert fetched.title == "Persistent Session"
        assert fetched.selected_doc_id == "doc_x"
        assert len(fetched.messages) == 1
        assert fetched.messages[0].content == "Persistent test query"
        repo2.close()

    def test_transaction_rollback_on_failed_message_persistence(self, tmp_path: Path) -> None:
        """Atomic add_turn transaction rolls back if session does not exist or failure occurs."""
        repo = SessionRepository(Database(str(tmp_path / "sessions.db")))

        user_msg = MessageRecord(
            message_id="msg_orphan_u",
            session_id="nonexistent_session",
            role="user",
            content="Will fail",
            created_at="2026-11-06T12:00:00Z",
        )
        asst_msg = MessageRecord(
            message_id="msg_orphan_a",
            session_id="nonexistent_session",
            role="assistant",
            content="Will fail",
            created_at="2026-11-06T12:00:01Z",
        )

        with pytest.raises(SessionNotFoundError):
            repo.add_turn("nonexistent_session", user_msg, asst_msg)

        # Verify neither message was persisted
        assert repo.get_messages("nonexistent_session") == []
