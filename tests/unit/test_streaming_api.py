"""Integration tests for FastAPI Server-Sent Events (SSE) endpoints (STING V2-003 Phase C).

Verifies standalone and session-aware SSE endpoints, event framing, progressive token delivery,
transactional turn persistence on stream completion, and failure resilience.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reliability_lab.api.main import (
    create_app,
    get_document_repository,
    get_ingestion_service,
    get_query_service,
    get_session_repository,
    get_session_service,
)
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import MockLLMClient
from reliability_lab.query.orchestrator import DocumentQueryService
from reliability_lab.session.repository import SessionRepository
from reliability_lab.session.service import SessionService
from reliability_lab.storage.database import Database


class ApiTestEmbedder(BaseEmbedder):
    """Deterministic 4D embedder for streaming API tests."""

    @property
    def model_name(self) -> str:
        return "api-stream-embedder"

    @property
    def artifact_fingerprint(self) -> str:
        return "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    @property
    def embedding_dim(self) -> int:
        return 4

    def embed_texts(self, texts: Any, batch_size: int = 32) -> np.ndarray:
        vecs = []
        for t in texts:
            if "alpha" in str(t).lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                vecs.append([0.2, 0.2, 0.2, 0.2])
        return np.array(vecs, dtype=np.float32)


def parse_sse_events(text: str) -> list[dict[str, Any]]:
    """Parse raw SSE text output into structured event dictionaries."""
    events: list[dict[str, Any]] = []
    current_event: str | None = None
    current_data: str | None = None

    for line in text.split("\n"):
        line = line.strip()
        if not line:
            if current_event is not None and current_data is not None:
                try:
                    payload = json.loads(current_data)
                except Exception:
                    payload = current_data
                events.append({"event": current_event, "data": payload})
                current_event = None
                current_data = None
            continue

        if line.startswith("event:"):
            current_event = line[len("event:") :].strip()
        elif line.startswith("data:"):
            data_val = line[len("data:") :].strip()
            if current_data is None:
                current_data = data_val
            else:
                current_data += "\n" + data_val

    if current_event is not None and current_data is not None:
        try:
            payload = json.loads(current_data)
        except Exception:
            payload = current_data
        events.append({"event": current_event, "data": payload})

    return events


@pytest.fixture
def stream_env(tmp_path: Path) -> dict[str, Any]:
    db = Database(str(tmp_path / "stream_api_test.db"))
    doc_repo = DocumentRepository(db)
    session_repo = SessionRepository(db)
    embedder = ApiTestEmbedder()
    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "documents",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )
    mock_llm = MockLLMClient(
        default_response="Alpha provides verified deterministic execution [CIT-001]."
    )
    query_service = DocumentQueryService(
        repository=doc_repo,
        ingestion_service=ingestion,
        embedder=embedder,
        llm_client=mock_llm,
    )
    session_service = SessionService(
        session_repository=session_repo,
        query_service=query_service,
        default_max_history=10,
    )

    app = create_app()
    app.dependency_overrides[get_document_repository] = lambda: doc_repo
    app.dependency_overrides[get_session_repository] = lambda: session_repo
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_query_service] = lambda: query_service
    app.dependency_overrides[get_session_service] = lambda: session_service

    client = TestClient(app)

    # Ingest a ready test document
    doc_content = "Alpha architecture provides high reliability and deterministic verification."
    rec = ingestion.save_upload_stream("doc_alpha.txt", io.BytesIO(doc_content.encode("utf-8")))
    ready_doc = ingestion.process_document(rec.doc_id)

    return {
        "client": client,
        "session_repo": session_repo,
        "doc_id": ready_doc.doc_id,
        "mock_llm": mock_llm,
        "ingestion": ingestion,
    }


class TestStreamingApiEndpoints:
    """Test suite for FastAPI SSE streaming endpoints."""

    def test_post_query_stream_success(self, stream_env: dict[str, Any]) -> None:
        """Verify POST /api/query/stream delivers valid SSE headers and ordered events."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]

        resp = client.post(
            "/api/query/stream",
            json={
                "query": "What does Alpha architecture provide?",
                "document_ids": [doc_id],
                "strategy": "hybrid",
            },
        )

        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/event-stream")
        assert "no-cache" in resp.headers["cache-control"]
        assert resp.headers["x-accel-buffering"] == "no"

        events = parse_sse_events(resp.text)
        assert len(events) >= 3

        event_names = [e["event"] for e in events]
        assert event_names[0] == "start"
        assert "token" in event_names
        assert "citation" in event_names
        assert event_names[-1] == "complete"

        complete_data = events[-1]["data"]
        assert "Alpha provides verified" in complete_data["answer"]
        assert len(complete_data["resolved_citations"]) == 1
        assert complete_data["resolved_citations"][0]["marker"] == "[CIT-001]"

    def test_post_query_stream_incremental_delivery(self, stream_env: dict[str, Any]) -> None:
        """Verify stream yields tokens progressively using client.stream."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]
        mock_llm: MockLLMClient = stream_env["mock_llm"]

        custom_tokens = ["Alpha ", "guarantees ", "safe ", "execution ", "[CIT-001]."]
        mock_llm.stream_chunks = custom_tokens

        with client.stream(
            "POST",
            "/api/query/stream",
            json={
                "query": "What does Alpha guarantee?",
                "document_ids": [doc_id],
                "strategy": "hybrid",
            },
        ) as response:
            assert response.status_code == 200
            token_count = 0
            for line in response.iter_lines():
                if line.startswith("event: token"):
                    token_count += 1

            assert token_count == len(custom_tokens)

    def test_post_query_stream_missing_document_returns_404(
        self, stream_env: dict[str, Any]
    ) -> None:
        """Verify querying non-existent document returns HTTP 404 upfront."""
        client: TestClient = stream_env["client"]
        resp = client.post(
            "/api/query/stream",
            json={
                "query": "Question",
                "document_ids": ["doc_nonexistent"],
            },
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_post_query_stream_skipped_document_returns_422(
        self, stream_env: dict[str, Any]
    ) -> None:
        """Verify querying empty/skipped document returns HTTP 422 upfront."""
        client: TestClient = stream_env["client"]
        ingestion: DocumentIngestionService = stream_env["ingestion"]

        rec = ingestion.save_upload_stream("empty.txt", io.BytesIO(b"   \n"))
        skipped_doc = ingestion.process_document(rec.doc_id)

        resp = client.post(
            "/api/query/stream",
            json={
                "query": "Question",
                "document_ids": [skipped_doc.doc_id],
            },
        )
        assert resp.status_code == 422
        assert "skipped" in resp.json()["detail"].lower()

    def test_post_session_query_stream_success_and_persistence(
        self, stream_env: dict[str, Any]
    ) -> None:
        """Verify session streaming persists exactly one user and assistant turn on success."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]

        # Create session
        create_resp = client.post(
            "/api/sessions",
            json={"title": "Streaming Test Session", "selected_doc_id": doc_id},
        )
        assert create_resp.status_code == 201
        session_id = create_resp.json()["session_id"]

        # Execute streaming turn
        stream_resp = client.post(
            f"/api/sessions/{session_id}/query/stream",
            json={"query": "Tell me about Alpha"},
        )
        assert stream_resp.status_code == 200

        events = parse_sse_events(stream_resp.text)
        complete_event = events[-1]
        assert complete_event["event"] == "complete"
        data = complete_event["data"]

        assert data["session_id"] == session_id
        assert "user_message_id" in data
        assert "assistant_message_id" in data

        # Verify database persistence: exactly 2 messages
        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["message_count"] == 2
        assert len(detail_data["messages"]) == 2

        user_msg = detail_data["messages"][0]
        asst_msg = detail_data["messages"][1]

        assert user_msg["role"] == "user"
        assert user_msg["content"] == "Tell me about Alpha"
        assert asst_msg["role"] == "assistant"
        assert asst_msg["content"] == data["answer"]
        assert len(asst_msg["resolved_citations"]) == 1

    def test_post_session_query_stream_multi_turn_history(self, stream_env: dict[str, Any]) -> None:
        """Verify multi-turn session streaming passes bounded context to subsequent turns."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]
        mock_llm: MockLLMClient = stream_env["mock_llm"]

        create_resp = client.post(
            "/api/sessions",
            json={"title": "Multi Turn", "selected_doc_id": doc_id},
        )
        session_id = create_resp.json()["session_id"]

        # Turn 1
        client.post(
            f"/api/sessions/{session_id}/query/stream",
            json={"query": "First question"},
        )

        # Turn 2
        mock_llm.reset()
        mock_llm.default_response = "Second turn response [CIT-001]."

        client.post(
            f"/api/sessions/{session_id}/query/stream",
            json={"query": "Follow-up question"},
        )

        detail_resp = client.get(f"/api/sessions/{session_id}")
        detail_data = detail_resp.json()
        assert detail_data["message_count"] == 4

        # Verify mock LLM received conversation history on Turn 2
        assert mock_llm.call_count == 1
        record = mock_llm.history[0]
        messages = record["messages"]
        roles = [m.role for m in messages]
        # Should have previous user, previous assistant, current user
        assert roles == ["user", "assistant", "user"]
        assert messages[0].content == "First question"
        assert "Follow-up question" in messages[2].content

    def test_post_session_query_stream_max_history_zero(self, stream_env: dict[str, Any]) -> None:
        """Verify max_history_messages=0 injects zero prior history turns in streaming."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]
        mock_llm: MockLLMClient = stream_env["mock_llm"]

        create_resp = client.post(
            "/api/sessions",
            json={"title": "Zero Bound Stream", "selected_doc_id": doc_id},
        )
        session_id = create_resp.json()["session_id"]

        # Turn 1
        client.post(
            f"/api/sessions/{session_id}/query/stream",
            json={"query": "First question"},
        )

        # Turn 2 with max_history_messages=0
        mock_llm.reset()
        mock_llm.default_response = "Zero history response [CIT-001]."

        client.post(
            f"/api/sessions/{session_id}/query/stream",
            json={"query": "Follow-up with zero history", "max_history_messages": 0},
        )

        assert mock_llm.call_count == 1
        record = mock_llm.history[0]
        messages = record["messages"]
        # Only current turn user prompt should be present; zero conversation context
        assert len(messages) == 1
        assert messages[0].role == "user"
        assert "Follow-up with zero history" in messages[0].content
        assert "First question" not in messages[0].content

    def test_post_session_query_stream_session_isolation(self, stream_env: dict[str, Any]) -> None:
        """Verify streaming query turns in Session A do not leak into Session B."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]

        s_a = client.post("/api/sessions", json={"title": "A", "selected_doc_id": doc_id}).json()[
            "session_id"
        ]
        s_b = client.post("/api/sessions", json={"title": "B", "selected_doc_id": doc_id}).json()[
            "session_id"
        ]

        client.post(f"/api/sessions/{s_a}/query/stream", json={"query": "Message for A"})
        client.post(f"/api/sessions/{s_b}/query/stream", json={"query": "Message for B"})

        detail_a = client.get(f"/api/sessions/{s_a}").json()
        detail_b = client.get(f"/api/sessions/{s_b}").json()

        assert detail_a["message_count"] == 2
        assert detail_b["message_count"] == 2
        assert detail_a["messages"][0]["content"] == "Message for A"
        assert detail_b["messages"][0]["content"] == "Message for B"

    def test_post_session_query_stream_provider_failure_does_not_persist(
        self, stream_env: dict[str, Any]
    ) -> None:
        """Verify that provider failure mid-stream does NOT persist an orphaned user turn."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]
        mock_llm: MockLLMClient = stream_env["mock_llm"]

        create_resp = client.post(
            "/api/sessions",
            json={"title": "Failure Test", "selected_doc_id": doc_id},
        )
        session_id = create_resp.json()["session_id"]

        # Configure LLM to fail mid-stream
        mock_llm.stream_chunks = ["First token ", RuntimeError("Provider crashed mid-stream")]

        resp = client.post(
            f"/api/sessions/{session_id}/query/stream",
            json={"query": "Question that will fail"},
        )
        assert resp.status_code == 200

        events = parse_sse_events(resp.text)
        event_names = [e["event"] for e in events]
        assert "error" in event_names
        assert "complete" not in event_names

        # Critical invariant: Zero messages persisted in SQLite session!
        detail_resp = client.get(f"/api/sessions/{session_id}")
        assert detail_resp.status_code == 200
        assert detail_resp.json()["message_count"] == 0
        assert detail_resp.json()["messages"] == []

    def test_post_session_query_stream_missing_session_returns_404(
        self, stream_env: dict[str, Any]
    ) -> None:
        """Verify streaming query to non-existent session returns HTTP 404 upfront."""
        client: TestClient = stream_env["client"]
        resp = client.post(
            "/api/sessions/sess_missing/query/stream",
            json={"query": "Question"},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_post_session_query_stream_no_doc_scope_returns_422(
        self, stream_env: dict[str, Any]
    ) -> None:
        """Verify session query without session doc scope or query doc scope returns 422."""
        client: TestClient = stream_env["client"]
        sess = client.post("/api/sessions", json={"title": "No Scope"}).json()["session_id"]

        resp = client.post(
            f"/api/sessions/{sess}/query/stream",
            json={"query": "Question"},
        )
        assert resp.status_code == 422
        assert "no document selected" in resp.json()["detail"].lower()

    def test_backward_compatibility_synchronous_endpoints(self, stream_env: dict[str, Any]) -> None:
        """Verify synchronous POST /api/query and session query remain 100% functional."""
        client: TestClient = stream_env["client"]
        doc_id = stream_env["doc_id"]

        # Synchronous standalone query
        resp_q = client.post(
            "/api/query",
            json={"query": "Alpha features", "document_ids": [doc_id]},
        )
        assert resp_q.status_code == 200
        assert "Alpha provides verified" in resp_q.json()["answer"]

        # Synchronous session query
        sess = client.post(
            "/api/sessions", json={"title": "Sync Session", "selected_doc_id": doc_id}
        ).json()["session_id"]

        resp_sq = client.post(
            f"/api/sessions/{sess}/query",
            json={"query": "Session sync query"},
        )
        assert resp_sq.status_code == 200
        assert resp_sq.json()["session_id"] == sess
        assert "Alpha provides verified" in resp_sq.json()["answer"]

    def test_sse_stream_generator_disconnect_closes_iterator(self) -> None:
        """Verify sse_stream_generator closes iterator off-thread when client disconnects."""
        import asyncio
        from unittest.mock import AsyncMock

        from reliability_lab.api.main import sse_stream_generator
        from reliability_lab.query.orchestrator import QueryStreamEvent

        state = {"closed": False}

        def mock_iter():
            try:
                yield QueryStreamEvent(event="start", data={"query": "test"})
                yield QueryStreamEvent(event="token", data={"token": "hello"})
            finally:
                state["closed"] = True

        async def run_test() -> None:
            mock_request = AsyncMock()
            mock_request.is_disconnected.side_effect = [False, True]
            gen = sse_stream_generator(mock_iter(), mock_request)
            chunks = []
            async for chunk in gen:
                chunks.append(chunk)

        asyncio.run(run_test())
        assert state["closed"] is True
