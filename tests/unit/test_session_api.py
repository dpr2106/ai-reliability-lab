"""Unit and integration tests for Session API endpoints (STING V2-003 Phase B)."""

from __future__ import annotations

import io
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
from reliability_lab.llm.client import BaseLLMClient
from reliability_lab.llm.models import LLMMessage, LLMResponse, TokenUsage
from reliability_lab.query.orchestrator import DocumentQueryService
from reliability_lab.session.repository import SessionRepository
from reliability_lab.session.service import SessionService
from reliability_lab.storage.database import Database


class ApiTestEmbedder(BaseEmbedder):
    """Deterministic 4D embedder for API tests."""

    @property
    def model_name(self) -> str:
        return "api-test-model"

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
            elif "beta" in str(t).lower():
                vecs.append([0.0, 1.0, 0.0, 0.0])
            else:
                vecs.append([0.5, 0.5, 0.0, 0.0])
        return np.array(vecs, dtype=np.float32)


class ApiTestLLM(BaseLLMClient):
    """Deterministic mock LLM for API tests."""

    def __init__(self, response_text: str = "Grounded response citing [CIT-001].") -> None:
        self.response_text = response_text
        self.generated_prompts: list[list[LLMMessage]] = []

    @property
    def model_name(self) -> str:
        return "api-mock-llm"

    @property
    def provider_name(self) -> str:
        return "mock-provider"

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.generated_prompts.append(messages)
        return LLMResponse(
            text=self.response_text,
            model=self.model_name,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=25, total_tokens=125),
        )


@pytest.fixture
def client_env(tmp_path: Path) -> dict[str, Any]:
    db = Database(str(tmp_path / "api_test.db"))
    doc_repo = DocumentRepository(db)
    session_repo = SessionRepository(db)
    embedder = ApiTestEmbedder()
    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "documents",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )
    llm = ApiTestLLM()
    query_service = DocumentQueryService(
        repository=doc_repo,
        ingestion_service=ingestion,
        embedder=embedder,
        llm_client=llm,
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
    doc_content = (
        "Alpha architecture provides high reliability. "
        "Alpha guarantees deterministic citation verification."
    )
    rec = ingestion.save_upload_stream("doc_alpha.txt", io.BytesIO(doc_content.encode("utf-8")))
    ready_doc = ingestion.process_document(rec.doc_id)

    return {
        "client": client,
        "session_repo": session_repo,
        "doc_id": ready_doc.doc_id,
        "llm": llm,
    }


class TestSessionApiEndpoints:
    """Test suite for /api/sessions and /api/sessions/{session_id}/query REST endpoints."""

    def test_create_session(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        doc_id = client_env["doc_id"]

        resp = client.post(
            "/api/sessions",
            json={
                "title": "API Test Session",
                "selected_doc_id": doc_id,
                "metadata": {"source": "unit_test"},
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["session_id"].startswith("ses_")
        assert data["title"] == "API Test Session"
        assert data["selected_doc_id"] == doc_id
        assert data["metadata"] == {"source": "unit_test"}
        assert data["message_count"] == 0
        assert data["messages"] == []

    def test_list_sessions(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        doc_id = client_env["doc_id"]

        client.post("/api/sessions", json={"title": "Session 1", "selected_doc_id": doc_id})
        client.post("/api/sessions", json={"title": "Session 2", "selected_doc_id": doc_id})

        resp = client.get("/api/sessions?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        assert len(data["sessions"]) >= 2
        assert data["limit"] == 10
        assert data["offset"] == 0

    def test_get_session_by_id_and_not_found(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]

        created = client.post("/api/sessions", json={"title": "Detail Session"}).json()
        sid = created["session_id"]

        # Valid fetch
        resp = client.get(f"/api/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["session_id"] == sid

        # Unknown ID returns 404
        resp404 = client.get("/api/sessions/nonexistent_sid")
        assert resp404.status_code == 404
        assert "not found" in resp404.json()["detail"].lower()

    def test_delete_session_and_not_found(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]

        created = client.post("/api/sessions", json={"title": "To Delete"}).json()
        sid = created["session_id"]

        del_resp = client.delete(f"/api/sessions/{sid}")
        assert del_resp.status_code == 200
        assert del_resp.json() == {"deleted": True, "session_id": sid}

        # Subsequent fetch is 404
        assert client.get(f"/api/sessions/{sid}").status_code == 404

        # Deleting nonexistent is 404
        assert client.delete("/api/sessions/nonexistent_sid").status_code == 404

    def test_session_query_success(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        doc_id = client_env["doc_id"]

        created = client.post(
            "/api/sessions", json={"title": "Chat", "selected_doc_id": doc_id}
        ).json()
        sid = created["session_id"]

        query_payload = {
            "query": "What does Alpha provide?",
            "strategy": "hybrid",
            "top_k": 3,
        }
        resp = client.post(f"/api/sessions/{sid}/query", json=query_payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["session_id"] == sid
        assert data["user_message_id"].startswith("msg_")
        assert data["assistant_message_id"].startswith("msg_")
        assert data["answer"] == "Grounded response citing [CIT-001]."
        assert data["abstained"] is False
        assert len(data["resolved_citations"]) > 0
        assert data["trace"]["trace_id"].startswith("trc_")

        # Verify session now has 2 messages
        sess_resp = client.get(f"/api/sessions/{sid}")
        assert sess_resp.status_code == 200
        sess_data = sess_resp.json()
        assert sess_data["message_count"] == 2
        assert len(sess_data["messages"]) == 2
        assert sess_data["messages"][0]["role"] == "user"
        assert sess_data["messages"][1]["role"] == "assistant"

    def test_session_query_session_not_found_returns_404(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        doc_id = client_env["doc_id"]

        resp = client.post(
            "/api/sessions/ses_unknown/query",
            json={"query": "Hello", "document_ids": [doc_id]},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_session_query_document_not_found_returns_404(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        created = client.post("/api/sessions", json={"title": "Chat"}).json()
        sid = created["session_id"]

        resp = client.post(
            f"/api/sessions/{sid}/query",
            json={"query": "Hello", "document_ids": ["nonexistent_doc_id"]},
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_session_query_no_document_scope_returns_422(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        # Session without selected_doc_id
        created = client.post("/api/sessions", json={"title": "Unscoped"}).json()
        sid = created["session_id"]

        resp = client.post(
            f"/api/sessions/{sid}/query",
            json={"query": "Hello with no doc scope", "document_ids": None},
        )
        assert resp.status_code == 422
        assert "no document selected" in resp.json()["detail"].lower()

    def test_session_query_invalid_params_returns_422(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        created = client.post("/api/sessions", json={"title": "Chat"}).json()
        sid = created["session_id"]

        # Empty query
        resp_empty = client.post(f"/api/sessions/{sid}/query", json={"query": "   "})
        assert resp_empty.status_code == 422

        # Invalid strategy
        resp_strat = client.post(
            f"/api/sessions/{sid}/query", json={"query": "Valid", "strategy": "magic_search"}
        )
        assert resp_strat.status_code == 422

        # Invalid top_k
        resp_topk = client.post(f"/api/sessions/{sid}/query", json={"query": "Valid", "top_k": -1})
        assert resp_topk.status_code == 422

    def test_contextual_follow_up_in_session(self, client_env: dict[str, Any]) -> None:
        client: TestClient = client_env["client"]
        llm: ApiTestLLM = client_env["llm"]
        doc_id = client_env["doc_id"]

        created = client.post(
            "/api/sessions", json={"title": "Multi-turn", "selected_doc_id": doc_id}
        ).json()
        sid = created["session_id"]

        # Turn 1
        client.post(f"/api/sessions/{sid}/query", json={"query": "First question"})
        # Turn 2
        client.post(f"/api/sessions/{sid}/query", json={"query": "Second question"})

        assert len(llm.generated_prompts) == 2
        turn2_messages = llm.generated_prompts[1]
        assert turn2_messages[0].role == "system"
        assert turn2_messages[1].role == "user"
        assert turn2_messages[1].content == "First question"
        assert turn2_messages[2].role == "assistant"
        assert turn2_messages[3].role == "user"
        assert "Second question" in turn2_messages[3].content

    def test_session_query_with_max_history_zero_suppresses_context(
        self, client_env: dict[str, Any]
    ) -> None:
        client: TestClient = client_env["client"]
        llm: ApiTestLLM = client_env["llm"]
        doc_id = client_env["doc_id"]

        created = client.post(
            "/api/sessions", json={"title": "Zero History API", "selected_doc_id": doc_id}
        ).json()
        sid = created["session_id"]

        # Turn 1
        client.post(f"/api/sessions/{sid}/query", json={"query": "First question"})
        # Turn 2 with max_history_messages=0
        resp = client.post(
            f"/api/sessions/{sid}/query",
            json={"query": "Second question", "max_history_messages": 0},
        )
        assert resp.status_code == 200
        assert len(llm.generated_prompts) == 2
        turn2_messages = llm.generated_prompts[1]
        assert len(turn2_messages) == 2
        assert turn2_messages[0].role == "system"
        assert turn2_messages[1].role == "user"
        assert "Second question" in turn2_messages[1].content
        assert "First question" not in turn2_messages[1].content

    def test_backward_compatibility_existing_api_query(self, client_env: dict[str, Any]) -> None:
        """The existing synchronous POST /api/query endpoint remains fully functional."""
        client: TestClient = client_env["client"]
        doc_id = client_env["doc_id"]

        resp = client.post(
            "/api/query",
            json={
                "query": "Direct query without session",
                "document_ids": [doc_id],
                "strategy": "hybrid",
                "top_k": 3,
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query_text"] == "Direct query without session"
        assert data["selected_doc_ids"] == [doc_id]
        assert data["answer"] == "Grounded response citing [CIT-001]."
        assert data["trace"]["trace_id"].startswith("trc_")
