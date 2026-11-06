"""Unit tests for SessionService (STING V2-003 Phase B)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import BaseLLMClient
from reliability_lab.llm.models import LLMMessage, LLMResponse, TokenUsage
from reliability_lab.query.orchestrator import DocumentQueryService
from reliability_lab.session.repository import SessionNotFoundError, SessionRepository
from reliability_lab.session.service import SessionService
from reliability_lab.storage.database import Database


class MockEmbedder(BaseEmbedder):
    """Deterministic 4D embedder for service testing."""

    @property
    def model_name(self) -> str:
        return "mock-embedder"

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


class MockLLM(BaseLLMClient):
    """Mock LLM capturing incoming prompts and returning deterministic answers."""

    def __init__(self, response_text: str = "Grounded response citing [CIT-001].") -> None:
        self.response_text = response_text
        self.call_history: list[list[LLMMessage]] = []

    @property
    def model_name(self) -> str:
        return "mock-llm-service"

    @property
    def provider_name(self) -> str:
        return "mock-provider"

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.call_history.append(messages)
        return LLMResponse(
            text=self.response_text,
            model=self.model_name,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=25, total_tokens=125),
        )


@pytest.fixture
def service_env(tmp_path: Path) -> dict[str, Any]:
    db = Database(str(tmp_path / "test_env.db"))
    doc_repo = DocumentRepository(db)
    session_repo = SessionRepository(db)
    embedder = MockEmbedder()
    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "documents",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )
    llm = MockLLM()
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

    # Ingest a ready test document
    doc_content = (
        "Alpha architecture provides high reliability. "
        "Alpha guarantees deterministic citation verification. "
        "Beta subsystem manages failover routing."
    )
    rec = ingestion.save_upload_stream("doc_alpha.txt", io.BytesIO(doc_content.encode("utf-8")))
    ready_doc = ingestion.process_document(rec.doc_id)

    return {
        "db": db,
        "doc_repo": doc_repo,
        "session_repo": session_repo,
        "session_service": session_service,
        "query_service": query_service,
        "llm": llm,
        "doc_id": ready_doc.doc_id,
    }


class TestSessionService:
    """Test suite for conversational session service and bounded query execution."""

    def test_session_lifecycle(self, service_env: dict[str, Any]) -> None:
        service: SessionService = service_env["session_service"]

        # Create
        created = service.create_session(
            title="My Chat", selected_doc_id=service_env["doc_id"], metadata={"topic": "rag"}
        )
        assert created.session_id.startswith("ses_")
        assert created.title == "My Chat"
        assert created.selected_doc_id == service_env["doc_id"]

        # Get
        fetched = service.get_session(created.session_id)
        assert fetched.session_id == created.session_id
        assert fetched.message_count == 0

        # List
        sessions, total = service.list_sessions(limit=10, offset=0)
        assert total >= 1
        assert any(s.session_id == created.session_id for s in sessions)

        # Delete
        assert service.delete_session(created.session_id) is True
        with pytest.raises(SessionNotFoundError):
            service.get_session(created.session_id)

    def test_execute_query_with_explicit_and_default_doc_scope(
        self, service_env: dict[str, Any]
    ) -> None:
        service: SessionService = service_env["session_service"]
        doc_id = service_env["doc_id"]

        session = service.create_session(title="Scoped Chat", selected_doc_id=doc_id)

        # Query using default doc scope from session
        answer, trace, u_msg, a_msg = service.execute_session_query(
            session_id=session.session_id,
            query_text="What does Alpha architecture provide?",
        )

        assert answer is not None
        assert trace.abstained is False
        assert u_msg.role == "user"
        assert a_msg.role == "assistant"
        assert len(a_msg.resolved_citations) > 0

        # Verify messages persisted in session
        refreshed = service.get_session(session.session_id)
        assert refreshed.message_count == 2
        assert len(refreshed.messages) == 2

    def test_execute_query_missing_doc_scope_raises_value_error(
        self, service_env: dict[str, Any]
    ) -> None:
        service: SessionService = service_env["session_service"]
        # Session without selected_doc_id
        session = service.create_session(title="No Scope Chat", selected_doc_id=None)

        with pytest.raises(ValueError, match="No document selected"):
            service.execute_session_query(
                session_id=session.session_id,
                query_text="Where is the data?",
                document_ids=None,
            )

    def test_contextual_follow_up_passes_history_to_llm(self, service_env: dict[str, Any]) -> None:
        """Verify that conversation history is passed to LLM as conversational context."""
        service: SessionService = service_env["session_service"]
        llm: MockLLM = service_env["llm"]
        doc_id = service_env["doc_id"]

        session = service.create_session(title="Multi-Turn Chat", selected_doc_id=doc_id)

        # Turn 1
        llm.response_text = "Alpha provides high reliability [CIT-001]."
        service.execute_session_query(
            session_id=session.session_id,
            query_text="What does Alpha provide?",
        )

        # Turn 2: Follow-up question
        llm.response_text = "Yes, and it also guarantees deterministic verification [CIT-001]."
        service.execute_session_query(
            session_id=session.session_id,
            query_text="Does it guarantee verification?",
        )

        assert len(llm.call_history) == 2
        turn2_messages = llm.call_history[1]

        # Structure must be: system -> user turn 1 -> asst turn 1 -> user turn 2 (with evidence)
        assert turn2_messages[0].role == "system"
        assert turn2_messages[1].role == "user"
        assert turn2_messages[1].content == "What does Alpha provide?"
        assert turn2_messages[2].role == "assistant"
        assert turn2_messages[2].content == "Alpha provides high reliability [CIT-001]."
        assert turn2_messages[3].role == "user"
        assert "Does it guarantee verification?" in turn2_messages[3].content

    def test_bounded_history_enforces_limit(self, service_env: dict[str, Any]) -> None:
        """Bounded history limits how many previous turns are supplied to the LLM."""
        service: SessionService = service_env["session_service"]
        llm: MockLLM = service_env["llm"]
        doc_id = service_env["doc_id"]

        session = service.create_session(title="Bound Test", selected_doc_id=doc_id)

        # Execute 3 complete turns (6 messages total)
        for i in range(3):
            llm.response_text = f"Answer {i} [CIT-001]."
            service.execute_session_query(
                session_id=session.session_id,
                query_text=f"Question {i} about Alpha?",
            )

        # Turn 4 with max_history_messages=2: should only include the last 2 messages (Turn 2)
        llm.response_text = "Answer 3 [CIT-001]."
        service.execute_session_query(
            session_id=session.session_id,
            query_text="Question 3 about Alpha?",
            max_history_messages=2,
        )

        turn4_messages = llm.call_history[-1]
        # System + 2 context messages + 1 current user prompt = 4 messages
        assert len(turn4_messages) == 4
        assert turn4_messages[0].role == "system"
        assert turn4_messages[1].role == "user"
        assert turn4_messages[1].content == "Question 2 about Alpha?"
        assert turn4_messages[2].role == "assistant"
        assert turn4_messages[2].content == "Answer 2 [CIT-001]."
        assert turn4_messages[3].role == "user"

    def test_citation_provenance_strictly_restricted_to_current_query(
        self, service_env: dict[str, Any]
    ) -> None:
        """Citations from earlier turns cannot be cited unless retrieved in the current turn."""
        service: SessionService = service_env["session_service"]
        llm: MockLLM = service_env["llm"]
        doc_id = service_env["doc_id"]

        session = service.create_session(title="Citation Scope", selected_doc_id=doc_id)

        # Turn 1: citations to CIT-001 are valid
        llm.response_text = "Alpha provides reliability [CIT-001]."
        _, trace1, _, _ = service.execute_session_query(
            session_id=session.session_id,
            query_text="Alpha reliability",
        )
        assert len(trace1.resolved_citations) > 0

        # Turn 2: LLM tries to hallucinate or cite [CIT-099] which is not in current evidence
        llm.response_text = "As seen earlier [CIT-001], and also referencing nonexistent [CIT-099]."
        _, trace2, _, _ = service.execute_session_query(
            session_id=session.session_id,
            query_text="Alpha verification",
        )

        # CIT-099 must be classified as unresolved
        unresolved_aliases = [u.alias for u in trace2.unresolved_citations]
        assert "CIT-099" in unresolved_aliases

    def test_evidence_first_abstention_preserved(self, service_env: dict[str, Any]) -> None:
        """When query retrieves no evidence, orchestrator abstains and skips LLM generation."""
        service: SessionService = service_env["session_service"]
        llm: MockLLM = service_env["llm"]
        doc_id = service_env["doc_id"]

        session = service.create_session(title="Abstention Chat", selected_doc_id=doc_id)

        # Turn 1: normal grounded query
        llm.response_text = "Answer 1 [CIT-001]."
        service.execute_session_query(
            session_id=session.session_id,
            query_text="Alpha reliability",
        )
        calls_before = len(llm.call_history)

        # Turn 2: unanswerable query with zero keyword/dense match
        answer, trace, _, a_msg = service.execute_session_query(
            session_id=session.session_id,
            query_text="Xylophone quantum zebra thermodynamics?",
            strategy="bm25",
        )

        assert trace.abstained is True
        assert trace.retrieval_abstained is True
        assert "unable to answer" in answer.lower()
        # LLM was NOT called on abstention
        assert len(llm.call_history) == calls_before
        # Assistant message in session recorded the abstention
        assert a_msg.query_trace is not None
        assert a_msg.query_trace["abstained"] is True
