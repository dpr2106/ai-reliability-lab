"""Unit tests for streaming query orchestration (STING V2-003 Phase C).

Verifies incremental query execution, token delivery, retrieval abstention without LLM calls,
and citation resolution during streaming.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import MockLLMClient
from reliability_lab.llm.models import LLMMessage
from reliability_lab.query.orchestrator import (
    DocumentNotFoundError,
    DocumentQueryService,
    QueryStreamEvent,
)


class MockEmbedder(BaseEmbedder):
    """Deterministic 4D embedder for orchestrator testing."""

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
            if "solar" in str(t).lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                vecs.append([0.1, 0.1, 0.1, 0.1])
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def repo(tmp_path: Path) -> DocumentRepository:
    return DocumentRepository(tmp_path / "stream_query_test.db")


@pytest.fixture
def ingestion_service(repo: DocumentRepository, tmp_path: Path) -> DocumentIngestionService:
    embedder = MockEmbedder()
    return DocumentIngestionService(
        repository=repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )


def _ingest_test_doc(service: DocumentIngestionService, filename: str, content: str) -> str:
    rec = service.save_upload_stream(filename, io.BytesIO(content.encode("utf-8")))
    processed = service.process_document(rec.doc_id)
    return processed.doc_id


class TestStreamingQueryOrchestration:
    """Test suite for DocumentQueryService.query_stream."""

    def test_query_stream_incremental_tokens_and_citations(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Verify tokens arrive progressively and citations resolve at completion."""
        doc_id = _ingest_test_doc(
            ingestion_service,
            "planets.txt",
            "The solar system consists of eight official planets revolving around the Sun.",
        )

        mock_tokens = [
            "There ",
            "are ",
            "eight ",
            "planets ",
            "in ",
            "the ",
            "solar ",
            "system ",
            "[CIT-001].",
        ]
        mock_llm = MockLLMClient(stream_chunks=mock_tokens)

        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=mock_llm,
        )

        events: list[QueryStreamEvent] = list(
            qs.query_stream(
                query_text="How many planets exist?",
                selected_doc_ids=[doc_id],
                strategy="hybrid",
            )
        )

        # 1. Event types in sequence
        event_names = [e.event for e in events]
        assert event_names[0] == "start"
        assert "token" in event_names
        assert "citation" in event_names
        assert event_names[-1] == "complete"

        # 2. Start event validation
        start_event = events[0]
        assert start_event.data["query_text"] == "How many planets exist?"
        assert start_event.data["selected_doc_ids"] == [doc_id]
        assert start_event.data["strategy"] == "hybrid"

        # 3. Progressive tokens validation
        token_deltas = [e.data["token"] for e in events if e.event == "token"]
        assert token_deltas == mock_tokens
        assert len(token_deltas) > 1

        # 4. Citation event validation
        cit_events = [e for e in events if e.event == "citation"]
        assert len(cit_events) == 1
        assert cit_events[0].data["marker"] == "[CIT-001]"
        assert cit_events[0].data["alias"] == "CIT-001"
        assert cit_events[0].data["filename"] == "planets.txt"

        # 5. Complete event validation
        complete_event = events[-1]
        assert complete_event.data["answer"] == "".join(mock_tokens)
        trace = complete_event.data["trace"]
        assert trace.abstained is False
        assert trace.retrieval_abstained is False
        assert len(trace.resolved_citations) == 1
        assert trace.resolved_citations[0].marker == "[CIT-001]"

    def test_query_stream_retrieval_abstention_skips_llm(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Verify empty retrieval triggers immediate abstention without calling LLM."""
        doc_id = _ingest_test_doc(
            ingestion_service,
            "biology.txt",
            "Mitochondria are the powerhouses of cellular respiration.",
        )

        mock_llm = MockLLMClient(default_response="Should never be called")
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=mock_llm,
        )

        events: list[QueryStreamEvent] = list(
            qs.query_stream(
                query_text="quantum computing qubits superconductor",
                selected_doc_ids=[doc_id],
                strategy="bm25",
            )
        )

        # Assert LLM was never called
        assert mock_llm.call_count == 0

        # Assert events emitted
        event_names = [e.event for e in events]
        assert event_names == ["start", "token", "complete"]

        complete_event = events[-1]
        assert (
            "unable to answer because no relevant evidence was retrieved"
            in complete_event.data["answer"]
        )
        trace = complete_event.data["trace"]
        assert trace.abstained is True
        assert trace.retrieval_abstained is True
        assert trace.generation_latency_ms == 0.0

    def test_query_stream_generation_abstention(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Verify LLM tier-2 abstention response is recorded in stream trace."""
        doc_id = _ingest_test_doc(
            ingestion_service,
            "notes.txt",
            "The solar system formed approximately 4.6 billion years ago.",
        )

        mock_llm = MockLLMClient(
            default_response=(
                "[ABSTAIN: INSUFFICIENT_EVIDENCE] The document does not describe Mars atmosphere."
            )
        )
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=mock_llm,
        )

        events = list(
            qs.query_stream(
                query_text="What is Mars atmosphere composed of?",
                selected_doc_ids=[doc_id],
                strategy="hybrid",
            )
        )

        complete_event = events[-1]
        trace = complete_event.data["trace"]
        assert trace.abstained is True
        assert trace.retrieval_abstained is False
        assert trace.generation_abstained is True
        assert trace.generation_abstention_reason == "insufficient_evidence"

    def test_query_stream_with_bounded_conversation_context(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Verify bounded conversation turns are passed to LLM during streaming."""
        doc_id = _ingest_test_doc(
            ingestion_service,
            "planets.txt",
            "The solar system consists of eight official planets.",
        )

        mock_llm = MockLLMClient(default_response="Jupiter is the largest.")
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=mock_llm,
        )

        history = [
            LLMMessage(role="user", content="How many planets?"),
            LLMMessage(role="assistant", content="There are eight planets."),
        ]

        list(
            qs.query_stream(
                query_text="Which one is the largest?",
                selected_doc_ids=[doc_id],
                conversation_context=history,
            )
        )

        # Check prompts passed to mock LLM
        assert mock_llm.call_count == 1
        record = mock_llm.history[0]
        assert record["system_prompt"] is not None
        messages: list[LLMMessage] = record["messages"]
        roles = [m.role for m in messages]
        # Should contain user (history), assistant (history), user (current query)
        assert roles == ["user", "assistant", "user"]
        assert messages[0].content == "How many planets?"
        assert messages[1].content == "There are eight planets."

    def test_query_stream_invalid_inputs_raise_before_streaming(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Verify bad inputs raise ValueError or DocumentNotFoundError immediately."""
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLMClient(),
        )

        with pytest.raises(ValueError, match="query_text must be a non-empty string"):
            list(qs.query_stream("", selected_doc_ids=["doc_1"]))

        with pytest.raises(ValueError, match="selected_doc_ids must contain at least one"):
            list(qs.query_stream("Valid query", selected_doc_ids=[]))

        with pytest.raises(DocumentNotFoundError, match="Document 'doc_missing' not found"):
            list(qs.query_stream("Valid query", selected_doc_ids=["doc_missing"]))
