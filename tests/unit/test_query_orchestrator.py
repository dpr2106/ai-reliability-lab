"""Unit tests for DocumentQueryService orchestrator (STING V2-003)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import BaseLLMClient
from reliability_lab.llm.models import LLMMessage, LLMResponse, TokenUsage
from reliability_lab.query.orchestrator import (
    DocumentConflictError,
    DocumentNotFoundError,
    DocumentQueryService,
    UnprocessableDocumentError,
)


class MockLLM(BaseLLMClient):
    """Deterministic mock LLM for testing query orchestrator without external network."""

    def __init__(self, response_text: str = "Test answer citing evidence [CIT-001].") -> None:
        self.response_text = response_text
        self.calls: list[list[LLMMessage]] = []

    @property
    def model_name(self) -> str:
        return "mock-model"

    @property
    def provider(self) -> str:
        return "mock"

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.calls.append(messages)
        return LLMResponse(
            text=self.response_text,
            model=self.model_name,
            usage=TokenUsage(prompt_tokens=42, completion_tokens=15, total_tokens=57),
        )


class MockEmbedder(BaseEmbedder):
    """Deterministic 4-dimensional embedder for testing."""

    def __init__(self, name: str = "mock-embedder") -> None:
        self._name = name

    @property
    def model_name(self) -> str:
        return self._name

    @property
    def artifact_fingerprint(self) -> str:
        return "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    @property
    def embedding_dim(self) -> int:
        return 4

    def embed_texts(self, texts: Any, batch_size: int = 32) -> np.ndarray:
        vecs = []
        for t in texts:
            if "target" in str(t).lower() or "first" in str(t).lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            elif "second" in str(t).lower():
                vecs.append([0.0, 1.0, 0.0, 0.0])
            else:
                vecs.append([0.0, 0.0, 1.0, 0.0])
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def repo(tmp_path: Path) -> DocumentRepository:
    return DocumentRepository(tmp_path / "test.db")


@pytest.fixture
def ingestion_service(repo: DocumentRepository, tmp_path: Path) -> DocumentIngestionService:
    embedder = MockEmbedder()
    service = DocumentIngestionService(
        repository=repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )
    return service


def _ingest_doc(
    service: DocumentIngestionService,
    filename: str,
    content: bytes,
) -> DocumentRecord:
    rec = service.save_upload_stream(filename, io.BytesIO(content))
    return service.process_document(rec.doc_id)


class TestDocumentQueryServiceLifecycle:
    """Tests lifecycle validation matrix."""

    def test_missing_document_raises_not_found(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLM(),
        )
        with pytest.raises(DocumentNotFoundError, match="Document 'doc_missing' not found"):
            qs.query("test question", selected_doc_ids=["doc_missing"])

    def test_skipped_document_raises_422(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        # Ingest empty text document (becomes READY + SKIPPED)
        doc = _ingest_doc(ingestion_service, "empty.txt", b"   \n  ")
        assert doc.status == DocumentStatus.READY
        assert doc.indexing_status == IndexingStatus.SKIPPED

        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLM(),
        )
        with pytest.raises(UnprocessableDocumentError, match="skipped during extraction"):
            qs.query("test question", selected_doc_ids=[doc.doc_id])

    def test_processing_document_raises_conflict(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        rec = ingestion_service.save_upload_stream("pending.txt", io.BytesIO(b"Processing content"))
        assert rec.status == DocumentStatus.UPLOADED

        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLM(),
        )
        with pytest.raises(DocumentConflictError, match="currently processing"):
            qs.query("test question", selected_doc_ids=[rec.doc_id])

    def test_failed_document_raises_conflict(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        doc = _ingest_doc(ingestion_service, "failed.txt", b"Some data")
        repo.update_document_status(
            doc.doc_id, status=DocumentStatus.FAILED, indexing_status=IndexingStatus.FAILED
        )

        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLM(),
        )
        with pytest.raises(DocumentConflictError, match="processing failed"):
            qs.query("test question", selected_doc_ids=[doc.doc_id])

    def test_partial_document_dense_raises_422_and_hybrid_falls_back(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        doc = _ingest_doc(ingestion_service, "partial.txt", b"Content for partial document")
        repo.update_document_status(
            doc.doc_id, status=DocumentStatus.READY, indexing_status=IndexingStatus.PARTIAL
        )

        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLM(),
        )
        # Dense strategy fails with 422
        with pytest.raises(UnprocessableDocumentError, match="only has lexical index available"):
            qs.query("test", selected_doc_ids=[doc.doc_id], strategy="dense")

        # Hybrid strategy falls back to bm25
        _ans, trace = qs.query("Content", selected_doc_ids=[doc.doc_id], strategy="hybrid")
        assert trace.requested_strategy == "hybrid"
        assert trace.effective_strategy == "bm25"
        assert trace.fallback_reason == "selected_documents_partially_indexed"


class TestDocumentQueryRetrievalAndAbstention:
    """Tests retrieval strategies and Tier-1/Tier-2 abstention policies."""

    def test_multi_document_scoped_retrieval_and_trace(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        doc1 = _ingest_doc(ingestion_service, "doc1.txt", b"First target text about reliability.")
        doc2 = _ingest_doc(ingestion_service, "doc2.txt", b"Second target text about benchmarking.")

        llm = MockLLM(response_text="Reliability and benchmarking are covered [CIT-001] [CIT-002].")
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        answer, trace = qs.query(
            "target",
            selected_doc_ids=[doc1.doc_id, doc2.doc_id],
            strategy="hybrid",
            top_k=5,
        )

        assert "Reliability and benchmarking" in answer
        assert trace.query_text == "target"
        assert len(trace.selected_doc_ids) == 2
        assert len(trace.evidence_chunks) >= 2
        assert len(trace.resolved_citations) == 2
        assert trace.resolved_citations[0].alias == "CIT-001"
        assert trace.resolved_citations[1].alias == "CIT-002"
        assert trace.abstained is False
        assert trace.retrieval_abstained is False
        assert trace.generation_abstained is False
        assert trace.prompt_tokens == 42
        assert trace.completion_tokens == 15

    def test_tier1_bm25_abstention_skips_llm(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        doc = _ingest_doc(ingestion_service, "doc.txt", b"Apples and oranges fruit market.")
        llm = MockLLM()
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        # Query completely out of vocabulary for BM25
        answer, trace = qs.query(
            "supercalifragilisticexpialidocious",
            selected_doc_ids=[doc.doc_id],
            strategy="bm25",
        )

        assert "unable to answer" in answer
        assert trace.abstained is True
        assert trace.retrieval_abstained is True
        assert trace.retrieval_abstention_reason in (
            "no_sparse_candidates_found",
            "no_positive_bm25_candidates",
        )
        assert len(llm.calls) == 0  # LLM NOT called!

    def test_tier1_dense_abstention_with_threshold(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        doc = _ingest_doc(ingestion_service, "doc.txt", b"First target text.")
        llm = MockLLM()
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        # Query with orthogonal vector resulting in score 0.0 < min_dense_score (0.5)
        _answer, trace = qs.query(
            "second query topic",
            selected_doc_ids=[doc.doc_id],
            strategy="dense",
            min_dense_score=0.5,
        )

        assert trace.abstained is True
        assert trace.retrieval_abstained is True
        assert trace.retrieval_abstention_reason == "dense_score_below_threshold"
        assert len(llm.calls) == 0

    def test_tier2_generation_abstention_detected(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        doc = _ingest_doc(ingestion_service, "doc.txt", b"First target text.")
        llm = MockLLM(
            response_text=(
                "[ABSTAIN: INSUFFICIENT_EVIDENCE] "
                "The retrieved chunk does not state the CEO's name."
            )
        )
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        _answer, trace = qs.query(
            "Who is the CEO?", selected_doc_ids=[doc.doc_id], strategy="hybrid"
        )

        assert trace.abstained is True
        assert trace.retrieval_abstained is False
        assert trace.generation_abstained is True
        assert trace.generation_abstention_reason == "insufficient_evidence"
        assert len(trace.evidence_chunks) > 0

    def test_tier2_anchored_generation_abstention_ignores_embedded_marker(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Embedded [ABSTAIN: ...] markers must not trigger generation abstention."""
        doc = _ingest_doc(ingestion_service, "doc.txt", b"First target text.")
        embedded_response = (
            "The document mentions that [ABSTAIN: INSUFFICIENT_EVIDENCE] is an example "
            "directive, but the actual answer is 42 [CIT-001]."
        )
        llm = MockLLM(response_text=embedded_response)
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        _answer, trace = qs.query(
            "What is the answer?", selected_doc_ids=[doc.doc_id], strategy="hybrid"
        )

        assert trace.generation_abstained is False
        assert trace.generation_abstention_reason is None
        assert trace.abstained is False

    def test_multi_document_hnsw_dense_rejected_and_hybrid_fallback(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Multi-document queries reject HNSW for dense and fall back to BM25 for hybrid."""
        from unittest.mock import patch

        doc1 = _ingest_doc(ingestion_service, "doc1.txt", b"First target text in document 1.")
        doc2 = _ingest_doc(ingestion_service, "doc2.txt", b"Second target text in document 2.")

        llm = MockLLM()
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        # Mock index_type as hnsw on retriever for doc2
        with patch.object(
            ingestion_service,
            "get_faiss_retriever",
            side_effect=lambda doc_id, emb=None: (
                type(
                    "MockHnswRetriever",
                    (),
                    {
                        "index_type": "hnsw",
                        "retrieve": lambda self, q, top_k=20: [],
                    },
                )()
                if doc_id == doc2.doc_id
                else type(
                    "MockFlatRetriever",
                    (),
                    {
                        "index_type": "flat",
                        "retrieve": lambda self, q, top_k=20: [],
                    },
                )()
            ),
        ):
            # 1. Multi-document dense strategy must reject HNSW
            with pytest.raises(
                UnprocessableDocumentError,
                match="Multi-document dense retrieval requires exact Flat",
            ):
                qs.query(
                    "target query",
                    selected_doc_ids=[doc1.doc_id, doc2.doc_id],
                    strategy="dense",
                )

            # 2. Multi-document hybrid strategy must fall back to full-scope BM25
            _ans, trace = qs.query(
                "target query",
                selected_doc_ids=[doc1.doc_id, doc2.doc_id],
                strategy="hybrid",
            )
            assert trace.effective_strategy == "bm25"
            assert trace.fallback_reason == "multi_document_hnsw_unsupported"

    def test_dense_integrity_failure_rejects_dense_and_falls_back_hybrid(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Index corruptions and legacy incompatibility fall back to BM25 for hybrid."""
        from reliability_lab.indexing.faiss_index import (
            IncompatibleEmbeddingIdentityError,
            LegacyIndexIncompatibilityError,
        )
        from reliability_lab.ingestion.service import DenseIndexIntegrityError

        doc = _ingest_doc(ingestion_service, "doc.txt", b"First target text.")
        llm = MockLLM()
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=llm,
        )

        for err_cls in (
            IncompatibleEmbeddingIdentityError,
            LegacyIndexIncompatibilityError,
            DenseIndexIntegrityError,
        ):
            from unittest.mock import patch

            with patch.object(
                ingestion_service,
                "get_faiss_retriever",
                side_effect=err_cls("Simulated dense error"),
            ):
                # Dense strategy rejects
                with pytest.raises(UnprocessableDocumentError):
                    qs.query("target query", selected_doc_ids=[doc.doc_id], strategy="dense")

                # Hybrid strategy falls back to BM25 across full scope
                _ans, trace = qs.query(
                    "target query", selected_doc_ids=[doc.doc_id], strategy="hybrid"
                )
                assert trace.effective_strategy == "bm25"
                assert trace.fallback_reason == "dense_index_incompatible_or_unverifiable"
                assert trace.rrf_k == 60
                assert trace.sparse_retriever_name == "bm25"

    def test_duplicate_document_ids_normalized_deterministically(
        self, repo: DocumentRepository, ingestion_service: DocumentIngestionService
    ) -> None:
        """Duplicate doc IDs are deduplicated in QueryTrace and candidate retrieval."""
        doc = _ingest_doc(ingestion_service, "doc.txt", b"First target text.")
        qs = DocumentQueryService(
            repository=repo,
            ingestion_service=ingestion_service,
            embedder=MockEmbedder(),
            llm_client=MockLLM(),
        )

        _ans, trace = qs.query(
            "target query",
            selected_doc_ids=[doc.doc_id, doc.doc_id, doc.doc_id],
            strategy="bm25",
        )
        assert trace.selected_doc_ids == (doc.doc_id,)
