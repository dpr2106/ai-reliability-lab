"""Unit tests for InvestigationService."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.investigation.repository import InvestigationRepository
from reliability_lab.investigation.service import (
    InvestigationNotFoundError,
    InvestigationService,
    InvestigationValidationError,
)
from reliability_lab.llm.client import MockLLMClient
from reliability_lab.query.orchestrator import DocumentNotFoundError
from reliability_lab.storage.database import Database


class MockEmbedder(BaseEmbedder):
    """Deterministic 4D embedder."""

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
            if "attention" in str(t).lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            else:
                vecs.append([0.2, 0.2, 0.2, 0.2])
        return np.array(vecs, dtype=np.float32)


@pytest.fixture
def test_setup(tmp_path: Path):
    db_path = tmp_path / "test.db"
    db = Database(str(db_path))
    doc_repo = DocumentRepository(db_path)
    inv_repo = InvestigationRepository(db)

    ingestion_service = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=MockEmbedder(),
    )

    return {
        "db": db,
        "doc_repo": doc_repo,
        "inv_repo": inv_repo,
        "ingestion_service": ingestion_service,
        "tmp_path": tmp_path,
    }


def _ingest_doc(service: DocumentIngestionService, filename: str, content: str) -> str:
    rec = service.save_upload_stream(filename, io.BytesIO(content.encode("utf-8")))
    processed = service.process_document(rec.doc_id)
    return processed.doc_id


class TestInvestigationServiceValidation:
    """Test validation constraints on investigation creation and execution."""

    def test_create_validation_empty_query(self, test_setup: dict[str, Any]) -> None:
        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=MockLLMClient(),
        )

        with pytest.raises(InvestigationValidationError, match="non-empty"):
            service.create_investigation(
                title="Test",
                query="   ",
                document_ids=["doc_1"],
            )

    def test_create_validation_empty_doc_ids(self, test_setup: dict[str, Any]) -> None:
        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=MockLLMClient(),
        )

        with pytest.raises(InvestigationValidationError, match="At least one"):
            service.create_investigation(
                title="Test",
                query="Valid query",
                document_ids=[],
            )

    def test_create_validation_nonexistent_doc(self, test_setup: dict[str, Any]) -> None:
        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=MockLLMClient(),
        )

        with pytest.raises(
            (InvestigationValidationError, DocumentNotFoundError), match="not found"
        ):
            service.create_investigation(
                title="Test",
                query="Valid query",
                document_ids=["non_existent_doc_id"],
            )

    def test_create_default_aspects(self, test_setup: dict[str, Any]) -> None:
        ingestion = test_setup["ingestion_service"]
        doc_id = _ingest_doc(ingestion, "doc1.txt", "Attention mechanism research and benchmarks.")

        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=MockLLMClient(),
        )

        inv = service.create_investigation(
            title="",
            query="Attention efficiency",
            document_ids=[doc_id],
            aspects=None,
        )

        assert inv.title == "Investigation: Attention efficiency"
        assert len(inv.aspects) >= 3
        assert "Methodology & Experimental Setup" in inv.aspects


class TestInvestigationServiceExecution:
    """Test investigation pipeline execution and synthesis."""

    def test_run_investigation_success(self, test_setup: dict[str, Any]) -> None:
        ingestion = test_setup["ingestion_service"]
        doc1_id = _ingest_doc(
            ingestion,
            "flash.txt",
            "FlashAttention paper. Throughput is 3x faster than baseline. Memory is linear.",
        )
        doc2_id = _ingest_doc(
            ingestion,
            "standard.txt",
            "Standard attention paper. Baseline throughput is 100 queries/sec. Quadratic memory.",
        )

        chunks_doc1 = test_setup["doc_repo"].get_chunks(doc1_id)
        chunks_doc2 = test_setup["doc_repo"].get_chunks(doc2_id)
        assert len(chunks_doc1) > 0
        assert len(chunks_doc2) > 0

        # LLM response for doc 1
        llm_response_1 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Throughput & Latency",
                        "claim": "Throughput is 3x faster than baseline standard attention.",
                        "chunk_id": chunks_doc1[0].chunk_id,
                        "page_number": 1,
                        "supporting_passage": "Throughput is 3x faster than baseline.",
                        "relationship": "supports",
                        "confidence": 0.96,
                        "methodology_notes": "Measured on A100 GPUs.",
                    }
                ]
            }
        )

        # LLM response for doc 2
        llm_response_2 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Throughput & Latency",
                        "claim": "Standard baseline operates at 100 queries/sec.",
                        "chunk_id": chunks_doc2[0].chunk_id,
                        "page_number": 1,
                        "supporting_passage": "Baseline throughput is 100 queries/sec.",
                        "relationship": "contradicts",
                        "confidence": 0.91,
                        "methodology_notes": "Measured on baseline hardware.",
                    }
                ]
            }
        )

        mock_llm = MockLLMClient(response_queue=[llm_response_1, llm_response_2])
        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=mock_llm,
        )

        inv = service.create_investigation(
            title="Attention Comparison",
            query="Compare throughput and memory",
            document_ids=[doc1_id, doc2_id],
            aspects=["Throughput & Latency", "Evaluation Metrics"],
        )

        report = service.run_investigation(inv.investigation_id)

        assert report.investigation_id == inv.investigation_id
        assert report.total_findings == 4
        assert report.relationship_counts.get("supports") == 1
        assert report.relationship_counts.get("contradicts") == 1
        assert report.relationship_counts.get("not_addressed") == 2

        topic_throughput = next(t for t in report.topics if t.topic_name == "Throughput & Latency")
        assert len(topic_throughput.findings) == 2
        assert topic_throughput.disagreements_summary is not None
        assert "Conflicting or opposing evidence" in topic_throughput.disagreements_summary

        topic_metrics = next(t for t in report.topics if t.topic_name == "Evaluation Metrics")
        assert len(topic_metrics.findings) == 2
        assert all(f.relationship == "not_addressed" for f in topic_metrics.findings)
        assert len(topic_metrics.evidence_gaps) > 0

        # Verify Markdown export
        md = service.export_investigation_markdown(inv.investigation_id)
        assert "# STING Investigation Report: Attention Comparison" in md
        assert "## 1. Executive Summary" in md

    def test_run_investigation_hallucinated_chunk_fail_closed(
        self, test_setup: dict[str, Any]
    ) -> None:
        """Verify that hallucinated chunk IDs are reconciled with real retrieved chunks."""
        ingestion = test_setup["ingestion_service"]
        doc1_id = _ingest_doc(
            ingestion,
            "sample.txt",
            "This is a verified document chunk with specific facts about deep learning.",
        )

        # LLM returns a made-up hallucinated chunk ID
        hallucinated_response = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Key Findings",
                        "claim": "Deep learning facts.",
                        "chunk_id": "hallucinated_fake_chunk_id_9999",
                        "page_number": 99,
                        "supporting_passage": "specific facts about deep learning",
                        "relationship": "supports",
                        "confidence": 0.85,
                    }
                ]
            }
        )

        mock_llm = MockLLMClient(response_queue=[hallucinated_response])
        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=mock_llm,
        )

        inv = service.create_investigation(
            title="Hallucination Test",
            query="deep learning",
            document_ids=[doc1_id],
            aspects=["Key Findings"],
        )

        report = service.run_investigation(inv.investigation_id)
        assert report.total_findings == 1
        finding = report.topics[0].findings[0]
        # Chunk ID must have been reconciled to the genuine chunk in the document!
        real_chunk_id = test_setup["doc_repo"].get_chunks(doc1_id)[0].chunk_id
        assert finding.chunk_id == real_chunk_id

    def test_run_nonexistent_raises_404(self, test_setup: dict[str, Any]) -> None:
        service = InvestigationService(
            investigation_repository=test_setup["inv_repo"],
            document_repository=test_setup["doc_repo"],
            llm_client=MockLLMClient(),
        )

        with pytest.raises(InvestigationNotFoundError):
            service.run_investigation("invalid_inv_id")
