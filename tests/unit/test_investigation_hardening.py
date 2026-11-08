"""Targeted correctness-hardening regression tests for STING Evidence Investigation."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from reliability_lab.core.models import Chunk
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.investigation.repository import InvestigationRepository
from reliability_lab.investigation.service import InvestigationService
from reliability_lab.llm.client import MockLLMClient
from reliability_lab.storage.database import Database


class DeterministicEmbedder(BaseEmbedder):
    """Deterministic 4D embedder for testing."""

    @property
    def model_name(self) -> str:
        return "deterministic-embedder"

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


class MockQueryServiceWithBoundedRetrieval:
    """Mock query service that simulates bounded top-k retrieval."""

    def __init__(self, top_chunks: list[Chunk]) -> None:
        self.top_chunks = top_chunks

    def query(
        self,
        query_text: str,
        selected_doc_ids: list[str] | None = None,
        strategy: str = "hybrid",
        top_k: int = 8,
    ) -> tuple[str, Any]:
        trace = MagicMock()
        trace.evidence_chunks = self.top_chunks[:top_k]
        return "mock answer", trace


@pytest.fixture
def test_env(tmp_path: Path) -> dict[str, Any]:
    db_path = tmp_path / "hardening_test.db"
    db = Database(str(db_path))
    doc_repo = DocumentRepository(db_path)
    inv_repo = InvestigationRepository(db)

    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=DeterministicEmbedder(),
    )

    return {
        "db": db,
        "doc_repo": doc_repo,
        "inv_repo": inv_repo,
        "ingestion": ingestion,
    }


def _ingest(service: DocumentIngestionService, filename: str, text: str) -> str:
    rec = service.save_upload_stream(filename, io.BytesIO(text.encode("utf-8")))
    processed = service.process_document(rec.doc_id)
    return processed.doc_id


class TestEvidenceGapSemantics:
    """1. Fix evidence-gap semantics and distinguish bounded retrieval from verified absence."""

    def test_evidence_outside_top_k_retrieval(self, test_env: dict[str, Any]) -> None:
        """Evidence outside top-k must be insufficient_evidence, not not_addressed."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        # Generate a large document with 20 distinct long paragraphs (ensures >= 16 chunks)
        head_p = ("Header paragraph discussing general architecture in deep neural networks. ") * 5
        paragraphs = [head_p]
        for i in range(1, 18):
            body_p = (f"Paragraph {i}: Generic discussion of network layers and forward pass. ") * 5
            paragraphs.append(body_p)
        p18 = ("Paragraph 18: PagedAttention eliminates 96% of KV cache memory waste. ") * 5
        paragraphs.append(p18)
        p19 = ("Paragraph 19: Summary of future scaling directions and memory management. ") * 5
        paragraphs.append(p19)
        content = "\n\n".join(paragraphs)

        doc_id = _ingest(ingestion, "large_transformer_paper.txt", content)
        chunks = doc_repo.get_chunks(doc_id)
        assert len(chunks) >= 16

        # Simulate query service that returns only top 8 chunks (which do not include chunk 18)
        query_service = MockQueryServiceWithBoundedRetrieval(top_chunks=chunks[:8])

        # LLM returns empty findings for Memory Efficiency since chunk 18 was not retrieved
        mock_llm = MockLLMClient(response_queue=[json.dumps({"findings": []})])
        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            query_service=query_service,  # type: ignore[arg-type]
            llm_client=mock_llm,
        )

        inv = service.create_investigation(
            title="KV Cache Gap Test",
            query="Memory efficiency of paging",
            document_ids=[doc_id],
            aspects=["Memory Efficiency"],
        )

        report = service.run_investigation(inv.investigation_id)
        finding = report.topics[0].findings[0]

        # Must NOT be classified as not_addressed because retrieval was bounded/sampled
        assert finding.relationship == "insufficient_evidence"
        assert "Insufficient retrieved evidence" in finding.claim
        assert "Partial retrieval coverage" in (finding.methodology_notes or "")

        # Coverage limitations must be visible in document summaries
        doc_summary = report.document_summaries[0]
        assert doc_summary["coverage_mode"] == "sampled_retrieval"
        assert doc_summary["total_chunks"] == len(chunks)
        assert doc_summary["retrieved_chunks"] < len(chunks)

    def test_document_with_genuinely_missing_coverage(self, test_env: dict[str, Any]) -> None:
        """Missing topic coverage under exhaustive examination gets not_addressed."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        # Document with 2 chunks about database b-trees (<= 15 chunks = exhaustive coverage)
        content = (
            "Database Indexing Research.\n\n"
            "This paper evaluates B+ tree balance factors and write-ahead logging."
        )
        doc_id = _ingest(ingestion, "database_paper.txt", content)

        mock_llm = MockLLMClient(response_queue=[json.dumps({"findings": []})])
        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=mock_llm,
        )

        inv = service.create_investigation(
            title="Exhaustive Absence Test",
            query="Transformer optimization",
            document_ids=[doc_id],
            aspects=["KV Cache Paging"],
        )

        report = service.run_investigation(inv.investigation_id)
        finding = report.topics[0].findings[0]

        # Legitimate verified absence because entire document was examined
        assert finding.relationship == "not_addressed"
        assert finding.confidence == 1.0
        assert "Verified absent across all" in (finding.methodology_notes or "")
        assert report.document_summaries[0]["coverage_mode"] == "exhaustive"


class TestCrossDocumentSynthesisLogic:
    """2. Improve cross-document agreement and contradiction logic."""

    def test_two_papers_reporting_matching_claims(self, test_env: dict[str, Any]) -> None:
        """Two papers reporting matching claims establish verified consensus."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc1_id = _ingest(
            ingestion,
            "paper_1.txt",
            "FlashAttention achieves 3x speedup on A100 GPU with exact tiling.",
        )
        doc2_id = _ingest(
            ingestion,
            "paper_2.txt",
            "Hardware-aware tiling achieves 3x speedup on A100 GPU for attention.",
        )

        c1 = doc_repo.get_chunks(doc1_id)[0]
        c2 = doc_repo.get_chunks(doc2_id)[0]

        resp_1 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Throughput",
                        "claim": "FlashAttention achieves 3x speedup on A100 GPU.",
                        "chunk_id": c1.chunk_id,
                        "page_number": c1.page_number,
                        "supporting_passage": "FlashAttention achieves 3x speedup on A100 GPU",
                        "relationship": "supports",
                        "confidence": 0.95,
                        "methodology_notes": "Measured on A100 GPU with FP16.",
                    }
                ]
            }
        )
        resp_2 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Throughput",
                        "claim": "Tiling achieves 3x speedup on A100 GPU.",
                        "chunk_id": c2.chunk_id,
                        "page_number": c2.page_number,
                        "supporting_passage": "tiling achieves 3x speedup on A100 GPU",
                        "relationship": "supports",
                        "confidence": 0.92,
                        "methodology_notes": "Measured on A100 GPU with FP16.",
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[resp_1, resp_2]),
        )

        inv = service.create_investigation(
            title="Consensus Verification",
            query="Attention speedup",
            document_ids=[doc1_id, doc2_id],
            aspects=["Throughput"],
        )

        report = service.run_investigation(inv.investigation_id)
        topic = report.topics[0]
        assert "Verified consensus across 2 sources" in topic.agreements_summary
        assert "paper_1.txt" in topic.agreements_summary
        assert "paper_2.txt" in topic.agreements_summary

    def test_two_papers_reporting_genuinely_conflicting_claims(
        self, test_env: dict[str, Any]
    ) -> None:
        """Two papers reporting genuinely conflicting claims are identified."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc1_id = _ingest(
            ingestion,
            "spec_a.txt",
            "Speculative decoding reduces latency by 2.5x on A100 GPU.",
        )
        doc2_id = _ingest(
            ingestion,
            "spec_b.txt",
            "Speculative decoding increases latency by 1.4x on A100 GPU.",
        )

        c1 = doc_repo.get_chunks(doc1_id)[0]
        c2 = doc_repo.get_chunks(doc2_id)[0]

        resp_1 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Latency",
                        "claim": "Speculative decoding reduces latency by 2.5x on A100 GPU.",
                        "chunk_id": c1.chunk_id,
                        "page_number": c1.page_number,
                        "supporting_passage": "Speculative decoding reduces latency by 2.5x",
                        "relationship": "supports",
                        "confidence": 0.96,
                        "methodology_notes": "Batch size 1 on A100 GPU.",
                    }
                ]
            }
        )
        resp_2 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Latency",
                        "claim": "Speculative decoding increases latency by 1.4x on A100 GPU.",
                        "chunk_id": c2.chunk_id,
                        "page_number": c2.page_number,
                        "supporting_passage": "Speculative decoding increases latency by 1.4x",
                        "relationship": "contradicts",
                        "confidence": 0.93,
                        "methodology_notes": "Batch size 1 on A100 GPU.",
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[resp_1, resp_2]),
        )

        inv = service.create_investigation(
            title="Conflict Audit",
            query="Speculative decoding latency",
            document_ids=[doc1_id, doc2_id],
            aspects=["Latency"],
        )

        report = service.run_investigation(inv.investigation_id)
        topic = report.topics[0]
        assert "Conflicting or opposing evidence" in topic.disagreements_summary
        assert "spec_b.txt" in topic.disagreements_summary

    def test_different_experimental_conditions_not_labelled_contradictions(
        self, test_env: dict[str, Any]
    ) -> None:
        """Differing conditions (GPU vs CPU) must not be called direct contradiction."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc1_id = _ingest(
            ingestion,
            "server_gpu.txt",
            "Model achieves 200 tok/sec on NVIDIA A100 GPU with FP16 precision.",
        )
        doc2_id = _ingest(
            ingestion,
            "mobile_cpu.txt",
            "Model achieves 15 tok/sec on Raspberry Pi 4 ARM CPU with INT4 precision.",
        )

        c1 = doc_repo.get_chunks(doc1_id)[0]
        c2 = doc_repo.get_chunks(doc2_id)[0]

        resp_1 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Throughput",
                        "claim": "Achieves 200 tok/sec on NVIDIA A100 GPU.",
                        "chunk_id": c1.chunk_id,
                        "page_number": 1,
                        "supporting_passage": "Model achieves 200 tok/sec on NVIDIA A100 GPU",
                        "relationship": "supports",
                        "confidence": 0.95,
                        "methodology_notes": "Measured on NVIDIA A100 GPU with FP16 precision.",
                    }
                ]
            }
        )
        resp_2 = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Throughput",
                        "claim": "Achieves 15 tok/sec on Raspberry Pi 4 CPU.",
                        "chunk_id": c2.chunk_id,
                        "page_number": 1,
                        "supporting_passage": "Model achieves 15 tok/sec on Raspberry Pi 4 ARM CPU",
                        "relationship": "contradicts",
                        "confidence": 0.90,
                        "methodology_notes": "Measured on Raspberry Pi 4 ARM CPU baseline INT4.",
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[resp_1, resp_2]),
        )

        inv = service.create_investigation(
            title="Setup Variance Audit",
            query="Throughput performance",
            document_ids=[doc1_id, doc2_id],
            aspects=["Throughput"],
        )

        report = service.run_investigation(inv.investigation_id)
        topic = report.topics[0]
        # Must acknowledge distinct experimental setups rather than pure conflict
        assert (
            "distinct experimental setups" in topic.disagreements_summary
            or "baseline conditions" in topic.disagreements_summary
        )


class TestEvidenceProvenanceHardening:
    """3. Harden evidence provenance verification, page numbers, and quote validity."""

    def test_empty_supporting_passage_fails_closed(self, test_env: dict[str, Any]) -> None:
        """Finding with empty supporting passage must fail closed to insufficient_evidence."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc_id = _ingest(
            ingestion,
            "doc_empty_passage.txt",
            "This document talks about linear attention complexity.",
        )
        c = doc_repo.get_chunks(doc_id)[0]

        # Model claims supports, but leaves supporting_passage empty
        mock_response = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Complexity",
                        "claim": "Attention is linear.",
                        "chunk_id": c.chunk_id,
                        "page_number": c.page_number,
                        "supporting_passage": "   ",
                        "relationship": "supports",
                        "confidence": 0.95,
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[mock_response]),
        )

        inv = service.create_investigation(
            title="Empty Passage Test",
            query="Linear attention",
            document_ids=[doc_id],
            aspects=["Complexity"],
        )

        report = service.run_investigation(inv.investigation_id)
        finding = report.topics[0].findings[0]

        assert finding.relationship == "insufficient_evidence"
        assert finding.confidence == 0.0
        assert "[PROVENANCE FAILED" in finding.supporting_passage

    def test_fabricated_passage_valid_chunk_id_fails_closed(self, test_env: dict[str, Any]) -> None:
        """Fabricated passage with valid chunk ID must fail closed."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc_id = _ingest(
            ingestion,
            "real_doc.txt",
            "This is genuine verified content about standard cross entropy loss.",
        )
        c = doc_repo.get_chunks(doc_id)[0]

        # Model gives valid chunk ID, but invents a quote not in the chunk
        fake_quote = "Quantum entanglement accelerates neural networks by 10000x."
        mock_response = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Quantum Speedup",
                        "claim": "Neural networks achieve quantum speedup.",
                        "chunk_id": c.chunk_id,
                        "page_number": c.page_number,
                        "supporting_passage": fake_quote,
                        "relationship": "supports",
                        "confidence": 0.99,
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[mock_response]),
        )

        inv = service.create_investigation(
            title="Fabricated Passage Test",
            query="Quantum speedup",
            document_ids=[doc_id],
            aspects=["Quantum Speedup"],
        )

        report = service.run_investigation(inv.investigation_id)
        finding = report.topics[0].findings[0]

        # Model quote must NEVER be shown as verified passage
        assert fake_quote not in finding.supporting_passage
        assert finding.relationship == "insufficient_evidence"
        assert finding.confidence == 0.0
        assert "[PROVENANCE FAILED" in finding.supporting_passage

    def test_incorrect_page_number_canonical_override(self, test_env: dict[str, Any]) -> None:
        """Canonical chunk metadata must override incorrect model-reported page numbers."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc_id = _ingest(
            ingestion,
            "paged_doc.txt",
            "This passage verifies deterministic memory allocation on page 1.",
        )
        c = doc_repo.get_chunks(doc_id)[0]
        canonical_page = c.page_number
        assert canonical_page is not None

        # Model reports incorrect page 99
        mock_response = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Memory",
                        "claim": "Deterministic memory allocation is verified.",
                        "chunk_id": c.chunk_id,
                        "page_number": 99,
                        "supporting_passage": "deterministic memory allocation",
                        "relationship": "supports",
                        "confidence": 0.90,
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[mock_response]),
        )

        inv = service.create_investigation(
            title="Page Correction Test",
            query="Memory allocation",
            document_ids=[doc_id],
            aspects=["Memory"],
        )

        report = service.run_investigation(inv.investigation_id)
        finding = report.topics[0].findings[0]

        # Canonical page must override 99
        assert finding.page_number == canonical_page
        assert finding.relationship == "supports"
        assert "overrides reported page 99" in (finding.methodology_notes or "")

    def test_fabricated_chunk_id_fails_closed(self, test_env: dict[str, Any]) -> None:
        """Fabricated chunk ID with unverified passage must fail closed."""
        ingestion = test_env["ingestion"]
        doc_repo = test_env["doc_repo"]

        doc_id = _ingest(
            ingestion,
            "doc_legit.txt",
            "Genuine content about convolutional filters.",
        )

        # Model outputs a non-existent chunk ID and non-existent passage
        mock_response = json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Filters",
                        "claim": "Filters optimize image processing.",
                        "chunk_id": "totally_fabricated_chunk_99999",
                        "page_number": 42,
                        "supporting_passage": "This sentence does not exist anywhere.",
                        "relationship": "supports",
                        "confidence": 0.88,
                    }
                ]
            }
        )

        service = InvestigationService(
            investigation_repository=test_env["inv_repo"],
            document_repository=doc_repo,
            llm_client=MockLLMClient(response_queue=[mock_response]),
        )

        inv = service.create_investigation(
            title="Fabricated Chunk Test",
            query="Filters",
            document_ids=[doc_id],
            aspects=["Filters"],
        )

        report = service.run_investigation(inv.investigation_id)
        finding = report.topics[0].findings[0]

        assert finding.relationship == "insufficient_evidence"
        assert finding.confidence == 0.0
        assert finding.chunk_id == "invalid_chunk_provenance"
        assert "[PROVENANCE FAILED" in finding.supporting_passage
