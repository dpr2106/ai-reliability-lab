"""Unit tests for Document Ingestion, Text Parsing, Repository, and Indexing Service."""

from __future__ import annotations

import io
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from reliability_lab.core.models import Chunk
from reliability_lab.indexing.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from reliability_lab.ingestion.chunker import RecursiveCharacterChunker
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.parser import FileNotFoundParserError
from reliability_lab.ingestion.repository import DocumentRepository, GenerationMismatchError
from reliability_lab.ingestion.service import (
    DenseIndexIntegrityError,
    DocumentDeletionError,
    DocumentIngestionService,
    EmptyFileError,
    IngestionError,
    InvalidDocumentFormatError,
    UploadSizeLimitExceededError,
    clear_shared_doc_locks,
)
from reliability_lab.ingestion.text_parser import InvalidTextDocumentError, TextDocumentParser
from reliability_lab.storage.database import Database


class DummyMockEmbedder(BaseEmbedder):
    """Deterministic in-memory embedder for hermetic FAISS testing."""

    def __init__(self, dim: int = 16) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "dummy-mock-embedder"

    @property
    def artifact_fingerprint(self) -> str:
        return "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str] | tuple[str, ...], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for text in texts:
            # Deterministic hash-based pseudo vector
            val = float(sum(ord(c) for c in text) % 100) / 100.0
            vec = np.full(self._dim, val, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


class ControllableEmbedder(DummyMockEmbedder):
    """Embedder with explicit synchronization barriers for deterministic concurrency tests."""

    def __init__(self, dim: int = 16) -> None:
        super().__init__(dim=dim)
        self.started_event = threading.Event()
        self.resume_event = threading.Event()
        self.enabled = False

    def embed_texts(self, texts: list[str] | tuple[str, ...], batch_size: int = 32) -> np.ndarray:
        if self.enabled:
            self.started_event.set()
            assert self.resume_event.wait(timeout=10.0), "Worker resume event timed out"
        return super().embed_texts(texts, batch_size)


def create_sample_pdf(content: str = "This is a test PDF document for ingestion.") -> bytes:
    """Generate a valid single-page PDF in-memory."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), content)
    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


def create_encrypted_pdf(password: str = "secret") -> bytes:
    """Generate a password-protected PDF in-memory."""
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), "Confidential encrypted content.")
    pdf_bytes = doc.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw=password,
        user_pw=password,
    )
    doc.close()
    return pdf_bytes


class TestTextDocumentParser:
    """Tests verifying plain text and Markdown document parsing."""

    def test_parse_plain_text(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.txt"
        file.write_text("Line 1 text.\nLine 2 text.\nLine 3 text.", encoding="utf-8")

        parser = TextDocumentParser()
        doc = parser.parse(file, doc_id="test_txt")

        assert doc.doc_id == "test_txt"
        assert doc.page_count == 1
        assert "Line 1 text." in doc.content
        assert doc.metadata["file_type"] == "txt"
        assert len(doc.metadata["pages"]) == 1
        page = doc.metadata["pages"][0]
        assert page["page_number"] == 1
        assert len(page["lines"]) == 3

    def test_parse_markdown(self, tmp_path: Path) -> None:
        file = tmp_path / "sample.md"
        file.write_text("# Title\n\nParagraph with **bold** text.", encoding="utf-8")

        parser = TextDocumentParser()
        doc = parser.parse(file)

        assert doc.doc_id == "sample"
        assert doc.page_count == 1
        assert doc.metadata["file_type"] == "md"
        assert doc.metadata["pages"][0]["page_number"] == 1

    def test_missing_file_raises_error(self, tmp_path: Path) -> None:
        parser = TextDocumentParser()
        with pytest.raises(FileNotFoundParserError):
            parser.parse(tmp_path / "nonexistent.txt")

    def test_binary_null_bytes_rejected(self, tmp_path: Path) -> None:
        file = tmp_path / "binary.txt"
        file.write_bytes(b"Text with null \x00 byte.")

        parser = TextDocumentParser()
        with pytest.raises(InvalidTextDocumentError, match="contains binary null bytes"):
            parser.parse(file)


class TestDocumentRepository:
    """Tests verifying SQLite document repository persistence, queries, and chunk storage."""

    @pytest.fixture
    def repo(self) -> DocumentRepository:
        db = Database(":memory:")
        return DocumentRepository(db)

    def test_insert_and_get_document(self, repo: DocumentRepository) -> None:
        record = DocumentRecord(
            doc_id="doc_001",
            filename="report.pdf",
            storage_path="/data/doc_001.pdf",
            file_type="pdf",
            file_size=1024,
            content_hash="abc123hash",
            status=DocumentStatus.UPLOADED,
        )
        repo.insert_document(record)

        fetched = repo.get_document("doc_001")
        assert fetched is not None
        assert fetched.doc_id == "doc_001"
        assert fetched.filename == "report.pdf"
        assert fetched.file_size == 1024
        assert fetched.content_hash == "abc123hash"
        assert fetched.status == DocumentStatus.UPLOADED
        assert fetched.indexing_status == IndexingStatus.PENDING

    def test_get_document_by_hash(self, repo: DocumentRepository) -> None:
        record = DocumentRecord(
            doc_id="doc_hash_test",
            filename="notes.txt",
            storage_path="/data/notes.txt",
            file_type="txt",
            file_size=50,
            content_hash="unique_hash_xyz",
        )
        repo.insert_document(record)

        fetched = repo.get_document_by_hash("unique_hash_xyz")
        assert fetched is not None
        assert fetched.doc_id == "doc_hash_test"

        assert repo.get_document_by_hash("nonexistent_hash") is None

    def test_update_document_status(self, repo: DocumentRepository) -> None:
        record = DocumentRecord(
            doc_id="doc_update",
            filename="update.md",
            storage_path="/data/update.md",
            file_type="md",
            file_size=200,
            content_hash="hash_up",
        )
        repo.insert_document(record)

        repo.update_document_status(
            "doc_update",
            status=DocumentStatus.READY,
            indexing_status=IndexingStatus.INDEXED,
            page_count=1,
            chunk_count=5,
            metadata={"custom_key": "val"},
        )

        fetched = repo.get_document("doc_update")
        assert fetched is not None
        assert fetched.status == DocumentStatus.READY
        assert fetched.indexing_status == IndexingStatus.INDEXED
        assert fetched.page_count == 1
        assert fetched.chunk_count == 5
        assert fetched.metadata.get("custom_key") == "val"

    def test_save_and_retrieve_chunks(self, repo: DocumentRepository) -> None:
        record = DocumentRecord(
            doc_id="doc_chunks",
            filename="chunks.txt",
            storage_path="/data/chunks.txt",
            file_type="txt",
            file_size=300,
            content_hash="hash_chunks",
        )
        repo.insert_document(record)

        chunks = [
            Chunk(
                chunk_id="chk_001",
                doc_id="doc_chunks",
                content="First chunk text.",
                page_number=1,
                char_start=0,
                char_end=17,
                metadata={"strategy": "recursive"},
            ),
            Chunk(
                chunk_id="chk_002",
                doc_id="doc_chunks",
                content="Second chunk text.",
                page_number=1,
                char_start=18,
                char_end=36,
                metadata={"strategy": "recursive"},
            ),
        ]

        repo.save_chunks("doc_chunks", chunks)

        loaded = repo.get_chunks("doc_chunks")
        assert len(loaded) == 2
        assert loaded[0].chunk_id == "chk_001"
        assert loaded[0].page_number == 1
        assert loaded[0].char_start == 0
        assert loaded[0].char_end == 17
        assert loaded[1].chunk_id == "chk_002"

        # Check document chunk_count update
        doc = repo.get_document("doc_chunks")
        assert doc is not None
        assert doc.chunk_count == 2

    def test_list_and_count_documents(self, repo: DocumentRepository) -> None:
        for i in range(10):
            status = DocumentStatus.READY if i % 2 == 0 else DocumentStatus.FAILED
            repo.insert_document(
                DocumentRecord(
                    doc_id=f"doc_{i}",
                    filename=f"file_{i}.txt",
                    storage_path=f"/data/{i}.txt",
                    file_type="txt",
                    file_size=100 + i,
                    content_hash=f"hash_{i}",
                    status=status,
                )
            )

        assert repo.count_documents() == 10
        assert repo.count_documents(status=DocumentStatus.READY) == 5
        assert repo.count_documents(status=DocumentStatus.FAILED) == 5

        page1 = repo.list_documents(limit=5, offset=0)
        assert len(page1) == 5

        ready_docs = repo.list_documents(status=DocumentStatus.READY)
        assert len(ready_docs) == 5
        assert all(d.status == DocumentStatus.READY for d in ready_docs)

    def test_delete_document(self, repo: DocumentRepository) -> None:
        record = DocumentRecord(
            doc_id="doc_del",
            filename="del.txt",
            storage_path="/data/del.txt",
            file_type="txt",
            file_size=10,
            content_hash="del_hash",
        )
        repo.insert_document(record)
        repo.save_chunks(
            "doc_del",
            [Chunk(chunk_id="c1", doc_id="doc_del", content="text", page_number=1)],
        )

        assert repo.delete_document("doc_del") is True
        assert repo.get_document("doc_del") is None
        assert len(repo.get_chunks("doc_del")) == 0
        assert repo.delete_document("doc_del") is False


class TestDocumentIngestionService:
    """Tests verifying upload streaming, format validation, chunking, and indexing pipeline."""

    @pytest.fixture
    def service(self, tmp_path: Path) -> DocumentIngestionService:
        db = Database(":memory:")
        repo = DocumentRepository(db)
        embedder = DummyMockEmbedder(dim=16)
        return DocumentIngestionService(
            repository=repo,
            storage_dir=tmp_path / "documents",
            index_dir=tmp_path / "indexes",
            max_upload_size=1024 * 1024,  # 1 MB
            embedder=embedder,
            chunk_size=100,
            chunk_overlap=20,
        )

    def test_upload_and_process_plain_text(self, service: DocumentIngestionService) -> None:
        text = (
            "This is a sample document for full end-to-end ingestion testing.\n"
            "It has multiple sentences."
        )
        stream = io.BytesIO(text.encode("utf-8"))

        record = service.save_upload_stream("article.txt", stream)
        assert record.status == DocumentStatus.UPLOADED
        assert record.file_type == "txt"
        assert record.file_size == len(text.encode("utf-8"))
        assert Path(record.storage_path).is_file()

        # Execute ingestion pipeline
        processed = service.process_document(record.doc_id)
        assert processed.status == DocumentStatus.READY
        assert processed.indexing_status == IndexingStatus.INDEXED
        assert processed.chunk_count > 0
        assert processed.page_count == 1

        # Check BM25 retrieval
        bm25 = service.get_bm25_retriever(record.doc_id)
        results = bm25.retrieve("sample document", top_k=2)
        assert len(results) > 0
        assert results[0].chunk.doc_id == record.doc_id
        assert results[0].chunk.page_number == 1

        # Check FAISS retrieval
        faiss_retriever = service.get_faiss_retriever(record.doc_id)
        dense_results = faiss_retriever.retrieve("sample document", top_k=2)
        assert len(dense_results) > 0
        assert dense_results[0].chunk.doc_id == record.doc_id

    def test_upload_and_process_pdf(self, service: DocumentIngestionService) -> None:
        pdf_bytes = create_sample_pdf(
            "Deep learning reliability requires rigorous evaluation baselines."
        )
        stream = io.BytesIO(pdf_bytes)

        record = service.save_upload_stream("deep_learning.pdf", stream)
        assert record.file_type == "pdf"

        processed = service.process_document(record.doc_id)
        assert processed.status == DocumentStatus.READY
        assert processed.indexing_status == IndexingStatus.INDEXED
        assert processed.chunk_count > 0
        assert processed.page_count == 1

        # Verify provenance retention
        chunks = service.repository.get_chunks(record.doc_id)
        assert all(c.page_number == 1 for c in chunks)
        assert all(c.char_start is not None and c.char_end is not None for c in chunks)

    def test_upload_empty_file_rejected(self, service: DocumentIngestionService) -> None:
        stream = io.BytesIO(b"")
        with pytest.raises(EmptyFileError, match="empty \\(0 bytes\\)"):
            service.save_upload_stream("empty.txt", stream)

    def test_upload_oversized_file_rejected(self, service: DocumentIngestionService) -> None:
        oversized = b"x" * (service.max_upload_size + 100)
        stream = io.BytesIO(oversized)
        with pytest.raises(UploadSizeLimitExceededError):
            service.save_upload_stream("huge.txt", stream)

    def test_upload_invalid_extension_rejected(self, service: DocumentIngestionService) -> None:
        stream = io.BytesIO(b"print('hello')")
        with pytest.raises(InvalidDocumentFormatError, match="Unsupported file extension"):
            service.save_upload_stream("script.py", stream)

    def test_upload_pdf_with_invalid_header_rejected(
        self, service: DocumentIngestionService
    ) -> None:
        stream = io.BytesIO(b"Not a PDF header")
        with pytest.raises(InvalidDocumentFormatError, match="invalid PDF magic header"):
            service.save_upload_stream("fake.pdf", stream)

    def test_upload_text_with_null_bytes_rejected(self, service: DocumentIngestionService) -> None:
        stream = io.BytesIO(b"Hello \x00 World")
        with pytest.raises(InvalidDocumentFormatError, match="contains binary null bytes"):
            service.save_upload_stream("null.txt", stream)

    def test_encrypted_pdf_marks_status_failed(self, service: DocumentIngestionService) -> None:
        pdf_bytes = create_encrypted_pdf(password="secret123")
        stream = io.BytesIO(pdf_bytes)

        record = service.save_upload_stream("locked.pdf", stream)
        processed = service.process_document(record.doc_id)

        assert processed.status == DocumentStatus.FAILED
        assert "password" in (processed.error_message or "").lower()

    def test_corrupted_pdf_marks_status_failed(self, service: DocumentIngestionService) -> None:
        corrupted_bytes = b"%PDF-1.4\nCorrupted binary junk \xff\xfe\xaa"
        stream = io.BytesIO(corrupted_bytes)

        record = service.save_upload_stream("corrupted.pdf", stream)
        processed = service.process_document(record.doc_id)

        assert processed.status == DocumentStatus.FAILED
        assert "extraction failed" in (processed.error_message or "").lower()

    def test_missing_dense_model_handles_gracefully(self, tmp_path: Path) -> None:
        # Service configured with no embedder and local model unavailable
        db = Database(":memory:")
        repo = DocumentRepository(db)
        no_model_service = DocumentIngestionService(
            repository=repo,
            storage_dir=tmp_path / "documents",
            index_dir=tmp_path / "indexes",
            embedder=None,  # triggers SentenceTransformerEmbedder(local_files_only=True)
        )

        stream = io.BytesIO(b"Simple test content for dense model fallback.")
        record = no_model_service.save_upload_stream("fallback.txt", stream)
        processed = no_model_service.process_document(record.doc_id)

        # Dense indexing skipped or indexed, but document remains READY with BM25 available
        assert processed.chunk_count > 0
        assert processed.status == DocumentStatus.READY
        assert processed.indexing_status in {IndexingStatus.PARTIAL, IndexingStatus.INDEXED}
        if processed.indexing_status == IndexingStatus.PARTIAL:
            assert "dense" in (processed.error_message or "").lower()
            bm25 = no_model_service.get_bm25_retriever(record.doc_id)
            assert len(bm25.retrieve("fallback", top_k=1)) == 1

    def test_partial_indexing_when_embedder_fails(self, tmp_path: Path) -> None:
        class FailingEmbedder(BaseEmbedder):
            @property
            def model_name(self) -> str:
                return "failing-embedder"

            @property
            def embedding_dim(self) -> int:
                return 16

            def embed_texts(self, texts, batch_size=32):
                raise RuntimeError("Neural inference failure (out of memory)")

        db = Database(":memory:")
        repo = DocumentRepository(db)
        service = DocumentIngestionService(
            repository=repo,
            storage_dir=tmp_path / "documents",
            index_dir=tmp_path / "indexes",
            embedder=FailingEmbedder(),
        )

        stream = io.BytesIO(b"Document content that indexes in BM25 but fails dense FAISS.")
        record = service.save_upload_stream("partial_doc.txt", stream)
        processed = service.process_document(record.doc_id)

        # Usable document with PARTIAL indexing and READY document status
        assert processed.status == DocumentStatus.READY
        assert processed.indexing_status == IndexingStatus.PARTIAL
        assert "dense indexing failed" in (processed.error_message or "").lower()

        # BM25 is fully usable
        bm25 = service.get_bm25_retriever(record.doc_id)
        results = bm25.retrieve("indexes", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.doc_id == record.doc_id

        # FAISS retrieval raises informative IngestionError
        with pytest.raises(IngestionError, match="PARTIAL"):
            service.get_faiss_retriever(record.doc_id)

    def test_delete_document_removes_files_and_indexes(
        self, service: DocumentIngestionService
    ) -> None:
        stream = io.BytesIO(b"Content to be deleted.")
        record = service.save_upload_stream("to_delete.txt", stream)
        service.process_document(record.doc_id)

        storage_file = Path(record.storage_path)
        index_dir = service.index_dir / record.doc_id
        assert storage_file.is_file()
        assert index_dir.is_dir()

        deleted = service.delete_document(record.doc_id)
        assert deleted is True
        assert not storage_file.exists()
        assert not index_dir.exists()
        assert service.repository.get_document(record.doc_id) is None

    def test_delete_document_preserves_db_on_filesystem_failure(
        self, service: DocumentIngestionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream = io.BytesIO(b"File protected from deletion.")
        record = service.save_upload_stream("locked_file.txt", stream)
        service.process_document(record.doc_id)

        # Simulate filesystem unlink failure
        def mock_unlink(self_path):
            raise OSError("Permission denied: file is locked by another process")

        monkeypatch.setattr(Path, "unlink", mock_unlink)

        with pytest.raises(DocumentDeletionError, match="Failed to delete file"):
            service.delete_document(record.doc_id)

        # Database record must still exist so deletion can be retried
        preserved = service.repository.get_document(record.doc_id)
        assert preserved is not None
        assert preserved.doc_id == record.doc_id

        # Undo monkeypatch and retry deletion successfully
        monkeypatch.undo()
        retry_deleted = service.delete_document(record.doc_id)
        assert retry_deleted is True
        assert service.repository.get_document(record.doc_id) is None

    def test_delete_document_idempotent_when_files_already_absent(
        self, service: DocumentIngestionService
    ) -> None:
        stream = io.BytesIO(b"Files deleted externally.")
        record = service.save_upload_stream("external_del.txt", stream)
        service.process_document(record.doc_id)

        # Manually remove disk artifacts beforehand
        storage_file = Path(record.storage_path)
        index_dir = service.index_dir / record.doc_id
        if storage_file.exists():
            storage_file.unlink()
        if index_dir.exists():
            shutil.rmtree(index_dir)

        # delete_document should handle cleanly and delete database record
        deleted = service.delete_document(record.doc_id)
        assert deleted is True
        assert service.repository.get_document(record.doc_id) is None

        # Calling again on already deleted document returns False
        assert service.delete_document(record.doc_id) is False

    def test_process_document_unexpected_error_sets_failed_status(
        self, service: DocumentIngestionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stream = io.BytesIO(b"Test text for unexpected chunker error.")
        record = service.save_upload_stream("error_doc.txt", stream)

        # Simulate unexpected exception during chunking
        def mock_chunk(self, doc):
            raise RuntimeError("Unexpected tokenizer memory fault")

        from reliability_lab.ingestion.chunker import RecursiveCharacterChunker

        monkeypatch.setattr(RecursiveCharacterChunker, "chunk", mock_chunk)

        processed = service.process_document(record.doc_id)
        assert processed.status == DocumentStatus.FAILED
        assert "citation-aware chunking failed" in (processed.error_message or "").lower()

    def test_upload_stream_rejects_null_byte_in_subsequent_chunks(
        self, service: DocumentIngestionService
    ) -> None:
        # 64KB of clean text followed by binary null byte in 2nd chunk
        clean_prefix = b"A" * 65536
        dirty_suffix = b"Hello\x00World"
        stream = io.BytesIO(clean_prefix + dirty_suffix)

        with pytest.raises(InvalidDocumentFormatError, match="contains binary null bytes"):
            service.save_upload_stream("streaming_binary.txt", stream)

    def test_faiss_retriever_rejects_pending_failed_and_skipped_states(
        self, service: DocumentIngestionService
    ) -> None:
        # 1a. UPLOADED state
        stream1 = io.BytesIO(b"Pending document text.")
        rec1 = service.save_upload_stream("pending.txt", stream1)
        with pytest.raises(IngestionError, match="UPLOADED"):
            service.get_faiss_retriever(rec1.doc_id)

        # 1b. PENDING state (READY status with PENDING indexing_status)
        service.repository.update_document_status(
            rec1.doc_id,
            status=DocumentStatus.READY,
            indexing_status=IndexingStatus.PENDING,
        )
        with pytest.raises(IngestionError, match="PENDING"):
            service.get_faiss_retriever(rec1.doc_id)

        # 2. FAILED state
        service.repository.update_document_status(
            rec1.doc_id,
            status=DocumentStatus.FAILED,
            indexing_status=IndexingStatus.FAILED,
            error_message="OOM during embedding",
        )
        with pytest.raises(IngestionError, match="FAILED: OOM during embedding"):
            service.get_faiss_retriever(rec1.doc_id)

        # 3. SKIPPED state
        service.repository.update_document_status(
            rec1.doc_id,
            status=DocumentStatus.READY,
            indexing_status=IndexingStatus.SKIPPED,
        )
        with pytest.raises(IngestionError, match="skipped"):
            service.get_faiss_retriever(rec1.doc_id)

    def test_faiss_retriever_rejects_stale_directory_if_status_not_indexed(
        self, service: DocumentIngestionService
    ) -> None:
        # Create cleanly indexed document
        stream = io.BytesIO(b"Valid text for FAISS indexing test.")
        rec = service.save_upload_stream("stale_test.txt", stream)
        service.process_document(rec.doc_id)

        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        assert faiss_dir.is_dir()
        assert service.get_faiss_retriever(rec.doc_id) is not None

        # Simulate failed re-indexing leaving stale directory on disk
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.FAILED,
            indexing_status=IndexingStatus.FAILED,
            error_message="Re-indexing tokenizer failed",
        )

        # Must reject loading stale directory
        with pytest.raises(IngestionError, match="FAILED"):
            service.get_faiss_retriever(rec.doc_id)

    def test_faiss_retriever_validates_corrupt_or_missing_index_files(
        self, service: DocumentIngestionService
    ) -> None:
        stream = io.BytesIO(b"Content for corruption tests.")
        rec = service.save_upload_stream("corrupt_test.txt", stream)
        service.process_document(rec.doc_id)

        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        index_file = faiss_dir / "index.faiss"
        meta_file = faiss_dir / "metadata.json"

        # Case 1: Missing index.faiss
        index_file.unlink()
        with pytest.raises(IngestionError, match=r"required index files.*missing"):
            service.get_faiss_retriever(rec.doc_id)

        # Restore index.faiss
        index_file.write_bytes(b"dummy")

        # Case 2: Corrupt JSON in metadata.json
        meta_file.write_text("invalid json junk", encoding="utf-8")
        with pytest.raises(IngestionError, match=r"failed to parse 'metadata\.json'"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 3: JSON without 'dimension' key
        meta_file.write_text('{"format_version": "1.0"}', encoding="utf-8")
        with pytest.raises(IngestionError, match="missing required 'dimension' key"):
            service.get_faiss_retriever(rec.doc_id)

    def test_process_document_cleans_stale_faiss_on_extraction_chunking_failure(
        self, service: DocumentIngestionService
    ) -> None:
        # Index document cleanly first
        stream = io.BytesIO(b"Valid text initially.")
        rec = service.save_upload_stream("wipe_stale.txt", stream)
        service.process_document(rec.doc_id)

        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        assert faiss_dir.is_dir()

        # Corrupt storage file so extraction fails on re-processing
        storage_file = Path(rec.storage_path)
        storage_file.write_bytes(b"corrupt\x00data")

        processed = service.process_document(rec.doc_id)
        assert processed.status == DocumentStatus.FAILED
        assert processed.indexing_status == IndexingStatus.FAILED

        # Stale FAISS directory must be cleaned up from disk
        assert not faiss_dir.exists()

    def test_faiss_retriever_during_reprocessing_is_rejected(
        self, service: DocumentIngestionService
    ) -> None:
        """FAISS retrieval must be blocked during reprocessing while status is not READY."""
        stream = io.BytesIO(b"Text for reprocessing test.")
        rec = service.save_upload_stream("reprocess_in_progress.txt", stream)
        service.process_document(rec.doc_id)
        assert service.get_faiss_retriever(rec.doc_id) is not None

        # Simulate reprocessing in progress: status transitioned to EXTRACTING / PENDING
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.EXTRACTING,
            indexing_status=IndexingStatus.PENDING,
        )

        with pytest.raises(IngestionError, match="processing in progress"):
            service.get_faiss_retriever(rec.doc_id)

        # Also test CHUNKING and INDEXING lifecycle phases
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.CHUNKING,
        )
        with pytest.raises(IngestionError, match="processing in progress"):
            service.get_faiss_retriever(rec.doc_id)

        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.INDEXING,
        )
        with pytest.raises(IngestionError, match="processing in progress"):
            service.get_faiss_retriever(rec.doc_id)

    def test_faiss_retriever_after_extraction_failure_during_reprocessing(
        self, service: DocumentIngestionService
    ) -> None:
        """FAISS retrieval must be rejected after extraction failure during reprocessing."""
        stream = io.BytesIO(b"Initially valid document content.")
        rec = service.save_upload_stream("extraction_fail_reprocess.txt", stream)
        service.process_document(rec.doc_id)
        assert service.get_faiss_retriever(rec.doc_id) is not None

        # Corrupt the storage file
        Path(rec.storage_path).write_bytes(b"bad\x00data")
        processed = service.process_document(rec.doc_id)

        assert processed.status == DocumentStatus.FAILED
        assert processed.indexing_status == IndexingStatus.FAILED

        with pytest.raises(IngestionError, match="FAILED"):
            service.get_faiss_retriever(rec.doc_id)

    def test_faiss_retriever_after_chunking_failure_during_reprocessing(
        self, service: DocumentIngestionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FAISS retrieval must be rejected after chunking failure during reprocessing."""
        stream = io.BytesIO(b"Content for chunking failure reprocess.")
        rec = service.save_upload_stream("chunk_fail_reprocess.txt", stream)
        service.process_document(rec.doc_id)
        assert service.get_faiss_retriever(rec.doc_id) is not None

        def mock_chunk(self, doc):
            raise RuntimeError("Chunking buffer overflow error")

        monkeypatch.setattr(RecursiveCharacterChunker, "chunk", mock_chunk)

        processed = service.process_document(rec.doc_id)
        assert processed.status == DocumentStatus.FAILED
        assert processed.indexing_status == IndexingStatus.FAILED

        with pytest.raises(IngestionError, match="FAILED"):
            service.get_faiss_retriever(rec.doc_id)

    def test_successful_reprocessing_restores_valid_faiss_retrieval(
        self, service: DocumentIngestionService
    ) -> None:
        """Successful reprocessing must restore valid FAISS retrieval with updated chunks."""
        stream = io.BytesIO(b"Version 1 text content.")
        rec = service.save_upload_stream("reprocess_restore.txt", stream)
        service.process_document(rec.doc_id)
        retriever_v1 = service.get_faiss_retriever(rec.doc_id)
        res1 = retriever_v1.retrieve("Version", top_k=1)
        assert len(res1) == 1
        assert "Version 1" in res1[0].chunk.content

        # Update file with Version 2 content
        Path(rec.storage_path).write_text("Version 2 updated content.", encoding="utf-8")
        processed = service.process_document(rec.doc_id)

        assert processed.status == DocumentStatus.READY
        assert processed.indexing_status == IndexingStatus.INDEXED

        retriever_v2 = service.get_faiss_retriever(rec.doc_id)
        res2 = retriever_v2.retrieve("Version", top_k=1)
        assert len(res2) == 1
        assert "Version 2" in res2[0].chunk.content

    def test_cleanup_faiss_filesystem_failure_does_not_mask_error_and_is_retryable(
        self, service: DocumentIngestionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Filesystem cleanup failure must be logged, retryable, and not mask errors."""
        stream = io.BytesIO(b"Cleanup failure test document.")
        rec = service.save_upload_stream("cleanup_fail.txt", stream)
        service.process_document(rec.doc_id)

        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        assert faiss_dir.is_dir()

        # Simulate locked directory
        def failing_rmtree(path, *args, **kwargs):
            raise PermissionError("Directory locked by security scanner")

        monkeypatch.setattr(shutil, "rmtree", failing_rmtree)

        # 1. Direct cleanup returns False, does not raise
        assert service.cleanup_faiss_index(rec.doc_id) is False
        assert faiss_dir.is_dir()

        # 2. Invalidate status in DB
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.FAILED,
            indexing_status=IndexingStatus.FAILED,
            error_message="Ingestion failure occurred",
        )

        # Stale files remain on disk, but retrieval is strictly blocked by state
        with pytest.raises(IngestionError, match="FAILED"):
            service.get_faiss_retriever(rec.doc_id)

        # 3. Restore rmtree: cleanup is retryable and succeeds
        monkeypatch.undo()
        assert service.cleanup_faiss_index(rec.doc_id) is True
        assert not faiss_dir.exists()

    def test_faiss_retriever_corrupted_artifacts_handling(
        self, service: DocumentIngestionService
    ) -> None:
        """Handle malformed metadata, incompatible dimensions, and corrupt binary index files."""
        stream = io.BytesIO(b"Content for artifact corruption tests.")
        rec = service.save_upload_stream("corrupt_artifacts.txt", stream)
        service.process_document(rec.doc_id)

        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        meta_file = faiss_dir / "metadata.json"
        index_file = faiss_dir / "index.faiss"

        # Case 1: Metadata is JSON array instead of object
        meta_file.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(IngestionError, match="must be an object"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 2: Dimension is negative or zero
        meta_file.write_text('{"dimension": -5}', encoding="utf-8")
        with pytest.raises(IngestionError, match="positive integer"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 3: Dimension is incompatible with embedder
        meta_file.write_text('{"dimension": 128}', encoding="utf-8")
        with pytest.raises(IngestionError, match="Incompatible embedding dimension"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 4: Corrupt binary index file
        # Restore valid metadata with correct dimension for the active embedder
        active_embedder = service._embedder or DummyMockEmbedder(16)
        active_dim = active_embedder.embedding_dim
        doc = service.repository.get_document(rec.doc_id)
        assert doc is not None and doc.metadata is not None
        prov = doc.metadata["__embedding_provenance__"]
        meta_file.write_text(
            json.dumps(
                {
                    "format_version": "2.0",
                    "embedding_identity": active_embedder.embedding_identity.to_dict(),
                    "dimension": active_dim,
                    "index_type": "flat",
                    "normalize": True,
                    "chunks": {},
                    "artifact_digest": prov["artifact_digest"],
                    "chunk_manifest_hash": prov["chunk_manifest_hash"],
                }
            ),
            encoding="utf-8",
        )
        index_file.write_bytes(b"NOT_A_VALID_FAISS_INDEX_BINARY_DATA")
        with pytest.raises(
            IngestionError,
            match=(
                r"(Failed to read FAISS index binary|Invalid or corrupt FAISS index|"
                r"artifact digest mismatch)"
            ),
        ):
            service.get_faiss_retriever(rec.doc_id)

    def test_process_document_invalidation_ordering_rejects_during_cleanup(
        self, service: DocumentIngestionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """State invalidation must precede cleanup so retrieval is blocked immediately."""
        stream = io.BytesIO(b"Initial document content.")
        rec = service.save_upload_stream("order_test.txt", stream)
        service.process_document(rec.doc_id)
        assert service.get_faiss_retriever(rec.doc_id) is not None
        assert service.get_bm25_retriever(rec.doc_id) is not None

        cleanup_called = False
        original_cleanup = service.cleanup_faiss_index

        def intercept_cleanup(doc_id: str) -> bool:
            nonlocal cleanup_called
            cleanup_called = True
            # At the moment cleanup runs, SQLite state must ALREADY be invalid
            with pytest.raises(IngestionError, match="processing in progress"):
                service.get_faiss_retriever(doc_id)
            with pytest.raises(IngestionError, match="processing in progress"):
                service.get_bm25_retriever(doc_id)
            return original_cleanup(doc_id)

        monkeypatch.setattr(service, "cleanup_faiss_index", intercept_cleanup)
        service.process_document(rec.doc_id)
        assert cleanup_called is True

    def test_bm25_retriever_lifecycle_consistency(
        self, service: DocumentIngestionService, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """BM25 retriever strictly checks document lifecycle and rejects unready/failed states."""
        # 1. Uploaded document (not ready)
        stream = io.BytesIO(b"Version 1 text content.")
        rec = service.save_upload_stream("bm25_lifecycle.txt", stream)
        with pytest.raises(IngestionError, match="UPLOADED"):
            service.get_bm25_retriever(rec.doc_id)

        # 2. Ingest document to READY / INDEXED
        service.process_document(rec.doc_id)
        bm25_v1 = service.get_bm25_retriever(rec.doc_id)
        res1 = bm25_v1.retrieve("Version", top_k=1)
        assert len(res1) == 1
        assert "Version 1" in res1[0].chunk.content

        # 3. Blocked during reprocessing
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.EXTRACTING,
            indexing_status=IndexingStatus.PENDING,
        )
        with pytest.raises(IngestionError, match="processing in progress"):
            service.get_bm25_retriever(rec.doc_id)

        # 4. Blocked after extraction failure during reprocessing
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.FAILED,
            indexing_status=IndexingStatus.FAILED,
            error_message="Extraction crashed",
        )
        with pytest.raises(IngestionError, match="FAILED: Extraction crashed"):
            service.get_bm25_retriever(rec.doc_id)

        # 5. Blocked after chunking failure during reprocessing
        Path(rec.storage_path).write_text("Valid text for chunk fail.", encoding="utf-8")

        def failing_chunk(self_c, doc):
            raise RuntimeError("Chunking exploded")

        monkeypatch.setattr(RecursiveCharacterChunker, "chunk", failing_chunk)
        processed = service.process_document(rec.doc_id)
        assert processed.status == DocumentStatus.FAILED
        with pytest.raises(IngestionError, match="FAILED"):
            service.get_bm25_retriever(rec.doc_id)

        # 6. Restored after successful reprocessing with Version 2
        monkeypatch.undo()
        Path(rec.storage_path).write_text("Version 2 updated content.", encoding="utf-8")
        processed2 = service.process_document(rec.doc_id)
        assert processed2.status == DocumentStatus.READY
        assert processed2.indexing_status == IndexingStatus.INDEXED

        bm25_v2 = service.get_bm25_retriever(rec.doc_id)
        res2 = bm25_v2.retrieve("Version", top_k=1)
        assert len(res2) == 1
        assert "Version 2" in res2[0].chunk.content

        # 7. Valid BM25 fallback for READY / PARTIAL state
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.READY,
            indexing_status=IndexingStatus.PARTIAL,
        )
        assert service.get_bm25_retriever(rec.doc_id) is not None
        with pytest.raises(IngestionError, match="PARTIAL: BM25 lexical retrieval is available"):
            service.get_faiss_retriever(rec.doc_id)

    def test_metadata_dimension_validation_strictness(
        self, service: DocumentIngestionService
    ) -> None:
        """Dimension validation strictly rejects boolean, fractional, zero, and negative values."""
        stream = io.BytesIO(b"Data for strict dimension testing.")
        rec = service.save_upload_stream("strict_dim.txt", stream)
        service.process_document(rec.doc_id)

        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        meta_file = faiss_dir / "metadata.json"

        # Case 1: Boolean True
        meta_file.write_text('{"dimension": true}', encoding="utf-8")
        with pytest.raises(IngestionError, match="positive integer"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 2: Boolean False
        meta_file.write_text('{"dimension": false}', encoding="utf-8")
        with pytest.raises(IngestionError, match="positive integer"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 3: Fractional float
        meta_file.write_text('{"dimension": 16.5}', encoding="utf-8")
        with pytest.raises(IngestionError, match="positive integer"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 4: Zero
        meta_file.write_text('{"dimension": 0}', encoding="utf-8")
        with pytest.raises(IngestionError, match="positive integer"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 5: Negative integer
        meta_file.write_text('{"dimension": -16}', encoding="utf-8")
        with pytest.raises(IngestionError, match="positive integer"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 6: Missing dimension key
        meta_file.write_text('{"format_version": "1.0"}', encoding="utf-8")
        with pytest.raises(IngestionError, match="missing required 'dimension' key"):
            service.get_faiss_retriever(rec.doc_id)

        # Case 7: Valid dimension
        active_embedder = service._embedder or SentenceTransformerEmbedder(local_files_only=True)
        active_dim = active_embedder.embedding_dim
        real_chunks = service.repository.get_chunks(rec.doc_id)
        chunks_dict = {str(i): c.to_dict() for i, c in enumerate(real_chunks)}
        doc = service.repository.get_document(rec.doc_id)
        assert doc is not None and doc.metadata is not None
        prov = doc.metadata["__embedding_provenance__"]
        meta_file.write_text(
            json.dumps(
                {
                    "format_version": "2.0",
                    "embedding_identity": active_embedder.embedding_identity.to_dict(),
                    "dimension": active_dim,
                    "index_type": "flat",
                    "normalize": True,
                    "chunks": chunks_dict,
                    "artifact_digest": prov["artifact_digest"],
                    "chunk_manifest_hash": prov["chunk_manifest_hash"],
                }
            ),
            encoding="utf-8",
        )
        assert service.get_faiss_retriever(rec.doc_id) is not None

    def test_error_messages_path_privacy(self, service: DocumentIngestionService) -> None:
        """Internal storage and index directory paths must never leak into client error messages."""
        stream = io.BytesIO(b"Path privacy test data.")
        rec = service.save_upload_stream("path_privacy.txt", stream)

        # 1. Missing storage file
        Path(rec.storage_path).unlink()
        failed_rec = service.process_document(rec.doc_id)
        assert failed_rec.status == DocumentStatus.FAILED
        assert str(service.storage_dir) not in str(failed_rec.error_message)

        # 2. Corrupt FAISS binary error does not leak index_dir
        faiss_dir = service.index_dir / rec.doc_id / "faiss"
        faiss_dir.mkdir(parents=True, exist_ok=True)
        active_embedder = service._embedder or DummyMockEmbedder(16)
        active_dim = active_embedder.embedding_dim
        (faiss_dir / "metadata.json").write_text(
            json.dumps(
                {
                    "format_version": "2.0",
                    "embedding_identity": active_embedder.embedding_identity.to_dict(),
                    "dimension": active_dim,
                    "index_type": "flat",
                    "normalize": True,
                    "chunks": {},
                }
            ),
            encoding="utf-8",
        )
        (faiss_dir / "index.faiss").write_bytes(b"INVALID_FAISS_DATA")
        service.repository.update_document_status(
            rec.doc_id,
            status=DocumentStatus.READY,
            indexing_status=IndexingStatus.INDEXED,
        )
        with pytest.raises(IngestionError) as exc_info:
            service.get_faiss_retriever(rec.doc_id)
        assert str(service.index_dir) not in str(exc_info.value)

    def test_server_controlled_embedding_provenance_lifecycle(
        self, service: DocumentIngestionService
    ) -> None:
        """Embedding provenance is set on success, cleared on reprocessing, and protected
        from injection."""
        # 1. User metadata injection stripped
        stream = io.BytesIO(b"Content for provenance lifecycle testing.")
        rec = service.save_upload_stream(
            "provenance_test.txt",
            stream,
            metadata={"user_tag": "test", "__embedding_provenance__": "malicious_override"},
        )
        assert "__embedding_provenance__" not in rec.metadata
        assert rec.metadata.get("user_tag") == "test"

        # 2. Process document sets server-controlled provenance
        processed = service.process_document(rec.doc_id)
        assert processed.status == DocumentStatus.READY
        assert processed.indexing_status == IndexingStatus.INDEXED
        assert "__embedding_provenance__" in processed.metadata
        prov = processed.metadata["__embedding_provenance__"]
        assert "embedding_identity" in prov
        assert "artifact_digest" in prov
        assert "chunk_manifest_hash" in prov

        # 3. Reprocessing removes previous provenance
        service.repository.update_document_status(
            rec.doc_id, status=DocumentStatus.EXTRACTING, indexing_status=IndexingStatus.PENDING
        )
        service.repository.remove_embedding_provenance(rec.doc_id)
        interim = service.repository.get_document(rec.doc_id)
        assert interim is not None
        assert "__embedding_provenance__" not in interim.metadata

    def test_ingestion_generation_mismatch_and_stale_attempt_safety(
        self, service: DocumentIngestionService
    ) -> None:
        """Stale concurrent ingestion attempts cannot publish provenance or leave valid indexes."""
        from unittest.mock import patch

        from reliability_lab.ingestion.repository import GenerationMismatchError
        from reliability_lab.ingestion.service import DenseIndexIntegrityError

        stream = io.BytesIO(b"Content for generation safety.")
        rec = service.save_upload_stream("gen_safety.txt", stream)

        # 1. Concurrent attempt generation mismatch
        gen1 = service.repository.create_ingestion_generation(rec.doc_id)
        gen2 = service.repository.create_ingestion_generation(rec.doc_id)
        assert gen1 != gen2

        prov_dummy = {"doc_id": rec.doc_id, "data": "test"}
        with pytest.raises(GenerationMismatchError, match="does not match active generation"):
            service.repository.publish_embedding_provenance(rec.doc_id, gen1, prov_dummy)

        # Worker with active generation succeeds
        service.repository.publish_embedding_provenance(rec.doc_id, gen2, prov_dummy)
        doc = service.repository.get_document(rec.doc_id)
        assert doc is not None
        assert doc.metadata["__embedding_provenance__"]["generation"] == gen2

        # 2. Failed provenance persistence cleans up filesystem and marks PARTIAL
        stream2 = io.BytesIO(b"Content for failure cleanup test.")
        rec2 = service.save_upload_stream("fail_cleanup.txt", stream2)

        with patch.object(
            service.repository,
            "publish_embedding_provenance",
            side_effect=RuntimeError("Storage disk failure"),
        ):
            failed_doc = service.process_document(rec2.doc_id)
            assert failed_doc.indexing_status == IndexingStatus.PARTIAL
            faiss_dir = service.index_dir / rec2.doc_id / "faiss"
            assert not faiss_dir.exists()

        # 3. Provenance generation mismatch raises DenseIndexIntegrityError
        service.repository.update_document_status(
            rec.doc_id, status=DocumentStatus.READY, indexing_status=IndexingStatus.INDEXED
        )
        # Modify active generation in doc metadata to simulate out-of-sync generation
        with service.repository._db.get_connection() as conn:
            meta = json.loads(
                conn.execute(
                    "SELECT metadata_json FROM documents WHERE doc_id = ?", (rec.doc_id,)
                ).fetchone()["metadata_json"]
            )
            meta["__active_generation__"] = "newer_generation_xyz"
            conn.execute(
                "UPDATE documents SET metadata_json = ? WHERE doc_id = ?",
                (json.dumps(meta), rec.doc_id),
            )

        faiss_dir2 = service.index_dir / rec.doc_id / "faiss"
        faiss_dir2.mkdir(parents=True, exist_ok=True)
        (faiss_dir2 / "index.faiss").write_bytes(b"dummy_index")
        (faiss_dir2 / "metadata.json").write_text(
            json.dumps({"dimension": 16, "format_version": "2.0", "normalize": True}),
            encoding="utf-8",
        )

        with pytest.raises(DenseIndexIntegrityError, match="generation mismatch"):
            service.get_faiss_retriever(rec.doc_id)

    def test_deterministic_overlapping_ingestion_old_worker_finishes_last(
        self, service: DocumentIngestionService
    ) -> None:
        """Deterministic overlapping ingestion test: old worker finishes last but cannot
        overwrite, downgrade, or destroy newer worker's state or FAISS index."""
        from reliability_lab.ingestion.repository import GenerationMismatchError

        stream = io.BytesIO(b"Document content for concurrent overlapping worker test.")
        rec = service.save_upload_stream("overlapping_test.txt", stream)
        doc_id = rec.doc_id

        # Worker 1 starts and acquires Generation 1
        gen1 = service.repository.create_ingestion_generation(doc_id)
        assert service.repository.get_active_generation(doc_id) == gen1

        # Worker 1 extracts chunks
        worker1_chunks = [
            Chunk(chunk_id=f"{doc_id}:w1:0", doc_id=doc_id, content="Worker 1 chunk content")
        ]

        # Worker 2 supersedes Worker 1 and completes ingestion to READY / INDEXED
        worker2_record = service.process_document(doc_id)
        assert worker2_record.status == DocumentStatus.READY
        assert worker2_record.indexing_status == IndexingStatus.INDEXED
        gen2 = service.repository.get_active_generation(doc_id)
        assert gen2 != gen1

        faiss_dir = service.index_dir / doc_id / "faiss"
        assert faiss_dir.is_dir()
        index_mtime_before = (faiss_dir / "index.faiss").stat().st_mtime

        # Worker 1 resumes (finishes last) and attempts state modifications with gen1

        # 1. Worker 1 cannot save chunks
        with pytest.raises(GenerationMismatchError, match="does not match active generation"):
            service.repository.save_chunks(doc_id, worker1_chunks, generation=gen1)

        # 2. Worker 1 cannot update status to FAILED or any other status
        with pytest.raises(GenerationMismatchError, match="does not match active generation"):
            service.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                indexing_status=IndexingStatus.FAILED,
                error_message="Worker 1 failed",
                generation=gen1,
            )

        # 3. Worker 1 cannot publish embedding provenance
        with pytest.raises(GenerationMismatchError, match="does not match active generation"):
            service.repository.publish_embedding_provenance(
                doc_id,
                generation=gen1,
                provenance={"embedding_identity": {}, "generation": gen1},
            )

        # 4. Worker 1 destructive cleanup is skipped and does NOT delete Worker 2's FAISS index
        cleanup_res = service.cleanup_faiss_index(doc_id, generation=gen1)
        assert cleanup_res is False
        assert faiss_dir.is_dir()
        assert (faiss_dir / "index.faiss").is_file()
        assert (faiss_dir / "index.faiss").stat().st_mtime == index_mtime_before

        # 5. Verify final document state remains completely valid, READY, and INDEXED
        final_doc = service.repository.get_document(doc_id)
        assert final_doc is not None
        assert final_doc.status == DocumentStatus.READY
        assert final_doc.indexing_status == IndexingStatus.INDEXED
        assert final_doc.metadata["__active_generation__"] == gen2
        assert final_doc.metadata["__embedding_provenance__"]["generation"] == gen2

        # 6. Retriever loads Worker 2's index without issue
        retriever = service.get_faiss_retriever(doc_id)
        assert retriever is not None
        assert retriever.size > 0

    def test_tampered_index_faiss_binary_rejected_by_get_faiss_retriever(
        self, service: DocumentIngestionService
    ) -> None:
        """Tampering with index.faiss on disk raises DenseIndexIntegrityError."""
        stream = io.BytesIO(b"Document content for binary tampering verification test.")
        rec = service.save_upload_stream("tamper_binary_test.txt", stream)
        doc = service.process_document(rec.doc_id)
        assert doc.status == DocumentStatus.READY
        assert doc.indexing_status == IndexingStatus.INDEXED

        # Tamper with the index.faiss binary file
        index_file = service.index_dir / rec.doc_id / "faiss" / "index.faiss"
        index_file.write_bytes(index_file.read_bytes() + b"\x00\x00\x00")

        with pytest.raises(DenseIndexIntegrityError, match="artifact digest mismatch"):
            service.get_faiss_retriever(rec.doc_id)

    def test_tampered_chunk_manifest_rejected_by_get_faiss_retriever(
        self, service: DocumentIngestionService
    ) -> None:
        """Tampering with chunk manifest hash in metadata raises DenseIndexIntegrityError."""
        stream = io.BytesIO(b"Document content for chunk manifest tampering test.")
        rec = service.save_upload_stream("tamper_manifest_test.txt", stream)
        doc = service.process_document(rec.doc_id)
        assert doc.status == DocumentStatus.READY
        assert doc.indexing_status == IndexingStatus.INDEXED

        # Tamper with chunk_manifest_hash in metadata.json
        meta_file = service.index_dir / rec.doc_id / "faiss" / "metadata.json"
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        meta["chunk_manifest_hash"] = "sha256:" + "0" * 64
        meta_file.write_text(json.dumps(meta), encoding="utf-8")

        with pytest.raises(DenseIndexIntegrityError, match="Chunk manifest hash mismatch"):
            service.get_faiss_retriever(rec.doc_id)

    def test_missing_storage_path_superseded_worker_does_not_downgrade(
        self, service: DocumentIngestionService
    ) -> None:
        """A worker encountering a missing storage file respects generation ownership."""
        stream = io.BytesIO(b"Valid document content for storage validation test.")
        rec = service.save_upload_stream("missing_storage_test.txt", stream)
        doc_id = rec.doc_id

        # Worker 2 successfully completes document processing
        doc2 = service.process_document(doc_id)
        assert doc2.status == DocumentStatus.READY
        assert doc2.indexing_status == IndexingStatus.INDEXED
        active_gen = service.repository.get_active_generation(doc_id)

        # Stale Worker 1 attempts status update with an old generation
        stale_gen = "gen_stale_12345"
        with pytest.raises(GenerationMismatchError):
            service.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                indexing_status=IndexingStatus.FAILED,
                error_message="Storage file missing: test.txt",
                generation=stale_gen,
            )

        # State remains unchanged
        current = service.repository.get_document(doc_id)
        assert current is not None
        assert current.status == DocumentStatus.READY
        assert current.indexing_status == IndexingStatus.INDEXED
        assert service.repository.get_active_generation(doc_id) == active_gen

    def test_get_faiss_retriever_missing_provenance_fields_fail_closed(
        self, service: DocumentIngestionService
    ) -> None:
        """Missing artifact_digest or chunk_manifest_hash in database provenance fails closed."""
        stream = io.BytesIO(b"Document content for missing provenance field checks.")
        rec = service.save_upload_stream("prov_fields_test.txt", stream)
        service.process_document(rec.doc_id)

        # 1. Pop artifact_digest from trusted provenance
        doc = service.repository.get_document(rec.doc_id)
        assert doc is not None and doc.metadata is not None
        orig_prov = dict(doc.metadata["__embedding_provenance__"])
        orig_digest = orig_prov["artifact_digest"]

        # 1. Pop artifact_digest from trusted provenance
        prov = dict(orig_prov)
        prov.pop("artifact_digest")
        service.repository.set_embedding_provenance(rec.doc_id, prov)

        with pytest.raises(DenseIndexIntegrityError, match="missing required 'artifact_digest'"):
            service.get_faiss_retriever(rec.doc_id)

        # 2. Restore artifact_digest but pop chunk_manifest_hash
        prov["artifact_digest"] = orig_digest
        prov.pop("chunk_manifest_hash", None)
        service.repository.set_embedding_provenance(rec.doc_id, prov)

        with pytest.raises(
            DenseIndexIntegrityError, match="missing required 'chunk_manifest_hash'"
        ):
            service.get_faiss_retriever(rec.doc_id)

    def test_get_faiss_retriever_missing_metadata_fields_fail_closed(
        self, service: DocumentIngestionService
    ) -> None:
        """Missing artifact_digest or chunk_manifest_hash in metadata.json fails closed."""
        stream = io.BytesIO(b"Document content for metadata missing field checks.")
        rec = service.save_upload_stream("meta_fields_test.txt", stream)
        service.process_document(rec.doc_id)

        meta_file = service.index_dir / rec.doc_id / "faiss" / "metadata.json"
        original_meta = json.loads(meta_file.read_text(encoding="utf-8"))

        # 1. Missing artifact_digest
        m1 = dict(original_meta)
        m1.pop("artifact_digest")
        meta_file.write_text(json.dumps(m1), encoding="utf-8")
        with pytest.raises(DenseIndexIntegrityError, match="missing required 'artifact_digest'"):
            service.get_faiss_retriever(rec.doc_id)

        # 2. Missing chunk_manifest_hash
        m2 = dict(original_meta)
        m2.pop("chunk_manifest_hash")
        meta_file.write_text(json.dumps(m2), encoding="utf-8")
        with pytest.raises(
            DenseIndexIntegrityError, match="missing required 'chunk_manifest_hash'"
        ):
            service.get_faiss_retriever(rec.doc_id)

    def test_get_faiss_retriever_repository_chunks_tampered_fails_closed(
        self, service: DocumentIngestionService
    ) -> None:
        """Tampering with chunks in database repository triggers manifest mismatch."""
        stream = io.BytesIO(b"Document content for database repository chunk integrity test.")
        rec = service.save_upload_stream("repo_chunk_tamper.txt", stream)
        service.process_document(rec.doc_id)

        # Tamper directly with chunk content in the SQLite repository
        with service.repository._db.transaction() as conn:
            conn.execute(
                "UPDATE document_chunks SET content = 'TAMPERED_IN_DB' WHERE doc_id = ?",
                (rec.doc_id,),
            )

        with pytest.raises(
            DenseIndexIntegrityError, match="Repository chunk manifest hash mismatch"
        ):
            service.get_faiss_retriever(rec.doc_id)

    def test_save_chunks_and_status_missing_document_fails_closed(
        self, service: DocumentIngestionService
    ) -> None:
        """Generation checks on non-existent documents fail closed with KeyError."""
        with pytest.raises(KeyError, match="document 'nonexistent_doc' not found"):
            service.repository.save_chunks(
                "nonexistent_doc",
                [Chunk(chunk_id="c1", doc_id="nonexistent_doc", content="text")],
                generation="gen_fake",
            )

        with pytest.raises(KeyError, match="document 'nonexistent_doc' not found"):
            service.repository.update_document_status(
                "nonexistent_doc",
                status=DocumentStatus.FAILED,
                generation="gen_fake",
            )

    def test_remove_embedding_provenance_generation_safety(
        self, service: DocumentIngestionService
    ) -> None:
        """Stale worker cannot remove newer worker's embedding provenance."""
        stream = io.BytesIO(b"Document content for provenance removal safety.")
        rec = service.save_upload_stream("prov_remove_test.txt", stream)
        doc = service.process_document(rec.doc_id)
        active_gen = service.repository.get_active_generation(rec.doc_id)
        assert active_gen is not None
        assert doc.metadata is not None
        orig_prov = doc.metadata["__embedding_provenance__"]

        stale_gen = "gen_stale_99999"
        with pytest.raises(GenerationMismatchError, match="does not match active generation"):
            service.repository.remove_embedding_provenance(rec.doc_id, generation=stale_gen)

        current = service.repository.get_document(rec.doc_id)
        assert current is not None and current.metadata is not None
        assert current.metadata["__embedding_provenance__"] == orig_prov

    def test_multithreaded_genuinely_overlapping_workers_race_condition(
        self, tmp_path: Path
    ) -> None:
        """Truly concurrent overlapping workers: Worker 1 is paused while Worker 2 completes;
        Worker 1 finishing last cannot overwrite Worker 2's index, state, or provenance."""
        db = Database(str(tmp_path / "overlap.db"))
        repo = DocumentRepository(db)
        embedder = ControllableEmbedder(dim=16)
        service = DocumentIngestionService(
            repository=repo,
            storage_dir=tmp_path / "documents",
            index_dir=tmp_path / "indexes",
            embedder=embedder,
        )

        stream = io.BytesIO(b"Concurrent document content for multi-threaded testing.")
        rec = service.save_upload_stream("concurrent_workers.txt", stream)
        doc_id = rec.doc_id

        worker1_error: list[Exception] = []

        def run_worker1() -> None:
            try:
                service.process_document(doc_id)
            except Exception as e:
                worker1_error.append(e)

        embedder.enabled = True
        t1 = threading.Thread(target=run_worker1, name="Worker-1-Thread")
        t1.start()

        # Assert Worker 1 started and reached embedding step without lock
        assert embedder.started_event.wait(timeout=5.0) is True

        # Now Worker 2 runs concurrently and completes processing to READY / INDEXED
        # Disable hook so Worker 2 completes without pausing
        embedder.enabled = False
        doc2 = service.process_document(doc_id)
        assert doc2.status == DocumentStatus.READY
        assert doc2.indexing_status == IndexingStatus.INDEXED
        gen2 = service.repository.get_active_generation(doc_id)

        # Now signal Worker 1 to resume and attempt its outdated publish
        embedder.resume_event.set()
        t1.join(timeout=5.0)
        assert not t1.is_alive()
        assert not worker1_error

        # Worker 1 was superseded: verify final state is still Worker 2's state
        final_doc = service.repository.get_document(doc_id)
        assert final_doc is not None and final_doc.metadata is not None
        assert final_doc.status == DocumentStatus.READY
        assert final_doc.indexing_status == IndexingStatus.INDEXED
        assert final_doc.metadata["__active_generation__"] == gen2
        assert final_doc.metadata["__embedding_provenance__"]["generation"] == gen2

        # Verify FAISS retriever loads Worker 2's index cleanly
        retriever = service.get_faiss_retriever(doc_id)
        assert retriever is not None
        assert retriever.size > 0

    def test_cross_instance_concurrency_overlapping_workers(self, tmp_path: Path) -> None:
        """Cross-instance concurrency: two independent DocumentIngestionService instances
        with separate DocumentRepository objects sharing the same SQLite DB file and index dir.
        A stale worker from service1 finishing last cannot overwrite service2's newer generation.
        """
        clear_shared_doc_locks()
        db_path = tmp_path / "cross_instance.db"
        storage_dir = tmp_path / "cross_storage"
        index_dir = tmp_path / "cross_indexes"

        repo1 = DocumentRepository(Database(str(db_path)))
        repo2 = DocumentRepository(Database(str(db_path)))

        embedder1 = ControllableEmbedder(dim=16)
        embedder2 = DummyMockEmbedder(dim=16)

        service1 = DocumentIngestionService(
            repository=repo1,
            storage_dir=storage_dir,
            index_dir=index_dir,
            embedder=embedder1,
        )
        service2 = DocumentIngestionService(
            repository=repo2,
            storage_dir=storage_dir,
            index_dir=index_dir,
            embedder=embedder2,
        )

        assert service1 is not service2
        assert service1.repository is not service2.repository

        stream = io.BytesIO(b"Shared file content for cross-instance overlapping test.")
        rec = service1.save_upload_stream("cross_test.txt", stream)
        doc_id = rec.doc_id

        worker1_error: list[Exception] = []

        def run_worker1() -> None:
            try:
                service1.process_document(doc_id)
            except Exception as e:
                worker1_error.append(e)

        embedder1.enabled = True
        t1 = threading.Thread(target=run_worker1, name="Worker-1-Cross-Instance")
        t1.start()

        # Wait for Worker 1 to reach embedding computation
        assert embedder1.started_event.wait(timeout=5.0) is True

        # Worker 2 processes the document on a completely separate service instance to completion
        doc2 = service2.process_document(doc_id)
        assert doc2.status == DocumentStatus.READY
        assert doc2.indexing_status == IndexingStatus.INDEXED
        gen2 = repo2.get_active_generation(doc_id)
        assert gen2 is not None

        # Signal Worker 1 to resume and attempt to publish its stale generation
        embedder1.resume_event.set()
        t1.join(timeout=5.0)
        assert not t1.is_alive()
        assert not worker1_error

        # Stale worker 1 was superseded: verify final state is still Worker 2's state
        final_doc = repo2.get_document(doc_id)
        assert final_doc is not None and final_doc.metadata is not None
        assert final_doc.status == DocumentStatus.READY
        assert final_doc.indexing_status == IndexingStatus.INDEXED
        assert final_doc.metadata["__active_generation__"] == gen2
        assert final_doc.metadata["__embedding_provenance__"]["generation"] == gen2

        # Both service instances load Worker 2's FAISS index successfully
        retriever2 = service2.get_faiss_retriever(doc_id)
        assert retriever2 is not None and retriever2.size > 0

        retriever1 = service1.get_faiss_retriever(doc_id)
        assert retriever1 is not None and retriever1.size > 0

    def test_cross_instance_independent_concurrency_different_documents(
        self, tmp_path: Path
    ) -> None:
        """Two separate DocumentIngestionService instances can process different documents
        concurrently without blocking or cross-talk."""
        clear_shared_doc_locks()
        db_path = tmp_path / "cross_diff.db"
        storage_dir = tmp_path / "cross_storage"
        index_dir = tmp_path / "cross_indexes"

        repo1 = DocumentRepository(Database(str(db_path)))
        repo2 = DocumentRepository(Database(str(db_path)))

        service1 = DocumentIngestionService(
            repository=repo1,
            storage_dir=storage_dir,
            index_dir=index_dir,
            embedder=DummyMockEmbedder(16),
        )
        service2 = DocumentIngestionService(
            repository=repo2,
            storage_dir=storage_dir,
            index_dir=index_dir,
            embedder=DummyMockEmbedder(16),
        )

        stream_a = io.BytesIO(b"Document A content for independent concurrency.")
        stream_b = io.BytesIO(b"Document B content for independent concurrency.")
        rec_a = service1.save_upload_stream("doc_a.txt", stream_a)
        rec_b = service2.save_upload_stream("doc_b.txt", stream_b)

        results: dict[str, DocumentRecord] = {}
        errors: list[Exception] = []

        def worker_a() -> None:
            try:
                results[rec_a.doc_id] = service1.process_document(rec_a.doc_id)
            except Exception as e:
                errors.append(e)

        def worker_b() -> None:
            try:
                results[rec_b.doc_id] = service2.process_document(rec_b.doc_id)
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=worker_a)
        t_b = threading.Thread(target=worker_b)
        t_a.start()
        t_b.start()
        t_a.join(timeout=10.0)
        t_b.join(timeout=10.0)

        assert not errors
        assert rec_a.doc_id in results and rec_b.doc_id in results
        assert results[rec_a.doc_id].status == DocumentStatus.READY
        assert results[rec_b.doc_id].status == DocumentStatus.READY
        assert results[rec_a.doc_id].indexing_status == IndexingStatus.INDEXED
        assert results[rec_b.doc_id].indexing_status == IndexingStatus.INDEXED

    def test_cross_instance_lock_identity_and_isolation(self, tmp_path: Path) -> None:
        """Verify lock sharing across service instances for the same document and index_dir,
        and isolation across different documents and different index dirs."""
        clear_shared_doc_locks()
        idx_dir_1 = tmp_path / "idx1"
        idx_dir_2 = tmp_path / "idx2"
        db = Database(str(tmp_path / "lock_test.db"))
        repo = DocumentRepository(db)

        svc1 = DocumentIngestionService(repository=repo, index_dir=idx_dir_1)
        svc2 = DocumentIngestionService(repository=repo, index_dir=idx_dir_1)
        svc3 = DocumentIngestionService(repository=repo, index_dir=idx_dir_2)

        # 1. Same index_dir, same doc_id -> identical lock instance
        assert svc1._get_doc_lock("doc_alpha") is svc2._get_doc_lock("doc_alpha")

        # 2. Same index_dir, different doc_id -> distinct lock instances
        assert svc1._get_doc_lock("doc_alpha") is not svc1._get_doc_lock("doc_beta")
        assert svc1._get_doc_lock("doc_alpha") is not svc2._get_doc_lock("doc_beta")

        # 3. Different index_dir, same doc_id -> distinct lock instances
        assert svc1._get_doc_lock("doc_alpha") is not svc3._get_doc_lock("doc_alpha")

    def test_fastapi_dependency_lifecycle_shares_per_document_locks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """FastAPI get_ingestion_service() dependency creates new service instances per request
        that safely share the same per-document lock registry."""
        clear_shared_doc_locks()
        monkeypatch.setenv("RELIABILITY_LAB_STORAGE", str(tmp_path / "storage"))
        monkeypatch.setenv("RELIABILITY_LAB_INDEXES", str(tmp_path / "indexes"))
        monkeypatch.setenv("RELIABILITY_LAB_DB", str(tmp_path / "api_test.db"))

        from reliability_lab.api.main import get_ingestion_service

        svc1 = get_ingestion_service()
        svc2 = get_ingestion_service()

        # Distinct instances created per request injection
        assert svc1 is not svc2
        assert svc1.repository is not svc2.repository

        # But document locks are shared for the same document
        lock1 = svc1._get_doc_lock("doc_lifecycle")
        lock2 = svc2._get_doc_lock("doc_lifecycle")
        assert lock1 is lock2

        # Distinct for different documents
        lock3 = svc1._get_doc_lock("doc_lifecycle_other")
        assert lock1 is not lock3

    def test_multithreaded_different_documents_independent_concurrency(
        self, service: DocumentIngestionService
    ) -> None:
        """Two different documents can be ingested concurrently without blocking each other."""
        stream_a = io.BytesIO(b"Document A content for independent concurrency.")
        stream_b = io.BytesIO(b"Document B content for independent concurrency.")
        rec_a = service.save_upload_stream("doc_a.txt", stream_a)
        rec_b = service.save_upload_stream("doc_b.txt", stream_b)

        results: dict[str, DocumentRecord] = {}
        errors: list[Exception] = []

        def ingest_doc(d_id: str) -> None:
            try:
                results[d_id] = service.process_document(d_id)
            except Exception as e:
                errors.append(e)

        t_a = threading.Thread(target=ingest_doc, args=(rec_a.doc_id,))
        t_b = threading.Thread(target=ingest_doc, args=(rec_b.doc_id,))

        t_a.start()
        t_b.start()
        t_a.join(timeout=15.0)
        t_b.join(timeout=15.0)

        assert not errors
        assert rec_a.doc_id in results and rec_b.doc_id in results
        assert results[rec_a.doc_id].status == DocumentStatus.READY
        assert results[rec_b.doc_id].status == DocumentStatus.READY
        assert results[rec_a.doc_id].indexing_status == IndexingStatus.INDEXED
        assert results[rec_b.doc_id].indexing_status == IndexingStatus.INDEXED
