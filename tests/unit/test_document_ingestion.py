"""Unit tests for Document Ingestion, Text Parsing, Repository, and Indexing Service."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pymupdf
import pytest

from reliability_lab.core.models import Chunk
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.parser import FileNotFoundParserError
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import (
    DocumentDeletionError,
    DocumentIngestionService,
    EmptyFileError,
    IngestionError,
    InvalidDocumentFormatError,
    UploadSizeLimitExceededError,
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
            import shutil

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
