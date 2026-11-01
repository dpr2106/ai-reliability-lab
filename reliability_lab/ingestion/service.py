"""Document ingestion pipeline orchestrating storage, parsing, chunking, and indexing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from pathlib import Path
from typing import BinaryIO
from uuid import uuid4

from reliability_lab.core.models import Chunk, Document
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.indexing.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from reliability_lab.indexing.faiss_index import FaissVectorRetriever
from reliability_lab.ingestion.chunker import RecursiveCharacterChunker
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.parser import (
    EncryptedPDFParserError,
    InvalidPDFParserError,
    PyMuPDFParser,
)
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.text_parser import InvalidTextDocumentError, TextDocumentParser

logger = logging.getLogger(__name__)

DEFAULT_MAX_UPLOAD_SIZE = 50 * 1024 * 1024  # 50 MB
ALLOWED_EXTENSIONS: set[str] = {".pdf", ".txt", ".md", ".markdown"}


class IngestionError(Exception):
    """Base exception for document ingestion and upload errors."""


class InvalidDocumentFormatError(IngestionError):
    """Raised when an uploaded document has an unsupported or mismatched format."""


class EmptyFileError(IngestionError):
    """Raised when an uploaded file is 0 bytes."""


class UploadSizeLimitExceededError(IngestionError):
    """Raised when an uploaded file exceeds the maximum allowed size limit."""


class DocumentNotFoundError(IngestionError):
    """Raised when a requested document does not exist in the repository."""


class DocumentIngestionService:
    """Service orchestrating file upload validation, extraction, chunking, and indexing."""

    def __init__(
        self,
        repository: DocumentRepository,
        storage_dir: str | Path | None = None,
        index_dir: str | Path | None = None,
        max_upload_size: int = DEFAULT_MAX_UPLOAD_SIZE,
        embedder: BaseEmbedder | None = None,
        chunk_size: int = 500,
        chunk_overlap: int = 100,
    ) -> None:
        self.repository = repository
        self.max_upload_size = max_upload_size
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._embedder = embedder

        # Base directories
        if storage_dir is not None:
            self.storage_dir = Path(storage_dir).resolve()
        else:
            raw_base = os.environ.get("RELIABILITY_LAB_STORAGE", ".reliability-lab/documents")
            self.storage_dir = Path(raw_base).resolve()

        if index_dir is not None:
            self.index_dir = Path(index_dir).resolve()
        else:
            raw_idx = os.environ.get("RELIABILITY_LAB_INDEXES", ".reliability-lab/indexes")
            self.index_dir = Path(raw_idx).resolve()

        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_dir.mkdir(parents=True, exist_ok=True)

    def validate_file_header(self, filename: str, first_bytes: bytes) -> str:
        """Validate file extension and magic byte signatures.

        Returns normalized file_type ('pdf', 'txt', 'md').
        """
        raw_name = Path(filename).name
        ext = Path(raw_name).suffix.lower()

        if ext not in ALLOWED_EXTENSIONS:
            raise InvalidDocumentFormatError(
                f"Unsupported file extension '{ext}'. "
                f"Allowed formats: {', '.join(sorted(ALLOWED_EXTENSIONS))}."
            )

        if ext == ".pdf":
            if not first_bytes.startswith(b"%PDF-"):
                raise InvalidDocumentFormatError(
                    f"File '{raw_name}' has .pdf extension but invalid PDF magic header."
                )
            return "pdf"

        # Plain text or Markdown: must not contain binary null bytes
        if b"\x00" in first_bytes:
            raise InvalidDocumentFormatError(
                f"File '{raw_name}' contains binary null bytes and is not valid text."
            )

        return "md" if ext in {".md", ".markdown"} else "txt"

    def save_upload_stream(
        self,
        filename: str,
        file_stream: BinaryIO,
        doc_id: str | None = None,
    ) -> DocumentRecord:
        """Save an incoming file stream with streaming SHA-256 calculation and bounds check.

        Args:
            filename: Original user filename (will be sanitized).
            file_stream: Readable binary file stream.
            doc_id: Optional explicit document ID.

        Returns:
            DocumentRecord initialized with status=UPLOADED.

        Raises:
            EmptyFileError: If file is 0 bytes.
            UploadSizeLimitExceededError: If file exceeds max_upload_size.
            InvalidDocumentFormatError: If extension or magic bytes are invalid.
        """
        sanitized_filename = Path(filename).name
        ext = Path(sanitized_filename).suffix.lower()
        if not ext:
            raise InvalidDocumentFormatError(f"Filename '{sanitized_filename}' has no extension.")

        effective_id = doc_id.strip() if doc_id and doc_id.strip() else f"doc_{uuid4().hex[:16]}"
        safe_storage_name = f"{effective_id}{ext}"
        storage_path = self.storage_dir / safe_storage_name

        # Ensure safe path containment
        if not storage_path.resolve().is_relative_to(self.storage_dir.resolve()):
            raise InvalidDocumentFormatError("Unsafe filename detected (path traversal attempt).")

        hasher = hashlib.sha256()
        total_size = 0
        first_chunk = True
        file_type = ""

        try:
            with open(storage_path, "wb") as out_f:
                while True:
                    chunk = file_stream.read(65536)
                    if not chunk:
                        break

                    if first_chunk:
                        file_type = self.validate_file_header(sanitized_filename, chunk[:1024])
                        first_chunk = False

                    total_size += len(chunk)
                    if total_size > self.max_upload_size:
                        raise UploadSizeLimitExceededError(
                            f"File size exceeds maximum upload limit "
                            f"({self.max_upload_size} bytes)."
                        )

                    hasher.update(chunk)
                    out_f.write(chunk)

            if total_size == 0:
                raise EmptyFileError(f"Uploaded file '{sanitized_filename}' is empty (0 bytes).")

        except Exception:
            # Delete partial or invalid file
            if storage_path.exists():
                try:
                    storage_path.unlink()
                except OSError:
                    pass
            raise

        content_hash = hasher.hexdigest()
        record = DocumentRecord(
            doc_id=effective_id,
            filename=sanitized_filename,
            storage_path=str(storage_path),
            file_type=file_type,
            file_size=total_size,
            content_hash=content_hash,
            status=DocumentStatus.UPLOADED,
            indexing_status=IndexingStatus.PENDING,
        )

        return self.repository.insert_document(record)

    def process_document(self, doc_id: str) -> DocumentRecord:
        """Execute extraction, citation-aware chunking, and index generation.

        Lifecycle: UPLOADED -> EXTRACTING -> CHUNKING -> INDEXING -> READY (or FAILED).
        """
        record = self.repository.get_document(doc_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found.")

        storage_path = Path(record.storage_path)
        if not storage_path.is_file():
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                error_message=f"Storage file missing: {record.storage_path}",
            )
            return self.repository.get_document(doc_id)  # type: ignore[return-value]

        # 1. EXTRACTING
        self.repository.update_document_status(doc_id, status=DocumentStatus.EXTRACTING)
        try:
            document: Document
            if record.file_type == "pdf":
                parser = PyMuPDFParser()
                document = parser.parse(storage_path, doc_id=doc_id)
            else:
                text_parser = TextDocumentParser()
                document = text_parser.parse(storage_path, doc_id=doc_id)
        except EncryptedPDFParserError as exc:
            logger.warning("PDF %s is encrypted: %s", doc_id, exc)
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                error_message=f"PDF is encrypted or requires password authentication: {exc}",
            )
            return self.repository.get_document(doc_id)  # type: ignore[return-value]
        except (InvalidPDFParserError, InvalidTextDocumentError, Exception) as exc:
            logger.error("Extraction failed for %s: %s", doc_id, exc)
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                error_message=f"Document text extraction failed: {exc}",
            )
            return self.repository.get_document(doc_id)  # type: ignore[return-value]

        page_count = document.page_count or 1

        # 2. CHUNKING
        self.repository.update_document_status(
            doc_id,
            status=DocumentStatus.CHUNKING,
            page_count=page_count,
        )
        try:
            chunker = RecursiveCharacterChunker(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
            )
            chunks = chunker.chunk(document)
        except Exception as exc:
            logger.error("Chunking failed for %s: %s", doc_id, exc)
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                error_message=f"Citation-aware chunking failed: {exc}",
            )
            return self.repository.get_document(doc_id)  # type: ignore[return-value]

        # Persist chunks in repository
        self.repository.save_chunks(doc_id, chunks)

        # 3. INDEXING
        self.repository.update_document_status(
            doc_id,
            status=DocumentStatus.INDEXING,
            chunk_count=len(chunks),
        )

        doc_index_dir = self.index_dir / doc_id
        doc_index_dir.mkdir(parents=True, exist_ok=True)

        # 3a. Persist BM25 Index Chunks
        bm25_chunks_file = doc_index_dir / "bm25_chunks.json"
        try:
            chunks_json = [c.to_dict() for c in chunks]
            bm25_chunks_file.write_text(
                json.dumps(chunks_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        except OSError as exc:
            logger.error("Failed saving BM25 chunks for %s: %s", doc_id, exc)

        # 3b. Dense FAISS Indexing
        faiss_dir = doc_index_dir / "faiss"
        indexing_status = IndexingStatus.INDEXED
        index_err_msg: str | None = None

        if len(chunks) == 0:
            indexing_status = IndexingStatus.SKIPPED
        else:
            embedder = self._embedder
            if embedder is None:
                try:
                    # Attempt local model loading without network downloads
                    embedder = SentenceTransformerEmbedder(local_files_only=True)
                except Exception as exc:
                    logger.info(
                        "Local sentence-transformers embedding model unavailable for %s: %s",
                        doc_id,
                        exc,
                    )
                    indexing_status = IndexingStatus.FAILED
                    index_err_msg = (
                        "Dense FAISS index generation skipped: local sentence-transformers model "
                        f"'all-MiniLM-L6-v2' not available ({exc}). "
                        "BM25 lexical retrieval is available."
                    )
                    # Clean up partial faiss dir if any
                    if faiss_dir.exists():
                        shutil.rmtree(faiss_dir, ignore_errors=True)

            if embedder is not None and indexing_status != IndexingStatus.FAILED:
                try:
                    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")
                    retriever.index(chunks)
                    retriever.save(faiss_dir)
                    indexing_status = IndexingStatus.INDEXED
                except Exception as exc:
                    logger.error("Dense FAISS indexing failed for %s: %s", doc_id, exc)
                    indexing_status = IndexingStatus.FAILED
                    index_err_msg = f"FAISS dense indexing failed: {exc}"
                    if faiss_dir.exists():
                        shutil.rmtree(faiss_dir, ignore_errors=True)

        final_status = (
            DocumentStatus.READY
            if indexing_status in {IndexingStatus.INDEXED, IndexingStatus.SKIPPED}
            else DocumentStatus.FAILED
        )

        self.repository.update_document_status(
            doc_id,
            status=final_status,
            indexing_status=indexing_status,
            error_message=index_err_msg,
            page_count=page_count,
            chunk_count=len(chunks),
        )

        return self.repository.get_document(doc_id)  # type: ignore[return-value]

    def get_bm25_retriever(self, doc_id: str) -> BM25Retriever:
        """Construct and populate a BM25Retriever from persisted document chunks."""
        chunks = self.repository.get_chunks(doc_id)
        if not chunks:
            # Fallback to index json file
            bm25_file = self.index_dir / doc_id / "bm25_chunks.json"
            if bm25_file.is_file():
                raw = json.loads(bm25_file.read_text(encoding="utf-8"))
                chunks = [Chunk.from_dict(item) for item in raw]

        if not chunks:
            record = self.repository.get_document(doc_id)
            if record is None:
                raise DocumentNotFoundError(f"Document '{doc_id}' not found.")
            raise IngestionError(f"No chunks available for document '{doc_id}'.")

        retriever = BM25Retriever()
        retriever.index(chunks)
        return retriever

    def get_faiss_retriever(
        self,
        doc_id: str,
        embedder: BaseEmbedder | None = None,
    ) -> FaissVectorRetriever:
        """Load a persisted FAISS vector retriever from disk."""
        faiss_dir = self.index_dir / doc_id / "faiss"
        if not faiss_dir.is_dir():
            raise FileNotFoundError(f"FAISS index directory not found for document '{doc_id}'.")

        effective_embedder = embedder or self._embedder
        if effective_embedder is None:
            effective_embedder = SentenceTransformerEmbedder(local_files_only=True)

        return FaissVectorRetriever.load(faiss_dir, effective_embedder)

    def delete_document(self, doc_id: str) -> bool:
        """Safely delete original document file, indexes, and database records."""
        record = self.repository.get_document(doc_id)
        if record is None:
            return False

        # 1. Remove original storage file
        storage_path = Path(record.storage_path)
        if storage_path.is_file():
            try:
                storage_path.unlink()
            except OSError as exc:
                logger.warning("Could not delete file %s: %s", storage_path, exc)

        # 2. Remove index directory
        doc_index_dir = self.index_dir / doc_id
        if doc_index_dir.is_dir():
            try:
                shutil.rmtree(doc_index_dir)
            except OSError as exc:
                logger.warning("Could not delete index dir %s: %s", doc_index_dir, exc)

        # 3. Remove database record and cascaded chunks
        return self.repository.delete_document(doc_id)
