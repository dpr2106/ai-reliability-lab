"""Document ingestion pipeline orchestrating storage, parsing, chunking, and indexing."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO
from uuid import uuid4

from reliability_lab.core.models import Chunk, Document
from reliability_lab.core.privacy import sanitize_path_message
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.indexing.embeddings import BaseEmbedder, SentenceTransformerEmbedder
from reliability_lab.indexing.faiss_index import (
    FaissVectorRetriever,
    IncompatibleEmbeddingIdentityError,
    LegacyIndexIncompatibilityError,
    UnsupportedIndexFormatError,
    compute_chunk_manifest_hash,
)
from reliability_lab.ingestion.chunker import RecursiveCharacterChunker
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.parser import (
    EncryptedPDFParserError,
    InvalidPDFParserError,
    PyMuPDFParser,
)
from reliability_lab.ingestion.repository import DocumentRepository, GenerationMismatchError
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


class DocumentDeletionError(IngestionError):
    """Raised when deletion of document files or indexes fails on the filesystem."""


class DenseIndexIntegrityError(IngestionError):
    """Raised when dense index metadata, files, or provenance are corrupt, tampered,
    or mismatched."""


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
        metadata: dict[str, Any] | None = None,
    ) -> DocumentRecord:
        """Save an incoming file stream with streaming SHA-256 calculation and bounds check.

        Args:
            filename: Original user filename (will be sanitized).
            file_stream: Readable binary file stream.
            doc_id: Optional explicit document ID.
            metadata: Optional user metadata (server-reserved '__*' keys will be stripped).

        Returns:
            DocumentRecord initialized with status=UPLOADED.

        Raises:
            EmptyFileError: If file is 0 bytes.
            UploadSizeLimitExceededError: If file exceeds max_upload_size.
            InvalidDocumentFormatError: If extension or magic bytes are invalid.
        """
        raw_name = Path(filename).name.strip().replace("\x00", "")
        if not raw_name or raw_name in {".", ".."}:
            sanitized_filename = "document.bin"
        else:
            sanitized_filename = raw_name

        ext = Path(sanitized_filename).suffix.lower()
        if not ext:
            raise InvalidDocumentFormatError(f"Filename '{sanitized_filename}' has no extension.")

        effective_id = doc_id.strip() if doc_id and doc_id.strip() else f"doc_{uuid4().hex[:16]}"
        safe_storage_name = f"{effective_id}{ext}"
        storage_path = (self.storage_dir / safe_storage_name).resolve()

        # Ensure safe path containment
        if not storage_path.is_relative_to(self.storage_dir.resolve()):
            raise InvalidDocumentFormatError("Unsafe filename detected (path traversal attempt).")

        # Strip user-supplied reserved __* keys
        clean_metadata = {k: v for k, v in (metadata or {}).items() if not k.startswith("__")}

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

                    # Security check: if text or md, check entire stream for binary null bytes
                    if file_type in {"txt", "md"} and b"\x00" in chunk:
                        raise InvalidDocumentFormatError(
                            f"File '{sanitized_filename}' contains binary null bytes."
                        )

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
            metadata=clean_metadata,
        )

        return self.repository.insert_document(record)

    def cleanup_faiss_index(self, doc_id: str, generation: str | None = None) -> bool:
        """Safely and retryably remove persisted FAISS index directory if present.

        Args:
            doc_id: Unique document identifier.
            generation: Optional worker generation. If provided and not matching
                the document's active generation, cleanup is skipped to prevent
                destructive deletion of a newer worker's index.

        Returns:
            True if the directory does not exist or was successfully removed.
            False if removal failed due to an OS/filesystem error or generation mismatch.

        Guarantees:
            - Safe error logging without raising, preventing cleanup errors from
              masking underlying extraction/chunking/indexing exceptions.
            - Calling this method is idempotent and retryable.
        """
        if generation is not None:
            active_gen = self.repository.get_active_generation(doc_id)
            if active_gen is not None and active_gen != generation:
                logger.info(
                    "Skipping FAISS cleanup for document '%s': worker generation '%s' "
                    "does not match active generation '%s'.",
                    doc_id,
                    generation,
                    active_gen,
                )
                return False

        faiss_dir = self.index_dir / doc_id / "faiss"
        if not faiss_dir.exists():
            return True

        try:
            shutil.rmtree(faiss_dir)
            logger.info("Cleaned up FAISS index directory for document '%s'", doc_id)
            return True
        except Exception as exc:
            logger.error(
                "Failed to remove FAISS index directory for document '%s': %s",
                doc_id,
                exc,
            )
            return False

    def _cleanup_faiss_dir(self, doc_id: str, generation: str | None = None) -> bool:
        """Backward-compatible alias for cleanup_faiss_index."""
        try:
            return self.cleanup_faiss_index(doc_id, generation=generation)
        except TypeError:
            return self.cleanup_faiss_index(doc_id)

    def process_document(self, doc_id: str) -> DocumentRecord:
        """Execute extraction, citation-aware chunking, and index generation.

        Lifecycle: UPLOADED -> EXTRACTING -> CHUNKING -> INDEXING -> READY (or FAILED).
        Guarantees that unhandled exceptions transition the document to FAILED
        with actionable diagnostics rather than leaving it indefinitely stuck.
        """
        record = self.repository.get_document(doc_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found.")

        storage_path = Path(record.storage_path)
        if not storage_path.is_file():
            # Invalidate state in SQLite FIRST before filesystem cleanup
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.FAILED,
                indexing_status=IndexingStatus.FAILED,
                error_message=f"Storage file missing: {storage_path.name}",
            )
            self._cleanup_faiss_dir(doc_id)
            return self.repository.get_document(doc_id)  # type: ignore[return-value]

        try:
            # 1. EXTRACTING & Ingestion Generation Initialization
            # Invalidate prior provenance and create new generation in SQLite FIRST
            generation = self.repository.create_ingestion_generation(doc_id)
            self._cleanup_faiss_dir(doc_id, generation=generation)
            doc_index_dir = self.index_dir / doc_id
            bm25_file = doc_index_dir / "bm25_chunks.json"
            if bm25_file.exists():
                bm25_file.unlink(missing_ok=True)

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
                try:
                    self.repository.update_document_status(
                        doc_id,
                        status=DocumentStatus.FAILED,
                        indexing_status=IndexingStatus.FAILED,
                        error_message=(
                            "PDF is encrypted or requires password authentication: "
                            f"{sanitize_path_message(str(exc))}"
                        ),
                        generation=generation,
                    )
                    self._cleanup_faiss_dir(doc_id, generation=generation)
                except GenerationMismatchError:
                    logger.warning("Worker superseded during encrypted PDF handling for %s", doc_id)
                return self.repository.get_document(doc_id)  # type: ignore[return-value]
            except (InvalidPDFParserError, InvalidTextDocumentError, Exception) as exc:
                logger.error("Extraction failed for %s: %s", doc_id, exc, exc_info=True)
                try:
                    self.repository.update_document_status(
                        doc_id,
                        status=DocumentStatus.FAILED,
                        indexing_status=IndexingStatus.FAILED,
                        error_message=(
                            f"Document text extraction failed: {sanitize_path_message(str(exc))}"
                        ),
                        generation=generation,
                    )
                    self._cleanup_faiss_dir(doc_id, generation=generation)
                except GenerationMismatchError:
                    logger.warning(
                        "Worker superseded during extraction failure handling for %s", doc_id
                    )
                return self.repository.get_document(doc_id)  # type: ignore[return-value]

            page_count = document.page_count or 1

            # 2. CHUNKING
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.CHUNKING,
                page_count=page_count,
                generation=generation,
            )
            try:
                chunker = RecursiveCharacterChunker(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunks = chunker.chunk(document)
            except Exception as exc:
                logger.error("Chunking failed for %s: %s", doc_id, exc, exc_info=True)
                try:
                    self.repository.update_document_status(
                        doc_id,
                        status=DocumentStatus.FAILED,
                        indexing_status=IndexingStatus.FAILED,
                        error_message=(
                            f"Citation-aware chunking failed: {sanitize_path_message(str(exc))}"
                        ),
                        generation=generation,
                    )
                    self._cleanup_faiss_dir(doc_id, generation=generation)
                except GenerationMismatchError:
                    logger.warning(
                        "Worker superseded during chunking failure handling for %s", doc_id
                    )
                return self.repository.get_document(doc_id)  # type: ignore[return-value]

            # Persist chunks in repository (safely replaces prior chunks to avoid duplicates)
            self.repository.save_chunks(doc_id, chunks, generation=generation)

            # 3. INDEXING
            self.repository.update_document_status(
                doc_id,
                status=DocumentStatus.INDEXING,
                chunk_count=len(chunks),
                generation=generation,
            )

            doc_index_dir = self.index_dir / doc_id
            doc_index_dir.mkdir(parents=True, exist_ok=True)

            # 3a. Persist BM25 Index Chunks atomically
            bm25_chunks_file = doc_index_dir / "bm25_chunks.json"
            bm25_tmp_file = doc_index_dir / "bm25_chunks.json.tmp"
            bm25_indexing_ok = False
            try:
                chunks_json = [c.to_dict() for c in chunks]
                bm25_tmp_file.write_text(
                    json.dumps(chunks_json, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                bm25_tmp_file.replace(bm25_chunks_file)
                bm25_indexing_ok = True
            except OSError as exc:
                logger.error("Failed saving BM25 chunks for %s: %s", doc_id, exc)
                if bm25_tmp_file.exists():
                    bm25_tmp_file.unlink(missing_ok=True)

            # 3b. Dense FAISS Indexing
            faiss_dir = doc_index_dir / "faiss"
            indexing_status: IndexingStatus
            index_err_msg: str | None = None
            final_status: DocumentStatus

            if len(chunks) == 0:
                self._cleanup_faiss_dir(doc_id, generation=generation)
                indexing_status = IndexingStatus.SKIPPED
                final_status = DocumentStatus.READY
            elif not bm25_indexing_ok:
                self._cleanup_faiss_dir(doc_id, generation=generation)
                indexing_status = IndexingStatus.FAILED
                index_err_msg = "BM25 index generation failed."
                final_status = DocumentStatus.FAILED
            else:
                # BM25 succeeded. Now try dense vector indexing
                embedder = self._embedder
                faiss_available = False
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
                        index_err_msg = (
                            "Dense FAISS index generation skipped: local model "
                            f"'all-MiniLM-L6-v2' not available ({exc}). "
                            "BM25 lexical retrieval is available."
                        )
                        self._cleanup_faiss_dir(doc_id, generation=generation)

                if embedder is not None:
                    try:
                        retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")
                        retriever.index(chunks)
                        # Atomic FAISS save via temporary directory
                        faiss_tmp = doc_index_dir / "faiss_tmp"
                        if faiss_tmp.exists():
                            shutil.rmtree(faiss_tmp, ignore_errors=True)
                        retriever.save(faiss_tmp)

                        # Check generation BEFORE publishing index directory
                        active_gen = self.repository.get_active_generation(doc_id)
                        if active_gen != generation:
                            if faiss_tmp.exists():
                                shutil.rmtree(faiss_tmp, ignore_errors=True)
                            raise GenerationMismatchError(
                                f"Worker generation '{generation}' does not match active "
                                f"generation '{active_gen}' before FAISS directory publish."
                            )

                        if faiss_dir.exists():
                            shutil.rmtree(faiss_dir, ignore_errors=True)
                        faiss_tmp.rename(faiss_dir)

                        # Record server-controlled provenance only after atomic rename succeeds
                        try:
                            index_file = faiss_dir / "index.faiss"
                            artifact_digest = (
                                f"sha256:{hashlib.sha256(index_file.read_bytes()).hexdigest()}"
                            )
                            chunk_manifest_hash = compute_chunk_manifest_hash(chunks)
                            provenance = {
                                "embedding_identity": retriever.embedding_identity.to_dict(),
                                "artifact_digest": artifact_digest,
                                "chunk_manifest_hash": chunk_manifest_hash,
                                "doc_id": doc_id,
                                "document_content_hash": record.content_hash,
                                "ingestion_timestamp": datetime.now(timezone.utc).isoformat(),
                                "generation": generation,
                            }
                            self.repository.publish_embedding_provenance(
                                doc_id, generation, provenance
                            )
                            faiss_available = True
                        except GenerationMismatchError:
                            raise
                        except Exception as p_exc:
                            logger.error(
                                "Failed recording embedding provenance for %s: %s", doc_id, p_exc
                            )
                            self._cleanup_faiss_dir(doc_id, generation=generation)
                            try:
                                self.repository.remove_embedding_provenance(doc_id)
                            except Exception:
                                pass
                            index_err_msg = (
                                f"FAISS dense indexing failed: failed to record provenance"
                                f" ({p_exc})"
                            )
                            faiss_available = False
                    except GenerationMismatchError:
                        raise
                    except Exception as exc:
                        logger.error("Dense FAISS indexing failed for %s: %s", doc_id, exc)
                        index_err_msg = f"FAISS dense indexing failed: {exc}"
                        self._cleanup_faiss_dir(doc_id, generation=generation)
                        try:
                            self.repository.remove_embedding_provenance(doc_id)
                        except Exception:
                            pass

                if faiss_available:
                    indexing_status = IndexingStatus.INDEXED
                    final_status = DocumentStatus.READY
                else:
                    # BM25 is available, but FAISS is not available
                    self._cleanup_faiss_dir(doc_id, generation=generation)
                    indexing_status = IndexingStatus.PARTIAL
                    final_status = DocumentStatus.READY

            self.repository.update_document_status(
                doc_id,
                status=final_status,
                indexing_status=indexing_status,
                error_message=index_err_msg,
                page_count=page_count,
                chunk_count=len(chunks),
                generation=generation,
            )

            return self.repository.get_document(doc_id)  # type: ignore[return-value]

        except GenerationMismatchError as g_err:
            logger.warning(
                "Ingestion worker superseded for %s (generation %s): %s",
                doc_id,
                generation,
                g_err,
            )
            if "faiss_tmp" in locals() and faiss_tmp.exists():
                shutil.rmtree(faiss_tmp, ignore_errors=True)
            return self.repository.get_document(doc_id)  # type: ignore[return-value]
        except Exception as unhandled_exc:
            logger.exception("Unexpected error processing document %s: %s", doc_id, unhandled_exc)
            try:
                self.repository.update_document_status(
                    doc_id,
                    status=DocumentStatus.FAILED,
                    indexing_status=IndexingStatus.FAILED,
                    error_message=(
                        f"Unexpected processing error: {sanitize_path_message(str(unhandled_exc))}"
                    ),
                    generation=generation,
                )
                self._cleanup_faiss_dir(doc_id, generation=generation)
            except GenerationMismatchError:
                logger.warning(
                    "Worker superseded during unhandled exception recovery for %s", doc_id
                )
            return self.repository.get_document(doc_id)  # type: ignore[return-value]

    @classmethod
    def validate_document_freshness(
        cls,
        record: DocumentRecord,
        resource_name: str = "Chunks",
    ) -> None:
        """Validate that a document is in a ready, freshly indexed state for retrieval.

        Permits retrieval only when status == READY and indexing_status in (INDEXED, PARTIAL).
        Rejects access when UPLOADED, EXTRACTING, CHUNKING, INDEXING, FAILED, or READY with
        PENDING, FAILED, or SKIPPED indexing status.

        Raises:
            IngestionError: If the document is not in a valid, fresh retrieval state.
        """
        if record.status != DocumentStatus.READY or record.indexing_status not in (
            IndexingStatus.INDEXED,
            IndexingStatus.PARTIAL,
        ):
            if record.status == DocumentStatus.FAILED:
                err_detail = f": {record.error_message}" if record.error_message else ""
                raise IngestionError(
                    f"{resource_name} not available for document '{record.doc_id}' "
                    f"(document status is FAILED{err_detail})."
                )
            if record.status in (
                DocumentStatus.EXTRACTING,
                DocumentStatus.CHUNKING,
                DocumentStatus.INDEXING,
            ):
                status_val = (
                    record.status.value if hasattr(record.status, "value") else str(record.status)
                )
                raise IngestionError(
                    f"{resource_name} not ready for document '{record.doc_id}' "
                    f"(document status is {status_val}: processing in progress)."
                )
            if record.status == DocumentStatus.UPLOADED:
                raise IngestionError(
                    f"{resource_name} not ready for document '{record.doc_id}' "
                    "(document status is UPLOADED: processing has not started)."
                )

            # Document is READY, but indexing status is not INDEXED or PARTIAL
            if record.indexing_status == IndexingStatus.PENDING:
                raise IngestionError(
                    f"{resource_name} not ready for document '{record.doc_id}' "
                    "(indexing status is PENDING: indexing in progress)."
                )
            if record.indexing_status == IndexingStatus.FAILED:
                err_detail = f": {record.error_message}" if record.error_message else ""
                raise IngestionError(
                    f"{resource_name} failed for document '{record.doc_id}' "
                    f"(indexing status is FAILED{err_detail})."
                )
            if record.indexing_status == IndexingStatus.SKIPPED:
                raise IngestionError(
                    f"{resource_name} skipped for document '{record.doc_id}' "
                    "(document has 0 chunks)."
                )
            raise IngestionError(
                f"{resource_name} not available for document '{record.doc_id}' "
                f"(status: {record.status}, indexing status: {record.indexing_status})."
            )

    def get_bm25_retriever(self, doc_id: str) -> BM25Retriever:
        """Construct and populate a BM25Retriever from persisted document chunks.

        Guarantees:
        1. Validates document existence in the repository.
        2. Strictly verifies document lifecycle freshness: BM25 retrieval is permitted
           ONLY when status == READY AND indexing_status in (INDEXED, PARTIAL).
        3. Rejects access during active processing (EXTRACTING, CHUNKING, INDEXING, UPLOADED)
           and after processing failures (FAILED, SKIPPED).
        4. Rejects stale previous-version chunks after a failed reprocessing attempt.
        """
        record = self.repository.get_document(doc_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found.")

        self.validate_document_freshness(record, resource_name="BM25 index")

        chunks = self.repository.get_chunks(doc_id)
        if not chunks:
            # Fallback to index json file
            bm25_file = self.index_dir / doc_id / "bm25_chunks.json"
            if bm25_file.is_file():
                try:
                    raw = json.loads(bm25_file.read_text(encoding="utf-8"))
                    chunks = [Chunk.from_dict(item) for item in raw]
                except Exception as exc:
                    logger.error("Failed to read BM25 chunks JSON for %s: %s", doc_id, exc)

        if not chunks:
            raise IngestionError(f"No chunks available for document '{doc_id}'.")

        retriever = BM25Retriever()
        retriever.index(chunks)
        return retriever

    def get_faiss_retriever(
        self,
        doc_id: str,
        embedder: BaseEmbedder | None = None,
    ) -> FaissVectorRetriever:
        """Load a persisted FAISS vector retriever from disk.

        Guarantees:
        1. Validates document existence in the repository.
        2. Strictly verifies that document status is READY AND indexing_status is INDEXED.
           Rejects access for non-ready document states (EXTRACTING, CHUNKING, INDEXING,
           UPLOADED, FAILED) and non-indexed states (PARTIAL, PENDING, FAILED, SKIPPED)
           with actionable diagnostics indicating whether BM25 lexical retrieval remains available.
        3. Protects against stale indexes: even if a legacy faiss directory exists on disk,
           it will not be loaded if the database record is not in READY and INDEXED states.
        4. Validates presence and integrity of index.faiss and metadata.json before loading.
        5. Validates positive integer embedding dimension and compatibility with embedder.
        """
        record = self.repository.get_document(doc_id)
        if record is None:
            raise DocumentNotFoundError(f"Document '{doc_id}' not found.")

        # Guard: FAISS retrieval is permitted ONLY when status == READY
        # AND indexing_status == INDEXED
        if (
            record.status != DocumentStatus.READY
            or record.indexing_status != IndexingStatus.INDEXED
        ):
            # 1. Document-level lifecycle checks
            if record.status == DocumentStatus.FAILED:
                err_detail = f": {record.error_message}" if record.error_message else ""
                raise IngestionError(
                    f"FAISS dense index not available for document '{doc_id}' "
                    f"(document status is FAILED{err_detail})."
                )
            if record.status in (
                DocumentStatus.EXTRACTING,
                DocumentStatus.CHUNKING,
                DocumentStatus.INDEXING,
            ):
                raise IngestionError(
                    f"FAISS dense index not ready for document '{doc_id}' "
                    f"(document status is {record.status}: processing in progress)."
                )
            if record.status == DocumentStatus.UPLOADED:
                raise IngestionError(
                    f"FAISS dense index not ready for document '{doc_id}' "
                    "(document status is UPLOADED: processing has not started)."
                )

            # 2. Indexing-level checks when document status is READY
            if record.indexing_status == IndexingStatus.PARTIAL:
                raise IngestionError(
                    f"FAISS dense index not available for document '{doc_id}' "
                    "(indexing status is PARTIAL: BM25 lexical retrieval is available)."
                )
            if record.indexing_status == IndexingStatus.PENDING:
                raise IngestionError(
                    f"FAISS dense index not ready for document '{doc_id}' "
                    "(indexing status is PENDING: indexing in progress)."
                )
            if record.indexing_status == IndexingStatus.FAILED:
                err_detail = f": {record.error_message}" if record.error_message else ""
                raise IngestionError(
                    f"FAISS dense index failed for document '{doc_id}' "
                    f"(indexing status is FAILED{err_detail})."
                )
            if record.indexing_status == IndexingStatus.SKIPPED:
                raise IngestionError(
                    f"FAISS dense index skipped for document '{doc_id}' (document has 0 chunks)."
                )
            raise IngestionError(
                f"FAISS dense index not available for document '{doc_id}' "
                f"(status: {record.status}, indexing status: {record.indexing_status})."
            )

        faiss_dir = self.index_dir / doc_id / "faiss"
        if not faiss_dir.is_dir():
            raise FileNotFoundError(
                f"FAISS index directory missing for indexed document '{doc_id}'."
            )

        index_file = faiss_dir / "index.faiss"
        meta_file = faiss_dir / "metadata.json"

        if not index_file.is_file() or not meta_file.is_file():
            raise DenseIndexIntegrityError(
                f"Corrupt FAISS index directory for document '{doc_id}': "
                "required index files ('index.faiss', 'metadata.json') are missing."
            )

        try:
            meta_content = json.loads(meta_file.read_text(encoding="utf-8"))
            if not isinstance(meta_content, dict):
                raise DenseIndexIntegrityError(
                    f"Corrupt FAISS index metadata for document '{doc_id}': "
                    "root JSON structure must be an object."
                )
            if "dimension" not in meta_content:
                raise DenseIndexIntegrityError(
                    f"Corrupt FAISS index metadata for document '{doc_id}': "
                    "missing required 'dimension' key."
                )
            dim_val = meta_content["dimension"]
            if isinstance(dim_val, bool) or not isinstance(dim_val, int) or dim_val <= 0:
                raise DenseIndexIntegrityError(
                    f"Corrupt FAISS index metadata for document '{doc_id}': "
                    f"'dimension' must be a positive integer, got {dim_val!r}."
                )
            expected_dimension = dim_val
        except json.JSONDecodeError as exc:
            raise DenseIndexIntegrityError(
                f"Corrupt FAISS index metadata for document '{doc_id}': "
                f"failed to parse 'metadata.json' ({exc})."
            ) from exc

        # Verify trusted provenance generation matches active generation
        trusted_provenance = (
            record.metadata.get("__embedding_provenance__") if record.metadata else None
        )
        active_gen = record.metadata.get("__active_generation__") if record.metadata else None
        if trusted_provenance is not None and active_gen is not None:
            prov_gen = trusted_provenance.get("generation")
            if prov_gen is not None and prov_gen != active_gen:
                raise DenseIndexIntegrityError(
                    f"Embedding provenance generation mismatch for document '{doc_id}': "
                    f"provenance has '{prov_gen}', active generation is '{active_gen}'."
                )

        effective_embedder = embedder or self._embedder
        if effective_embedder is None:
            effective_embedder = SentenceTransformerEmbedder(local_files_only=True)

        if effective_embedder.embedding_dim != expected_dimension:
            raise IngestionError(
                f"Incompatible embedding dimension for document '{doc_id}': "
                f"saved index dimension ({expected_dimension}) does not match "
                f"embedder dimension ({effective_embedder.embedding_dim})."
            )

        format_version = str(meta_content.get("format_version", "1.0"))
        try:
            if format_version == "1.0":
                if not trusted_provenance:
                    raise LegacyIndexIncompatibilityError(
                        f"Legacy format 1.0 index for document '{doc_id}' lacks "
                        "trusted server provenance; reindexing is required."
                    )
                return FaissVectorRetriever.load_legacy_verified(
                    faiss_dir, effective_embedder, trusted_provenance=trusted_provenance
                )
            elif format_version == "2.0":
                return FaissVectorRetriever.load(faiss_dir, effective_embedder)
            else:
                raise UnsupportedIndexFormatError(
                    f"Unsupported index format version '{format_version}' for document '{doc_id}'."
                )
        except (
            IncompatibleEmbeddingIdentityError,
            LegacyIndexIncompatibilityError,
            UnsupportedIndexFormatError,
            DenseIndexIntegrityError,
            IngestionError,
        ):
            raise
        except (ValueError, FileNotFoundError, RuntimeError, Exception) as exc:
            logger.error("Failed loading FAISS index for document '%s': %s", doc_id, exc)
            raise IngestionError(
                f"Invalid or corrupt FAISS index for document '{doc_id}': {exc}"
            ) from exc

    def delete_document(self, doc_id: str) -> bool:
        """Safely delete original document file, indexes, and database records.

        Guarantees that database records are only deleted when filesystem cleanup
        has succeeded or target files/directories were already absent (idempotent).
        If filesystem cleanup fails, preserves the database record and raises
        DocumentDeletionError to allow subsequent safe retries.

        Returns:
            True if document existed and was deleted, False if document was not found.

        Raises:
            DocumentDeletionError: If filesystem cleanup fails.
        """
        record = self.repository.get_document(doc_id)
        if record is None:
            return False

        # 1. Remove original storage file if it exists
        storage_path = Path(record.storage_path)
        if storage_path.exists():
            try:
                storage_path.unlink()
            except OSError as exc:
                logger.error(
                    "Could not delete storage file %s for %s: %s", storage_path, doc_id, exc
                )
                raise DocumentDeletionError(
                    f"Failed to delete file '{storage_path.name}' for doc '{doc_id}': {exc}"
                ) from exc

        # 2. Remove index directory if it exists
        doc_index_dir = self.index_dir / doc_id
        if doc_index_dir.exists():
            try:
                shutil.rmtree(doc_index_dir)
            except OSError as exc:
                logger.error(
                    "Could not delete index directory %s for %s: %s", doc_index_dir, doc_id, exc
                )
                raise DocumentDeletionError(
                    f"Failed to delete index directory for document '{doc_id}': {exc}"
                ) from exc

        # 3. Only when filesystem cleanup has succeeded, remove database record and cascaded chunks
        return self.repository.delete_document(doc_id)
