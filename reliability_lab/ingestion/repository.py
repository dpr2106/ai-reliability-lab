"""Local SQLite repository for documents, chunk persistence, and ingestion state."""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reliability_lab.core.models import Chunk
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.storage.database import Database


class GenerationMismatchError(ValueError):
    """Raised when active ingestion generation does not match worker generation."""


CREATE_DOCUMENT_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS documents (
    doc_id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    storage_path TEXT NOT NULL,
    file_type TEXT NOT NULL,
    file_size INTEGER NOT NULL,
    content_hash TEXT NOT NULL,
    status TEXT NOT NULL,
    page_count INTEGER,
    chunk_count INTEGER DEFAULT 0,
    indexing_status TEXT NOT NULL,
    error_message TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS document_chunks (
    chunk_id TEXT PRIMARY KEY,
    doc_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    page_number INTEGER,
    char_start INTEGER,
    char_end INTEGER,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (doc_id) REFERENCES documents(doc_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_documents_status ON documents(status);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_created ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_document_chunks_doc_id ON document_chunks(doc_id);
"""


class DocumentRepository:
    """SQLite-backed persistent store for ingested documents and extracted chunks."""

    def __init__(self, database: Database | str | Path = ":memory:") -> None:
        if isinstance(database, Database):
            self._db = database
            self._owns_db = False
        else:
            self._db = Database(database)
            self._owns_db = True
        self._init_tables()

    @property
    def database(self) -> Database:
        """Return the underlying Database instance."""
        return self._db

    def _init_tables(self) -> None:
        """Initialize documents and document_chunks tables idempotently."""
        with self._db.get_connection() as conn:
            conn.executescript(CREATE_DOCUMENT_TABLES_SQL)

    def close(self) -> None:
        """Close the database connection if owned."""
        if self._owns_db:
            self._db.close()

    def __enter__(self) -> DocumentRepository:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def insert_document(self, record: DocumentRecord) -> DocumentRecord:
        """Insert a newly uploaded document record."""
        now = datetime.now(timezone.utc).isoformat()
        status_val = (
            record.status.value if isinstance(record.status, DocumentStatus) else str(record.status)
        )
        idx_val = (
            record.indexing_status.value
            if isinstance(record.indexing_status, IndexingStatus)
            else str(record.indexing_status)
        )
        meta_json = json.dumps(record.metadata or {}, ensure_ascii=False)

        with self._db.get_connection() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    doc_id, filename, storage_path, file_type, file_size, content_hash,
                    status, page_count, chunk_count, indexing_status, error_message,
                    metadata_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.doc_id,
                    record.filename,
                    record.storage_path,
                    record.file_type,
                    record.file_size,
                    record.content_hash,
                    status_val,
                    record.page_count,
                    record.chunk_count,
                    idx_val,
                    record.error_message,
                    meta_json,
                    record.created_at or now,
                    record.updated_at or now,
                ),
            )
        return record

    def get_document(self, doc_id: str) -> DocumentRecord | None:
        """Retrieve a document record by its unique doc_id."""
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if row is None:
                return None
            return DocumentRecord.from_row(row)

    def get_document_by_hash(self, content_hash: str) -> DocumentRecord | None:
        """Retrieve the most recent document record matching a content SHA-256 hash."""
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE content_hash = ? ORDER BY created_at DESC LIMIT 1",
                (content_hash,),
            ).fetchone()
            if row is None:
                return None
            return DocumentRecord.from_row(row)

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
        status: DocumentStatus | str | None = None,
    ) -> list[DocumentRecord]:
        """Return a paginated list of documents with optional status filtering."""
        with self._db.get_connection() as conn:
            if status is not None:
                status_val = status.value if isinstance(status, DocumentStatus) else str(status)
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    WHERE status = ?
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (status_val, limit, offset),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM documents
                    ORDER BY created_at DESC
                    LIMIT ? OFFSET ?
                    """,
                    (limit, offset),
                ).fetchall()
            return [DocumentRecord.from_row(r) for r in rows]

    def count_documents(self, status: DocumentStatus | str | None = None) -> int:
        """Count total documents matching an optional status filter."""
        with self._db.get_connection() as conn:
            if status is not None:
                status_val = status.value if isinstance(status, DocumentStatus) else str(status)
                row = conn.execute(
                    "SELECT COUNT(*) as cnt FROM documents WHERE status = ?",
                    (status_val,),
                ).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) as cnt FROM documents").fetchone()
            return int(row["cnt"]) if row else 0

    def update_document_status(
        self,
        doc_id: str,
        status: DocumentStatus | str,
        error_message: str | None = None,
        page_count: int | None = None,
        chunk_count: int | None = None,
        indexing_status: IndexingStatus | str | None = None,
        metadata: dict[str, Any] | None = None,
        generation: str | None = None,
    ) -> None:
        """Update processing status, counts, and errors for a document."""
        now = datetime.now(timezone.utc).isoformat()
        status_val = status.value if isinstance(status, DocumentStatus) else str(status)

        with self._db.transaction() as conn:
            current = conn.execute(
                "SELECT * FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not current:
                return

            try:
                cur_meta = json.loads(current["metadata_json"]) if current["metadata_json"] else {}
            except Exception:
                cur_meta = {}

            if generation is not None:
                active_gen = cur_meta.get("__active_generation__")
                if active_gen != generation:
                    raise GenerationMismatchError(
                        f"Cannot update status for document '{doc_id}': worker generation "
                        f"'{generation}' does not match active generation '{active_gen}'."
                    )

            eff_page_count = page_count if page_count is not None else current["page_count"]
            eff_chunk_count = chunk_count if chunk_count is not None else current["chunk_count"]

            if indexing_status is not None:
                idx_val = (
                    indexing_status.value
                    if isinstance(indexing_status, IndexingStatus)
                    else str(indexing_status)
                )
            else:
                idx_val = current["indexing_status"]

            if metadata is not None:
                cur_meta.update(metadata)
                meta_json = json.dumps(cur_meta, ensure_ascii=False)
            else:
                meta_json = (
                    json.dumps(cur_meta, ensure_ascii=False)
                    if cur_meta
                    else current["metadata_json"]
                )

            conn.execute(
                """
                UPDATE documents
                SET status = ?,
                    error_message = ?,
                    page_count = ?,
                    chunk_count = ?,
                    indexing_status = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE doc_id = ?
                """,
                (
                    status_val,
                    error_message,
                    eff_page_count,
                    eff_chunk_count,
                    idx_val,
                    meta_json,
                    now,
                    doc_id,
                ),
            )

    def save_chunks(
        self, doc_id: str, chunks: Sequence[Chunk], generation: str | None = None
    ) -> None:
        """Persist extracted chunks in the database for a document."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            if generation is not None:
                current = conn.execute(
                    "SELECT metadata_json FROM documents WHERE doc_id = ?",
                    (doc_id,),
                ).fetchone()
                if current:
                    try:
                        cur_meta = (
                            json.loads(current["metadata_json"]) if current["metadata_json"] else {}
                        )
                    except Exception:
                        cur_meta = {}
                    active_gen = cur_meta.get("__active_generation__")
                    if active_gen != generation:
                        raise GenerationMismatchError(
                            f"Cannot save chunks for document '{doc_id}': worker generation "
                            f"'{generation}' does not match active generation '{active_gen}'."
                        )

            # Delete existing chunks for this doc_id to avoid duplicates
            conn.execute("DELETE FROM document_chunks WHERE doc_id = ?", (doc_id,))
            for idx, chunk in enumerate(chunks):
                meta_json = json.dumps(chunk.metadata or {}, ensure_ascii=False)
                conn.execute(
                    """
                    INSERT INTO document_chunks (
                        chunk_id, doc_id, chunk_index, content, page_number,
                        char_start, char_end, metadata_json, created_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        chunk.chunk_id,
                        chunk.doc_id,
                        idx,
                        chunk.content,
                        chunk.page_number,
                        chunk.char_start,
                        chunk.char_end,
                        meta_json,
                        now,
                    ),
                )
            conn.execute(
                "UPDATE documents SET chunk_count = ?, updated_at = ? WHERE doc_id = ?",
                (len(chunks), now, doc_id),
            )

    def count_chunks(self, doc_id: str) -> int:
        """Return the count of stored chunks for a specific document."""
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM document_chunks WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def list_chunks(
        self,
        doc_id: str,
        limit: int = 50,
        offset: int = 0,
    ) -> list[tuple[int, Chunk]]:
        """Retrieve paginated stored chunks with chunk_index for a document."""
        with self._db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_chunks
                WHERE doc_id = ?
                ORDER BY chunk_index ASC
                LIMIT ? OFFSET ?
                """,
                (doc_id, limit, offset),
            ).fetchall()

            results: list[tuple[int, Chunk]] = []
            for r in rows:
                try:
                    meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                except Exception:
                    meta = {}
                chunk = Chunk(
                    chunk_id=str(r["chunk_id"]),
                    doc_id=str(r["doc_id"]),
                    content=str(r["content"]),
                    page_number=int(r["page_number"]) if r["page_number"] is not None else None,
                    char_start=int(r["char_start"]) if r["char_start"] is not None else None,
                    char_end=int(r["char_end"]) if r["char_end"] is not None else None,
                    metadata=meta,
                )
                results.append((int(r["chunk_index"]), chunk))
            return results

    def get_chunks(self, doc_id: str) -> list[Chunk]:
        """Retrieve all stored chunks for a document, ordered by chunk_index."""
        with self._db.get_connection() as conn:
            rows = conn.execute(
                """
                SELECT * FROM document_chunks
                WHERE doc_id = ?
                ORDER BY chunk_index ASC
                """,
                (doc_id,),
            ).fetchall()

            chunks: list[Chunk] = []
            for r in rows:
                try:
                    meta = json.loads(r["metadata_json"]) if r["metadata_json"] else {}
                except Exception:
                    meta = {}
                chunks.append(
                    Chunk(
                        chunk_id=str(r["chunk_id"]),
                        doc_id=str(r["doc_id"]),
                        content=str(r["content"]),
                        page_number=int(r["page_number"]) if r["page_number"] is not None else None,
                        char_start=int(r["char_start"]) if r["char_start"] is not None else None,
                        char_end=int(r["char_end"]) if r["char_end"] is not None else None,
                        metadata=meta,
                    )
                )
            return chunks

    def delete_document(self, doc_id: str) -> bool:
        """Delete document record and its associated chunks from the database.

        Returns True if a document was deleted, False if not found.
        """
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE doc_id = ?", (doc_id,))
            return cursor.rowcount > 0

    def create_ingestion_generation(self, doc_id: str) -> str:
        """Create a new unique ingestion generation and clear prior provenance in a transaction."""
        generation_id = f"gen_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Document '{doc_id}' not found.")
            try:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                meta = {}
            meta["__active_generation__"] = generation_id
            meta.pop("__embedding_provenance__", None)
            conn.execute(
                """
                UPDATE documents
                SET status = ?,
                    indexing_status = ?,
                    metadata_json = ?,
                    updated_at = ?
                WHERE doc_id = ?
                """,
                (
                    DocumentStatus.EXTRACTING.value,
                    IndexingStatus.PENDING.value,
                    json.dumps(meta, ensure_ascii=False),
                    now,
                    doc_id,
                ),
            )
        return generation_id

    def get_active_generation(self, doc_id: str) -> str | None:
        """Return the current active ingestion generation for the document, if any."""
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                return None
            try:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
                return meta.get("__active_generation__")
            except Exception:
                return None

    def publish_embedding_provenance(
        self, doc_id: str, generation: str, provenance: dict[str, Any]
    ) -> None:
        """Publish server-controlled provenance after verifying active generation matches."""
        now = datetime.now(timezone.utc).isoformat()
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Document '{doc_id}' not found.")
            try:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                meta = {}

            active_gen = meta.get("__active_generation__")
            if active_gen != generation:
                raise GenerationMismatchError(
                    f"Cannot publish embedding provenance for document '{doc_id}': "
                    f"worker generation '{generation}' does not match active generation "
                    f"'{active_gen}'."
                )

            prov_with_gen = dict(provenance)
            prov_with_gen["generation"] = generation
            meta["__embedding_provenance__"] = prov_with_gen
            conn.execute(
                "UPDATE documents SET metadata_json = ?, updated_at = ? WHERE doc_id = ?",
                (
                    json.dumps(meta, ensure_ascii=False),
                    now,
                    doc_id,
                ),
            )

    def set_embedding_provenance(self, doc_id: str, provenance: dict[str, Any]) -> None:
        """Store server-controlled embedding provenance in documents.metadata_json."""
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Document '{doc_id}' not found.")
            try:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                meta = {}
            gen = provenance.get("generation", meta.get("__active_generation__"))
            if gen:
                meta["__active_generation__"] = gen
            meta["__embedding_provenance__"] = provenance
            conn.execute(
                "UPDATE documents SET metadata_json = ?, updated_at = ? WHERE doc_id = ?",
                (
                    json.dumps(meta, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    doc_id,
                ),
            )

    def remove_embedding_provenance(self, doc_id: str) -> None:
        """Explicitly delete the reserved embedding provenance key from metadata."""
        with self._db.transaction() as conn:
            row = conn.execute(
                "SELECT metadata_json FROM documents WHERE doc_id = ?",
                (doc_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"Document '{doc_id}' not found.")
            try:
                meta = json.loads(row["metadata_json"]) if row["metadata_json"] else {}
            except Exception:
                meta = {}
            meta.pop("__embedding_provenance__", None)
            conn.execute(
                "UPDATE documents SET metadata_json = ?, updated_at = ? WHERE doc_id = ?",
                (
                    json.dumps(meta, ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                    doc_id,
                ),
            )
