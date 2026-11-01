"""Domain models and lifecycle states for local document ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class DocumentStatus(str, Enum):
    """Lifecycle state of an ingested document."""

    UPLOADED = "UPLOADED"
    EXTRACTING = "EXTRACTING"
    CHUNKING = "CHUNKING"
    INDEXING = "INDEXING"
    READY = "READY"
    FAILED = "FAILED"


class IndexingStatus(str, Enum):
    """Retrieval index generation state."""

    PENDING = "PENDING"
    INDEXED = "INDEXED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


@dataclass
class DocumentRecord:
    """Persistent database record representing an uploaded document."""

    doc_id: str
    filename: str
    storage_path: str
    file_type: str
    file_size: int
    content_hash: str
    status: DocumentStatus = DocumentStatus.UPLOADED
    page_count: int | None = None
    chunk_count: int = 0
    indexing_status: IndexingStatus = IndexingStatus.PENDING
    error_message: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Serialize document record to dictionary."""
        return {
            "doc_id": self.doc_id,
            "filename": self.filename,
            "storage_path": self.storage_path,
            "file_type": self.file_type,
            "file_size": self.file_size,
            "content_hash": self.content_hash,
            "status": (
                self.status.value if isinstance(self.status, DocumentStatus) else str(self.status)
            ),
            "page_count": self.page_count,
            "chunk_count": self.chunk_count,
            "indexing_status": (
                self.indexing_status.value
                if isinstance(self.indexing_status, IndexingStatus)
                else str(self.indexing_status)
            ),
            "error_message": self.error_message,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_row(cls, row: Any) -> DocumentRecord:
        """Construct a DocumentRecord from an SQLite row or mapping."""
        meta_raw = row["metadata_json"] if "metadata_json" in row.keys() else "{}"
        try:
            metadata = json.loads(meta_raw) if meta_raw else {}
        except Exception:
            metadata = {}

        raw_status = row["status"]
        try:
            status = DocumentStatus(raw_status)
        except ValueError:
            status = DocumentStatus.FAILED

        raw_idx = row["indexing_status"]
        try:
            indexing_status = IndexingStatus(raw_idx)
        except ValueError:
            indexing_status = IndexingStatus.FAILED

        return cls(
            doc_id=str(row["doc_id"]),
            filename=str(row["filename"]),
            storage_path=str(row["storage_path"]),
            file_type=str(row["file_type"]),
            file_size=int(row["file_size"]),
            content_hash=str(row["content_hash"]),
            status=status,
            page_count=int(row["page_count"]) if row["page_count"] is not None else None,
            chunk_count=int(row["chunk_count"]) if row["chunk_count"] is not None else 0,
            indexing_status=indexing_status,
            error_message=str(row["error_message"]) if row["error_message"] is not None else None,
            metadata=metadata,
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
