"""Text and Markdown Document Parser with line and span provenance tracking.

Extracts text content and character-level provenance from plain text (.txt) and
Markdown (.md) files for citation-aware chunking and auditing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from reliability_lab.core.models import Document
from reliability_lab.ingestion.parser import FileNotFoundParserError, ParserError


class InvalidTextDocumentError(ParserError):
    """Raised when a text file is corrupted, unreadable, or contains null bytes."""


class TextDocumentParser:
    """Parser for plain text (.txt) and Markdown (.md) documents.

    Constructs a Document entity with page_count=1 and granular line/offset
    provenance, enabling citation-aware chunkers to preserve source references.
    """

    def __init__(self, encoding: str = "utf-8") -> None:
        self.encoding = encoding

    def parse(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
    ) -> Document:
        """Parse a text or Markdown document into a Document entity.

        Args:
            file_path: Path to the target text or Markdown file.
            doc_id: Optional explicit document identifier. Defaults to file stem.

        Returns:
            A core Document entity with page_count=1 and structured provenance.

        Raises:
            FileNotFoundParserError: If file_path does not exist or is not a file.
            InvalidTextDocumentError: If the file cannot be decoded or contains null bytes.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundParserError(f"Target document not found: {path.name}")

        effective_doc_id = doc_id.strip() if doc_id and doc_id.strip() else path.stem
        if not effective_doc_id:
            effective_doc_id = "document"

        try:
            raw_bytes = path.read_bytes()
        except OSError as exc:
            raise InvalidTextDocumentError(f"Failed to read file '{path.name}': {exc}") from exc

        # Check for binary null bytes
        if b"\x00" in raw_bytes:
            raise InvalidTextDocumentError(
                f"File '{path.name}' contains binary null bytes and is not valid text."
            )

        try:
            content = raw_bytes.decode(self.encoding)
        except UnicodeDecodeError:
            # Fallback to UTF-8 with replacement or latin-1 if configured encoding fails
            try:
                content = raw_bytes.decode("latin-1")
            except Exception as exc:
                raise InvalidTextDocumentError(
                    f"Failed to decode text file '{path.name}': {exc}"
                ) from exc

        # Build page 1 line spans for provenance tracking
        lines_meta: list[dict[str, Any]] = []
        cur_offset = 0
        for line_idx, line in enumerate(content.splitlines(keepends=True), start=1):
            line_len = len(line)
            lines_meta.append(
                {
                    "line_number": line_idx,
                    "char_start": cur_offset,
                    "char_end": cur_offset + line_len,
                    "text": line.rstrip("\r\n"),
                }
            )
            cur_offset += line_len

        file_type = "md" if path.suffix.lower() in {".md", ".markdown"} else "txt"
        metadata: dict[str, Any] = {
            "source_file": path.name,
            "file_type": file_type,
            "encoding": self.encoding,
            "pages": [
                {
                    "page_number": 1,
                    "char_start": 0,
                    "char_end": len(content),
                    "lines": lines_meta,
                }
            ],
        }

        return Document(
            doc_id=effective_doc_id,
            content=content,
            metadata=metadata,
            page_count=1,
        )
