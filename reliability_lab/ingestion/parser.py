"""PyMuPDF Document Parser with page, line, and bounding span tracking.

This module provides deterministic text and geometric provenance extraction
from PDF documents for citation-aware chunking and auditing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pymupdf

from reliability_lab.core.models import Document


class ParserError(Exception):
    """Base exception for all document parsing errors."""


class FileNotFoundParserError(ParserError):
    """Raised when the target document does not exist."""


class InvalidPDFParserError(ParserError):
    """Raised when the target file is not a valid or readable PDF."""


class EncryptedPDFParserError(ParserError):
    """Raised when an encrypted PDF cannot be accessed or authenticated."""


class PyMuPDFParser:
    """Extracts text and geometric provenance from PDFs using PyMuPDF.

    Separators:
        PAGE_SEPARATOR = "\\n\\n"
        LINE_SEPARATOR = "\\n"

    Offset convention:
        All character offsets are 0-indexed half-open intervals [char_start, char_end).
        Slicing `Document.content[char_start:char_end]` exactly reproduces the recorded
        page, line, or span text.
    """

    PAGE_SEPARATOR: str = "\n\n"
    LINE_SEPARATOR: str = "\n"

    def __init__(
        self, page_separator: str | None = None, line_separator: str | None = None
    ) -> None:
        self.page_separator = page_separator if page_separator is not None else self.PAGE_SEPARATOR
        self.line_separator = line_separator if line_separator is not None else self.LINE_SEPARATOR

    def parse(
        self,
        file_path: str | Path,
        doc_id: str | None = None,
        password: str | None = None,
    ) -> Document:
        """Parse a PDF document into a Document entity with detailed provenance metadata.

        Args:
            file_path: Path to the target PDF file.
            doc_id: Optional explicit document identifier. Defaults to file stem.
            password: Optional decryption password for password-protected PDFs.

        Returns:
            A core Document entity containing the complete assembled text and
            structured geometric metadata for all pages, lines, and spans.

        Raises:
            FileNotFoundParserError: If file_path does not exist or is a directory.
            InvalidPDFParserError: If the file is not a valid PDF or is corrupted.
            EncryptedPDFParserError: If the PDF is encrypted and requires a password,
                or if authentication fails.
        """
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundParserError(f"Target document not found: {path.name}")

        effective_doc_id = doc_id.strip() if doc_id and doc_id.strip() else path.stem
        if not effective_doc_id:
            effective_doc_id = "document"

        try:
            pdf_doc = pymupdf.open(path)
        except Exception as exc:
            raise InvalidPDFParserError(f"Failed to open '{path.name}' as PDF: {exc}") from exc

        try:
            if pdf_doc.is_encrypted:
                if password is None:
                    raise EncryptedPDFParserError(
                        f"PDF '{path.name}' is encrypted and requires a password."
                    )
                auth_result = pdf_doc.authenticate(password)
                if auth_result == 0:
                    raise EncryptedPDFParserError(
                        f"Authentication failed for encrypted PDF '{path.name}'."
                    )

            assembled_content_parts: list[str] = []
            pages_metadata: list[dict[str, Any]] = []
            current_global_offset = 0

            page_count = len(pdf_doc)

            for page_index in range(page_count):
                page = pdf_doc[page_index]
                page_number = page_index + 1
                page_rect = page.rect

                # If subsequent page and previous text exists, account for page separator
                if page_index > 0 and assembled_content_parts:
                    assembled_content_parts.append(self.page_separator)
                    current_global_offset += len(self.page_separator)

                page_start_offset = current_global_offset
                page_lines_metadata: list[dict[str, Any]] = []
                page_text_parts: list[str] = []

                # Extract structured page dictionary containing blocks, lines, and spans
                try:
                    page_dict = page.get_text("dict", flags=pymupdf.TEXT_DEHYPHENATE)
                except Exception as exc:
                    raise InvalidPDFParserError(
                        f"Failed extracting text on page {page_number} of '{path.name}': {exc}"
                    ) from exc

                line_counter = 0

                for block in page_dict.get("blocks", []):
                    # Filter for text blocks (type 0)
                    if block.get("type", 0) != 0:
                        continue

                    for line in block.get("lines", []):
                        raw_spans = line.get("spans", [])
                        if not raw_spans:
                            continue

                        # If not the first line on this page, append line separator
                        if page_lines_metadata:
                            assembled_content_parts.append(self.line_separator)
                            page_text_parts.append(self.line_separator)
                            current_global_offset += len(self.line_separator)

                        line_start_offset = current_global_offset
                        line_counter += 1
                        spans_metadata: list[dict[str, Any]] = []
                        line_text_parts: list[str] = []

                        for span in raw_spans:
                            span_text = span.get("text", "")
                            if not span_text:
                                continue

                            span_start_offset = current_global_offset
                            assembled_content_parts.append(span_text)
                            line_text_parts.append(span_text)
                            page_text_parts.append(span_text)
                            current_global_offset += len(span_text)
                            span_end_offset = current_global_offset

                            span_bbox = [
                                round(coord, 2) for coord in span.get("bbox", (0, 0, 0, 0))
                            ]
                            spans_metadata.append(
                                {
                                    "text": span_text,
                                    "char_start": span_start_offset,
                                    "char_end": span_end_offset,
                                    "bbox": span_bbox,
                                    "font": span.get("font", ""),
                                    "size": round(float(span.get("size", 0.0)), 2),
                                    "flags": int(span.get("flags", 0)),
                                }
                            )

                        line_end_offset = current_global_offset
                        assembled_line_text = "".join(line_text_parts)

                        if spans_metadata:
                            line_bbox = [
                                round(coord, 2) for coord in line.get("bbox", (0, 0, 0, 0))
                            ]
                            page_lines_metadata.append(
                                {
                                    "line_number": line_counter,
                                    "text": assembled_line_text,
                                    "char_start": line_start_offset,
                                    "char_end": line_end_offset,
                                    "bbox": line_bbox,
                                    "spans": spans_metadata,
                                }
                            )

                page_end_offset = current_global_offset
                assembled_page_text = "".join(page_text_parts)

                pages_metadata.append(
                    {
                        "page_index": page_index,
                        "page_number": page_number,
                        "text": assembled_page_text,
                        "char_start": page_start_offset,
                        "char_end": page_end_offset,
                        "width": round(float(page_rect.width), 2),
                        "height": round(float(page_rect.height), 2),
                        "lines": page_lines_metadata,
                    }
                )

            final_content = "".join(assembled_content_parts)

            document_metadata: dict[str, Any] = {
                "source_file": path.name,
                "parser": "PyMuPDFParser",
                "format": "PDF",
                "page_count": page_count,
                "total_characters": len(final_content),
                "has_extractable_text": len(final_content.strip()) > 0,
                "page_separator": self.page_separator,
                "line_separator": self.line_separator,
                "pages": pages_metadata,
            }

            return Document(
                doc_id=effective_doc_id,
                content=final_content,
                metadata=document_metadata,
                page_count=page_count,
            )

        finally:
            pdf_doc.close()
