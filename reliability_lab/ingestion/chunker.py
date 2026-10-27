"""Citation-aware document chunkers for AI Reliability Lab.

This module implements deterministic sliding-window and recursive character
chunking strategies that preserve character-level provenance, bounding boxes,
and page associations for citation auditing.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from reliability_lab.core.interfaces import BaseChunker
from reliability_lab.core.models import Chunk, Document

DEFAULT_SEPARATORS: tuple[str, ...] = ("\n\n", "\n", ". ", " ", "")


def _validate_chunk_params(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk size and overlap parameters."""
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool):
        raise TypeError("chunk_size must be an integer.")
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be a positive integer, got {chunk_size}.")

    if not isinstance(chunk_overlap, int) or isinstance(chunk_overlap, bool):
        raise TypeError("chunk_overlap must be an integer.")
    if chunk_overlap < 0:
        raise ValueError(f"chunk_overlap must be non-negative, got {chunk_overlap}.")
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly less than chunk_size ({chunk_size})."
        )


def _extract_provenance(
    document: Document,
    char_start: int,
    char_end: int,
    chunker_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[int | None, dict[str, Any]]:
    """Extract strictly overlapping page, line, and span provenance for a chunk.

    Enforces zero metadata leakage: only pages, lines, and spans that overlap
    [char_start, char_end) are included. Spans that cross chunk boundaries are
    clipped to the exact intersecting substring while recording their original
    offsets and bounding geometry.

    Args:
        document: The source Document entity.
        char_start: 0-indexed half-open start offset of the chunk.
        char_end: 0-indexed half-open end offset of the chunk.
        chunker_name: The name of the chunking strategy.
        chunk_size: The configured chunk size.
        chunk_overlap: The configured chunk overlap.

    Returns:
        A tuple of (single_page_number_or_none, chunk_metadata_dict).
    """
    metadata: dict[str, Any] = {
        "chunker": chunker_name,
        "strategy": chunker_name,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "pages": [],
        "page_numbers": [],
    }

    if "source_file" in document.metadata:
        metadata["source_file"] = document.metadata["source_file"]

    raw_pages = document.metadata.get("pages")
    if not isinstance(raw_pages, list) or not raw_pages:
        # Fallback if document has page_count but no granular PyMuPDF page list
        page_num = document.page_count if document.page_count == 1 else None
        if page_num is not None:
            metadata["page_numbers"] = [page_num]
        return page_num, metadata

    overlapping_pages: list[dict[str, Any]] = []
    overlapping_page_numbers: list[int] = []

    for page in raw_pages:
        if not isinstance(page, dict):
            continue

        p_start = page.get("char_start")
        p_end = page.get("char_end")
        if not isinstance(p_start, int) or not isinstance(p_end, int):
            continue

        # Half-open intersection check
        p_inter_start = max(p_start, char_start)
        p_inter_end = min(p_end, char_end)
        if p_inter_start >= p_inter_end:
            continue

        p_number = page.get("page_number")
        if isinstance(p_number, int) and p_number not in overlapping_page_numbers:
            overlapping_page_numbers.append(p_number)

        page_record: dict[str, Any] = {
            "page_index": page.get("page_index"),
            "page_number": p_number,
            "char_start": p_inter_start,
            "char_end": p_inter_end,
        }
        if "width" in page:
            page_record["width"] = page["width"]
        if "height" in page:
            page_record["height"] = page["height"]

        overlapping_lines: list[dict[str, Any]] = []
        for line in page.get("lines", []):
            if not isinstance(line, dict):
                continue

            l_start = line.get("char_start")
            l_end = line.get("char_end")
            if not isinstance(l_start, int) or not isinstance(l_end, int):
                continue

            l_inter_start = max(l_start, char_start)
            l_inter_end = min(l_end, char_end)
            if l_inter_start >= l_inter_end:
                continue

            overlapping_spans: list[dict[str, Any]] = []
            for span in line.get("spans", []):
                if not isinstance(span, dict):
                    continue

                s_start = span.get("char_start")
                s_end = span.get("char_end")
                if not isinstance(s_start, int) or not isinstance(s_end, int):
                    continue

                s_inter_start = max(s_start, char_start)
                s_inter_end = min(s_end, char_end)
                if s_inter_start >= s_inter_end:
                    continue

                # Exact clipped slice from source document
                clipped_text = document.content[s_inter_start:s_inter_end]
                is_clipped = (s_inter_start > s_start) or (s_inter_end < s_end)

                span_record: dict[str, Any] = {
                    "text": clipped_text,
                    "char_start": s_inter_start,
                    "char_end": s_inter_end,
                    "bbox": span.get("bbox"),
                    "is_clipped": is_clipped,
                }
                if is_clipped:
                    span_record["original_char_start"] = s_start
                    span_record["original_char_end"] = s_end
                    span_record["original_text"] = span.get("text")
                if "font" in span:
                    span_record["font"] = span["font"]
                if "size" in span:
                    span_record["size"] = span["size"]
                if "flags" in span:
                    span_record["flags"] = span["flags"]

                overlapping_spans.append(span_record)

            line_clipped_text = document.content[l_inter_start:l_inter_end]
            is_line_clipped = (l_inter_start > l_start) or (l_inter_end < l_end)

            line_record: dict[str, Any] = {
                "line_number": line.get("line_number"),
                "text": line_clipped_text,
                "char_start": l_inter_start,
                "char_end": l_inter_end,
                "bbox": line.get("bbox"),
                "is_clipped": is_line_clipped,
                "spans": overlapping_spans,
            }
            if is_line_clipped:
                line_record["original_char_start"] = l_start
                line_record["original_char_end"] = l_end

            overlapping_lines.append(line_record)

        page_record["lines"] = overlapping_lines
        overlapping_pages.append(page_record)

    metadata["pages"] = overlapping_pages
    metadata["page_numbers"] = overlapping_page_numbers

    # Populate single page_number only if chunk overlaps exactly one identifiable page
    page_number: int | None = (
        overlapping_page_numbers[0] if len(overlapping_page_numbers) == 1 else None
    )

    return page_number, metadata


class SlidingWindowChunker(BaseChunker):
    """Produces configurable, fixed-size overlapping character windows.

    All character intervals are zero-indexed, half-open [char_start, char_end)
    referencing Document.content exactly.
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50) -> None:
        """Initialize sliding window chunker.

        Args:
            chunk_size: Maximum character length of each chunk (> 0).
            chunk_overlap: Character overlap between consecutive chunks (>= 0 and < chunk_size).

        Raises:
            TypeError: If chunk_size or chunk_overlap are not integers.
            ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
        """
        _validate_chunk_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    @property
    def name(self) -> str:
        """Return unique strategy identifier."""
        return "sliding_window"

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into overlapping character window Chunks.

        Args:
            document: Source Document to split.

        Returns:
            An ordered list of Chunk domain entities with verified provenance.
        """
        content = document.content
        content_len = len(content)

        # Empty or whitespace-only documents produce zero chunks
        if not content.strip():
            return []

        step = self.chunk_size - self.chunk_overlap
        chunks: list[Chunk] = []
        start = 0

        while start < content_len:
            end = min(start + self.chunk_size, content_len)
            slice_text = content[start:end]

            # Avoid degenerate chunks containing only whitespace
            if slice_text.strip():
                page_num, chunk_meta = _extract_provenance(
                    document=document,
                    char_start=start,
                    char_end=end,
                    chunker_name=self.name,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunk_id = f"{document.doc_id}:{self.name}:{start}:{end}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        content=slice_text,
                        page_number=page_num,
                        char_start=start,
                        char_end=end,
                        metadata=chunk_meta,
                    )
                )

            if end == content_len:
                break
            start += step

        return chunks


class RecursiveCharacterChunker(BaseChunker):
    """Recursively splits text on semantic boundaries while preserving exact offsets.

    Separators are evaluated in prioritized order (e.g., paragraph, newline,
    sentence, whitespace, character). The resulting pieces are merged up to
    `chunk_size` while maintaining up to `chunk_overlap` character overlap.
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: Sequence[str] | None = None,
    ) -> None:
        """Initialize recursive character chunker.

        Args:
            chunk_size: Maximum character length of each chunk (> 0).
            chunk_overlap: Character overlap between consecutive chunks (>= 0 and < chunk_size).
            separators: Ordered list of separators to split on. Defaults to
                ("\\n\\n", "\\n", ". ", " ", "").

        Raises:
            TypeError: If configuration parameters have invalid types.
            ValueError: If configuration values violate boundary constraints.
        """
        _validate_chunk_params(chunk_size, chunk_overlap)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        if separators is None:
            self.separators: list[str] = list(DEFAULT_SEPARATORS)
        else:
            if not isinstance(separators, (list, tuple)):
                raise TypeError("separators must be a list or tuple of strings.")
            if not separators:
                raise ValueError("separators sequence must not be empty.")
            for sep in separators:
                if not isinstance(sep, str):
                    raise TypeError("All separators must be strings.")
            self.separators = list(separators)
            # Ensure character-level fallback is present to guarantee progress
            if "" not in self.separators:
                self.separators.append("")

    @property
    def name(self) -> str:
        """Return unique strategy identifier."""
        return "recursive_character"

    def _split_interval(
        self,
        content: str,
        start: int,
        end: int,
        sep_idx: int,
    ) -> list[tuple[int, int]]:
        """Recursively partition [start, end) into atomic intervals <= chunk_size.

        Separators stay attached to the preceding slice so no characters or
        separators are lost from the original Document.content.
        """
        if end - start <= self.chunk_size:
            return [(start, end)]

        if sep_idx >= len(self.separators):
            # Fallback: character-by-character split
            return [(start + k, start + k + 1) for k in range(end - start)]

        sep = self.separators[sep_idx]
        if sep == "":
            return [(start + k, start + k + 1) for k in range(end - start)]

        slice_text = content[start:end]
        if sep not in slice_text:
            return self._split_interval(content, start, end, sep_idx + 1)

        # Split slice_text keeping the separator attached to the preceding slice
        raw_intervals: list[tuple[int, int]] = []
        curr_local = 0
        slice_len = len(slice_text)

        while curr_local < slice_len:
            next_find = slice_text.find(sep, curr_local)
            if next_find == -1:
                raw_intervals.append((start + curr_local, end))
                break

            split_local = next_find + len(sep)
            if split_local > curr_local:
                raw_intervals.append((start + curr_local, start + split_local))
                curr_local = split_local
            else:
                raw_intervals.append((start + curr_local, start + curr_local + 1))
                curr_local += 1

        atomic_intervals: list[tuple[int, int]] = []
        for p_start, p_end in raw_intervals:
            if p_end <= p_start:
                continue
            if p_end - p_start <= self.chunk_size:
                atomic_intervals.append((p_start, p_end))
            else:
                sub_intervals = self._split_interval(content, p_start, p_end, sep_idx + 1)
                atomic_intervals.extend(sub_intervals)

        return atomic_intervals

    def chunk(self, document: Document) -> list[Chunk]:
        """Split document into semantic Chunks using recursive boundary splitting.

        Args:
            document: Source Document to split.

        Returns:
            An ordered list of Chunk domain entities with verified provenance.
        """
        content = document.content
        content_len = len(content)

        if not content.strip():
            return []

        # Step 1: Partition text into contiguous atomic pieces <= chunk_size
        atomic_pieces = self._split_interval(content, 0, content_len, 0)
        if not atomic_pieces:
            return []

        # Step 2: Merge adjacent atomic pieces into chunks <= chunk_size with overlap
        chunks: list[Chunk] = []
        i = 0
        num_pieces = len(atomic_pieces)

        while i < num_pieces:
            chunk_start = atomic_pieces[i][0]
            j = i
            while j < num_pieces and (atomic_pieces[j][1] - chunk_start <= self.chunk_size):
                j += 1

            chunk_end = atomic_pieces[j - 1][1]
            chunk_content = content[chunk_start:chunk_end]

            if chunk_content.strip():
                page_num, chunk_meta = _extract_provenance(
                    document=document,
                    char_start=chunk_start,
                    char_end=chunk_end,
                    chunker_name=self.name,
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap,
                )
                chunk_id = f"{document.doc_id}:{self.name}:{chunk_start}:{chunk_end}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        doc_id=document.doc_id,
                        content=chunk_content,
                        page_number=page_num,
                        char_start=chunk_start,
                        char_end=chunk_end,
                        metadata=chunk_meta,
                    )
                )

            if j >= num_pieces:
                break

            # Choose next starting piece to achieve configured overlap without exceeding it
            next_i = j
            for k in range(i + 1, j):
                overlap = chunk_end - atomic_pieces[k][0]
                if overlap <= self.chunk_overlap:
                    next_i = k
                    break

            # Guarantee forward progress
            if next_i <= i:
                next_i = i + 1
            i = next_i

        return chunks
