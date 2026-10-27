"""Unit tests for citation-aware sliding and recursive chunkers.

Verifies configuration validation, windowing, recursive splitting, and exact
character provenance invariants for citation auditing (ING-002).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from reliability_lab.core.models import Document
from reliability_lab.ingestion.chunker import (
    RecursiveCharacterChunker,
    SlidingWindowChunker,
)


def _make_mock_document(
    doc_id: str = "doc-test-1",
    content: str = "",
    pages: list[dict[str, Any]] | None = None,
    page_count: int | None = 1,
) -> Document:
    """Helper to construct Document entities with optional PyMuPDF-like page metadata."""
    metadata: dict[str, Any] = {"source_file": "test.pdf"}
    if pages is not None:
        metadata["pages"] = pages
    return Document(
        doc_id=doc_id,
        content=content,
        metadata=metadata,
        page_count=page_count,
    )


class TestChunkerConfiguration:
    """Test parameter validation, defaults, and type enforcement."""

    def test_sliding_window_valid_defaults(self) -> None:
        chunker = SlidingWindowChunker()
        assert chunker.name == "sliding_window"
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_recursive_valid_defaults(self) -> None:
        chunker = RecursiveCharacterChunker()
        assert chunker.name == "recursive_character"
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50
        assert "" in chunker.separators

    @pytest.mark.parametrize("chunker_cls", [SlidingWindowChunker, RecursiveCharacterChunker])
    def test_reject_invalid_chunk_sizes(self, chunker_cls: type) -> None:
        with pytest.raises(ValueError, match="positive integer"):
            chunker_cls(chunk_size=0, chunk_overlap=0)
        with pytest.raises(ValueError, match="positive integer"):
            chunker_cls(chunk_size=-10, chunk_overlap=0)
        with pytest.raises(TypeError, match="integer"):
            chunker_cls(chunk_size="500", chunk_overlap=50)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="integer"):
            chunker_cls(chunk_size=500.5, chunk_overlap=50)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="integer"):
            chunker_cls(chunk_size=True, chunk_overlap=0)  # type: ignore[arg-type]

    @pytest.mark.parametrize("chunker_cls", [SlidingWindowChunker, RecursiveCharacterChunker])
    def test_reject_invalid_chunk_overlaps(self, chunker_cls: type) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            chunker_cls(chunk_size=100, chunk_overlap=-1)
        with pytest.raises(ValueError, match="strictly less than"):
            chunker_cls(chunk_size=100, chunk_overlap=100)
        with pytest.raises(ValueError, match="strictly less than"):
            chunker_cls(chunk_size=100, chunk_overlap=150)
        with pytest.raises(TypeError, match="integer"):
            chunker_cls(chunk_size=100, chunk_overlap="10")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="integer"):
            chunker_cls(chunk_size=100, chunk_overlap=10.5)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="integer"):
            chunker_cls(chunk_size=100, chunk_overlap=False)  # type: ignore[arg-type]

    def test_recursive_reject_invalid_separators(self) -> None:
        with pytest.raises(TypeError, match="list or tuple"):
            RecursiveCharacterChunker(separators="invalid")  # type: ignore[arg-type]
        with pytest.raises(ValueError, match="must not be empty"):
            RecursiveCharacterChunker(separators=[])
        with pytest.raises(TypeError, match="All separators must be strings"):
            RecursiveCharacterChunker(separators=["\n", 123])  # type: ignore[list-item]


class TestSlidingWindowChunker:
    """Test sliding window character segmentation and edge cases."""

    def test_empty_or_whitespace_document(self) -> None:
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk(_make_mock_document(content="")) == []
        assert chunker.chunk(_make_mock_document(content="   \n\t  \n")) == []

    def test_document_shorter_than_chunk_size(self) -> None:
        text = "Short text under chunk size."
        doc = _make_mock_document(content=text)
        chunker = SlidingWindowChunker(chunk_size=100, chunk_overlap=20)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == len(text)
        assert chunks[0].content == text
        assert doc.content[chunks[0].char_start : chunks[0].char_end] == chunks[0].content

    def test_exact_size_document(self) -> None:
        text = "1234567890"
        doc = _make_mock_document(content=text)
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=2)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].char_start == 0
        assert chunks[0].char_end == 10
        assert chunks[0].content == text

    def test_deterministic_windowing_and_overlap(self) -> None:
        text = "0123456789abcdefghijklmnopqrstuvwxyz"  # 36 chars
        doc = _make_mock_document(content=text)
        chunker = SlidingWindowChunker(chunk_size=10, chunk_overlap=4)
        # step = 10 - 4 = 6
        chunks = chunker.chunk(doc)

        assert len(chunks) == 6
        expected_ranges = [(0, 10), (6, 16), (12, 22), (18, 28), (24, 34), (30, 36)]
        for chunk, (expected_start, expected_end) in zip(chunks, expected_ranges, strict=True):
            assert chunk.char_start == expected_start
            assert chunk.char_end == expected_end
            assert chunk.content == text[expected_start:expected_end]
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content

    def test_zero_overlap(self) -> None:
        text = "ABCDEFGHIJ1234567890"  # 20 chars
        doc = _make_mock_document(content=text)
        chunker = SlidingWindowChunker(chunk_size=5, chunk_overlap=0)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 4
        assert [c.content for c in chunks] == ["ABCDE", "FGHIJ", "12345", "67890"]

    def test_unicode_content(self) -> None:
        text = "Hello 🌍 World! Привет мир! こんにちは世界!"
        doc = _make_mock_document(content=text)
        chunker = SlidingWindowChunker(chunk_size=15, chunk_overlap=5)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content


class TestRecursiveCharacterChunker:
    """Test semantic recursive splitting and fallback behavior."""

    def test_empty_or_whitespace_document(self) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=20)
        assert chunker.chunk(_make_mock_document(content="")) == []
        assert chunker.chunk(_make_mock_document(content="   \n\t  \n")) == []

    def test_short_document_produces_single_chunk(self) -> None:
        text = "A concise sentence."
        doc = _make_mock_document(content=text)
        chunker = RecursiveCharacterChunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk(doc)

        assert len(chunks) == 1
        assert chunks[0].content == text
        assert doc.content[chunks[0].char_start : chunks[0].char_end] == chunks[0].content

    def test_paragraph_split_priority(self) -> None:
        p1 = "Paragraph one with detailed text."
        p2 = "Paragraph two with additional notes."
        text = f"{p1}\n\n{p2}"
        doc = _make_mock_document(content=text)

        # chunk_size chosen so p1 and p2 fit individually but not together
        chunker = RecursiveCharacterChunker(
            chunk_size=max(len(p1) + 2, len(p2)),
            chunk_overlap=0,
            separators=["\n\n", "\n", " ", ""],
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) == 2
        assert chunks[0].content == f"{p1}\n\n"
        assert chunks[1].content == p2
        for chunk in chunks:
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content

    def test_sentence_and_whitespace_boundaries(self) -> None:
        text = "First sentence. Second sentence. Third sentence. Fourth sentence."
        doc = _make_mock_document(content=text)
        chunker = RecursiveCharacterChunker(
            chunk_size=35, chunk_overlap=10, separators=[". ", " ", ""]
        )
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= 35
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content

    def test_character_fallback_on_unbroken_text(self) -> None:
        text = "A" * 150  # No whitespace or punctuation
        doc = _make_mock_document(content=text)
        chunker = RecursiveCharacterChunker(chunk_size=40, chunk_overlap=10)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.content) <= 40
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content

    def test_repeated_separators_do_not_hang(self) -> None:
        text = "Start\n\n\n\n\n\n\n\nMiddle\n\n\n\n\n\nEnd"
        doc = _make_mock_document(content=text)
        chunker = RecursiveCharacterChunker(chunk_size=15, chunk_overlap=5)
        chunks = chunker.chunk(doc)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert len(chunk.content) <= 15
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content


class TestProvenanceAndGeometricTracking:
    """Test page, line, span provenance tracking and zero metadata leakage."""

    @pytest.fixture
    def structured_doc(self) -> Document:
        """Create a 2-page document with known lines and bounding boxes."""
        # Page 1 text: "Page 1 line 1\nPage 1 line 2" (chars 0..27)
        # Separator: "\n\n" (chars 27..29)
        # Page 2 text: "Page 2 line 1\nPage 2 line 2" (chars 29..56)
        content = "Page 1 line 1\nPage 1 line 2\n\nPage 2 line 1\nPage 2 line 2"

        page_1 = {
            "page_index": 0,
            "page_number": 1,
            "char_start": 0,
            "char_end": 27,
            "width": 612.0,
            "height": 792.0,
            "lines": [
                {
                    "line_number": 1,
                    "text": "Page 1 line 1",
                    "char_start": 0,
                    "char_end": 13,
                    "bbox": [50.0, 700.0, 150.0, 715.0],
                    "spans": [
                        {
                            "text": "Page 1 line 1",
                            "char_start": 0,
                            "char_end": 13,
                            "bbox": [50.0, 700.0, 150.0, 715.0],
                            "font": "Helvetica",
                            "size": 12.0,
                            "flags": 0,
                        }
                    ],
                },
                {
                    "line_number": 2,
                    "text": "Page 1 line 2",
                    "char_start": 14,
                    "char_end": 27,
                    "bbox": [50.0, 680.0, 150.0, 695.0],
                    "spans": [
                        {
                            "text": "Page 1 line 2",
                            "char_start": 14,
                            "char_end": 27,
                            "bbox": [50.0, 680.0, 150.0, 695.0],
                            "font": "Helvetica",
                            "size": 12.0,
                            "flags": 0,
                        }
                    ],
                },
            ],
        }

        page_2 = {
            "page_index": 1,
            "page_number": 2,
            "char_start": 29,
            "char_end": 56,
            "width": 612.0,
            "height": 792.0,
            "lines": [
                {
                    "line_number": 1,
                    "text": "Page 2 line 1",
                    "char_start": 29,
                    "char_end": 42,
                    "bbox": [50.0, 700.0, 150.0, 715.0],
                    "spans": [
                        {
                            "text": "Page 2 line 1",
                            "char_start": 29,
                            "char_end": 42,
                            "bbox": [50.0, 700.0, 150.0, 715.0],
                            "font": "Helvetica",
                            "size": 12.0,
                            "flags": 0,
                        }
                    ],
                },
                {
                    "line_number": 2,
                    "text": "Page 2 line 2",
                    "char_start": 43,
                    "char_end": 56,
                    "bbox": [50.0, 680.0, 150.0, 695.0],
                    "spans": [
                        {
                            "text": "Page 2 line 2",
                            "char_start": 43,
                            "char_end": 56,
                            "bbox": [50.0, 680.0, 150.0, 695.0],
                            "font": "Helvetica",
                            "size": 12.0,
                            "flags": 0,
                        }
                    ],
                },
            ],
        }

        return _make_mock_document(
            doc_id="provenance-doc-1",
            content=content,
            pages=[page_1, page_2],
            page_count=2,
        )

    def test_single_page_chunk_provenance(self, structured_doc: Document) -> None:
        chunker = SlidingWindowChunker(chunk_size=20, chunk_overlap=0)
        chunks = chunker.chunk(structured_doc)

        # First chunk covers 0..20, strictly within page 1 (0..27)
        c0 = chunks[0]
        assert c0.page_number == 1
        assert c0.metadata["page_numbers"] == [1]
        assert len(c0.metadata["pages"]) == 1
        p_rec = c0.metadata["pages"][0]
        assert p_rec["page_number"] == 1
        assert p_rec["char_start"] == 0
        assert p_rec["char_end"] == 20

        # Lines in chunk 0:
        # line 1 (0..13) is fully within 0..20
        # line 2 (14..27) is partially within 0..20 (clipped to 14..20)
        lines = p_rec["lines"]
        assert len(lines) == 2
        assert lines[0]["text"] == "Page 1 line 1"
        assert not lines[0]["is_clipped"]
        assert lines[1]["text"] == "Page 1"  # chars 14..20 of "Page 1 line 2"
        assert lines[1]["is_clipped"]
        assert lines[1]["char_start"] == 14
        assert lines[1]["char_end"] == 20
        assert lines[1]["original_char_start"] == 14
        assert lines[1]["original_char_end"] == 27

        # Zero leakage: page 2 elements must not be present
        assert not any(p.get("page_number") == 2 for p in c0.metadata["pages"])

    def test_cross_page_chunk_provenance(self, structured_doc: Document) -> None:
        # Create a chunk that spans from page 1 into page 2
        # Page 1 ends at 27. Separator is 27..29. Page 2 starts at 29.
        doc = structured_doc
        chunker_boundary = SlidingWindowChunker(chunk_size=20, chunk_overlap=0)
        # Offset 20..40 covers page 1 (20..27), separator (27..29), and page 2 (29..40)
        c = chunker_boundary.chunk(doc)[1]  # 20..40

        assert c.char_start == 20
        assert c.char_end == 40
        # Chunk spans both pages, so page_number must be None
        assert c.page_number is None
        assert sorted(c.metadata["page_numbers"]) == [1, 2]
        assert len(c.metadata["pages"]) == 2

        # Page 1 record: char_start=20, char_end=27
        p1 = c.metadata["pages"][0]
        assert p1["page_number"] == 1
        assert p1["char_start"] == 20
        assert p1["char_end"] == 27
        # Page 2 record: char_start=29, char_end=40
        p2 = c.metadata["pages"][1]
        assert p2["page_number"] == 2
        assert p2["char_start"] == 29
        assert p2["char_end"] == 40

        # Separator (27..29) is neither attributed to page 1 nor page 2
        assert p1["char_end"] == 27
        assert p2["char_start"] == 29

    def test_zero_metadata_leakage_on_non_overlapping_lines(self, structured_doc: Document) -> None:
        chunker = SlidingWindowChunker(chunk_size=13, chunk_overlap=0)
        chunks = chunker.chunk(structured_doc)

        # Chunk 0 covers 0..13 (only line 1: 0..13)
        c0 = chunks[0]
        lines = c0.metadata["pages"][0]["lines"]
        assert len(lines) == 1
        assert lines[0]["line_number"] == 1
        # Line 2 (14..27) must NOT appear in chunk 0
        assert not any(line_entry["line_number"] == 2 for line_entry in lines)

    def test_exact_substring_slicing_invariant_for_all_elements(
        self, structured_doc: Document
    ) -> None:
        for chunker in [
            SlidingWindowChunker(chunk_size=16, chunk_overlap=4),
            RecursiveCharacterChunker(chunk_size=16, chunk_overlap=4),
        ]:
            chunks = chunker.chunk(structured_doc)
            assert len(chunks) > 0

            for chunk in chunks:
                # 1. Chunk content matches global Document slice
                assert structured_doc.content[chunk.char_start : chunk.char_end] == chunk.content
                assert chunk.doc_id == structured_doc.doc_id

                # 2. Every line matches its global Document slice
                for page in chunk.metadata["pages"]:
                    for line in page["lines"]:
                        assert (
                            structured_doc.content[line["char_start"] : line["char_end"]]
                            == line["text"]
                        )
                        # 3. Every span matches its global Document slice
                        for span in line["spans"]:
                            assert (
                                structured_doc.content[span["char_start"] : span["char_end"]]
                                == span["text"]
                            )

    def test_stable_deterministic_chunk_ids(self, structured_doc: Document) -> None:
        chunker1 = SlidingWindowChunker(chunk_size=20, chunk_overlap=5)
        chunker2 = SlidingWindowChunker(chunk_size=20, chunk_overlap=5)

        chunks1 = chunker1.chunk(structured_doc)
        chunks2 = chunker2.chunk(structured_doc)

        assert len(chunks1) == len(chunks2)
        for c1, c2 in zip(chunks1, chunks2, strict=True):
            assert c1.chunk_id == c2.chunk_id
            assert c1.char_start == c2.char_start
            assert c1.char_end == c2.char_end

    def test_metadata_is_json_serializable(self, structured_doc: Document) -> None:
        chunker = RecursiveCharacterChunker(chunk_size=25, chunk_overlap=5)
        chunks = chunker.chunk(structured_doc)

        for chunk in chunks:
            serialized = json.dumps(chunk.metadata)
            assert isinstance(serialized, str)
            round_trip = json.loads(serialized)
            assert round_trip["chunker"] == "recursive_character"
            assert "pages" in round_trip

    def test_document_without_page_metadata(self) -> None:
        doc = Document(doc_id="plain-doc", content="A simple document without PDF page metadata.")
        chunker = SlidingWindowChunker(chunk_size=20, chunk_overlap=5)
        chunks = chunker.chunk(doc)

        assert len(chunks) > 0
        for chunk in chunks:
            assert doc.content[chunk.char_start : chunk.char_end] == chunk.content
            assert chunk.metadata["pages"] == []


class TestPyMuPDFIntegration:
    """Test chunker behavior when processing output directly from PyMuPDFParser."""

    def test_end_to_end_parser_and_chunker_pipeline(self, tmp_path: Any) -> None:
        import pymupdf

        from reliability_lab.ingestion.parser import PyMuPDFParser

        pdf_path = tmp_path / "integration_sample.pdf"
        doc = pymupdf.open()
        p1 = doc.new_page(width=595, height=842)
        p1.insert_text((50, 50), "First page title\nFirst page body sentence.", fontsize=12)
        p2 = doc.new_page(width=595, height=842)
        p2.insert_text((50, 50), "Second page title\nSecond page content.", fontsize=12)
        doc.save(pdf_path)
        doc.close()

        parser = PyMuPDFParser()
        parsed_doc = parser.parse(pdf_path, doc_id="integration-doc")

        sliding_chunker = SlidingWindowChunker(chunk_size=30, chunk_overlap=10)
        sliding_chunks = sliding_chunker.chunk(parsed_doc)
        assert len(sliding_chunks) > 1

        for c in sliding_chunks:
            assert parsed_doc.content[c.char_start : c.char_end] == c.content
            assert c.doc_id == "integration-doc"
            for p in c.metadata["pages"]:
                for line in p["lines"]:
                    assert parsed_doc.content[line["char_start"] : line["char_end"]] == line["text"]
                    for span in line["spans"]:
                        assert (
                            parsed_doc.content[span["char_start"] : span["char_end"]]
                            == span["text"]
                        )

        recursive_chunker = RecursiveCharacterChunker(chunk_size=35, chunk_overlap=10)
        recursive_chunks = recursive_chunker.chunk(parsed_doc)
        assert len(recursive_chunks) > 1

        for c in recursive_chunks:
            assert parsed_doc.content[c.char_start : c.char_end] == c.content
            assert c.doc_id == "integration-doc"
