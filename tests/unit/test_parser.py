"""Unit tests for PyMuPDF Document Parser and provenance tracking."""

from pathlib import Path

import pymupdf
import pytest

from reliability_lab.core.models import Document
from reliability_lab.ingestion.parser import (
    EncryptedPDFParserError,
    FileNotFoundParserError,
    InvalidPDFParserError,
    PyMuPDFParser,
)


def _create_pdf(path: Path, pages: list[list[str]]) -> None:
    """Helper to dynamically generate a clean test PDF using PyMuPDF."""
    doc = pymupdf.open()
    for lines in pages:
        page = doc.new_page(width=595, height=842)
        y = 50
        for line in lines:
            page.insert_text((50, y), line, fontsize=12)
            y += 20
    doc.save(path)
    doc.close()


def _create_encrypted_pdf(path: Path, text: str, user_pw: str, owner_pw: str = "owner123") -> None:
    """Helper to dynamically generate an encrypted PDF using PyMuPDF."""
    doc = pymupdf.open()
    page = doc.new_page(width=595, height=842)
    page.insert_text((50, 50), text, fontsize=12)
    doc.save(
        path,
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        user_pw=user_pw,
        owner_pw=owner_pw,
    )
    doc.close()


class TestPyMuPDFParserBasics:
    """Basic extraction, metadata, and provenance slicing tests."""

    @pytest.fixture
    def single_page_pdf(self, tmp_path: Path) -> Path:
        pdf_path = tmp_path / "single_page.pdf"
        _create_pdf(
            pdf_path,
            pages=[
                [
                    "Acme Corporation Q3 Financial Report",
                    "Total Revenue: $4.2B",
                    "Net Income: $820M",
                ]
            ],
        )
        return pdf_path

    @pytest.fixture
    def multi_page_pdf(self, tmp_path: Path) -> Path:
        pdf_path = tmp_path / "multi_page.pdf"
        _create_pdf(
            pdf_path,
            pages=[
                ["Page 1 Header", "Introductory paragraph on page 1."],
                ["Page 2 Header", "Financial data table row 1."],
                ["Page 3 Header", "Summary conclusions and citations."],
            ],
        )
        return pdf_path

    def test_parse_single_page_success(self, single_page_pdf: Path):
        parser = PyMuPDFParser()
        doc = parser.parse(single_page_pdf, doc_id="acme_q3")

        assert isinstance(doc, Document)
        assert doc.doc_id == "acme_q3"
        assert doc.page_count == 1
        assert "Total Revenue: $4.2B" in doc.content
        assert doc.metadata["source_file"] == "single_page.pdf"
        assert doc.metadata["has_extractable_text"] is True

    def test_default_doc_id_from_stem(self, single_page_pdf: Path):
        parser = PyMuPDFParser()
        doc = parser.parse(single_page_pdf)
        assert doc.doc_id == "single_page"

    def test_multi_page_extraction_and_order(self, multi_page_pdf: Path):
        parser = PyMuPDFParser()
        doc = parser.parse(multi_page_pdf)

        assert doc.page_count == 3
        pages = doc.metadata["pages"]
        assert len(pages) == 3

        assert pages[0]["page_number"] == 1
        assert pages[0]["page_index"] == 0
        assert "Page 1 Header" in pages[0]["text"]

        assert pages[1]["page_number"] == 2
        assert pages[1]["page_index"] == 1
        assert "Page 2 Header" in pages[1]["text"]

        assert pages[2]["page_number"] == 3
        assert pages[2]["page_index"] == 2
        assert "Page 3 Header" in pages[2]["text"]

    def test_exact_slicing_provenance_invariant(self, multi_page_pdf: Path):
        """CRITICAL: Slicing Document.content with char_start:char_end must yield exact text."""
        parser = PyMuPDFParser()
        doc = parser.parse(multi_page_pdf)
        content = doc.content

        for page in doc.metadata["pages"]:
            p_start, p_end = page["char_start"], page["char_end"]
            assert content[p_start:p_end] == page["text"], "Page slice mismatch"

            for line in page["lines"]:
                l_start, l_end = line["char_start"], line["char_end"]
                assert content[l_start:l_end] == line["text"], "Line slice mismatch"

                for span in line["spans"]:
                    s_start, s_end = span["char_start"], span["char_end"]
                    assert content[s_start:s_end] == span["text"], "Span slice mismatch"

    def test_custom_separators(self, multi_page_pdf: Path):
        parser = PyMuPDFParser(page_separator="\n===PAGE===\n", line_separator=" | ")
        doc = parser.parse(multi_page_pdf)

        assert "\n===PAGE===\n" in doc.content
        assert " | " in doc.content

        # Verify slicing remains 100% exact with custom separators
        for page in doc.metadata["pages"]:
            assert doc.content[page["char_start"] : page["char_end"]] == page["text"]
            for line in page["lines"]:
                assert doc.content[line["char_start"] : line["char_end"]] == line["text"]
                for span in line["spans"]:
                    assert doc.content[span["char_start"] : span["char_end"]] == span["text"]

    def test_bounding_box_geometry_invariants(self, single_page_pdf: Path):
        parser = PyMuPDFParser()
        doc = parser.parse(single_page_pdf)
        page = doc.metadata["pages"][0]

        assert page["width"] > 0
        assert page["height"] > 0

        for line in page["lines"]:
            bbox = line["bbox"]
            assert len(bbox) == 4
            x0, y0, x1, y1 = bbox
            assert x0 <= x1
            assert y0 <= y1

            for span in line["spans"]:
                s_bbox = span["bbox"]
                assert len(s_bbox) == 4
                assert s_bbox[0] <= s_bbox[2]
                assert s_bbox[1] <= s_bbox[3]

    def test_unicode_text_handling(self, tmp_path: Path):
        pdf_path = tmp_path / "unicode.pdf"
        _create_pdf(
            pdf_path,
            pages=[["Euro: 100 EUR, Yen: 10,000 JPY", "Mathematical Symbols: sum(x) >= 0"]],
        )
        parser = PyMuPDFParser()
        doc = parser.parse(pdf_path)

        assert "100 EUR" in doc.content
        for page in doc.metadata["pages"]:
            assert doc.content[page["char_start"] : page["char_end"]] == page["text"]

    def test_blank_pages_and_no_extractable_text(self, tmp_path: Path):
        pdf_path = tmp_path / "blank.pdf"
        doc = pymupdf.open()
        doc.new_page(width=400, height=400)  # Blank page
        doc.new_page(width=400, height=400)  # Blank page
        doc.save(pdf_path)
        doc.close()

        parser = PyMuPDFParser()
        parsed_doc = parser.parse(pdf_path)

        assert parsed_doc.page_count == 2
        assert parsed_doc.content == ""
        assert parsed_doc.metadata["has_extractable_text"] is False
        assert len(parsed_doc.metadata["pages"]) == 2
        assert parsed_doc.metadata["pages"][0]["text"] == ""


class TestPyMuPDFParserErrors:
    """Error handling, exception hierarchies, and resource cleanup tests."""

    def test_missing_file_raises_file_not_found_parser_error(self, tmp_path: Path):
        parser = PyMuPDFParser()
        missing = tmp_path / "non_existent.pdf"
        with pytest.raises(FileNotFoundParserError, match="Target document not found"):
            parser.parse(missing)

    def test_directory_path_raises_file_not_found_parser_error(self, tmp_path: Path):
        parser = PyMuPDFParser()
        with pytest.raises(FileNotFoundParserError):
            parser.parse(tmp_path)

    def test_corrupt_file_raises_invalid_pdf_parser_error(self, tmp_path: Path):
        corrupt_path = tmp_path / "corrupt.pdf"
        corrupt_path.write_bytes(b"%PDF-1.4\nCorrupt bytes that cannot be parsed as a valid PDF")

        parser = PyMuPDFParser()
        with pytest.raises(InvalidPDFParserError, match="Failed to open"):
            parser.parse(corrupt_path)

    def test_encrypted_pdf_without_password_raises_encrypted_error(self, tmp_path: Path):
        enc_path = tmp_path / "encrypted.pdf"
        _create_encrypted_pdf(enc_path, "Secret Financial Data", user_pw="secure_pass")

        parser = PyMuPDFParser()
        with pytest.raises(EncryptedPDFParserError, match="encrypted and requires a password"):
            parser.parse(enc_path)

    def test_encrypted_pdf_with_wrong_password_raises_encrypted_error(self, tmp_path: Path):
        enc_path = tmp_path / "encrypted.pdf"
        _create_encrypted_pdf(enc_path, "Secret Financial Data", user_pw="secure_pass")

        parser = PyMuPDFParser()
        with pytest.raises(EncryptedPDFParserError, match="Authentication failed"):
            parser.parse(enc_path, password="wrong_password")

    def test_encrypted_pdf_with_correct_password_succeeds(self, tmp_path: Path):
        enc_path = tmp_path / "encrypted.pdf"
        _create_encrypted_pdf(enc_path, "Secret Financial Data", user_pw="correct_pw")

        parser = PyMuPDFParser()
        doc = parser.parse(enc_path, password="correct_pw")

        assert "Secret Financial Data" in doc.content
        assert doc.page_count == 1
        # Password must not be leaked into metadata or content
        assert "correct_pw" not in str(doc.metadata)

    def test_no_absolute_paths_in_metadata(self, tmp_path: Path):
        pdf_path = tmp_path / "my_secret_doc.pdf"
        _create_pdf(pdf_path, pages=[["Confidential statement"]])

        parser = PyMuPDFParser()
        doc = parser.parse(pdf_path)

        # source_file should only be the filename, not full directory path
        assert doc.metadata["source_file"] == "my_secret_doc.pdf"
        assert str(tmp_path) not in doc.metadata["source_file"]
