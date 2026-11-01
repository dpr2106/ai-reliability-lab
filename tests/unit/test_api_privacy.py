"""Unit tests for path privacy and message sanitization utilities."""

from __future__ import annotations

import pytest

from reliability_lab.api.privacy import sanitize_path_message


class TestPathPrivacySanitization:
    """Tests for sanitize_path_message()."""

    def test_sanitize_none_and_empty(self) -> None:
        assert sanitize_path_message(None) is None
        assert sanitize_path_message("") == ""

    def test_sanitize_plain_message_without_paths(self) -> None:
        msg = "Document text extraction failed: Invalid format or corrupted file."
        assert sanitize_path_message(msg) == msg

    @pytest.mark.parametrize(
        "raw,expected",
        [
            (
                "Failed to open C:\\Users\\admin\\app\\data\\report.pdf: File not found",
                "Failed to open report.pdf: File not found",
            ),
            (
                "Error reading C:/Users/developer/workspace/project/data.txt: bad encoding",
                "Error reading data.txt: bad encoding",
            ),
            (
                "[Errno 2] No such file: 'C:\\\\Users\\\\runneradmin\\\\data\\\\doc.pdf'",
                "[Errno 2] No such file: 'doc.pdf'",
            ),
            (
                "Permission denied: D:\\data\\storage\\document.bin",
                "Permission denied: document.bin",
            ),
            (
                "Root path error at C:\\",
                "Root path error at [path]",
            ),
            (
                "Failed at C:/",
                "Failed at [path]",
            ),
            (
                "Failed on C:\\Users\\prash\\My Documents\\file(1).pdf",
                "Failed on file(1).pdf",
            ),
            (
                "Failed on C:\\Users\\prash\\naïve résumé#1.pdf",
                "Failed on naïve résumé#1.pdf",
            ),
        ],
    )
    def test_sanitize_windows_paths(self, raw: str, expected: str) -> None:
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Legal filename characters: spaces, Unicode, parentheses, #
            (
                "/mnt/data/private/file name.pdf",
                "file name.pdf",
            ),
            (
                "'/mnt/data/private/file name.pdf'",
                "'file name.pdf'",
            ),
            (
                '"/mnt/data/private/file name.pdf"',
                '"file name.pdf"',
            ),
            (
                "Error accessing /mnt/data/private/file name.pdf: not found",
                "Error accessing file name.pdf: not found",
            ),
            (
                "[Errno 2] No such file or directory: '/mnt/data/private/file name.pdf'",
                "[Errno 2] No such file or directory: 'file name.pdf'",
            ),
            (
                "/mnt/data/naïve résumé.pdf",
                "naïve résumé.pdf",
            ),
            (
                "'/mnt/data/naïve résumé.pdf'",
                "'naïve résumé.pdf'",
            ),
            (
                "Error in /mnt/data/naïve résumé.pdf: corrupted",
                "Error in naïve résumé.pdf: corrupted",
            ),
            (
                "/mnt/data/file(1).pdf",
                "file(1).pdf",
            ),
            (
                "'/mnt/data/file(1).pdf'",
                "'file(1).pdf'",
            ),
            (
                "Failed at /mnt/data/file(1).pdf",
                "Failed at file(1).pdf",
            ),
            (
                "Missing file in (/mnt/data/file(1).pdf)",
                "Missing file in (file(1).pdf)",
            ),
            (
                "/mnt/data/file#1.pdf",
                "file#1.pdf",
            ),
            (
                "'/mnt/data/file#1.pdf'",
                "'file#1.pdf'",
            ),
            (
                "Error reading /mnt/data/file#1.pdf",
                "Error reading file#1.pdf",
            ),
            # Paths with spaces and Unicode in intermediate directory names
            (
                "/workspace/my documents/naïve test/file.pdf",
                "file.pdf",
            ),
            (
                "'/workspace/my documents/naïve test/file.pdf'",
                "'file.pdf'",
            ),
            (
                "Extraction failed on /workspace/my documents/naïve test/file.pdf: syntax error",
                "Extraction failed on file.pdf: syntax error",
            ),
            (
                "Directory /workspace/my documents/naïve test/ does not exist",
                "Directory naïve test does not exist",
            ),
            # Standard POSIX directories and arbitrary mounts
            (
                "/srv/storage/file.pdf",
                "file.pdf",
            ),
            (
                "/data/documents/file.pdf",
                "file.pdf",
            ),
            (
                "Directory /srv/storage/ does not exist",
                "Directory storage does not exist",
            ),
            (
                "Directory /srv/storage does not exist",
                "Directory storage does not exist",
            ),
            (
                "Permission denied: /home/runner/work/ai-reliability-lab/storage/data.bin",
                "Permission denied: data.bin",
            ),
            (
                "Cannot access /tmp/ai_lab_scratch/test.pdf: no permission",
                "Cannot access test.pdf: no permission",
            ),
            (
                "Error opening file '/var/data/models/faiss.index'",
                "Error opening file 'faiss.index'",
            ),
            (
                "Missing file in (/Users/prash/repo/file.txt)",
                "Missing file in (file.txt)",
            ),
            (
                "Failed at /opt/app/config.json",
                "Failed at config.json",
            ),
            (
                "Sentence ending with path /opt/data/file.txt.",
                "Sentence ending with path file.txt.",
            ),
            (
                "Multiple: /var/log/app.log and /tmp/run.pid.",
                "Multiple: app.log and run.pid.",
            ),
            (
                "Failed to copy from /mnt/data/my file (1).pdf to /srv/storage/backup #2.pdf",
                "Failed to copy from my file (1).pdf to backup #2.pdf",
            ),
            # Root paths:
            (
                "Failed at root /",
                "Failed at root [path]",
            ),
            (
                "Error accessing '/'",
                "Error accessing '[path]'",
            ),
            (
                'Error accessing "/"',
                'Error accessing "[path]"',
            ),
        ],
    )
    def test_sanitize_posix_paths(self, raw: str, expected: str) -> None:
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # URLs
            "See https://example.com/api/v1/query?status=ready for details",
            "Check http://localhost:8000/api/documents/doc_123/chunks",
            "API at http://127.0.0.1:8000/api/documents",
            # API routes
            "Check endpoint /api/v1/documents for details",
            "Endpoint '/api/v1/documents' returned 404",
            "GET /api/documents/{id}/chunks",
            "Schema at /openapi.json",
            "Docs at /docs",
            # Documentation placeholders
            "Pass an /absolute/path to the command",
            "Replace with /path/to/file before running",
            # Ordinary syntax and expressions
            "Score: 5/10, status: ok/ready",
            "5 / 10 ratio",
            "Document status is READY / PARTIAL",
            "Supports text/plain and application/pdf formats",
            "Use flag -a/--all for complete listing",
        ],
    )
    def test_preserves_urls_api_routes_and_ordinary_syntax(self, text: str) -> None:
        assert sanitize_path_message(text) == text
