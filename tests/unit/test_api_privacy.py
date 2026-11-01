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
        ],
    )
    def test_sanitize_windows_paths(self, raw: str, expected: str) -> None:
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Previously missing paths from review:
            (
                "Error accessing /mnt/data/private/file.pdf: not found",
                "Error accessing file.pdf: not found",
            ),
            (
                "Failed to read /workspace/project/storage/file.pdf",
                "Failed to read file.pdf",
            ),
            (
                "Failed to parse /srv/storage/file.pdf: corrupted header",
                "Failed to parse file.pdf: corrupted header",
            ),
            (
                "[Errno 13] Permission denied: '/data/documents/file.pdf'",
                "[Errno 13] Permission denied: 'file.pdf'",
            ),
            (
                'Corrupted file at "/data/documents/file.pdf"',
                'Corrupted file at "file.pdf"',
            ),
            (
                "Directory /srv/storage/ does not exist",
                "Directory storage does not exist",
            ),
            (
                "Directory /srv/storage does not exist",
                "Directory storage does not exist",
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
            # Standard POSIX directories:
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
        ],
    )
    def test_sanitize_posix_paths(self, raw: str, expected: str) -> None:
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "text",
        [
            "See https://example.com/api/v1/query?status=ready for details",
            "Check http://localhost:8000/api/documents/doc_123/chunks",
            "API at http://127.0.0.1:8000/api/documents",
            "Score: 5/10, status: ok/ready",
            "5 / 10 ratio",
            "Document status is READY / PARTIAL",
            "Supports text/plain and application/pdf formats",
            "Use flag -a/--all for complete listing",
        ],
    )
    def test_preserves_urls_and_ordinary_slashes(self, text: str) -> None:
        assert sanitize_path_message(text) == text
