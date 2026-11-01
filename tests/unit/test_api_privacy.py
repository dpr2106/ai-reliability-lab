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
        ],
    )
    def test_sanitize_windows_paths(self, raw: str, expected: str) -> None:
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
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
        ],
    )
    def test_sanitize_posix_paths(self, raw: str, expected: str) -> None:
        assert sanitize_path_message(raw) == expected
