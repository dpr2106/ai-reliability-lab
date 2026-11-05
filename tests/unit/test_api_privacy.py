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
            # Legal filename characters: spaces, Unicode, parentheses, #, [], +
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
            (
                "/mnt/data/file[2].pdf",
                "file[2].pdf",
            ),
            (
                "/mnt/data/file+v2.pdf",
                "file+v2.pdf",
            ),
            (
                "'/workspace/my documents/naïve test/file[1]+v2.pdf'",
                "'file[1]+v2.pdf'",
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
        "raw,expected",
        [
            # Protected-prefix false negatives (MUST BE SANITIZED)
            (
                "Permission denied: /api/private/customer-data/file.pdf",
                "Permission denied: file.pdf",
            ),
            (
                "Failed to open /docs/private/storage/credentials.txt",
                "Failed to open credentials.txt",
            ),
            (
                "[Errno 2] No such file or directory: '/api/v1/internal/secrets/config.json'",
                "[Errno 2] No such file or directory: 'config.json'",
            ),
            (
                "Cannot access /health/runtime/private/file.bin: permission denied",
                "Cannot access file.bin: permission denied",
            ),
            (
                "Error reading /v1/internal/data/document.pdf: bad format",
                "Error reading document.pdf: bad format",
            ),
            (
                "Failed to delete /redoc/cache/temporary/file.tmp",
                "Failed to delete file.tmp",
            ),
            # Extensionless & directory paths under protected-prefixes
            (
                "Directory /docs/private/storage/ does not exist",
                "Directory storage does not exist",
            ),
            (
                "Failed to access directory /api/private/storage",
                "Failed to access directory storage",
            ),
            (
                "Error accessing /api/storage/records",
                "Error accessing records",
            ),
        ],
    )
    def test_sanitize_protected_prefix_paths(self, raw: str, expected: str) -> None:
        """Verify that filesystem paths beginning with /api, /docs, etc. are sanitized."""
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Case A: Brace-based route parameter bypass
            (
                "Permission denied: /api/private/{id}/storage/file.pdf",
                "Permission denied: file.pdf",
            ),
            (
                "Failed to access /api/v1/{tenant_id}/data/document.txt",
                "Failed to access document.txt",
            ),
            (
                "[Errno 2] No such file: '/srv/{user}/storage/data.bin'",
                "[Errno 2] No such file: 'data.bin'",
            ),
            # Case B: Quoted filesystem path bypass
            (
                "Permission denied: '/api/tenant/projects/alpha/current'",
                "Permission denied: 'current'",
            ),
            (
                "Cannot open '/api/v1/internal/config'",
                "Cannot open 'config'",
            ),
            (
                "Error accessing '/api/storage'",
                "Error accessing 'storage'",
            ),
            # Case C: Documentation placeholder bypass
            (
                "Permission denied: /path/to/private/file.pdf",
                "Permission denied: file.pdf",
            ),
            (
                "Failed to read /path/to/my data/report.pdf",
                "Failed to read report.pdf",
            ),
            (
                "[Errno 2] No such file or directory: '/path/to/system/config.json'",
                "[Errno 2] No such file or directory: 'config.json'",
            ),
            (
                "Error accessing /path/to/records",
                "Error accessing records",
            ),
            (
                "Failed to open /path/to/file",
                "Failed to open file",
            ),
            (
                "Failed to open /path/to/dir",
                "Failed to open dir",
            ),
            (
                "Error reading '/path/to/file': permission denied",
                "Error reading 'file': permission denied",
            ),
            (
                "[Errno 13] Permission denied: '/path/to/file'",
                "[Errno 13] Permission denied: 'file'",
            ),
            # Filesystem paths with unfamiliar directory names and extensions
            (
                "Failed to open /custom_mount/archive/blob.xyz",
                "Failed to open blob.xyz",
            ),
            (
                "Permission denied: /srv/node_modules_cache/pkg.bundle",
                "Permission denied: pkg.bundle",
            ),
            (
                "Cannot write to /opt/mycorp/logs/service.trace: disk full",
                "Cannot write to service.trace: disk full",
            ),
            # Filesystem paths mixed with ordinary endpoint descriptions
            (
                "Error in /api/v1/documents endpoint handler: failed to open /var/storage/cache.db",
                "Error in /api/v1/documents endpoint handler: failed to open cache.db",
            ),
            (
                "POST /api/documents/upload failed to write to /mnt/uploads/temp_file.tmp",
                "POST /api/documents/upload failed to write to temp_file.tmp",
            ),
            # Multiple paths in one error message
            (
                "Permission denied for /api/internal/secret.key and /path/to/private/file.pdf",
                "Permission denied for secret.key and file.pdf",
            ),
            (
                (
                    "Pass an /absolute/path to the command. "
                    "However, failed to open /path/to/private/file.pdf."
                ),
                ("Pass an /absolute/path to the command. However, failed to open file.pdf."),
            ),
            (
                (
                    "Check endpoint /api/v1/documents for details, "
                    "but failed to write /var/log/api.log"
                ),
                ("Check endpoint /api/v1/documents for details, but failed to write api.log"),
            ),
        ],
    )
    def test_sanitize_reproduced_bypasses_and_contextual_negatives(
        self, raw: str, expected: str
    ) -> None:
        """Verify that Cases A, B, C, and all contextual negatives are properly sanitized."""
        assert sanitize_path_message(raw) == expected

    @pytest.mark.parametrize(
        "text",
        [
            # URLs
            "See https://example.com/api/v1/query?status=ready for details",
            "Check http://localhost:8000/api/documents/doc_123/chunks",
            "API at http://127.0.0.1:8000/api/documents",
            # API routes and operations (quoted and unquoted)
            "Failed to reach /api/v1/documents",
            "Client error when calling /api/documents",
            "Check endpoint /api/v1/documents for details",
            "Endpoint: /api/v1/documents",
            "Endpoint '/api/v1/documents' returned 404",
            "GET /api/documents/{id}/chunks",
            "GET '/api/documents/{id}/chunks'",
            "POST /api/documents/upload returned 201",
            "Schema at /openapi.json",
            "Docs at /docs",
            "Documentation at /docs",
            "API documentation at /redoc",
            "Check service status at /health",
            "GET /health returned status ok",
            "Query /v1/models endpoint",
            "Route /api/runs was queried",
            # Documentation placeholders in genuine instructional context
            "Pass an /absolute/path to the command",
            "Replace with /path/to/file before running",
            "Replace with /path/to/dir before running",
            "Set directory to /path/to/dir",
            "Example path: /path/to/file",
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

    @pytest.mark.parametrize(
        "raw,expected",
        [
            # Case A: Earlier error must not affect a later route
            (
                "Previous operation: permission denied.\nGET /api/v1/documents",
                "Previous operation: permission denied.\nGET /api/v1/documents",
            ),
            (
                "Previous operation: permission denied. Failed to reach /api/v1/documents",
                "Previous operation: permission denied. Failed to reach /api/v1/documents",
            ),
            # Case B: Earlier error must not affect documentation
            (
                "Previous operation: permission denied.\nExample path: /path/to/file",
                "Previous operation: permission denied.\nExample path: /path/to/file",
            ),
            (
                "Previous operation: permission denied. Example path: /path/to/file",
                "Previous operation: permission denied. Example path: /path/to/file",
            ),
            # Case C: Route and filesystem path in the same message
            (
                "Endpoint: /api/v1/documents returned 404. Failed to open /mnt/private/data.pdf",
                "Endpoint: /api/v1/documents returned 404. Failed to open data.pdf",
            ),
            (
                "Query /api/v1/documents endpoint; error reading /srv/storage/report.pdf",
                "Query /api/v1/documents endpoint; error reading report.pdf",
            ),
            # Case D: Documentation placeholder followed by actual filesystem path
            (
                "Example path: /path/to/file. Permission denied: /path/to/private/file.pdf",
                "Example path: /path/to/file. Permission denied: file.pdf",
            ),
            (
                "Pass an /absolute/path to the command\nFailed to open /var/data/secret.bin",
                "Pass an /absolute/path to the command\nFailed to open secret.bin",
            ),
            # Case E: Multiple independent clauses (newlines, semicolons, periods in filenames)
            (
                (
                    "Previous operation: permission denied.\n"
                    "Check endpoint /api/v1/documents for details.\n"
                    "Failed to open /var/log/app.log"
                ),
                (
                    "Previous operation: permission denied.\n"
                    "Check endpoint /api/v1/documents for details.\n"
                    "Failed to open app.log"
                ),
            ),
            (
                (
                    "Previous operation: permission denied; "
                    "Check endpoint /api/v1/documents for details; "
                    "failed to open /var/log/app.log"
                ),
                (
                    "Previous operation: permission denied; "
                    "Check endpoint /api/v1/documents for details; "
                    "failed to open app.log"
                ),
            ),
            (
                (
                    "GET /api/documents/{id}/chunks succeeded. "
                    "Next, failed to open /custom/path/file.v2.tar.gz."
                ),
                ("GET /api/documents/{id}/chunks succeeded. Next, failed to open file.v2.tar.gz."),
            ),
            (
                "Cannot open /var/data/archive.tar.gz: permission denied. See /docs for manual.",
                "Cannot open archive.tar.gz: permission denied. See /docs for manual.",
            ),
        ],
    )
    def test_scoped_context_isolation(self, raw: str, expected: str) -> None:
        """Verify that filesystem error context is locally scoped to the clause."""
        assert sanitize_path_message(raw) == expected
