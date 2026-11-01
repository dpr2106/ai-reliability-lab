"""Path privacy and message sanitization utilities."""

from __future__ import annotations

import re


def sanitize_path_message(text: str | None) -> str | None:
    """Sanitize error and diagnostic messages to remove absolute host filesystem paths.

    Replaces identifiable host paths (Windows drive letters, Unix absolute paths)
    with file basenames or redacted placeholders, ensuring internal infrastructure
    locations are never disclosed in client responses.
    """
    if not text:
        return text

    # Pattern for Windows drive-letter paths (single or double backslashes, forward slashes)
    win_pattern = re.compile(
        r"[a-zA-Z]:(?:\\\\|[\\/])(?:[^\\/\s'\":<>\?]+(?:\\\\|[\\/]))*(?:[^\\/\s'\":<>\?]+)?"
    )

    def replace_win(m: re.Match[str]) -> str:
        matched = m.group(0)
        parts = [p for p in re.split(r"\\\\|[\\/]", matched) if p and not p.endswith(":")]
        return parts[-1] if parts else "[path]"

    cleaned = win_pattern.sub(replace_win, text)

    # Pattern for POSIX absolute directory paths: e.g. /home/..., /tmp/..., /Users/...
    posix_pattern = re.compile(
        r"(?:^|[\s'\":`\(])(/(?:Users|home|tmp|var|etc|app|root|opt|private)[^'\":\s`\)]*)"
    )

    def replace_posix(m: re.Match[str]) -> str:
        matched = m.group(1)
        parts = [p for p in matched.split("/") if p]
        return m.group(0).replace(matched, parts[-1] if parts else "[path]")

    cleaned = posix_pattern.sub(replace_posix, cleaned)
    return cleaned
