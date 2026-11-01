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

    # 1. Protect URLs by replacing them with opaque tokens
    url_pattern = re.compile(r"https?://[^\s'\"<>()]+", re.IGNORECASE)
    urls: list[str] = []

    def save_url(m: re.Match[str]) -> str:
        urls.append(m.group(0))
        return f"__URL_TOKEN_{len(urls) - 1}__"

    text = url_pattern.sub(save_url, text)

    # 2. Windows drive-letter paths (single/double backslashes or forward slashes)
    win_pattern = re.compile(
        r"[a-zA-Z]:(?:\\\\|[\\/])(?:[^\\\\/\s'\"`:<>\?]+(?:\\\\|[\\/]))*(?:[^\\\\/\s'\"`:<>\?]+)?"
    )

    def replace_win(m: re.Match[str]) -> str:
        matched = m.group(0)
        parts = [p for p in re.split(r"\\\\|[\\/]", matched) if p and not p.endswith(":")]
        return parts[-1] if parts else "[path]"

    text = win_pattern.sub(replace_win, text)

    # 3. Quoted POSIX root or contextual standalone root
    text = re.sub(r"(['\"`])/(['\"`])", r"\1[path]\2", text)
    text = re.sub(
        r"(\b(?:at|in|to|from|root|directory|path)\s+)/(?=[,.;:!?\s'\"`\)\]]|$)",
        r"\1[path]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^/(?=[,.;:!?\s'\"`\)\]]|$)", "[path]", text)

    # 4. POSIX absolute paths with directory/file segments
    posix_pattern = re.compile(r"(?<![a-zA-Z0-9_.-])/(?:[a-zA-Z0-9_.-]+(?:/[a-zA-Z0-9_.-]*)*)")

    def replace_posix(m: re.Match[str]) -> str:
        matched = m.group(0)
        trailing = ""
        while matched.endswith(".") and not matched.endswith(".."):
            trailing = "." + trailing
            matched = matched[:-1]

        parts = [p for p in matched.split("/") if p]
        if not parts:
            return f"[path]{trailing}"
        return f"{parts[-1]}{trailing}"

    text = posix_pattern.sub(replace_posix, text)

    # 5. Restore protected URLs
    for idx, u in enumerate(urls):
        text = text.replace(f"__URL_TOKEN_{idx}__", u)

    return text
