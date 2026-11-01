"""Path privacy and message sanitization utilities."""

from __future__ import annotations

import re


def sanitize_path_message(text: str | None) -> str | None:
    """Sanitize error and diagnostic messages to remove absolute host filesystem paths.

    Replaces identifiable host paths (Windows drive letters, Unix absolute paths)
    with file basenames or redacted placeholders, ensuring internal infrastructure
    locations are never disclosed in client responses.

    Contextual Path Boundary:
    - Preserves URLs (e.g. 'http://...', 'https://...').
    - Preserves API routes (e.g. '/api/v1/documents', '/openapi.json', '/docs', '/health').
    - Preserves documentation placeholders (e.g. '/absolute/path', '/path/to/file').
    - Preserves status expressions, division ratios, MIME types, and CLI options
      (e.g. 'READY / PARTIAL', '5/10', 'text/plain', '-a/--all').
    - Redacts quoted and unquoted filesystem paths with arbitrary legal filename characters,
      including spaces, Unicode, parentheses, and hashes ('#').
    - Replaces standalone root paths ('/', 'C:\\', 'C:/') with '[path]'.
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

    # 2. Protect API routes and documentation placeholders
    api_doc_pattern = re.compile(
        r"(?<![a-zA-Z0-9_.-])/(?:(?:api|v[0-9]+|openapi\.json|docs|redoc|health)\b(?:/[^\s'\"`\(\)<>]+)*|(?:absolute/path|path/to(?:/[a-zA-Z0-9_.-]+)*|example/path)\b)"
    )
    api_doc_tokens: list[str] = []

    def save_api_doc(m: re.Match[str]) -> str:
        api_doc_tokens.append(m.group(0))
        return f"__API_DOC_TOKEN_{len(api_doc_tokens) - 1}__"

    text = api_doc_pattern.sub(save_api_doc, text)

    # 3. Quoted POSIX & Windows paths: e.g. '/path/...', "/path/...", `C:\...`
    # Allows spaces, Unicode, parentheses, #, etc. inside matching quotes
    quoted_path_pattern = re.compile(r"(['\"`])((?:[a-zA-Z]:(?:\\\\|[\\/])|/)[^'\"`\r\n]*)\1")

    def replace_quoted(m: re.Match[str]) -> str:
        quote = m.group(1)
        raw_path = m.group(2)
        if re.match(r"^[a-zA-Z]:", raw_path):
            parts = [p for p in re.split(r"\\\\|[\\/]", raw_path) if p and not p.endswith(":")]
            name = parts[-1] if parts else "[path]"
            return f"{quote}{name}{quote}"
        parts = [p for p in raw_path.split("/") if p]
        name = parts[-1] if parts else "[path]"
        return f"{quote}{name}{quote}"

    text = quoted_path_pattern.sub(replace_quoted, text)

    # 4. Standalone & contextual root paths: e.g. "at /", "root /", "path /", "C:\", "C:/"
    text = re.sub(
        r"(\b(?:at|in|to|from|root|directory|path)\s+)/(?=[,.;:!?\s'\"`\)\]]|$)",
        r"\1[path]",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"^/(?=[,.;:!?\s'\"`\)\]]|$)", "[path]", text)
    text = re.sub(r"\b[a-zA-Z]:[\\/](?=[,.;:!?\s'\"`\)\]]|$)", "[path]", text)

    # 5. Unquoted Windows paths: e.g. C:\Users\admin\report.pdf or D:/data/doc.bin
    win_unquoted_pattern = re.compile(
        r"\b[a-zA-Z]:(?:\\\\|[\\/])(?:(?![^\\\\/\r\n]*\.[a-zA-Z0-9]{1,5}\s)[^\\\\/\r\n'\"`:<>\?]+(?:\\\\|[\\/]))*(?:[^\\\\/\r\n'\"`:<>\?\s,;!)]+(?:\s+[^\\\\/\r\n'\"`:<>\?\s,;!)]+)*\.[a-zA-Z0-9]+|[^\\\\/\r\n'\"`:<>\?\s,;!)]+)"
    )

    def replace_win_unquoted(m: re.Match[str]) -> str:
        matched = m.group(0)
        parts = [p for p in re.split(r"\\\\|[\\/]", matched) if p and not p.endswith(":")]
        return parts[-1] if parts else "[path]"

    text = win_unquoted_pattern.sub(replace_win_unquoted, text)

    # 6. Unquoted POSIX absolute paths
    # Matches:
    # 1) Directories ending with a slash: /(?:dir/)+(?=[\s,;!?\'\"\`\)\]]|$)
    # 2) Paths with filename (including spaces/Unicode/parens/#) and extension
    # 3) Directory path without extension: /(?:dir/)*segment
    posix_unquoted_pattern = re.compile(
        r"(?<![a-zA-Z0-9_.-])(?:/(?:(?![^/\r\n]*\.[a-zA-Z0-9]{1,5}\s)[^/\\:\r\n'\"`<>]+/)+(?=[\s,;!?\'\"\`\)\]]|$)|/(?:(?![^/\r\n]*\.[a-zA-Z0-9]{1,5}\s)[^/\\:\r\n'\"`<>]+/)*(?:[^/\\:\r\n'\"`<>\s,;!?\)]+(?:\s+[^/\\:\r\n'\"`<>\s,;!?\)]+)*\.[a-zA-Z0-9]+|[^/\\:\r\n'\"`<>\s,;!?\)]+))"
    )

    def replace_posix_unquoted(m: re.Match[str]) -> str:
        matched = m.group(0)
        trailing = ""
        while matched.endswith(".") and not matched.endswith(".."):
            trailing = "." + trailing
            matched = matched[:-1]

        parts = [p for p in matched.split("/") if p]
        if not parts:
            return f"[path]{trailing}"
        return f"{parts[-1]}{trailing}"

    text = posix_unquoted_pattern.sub(replace_posix_unquoted, text)

    # 7. Restore protected API routes, documentation placeholders, and URLs
    for idx, token in enumerate(api_doc_tokens):
        text = text.replace(f"__API_DOC_TOKEN_{idx}__", token)

    for idx, u in enumerate(urls):
        text = text.replace(f"__URL_TOKEN_{idx}__", u)

    return text
