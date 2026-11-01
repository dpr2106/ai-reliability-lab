"""Path privacy and message sanitization utilities."""

from __future__ import annotations

import re


def sanitize_path_message(text: str | None) -> str | None:
    """Sanitize error and diagnostic messages to remove absolute host filesystem paths.

    Replaces identifiable host paths (Windows drive letters, Unix absolute paths)
    with file basenames or redacted placeholders, ensuring internal infrastructure
    locations are never disclosed in client responses.

    Context-Sensitive Boundary Principles:
    1. Explicit filesystem errors and operations take priority over route or
       documentation matching.
    2. Quoted filesystem paths are sanitized when filesystem context identifies them
       as paths. Quoted strings are preserved as API routes only when introduced by
       an HTTP method ('GET /...') or explicit endpoint keyword ('Endpoint: /...').
    3. Brace parameters alone ('{id}') never exempt a path from sanitization.
    4. Documentation placeholders ('/absolute/path', '/path/to/file', '/path/to/dir')
       are protected only in genuine documentation or instructional context and never
       when introduced by filesystem error verbs.
    5. Genuine API routes remain intact in HTTP methods, explicit endpoint labels,
       standalone metadata routes ('/openapi.json', '/docs', '/redoc', '/health'),
       and clean unquoted client/network call messages.
    6. When context is ambiguous, prioritize privacy and redact directory components.
    """
    if not text:
        return text

    current_text: str = text

    # 1. Protect URLs by replacing them with opaque tokens
    url_pattern = re.compile(r"https?://[^\s'\"<>()]+", re.IGNORECASE)
    urls: list[str] = []

    def save_url(m: re.Match[str]) -> str:
        urls.append(m.group(0))
        return f"__URL_TOKEN_{len(urls) - 1}__"

    current_text = url_pattern.sub(save_url, current_text)

    # 2. Context-Sensitive Protection for Genuine API Routes & Documentation Placeholders
    protected_tokens: list[str] = []

    def save_protected(token: str) -> str:
        protected_tokens.append(token)
        return f"__PROTECTED_TOKEN_{len(protected_tokens) - 1}__"

    # Regex detecting explicit filesystem error or operation context
    fs_error_context_pattern = re.compile(
        r"\b(?:permission denied|"
        r"failed to (?:open|read|write|delete|access|create|remove|copy|find|stat)|"
        r"cannot (?:open|read|write|delete|access|create|remove)|"
        r"error (?:opening|reading|writing|deleting|accessing|creating|removing|in)|"
        r"no such file|file not found|directory not found|does not exist|\[errno \d+\])\b",
        re.IGNORECASE,
    )

    # 2a. HTTP method + route (quoted or unquoted):
    # e.g. GET /api/documents/{id}/chunks, GET '/api/documents/{id}/chunks', POST /api/upload
    def protect_http_route(m: re.Match[str]) -> str:
        method = m.group(1)
        quote = m.group(2) or ""
        route = m.group(3)
        return f"{method} {quote}{save_protected(route)}{quote}"

    current_text = re.sub(
        r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+"
        r"(['\"`]?)(/(?:api|v[0-9]+|docs|redoc|health|openapi\.json)[^\s'\"`\(\)<>]*)\2",
        protect_http_route,
        current_text,
    )

    # 2b. Explicit endpoint/route contextual wording:
    # e.g. "Endpoint: /api/v1/documents", "Endpoint '/api/v1/documents'", "Route /api/runs"
    def protect_contextual_route(m: re.Match[str]) -> str:
        prefix = m.group(1)
        quote = m.group(2) or ""
        route = m.group(3)
        # Never protect as endpoint if it looks like a file with extension
        if re.search(r"\.[a-zA-Z0-9]{1,5}$", route):
            return m.group(0)
        return f"{prefix}{quote}{save_protected(route)}{quote}"

    current_text = re.sub(
        r"(\b(?:endpoint|route)\s*[:\s]\s*)(['\"`]?)(/(?:api|v[0-9]+)[^\s'\"`\(\)<>]*)\2",
        protect_contextual_route,
        current_text,
        flags=re.IGNORECASE,
    )

    # 2c. Routes followed by "endpoint": e.g. "Query /v1/models endpoint"
    def protect_trailing_endpoint(m: re.Match[str]) -> str:
        route = m.group(1)
        suffix = m.group(2)
        if re.search(r"\.[a-zA-Z0-9]{1,5}$", route):
            return m.group(0)
        return f"{save_protected(route)}{suffix}"

    current_text = re.sub(
        r"(/(?:api|v[0-9]+)[^\s'\"`\(\)<>]*?)(\s+endpoint\b)",
        protect_trailing_endpoint,
        current_text,
        flags=re.IGNORECASE,
    )

    # 2d. Standalone documentation/meta routes (ONLY when not followed by subpaths):
    # e.g. /openapi.json, /docs, /redoc, /health
    def protect_standalone_meta(m: re.Match[str]) -> str:
        route = m.group(1)
        return save_protected(route)

    current_text = re.sub(
        r"(?<![a-zA-Z0-9_.-])(/(?:openapi\.json|docs|redoc|health))(?=[,.;:!?\s'\"`\)\]]|$)",
        protect_standalone_meta,
        current_text,
    )

    # 2e. Documentation placeholders: e.g. /absolute/path, /path/to/file, /path/to/dir
    # Protected ONLY when not in a filesystem error context, and limited to canonical examples
    def protect_doc_placeholders(m: re.Match[str]) -> str:
        start_pos = m.start()
        prefix_context = current_text[:start_pos]
        if fs_error_context_pattern.search(prefix_context):
            return m.group(0)  # In filesystem error context: do not protect!
        placeholder = m.group(0)
        return save_protected(placeholder)

    current_text = re.sub(
        r"(?<![a-zA-Z0-9_.-])/(?:absolute/path|path/to/(?:file|dir|directory)|example/path)\b",
        protect_doc_placeholders,
        current_text,
    )

    # 2f. Clean unquoted API routes in client/network messages:
    # e.g. "Failed to reach /api/v1/documents", "Client error when calling /api/documents"
    fs_indicator_pattern = re.compile(
        r"\.[a-zA-Z0-9]{1,5}\b|"
        r"/(?:private|storage|secrets|internal|customer-data|runtime|cache|data|tenant|projects)\b|"
        r"\{[a-zA-Z0-9_]+\}",
        re.IGNORECASE,
    )

    def protect_clean_api_route(m: re.Match[str]) -> str:
        route_part = m.group(1)
        start_pos = m.start()
        prefix_context = current_text[max(0, start_pos - 60) : start_pos]
        if fs_error_context_pattern.search(prefix_context):
            return m.group(0)
        if re.search(r"\b(?:directory|file|folder)\s*$", prefix_context, re.IGNORECASE):
            return m.group(0)
        if fs_indicator_pattern.search(route_part):
            return m.group(0)
        return save_protected(route_part)

    # Strictly UNQUOTED: must not be preceded or followed by quotes
    current_text = re.sub(
        r"(?<!['\"`a-zA-Z0-9_.-])(/(?:api|v[0-9]+)/[a-zA-Z0-9_/-]+)(?!['\"`])",
        protect_clean_api_route,
        current_text,
    )

    # 3. Quoted POSIX & Windows paths: e.g. '/path/...', "/path/...", `C:\...`
    # Allows spaces, Unicode, parentheses, #, braces, etc. inside matching quotes
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

    current_text = quoted_path_pattern.sub(replace_quoted, current_text)

    # 4. Standalone & contextual root paths: e.g. "at /", "root /", "path /", "C:\", "C:/"
    current_text = re.sub(
        r"(\b(?:at|in|to|from|root|directory|path)\s+)/(?=[,.;:!?\s'\"`\)\]]|$)",
        r"\1[path]",
        current_text,
        flags=re.IGNORECASE,
    )
    current_text = re.sub(r"^/(?=[,.;:!?\s'\"`\)\]]|$)", "[path]", current_text)
    current_text = re.sub(r"\b[a-zA-Z]:[\\/](?=[,.;:!?\s'\"`\)\]]|$)", "[path]", current_text)

    # 5. Unquoted Windows paths: e.g. C:\Users\admin\report.pdf or D:/data/doc.bin
    win_unquoted_pattern = re.compile(
        r"\b[a-zA-Z]:(?:\\\\|[\\/])(?:(?![^\\\\/\r\n]*\.[a-zA-Z0-9]{1,5}\s)"
        r"[^\\\\/\r\n'\"`:<>\?]+(?:\\\\|[\\/]))*"
        r"(?:[^\\\\/\r\n'\"`:<>\?\s,;!)]+(?:\s+[^\\\\/\r\n'\"`:<>\?\s,;!)]+)*\.[a-zA-Z0-9]+|"
        r"[^\\\\/\r\n'\"`:<>\?\s,;!)]+)"
    )

    def replace_win_unquoted(m: re.Match[str]) -> str:
        matched = m.group(0)
        parts = [p for p in re.split(r"\\\\|[\\/]", matched) if p and not p.endswith(":")]
        return parts[-1] if parts else "[path]"

    current_text = win_unquoted_pattern.sub(replace_win_unquoted, current_text)

    # 6. Unquoted POSIX absolute paths
    posix_unquoted_pattern = re.compile(
        r"(?<![a-zA-Z0-9_.-])(?:/(?:(?![^/\r\n]*\.[a-zA-Z0-9]{1,5}\s)"
        r"[^/\\:\r\n'\"`<>]+/)+(?=[\s,;!?\'\"\`\)\]]|$)|"
        r"/(?:(?![^/\r\n]*\.[a-zA-Z0-9]{1,5}\s)[^/\\:\r\n'\"`<>]+/)*"
        r"(?:[^/\\:\r\n'\"`<>\s,;!?\)]+(?:\s+[^/\\:\r\n'\"`<>\s,;!?\)]+)*\.[a-zA-Z0-9]+|"
        r"[^/\\:\r\n'\"`<>\s,;!?\)]+))"
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

    current_text = posix_unquoted_pattern.sub(replace_posix_unquoted, current_text)

    # 7. Restore protected tokens and URLs
    for idx, token in enumerate(protected_tokens):
        current_text = current_text.replace(f"__PROTECTED_TOKEN_{idx}__", token)

    for idx, u in enumerate(urls):
        current_text = current_text.replace(f"__URL_TOKEN_{idx}__", u)

    return current_text
