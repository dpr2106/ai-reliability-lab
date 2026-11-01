"""Path privacy and message sanitization utilities."""

from __future__ import annotations

import re


def sanitize_path_message(text: str | None) -> str | None:
    """Sanitize error and diagnostic messages to remove absolute host filesystem paths.

    Replaces identifiable host paths (Windows drive letters, Unix absolute paths)
    with file basenames or redacted placeholders, ensuring internal infrastructure
    locations are never disclosed in client responses.

    Context-Sensitive Boundary:
    - Protects URLs (e.g. 'http://...', 'https://...').
    - Distinguishes API routes and documentation placeholders from host filesystem paths:
      Routes are preserved when introduced by HTTP methods ('GET /api/documents/{id}/chunks'),
      route parameter syntax ('{id}'), route/endpoint markers ('Endpoint: /api/v1/documents'),
      trailing endpoint labels ('/v1/models endpoint'), standalone metadata endpoints
      ('/openapi.json', '/docs', '/redoc', '/health'), or clean API routes without file
      extensions or internal filesystem directories.
    - Eliminates prefix false negatives: paths under /api, /v1, /docs, /health, /redoc
      that contain file extensions (e.g. '/api/private/customer-data/file.pdf'), filesystem
      directory markers (e.g. '/docs/private/storage/'), or are referenced in filesystem
      operations are sanitized to basenames.
    - Preserves status expressions, division ratios, MIME types, and CLI options
      (e.g. 'READY / PARTIAL', '5/10', 'text/plain', '-a/--all').
    - Redacts quoted and unquoted filesystem paths with arbitrary legal filename characters,
      including spaces, Unicode, parentheses, brackets, plus signs, and hashes ('#').
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

    # 2. Context-Sensitive Protection for Genuine API Routes & Documentation Placeholders
    #
    # Distinguishing Route vs. Filesystem Boundary:
    # A slash-prefixed token is a genuine API route or documentation syntax ONLY when:
    # 1) It is explicitly introduced by HTTP methods: GET, POST, PUT, DELETE, PATCH, OPTIONS, HEAD
    # 2) It contains API route parameter syntax: e.g. {id}, {run_id}
    # 3) It is preceded by route/endpoint context keywords: "endpoint", "route", "query"
    # 4) It is followed by "endpoint"
    # 5) It is a recognized documentation placeholder: /absolute/path, /path/to/..., /example/path
    # 6) It is an exact standalone metadata endpoint: /openapi.json, /docs, /redoc, /health
    # 7) It is an unquoted standard API route without file extensions and without
    #    filesystem operation context.
    #
    # A path is NEVER protected as an API route if:
    # - It contains a file extension (.pdf, .txt, .bin, .tmp, etc., except openapi.json)
    # - It contains internal filesystem directory markers: private, storage, secrets,
    #   internal, customer-data, runtime, cache, data
    # - It is preceded by filesystem error/operation verbs: permission denied, failed to open,
    #   no such file, directory, etc.

    protected_tokens: list[str] = []

    def save_protected(token: str) -> str:
        protected_tokens.append(token)
        return f"__PROTECTED_TOKEN_{len(protected_tokens) - 1}__"

    # 2a. HTTP method + route: e.g. GET /api/documents/{id}/chunks, POST /api/documents/upload
    def protect_http_route(m: re.Match[str]) -> str:
        method = m.group(1)
        route = m.group(2)
        return f"{method} {save_protected(route)}"

    text = re.sub(
        r"\b(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+(/(?:api|v[0-9]+|docs|redoc|health|openapi\.json)[^\s'\"`\(\)<>]*)",
        protect_http_route,
        text,
    )

    # 2b. Route parameter syntax in braces: e.g. /api/documents/{id}/chunks
    def protect_brace_route(m: re.Match[str]) -> str:
        return save_protected(m.group(0))

    text = re.sub(
        r"(?<![a-zA-Z0-9_.-])/(?:api|v[0-9]+)/[a-zA-Z0-9_./-]*\{[a-zA-Z0-9_]+\}[a-zA-Z0-9_{}./-]*",
        protect_brace_route,
        text,
    )

    # 2c. Explicit endpoint/route contextual wording:
    # e.g. "Endpoint: /api/v1/documents", "Route /api/runs was queried"
    def protect_contextual_route(m: re.Match[str]) -> str:
        prefix = m.group(1)
        quote = m.group(2) or ""
        route = m.group(3)
        return f"{prefix}{quote}{save_protected(route)}{quote}"

    text = re.sub(
        r"(\b(?:endpoint|route)\s*[:\s]\s*)(['\"`]?)(/(?:api|v[0-9]+)[^\s'\"`\(\)<>]*)\2",
        protect_contextual_route,
        text,
        flags=re.IGNORECASE,
    )

    # 2d. Routes followed by "endpoint": e.g. "Query /v1/models endpoint"
    def protect_trailing_endpoint(m: re.Match[str]) -> str:
        route = m.group(1)
        suffix = m.group(2)
        return f"{save_protected(route)}{suffix}"

    text = re.sub(
        r"(/(?:api|v[0-9]+)[^\s'\"`\(\)<>]*?)(\s+endpoint\b)",
        protect_trailing_endpoint,
        text,
        flags=re.IGNORECASE,
    )

    # 2e. Standalone documentation/meta routes (ONLY when not followed by subpaths):
    # e.g. /openapi.json, /docs, /redoc, /health
    def protect_standalone_meta(m: re.Match[str]) -> str:
        route = m.group(1)
        return save_protected(route)

    text = re.sub(
        r"(?<![a-zA-Z0-9_.-])(/(?:openapi\.json|docs|redoc|health))(?=[,.;:!?\s'\"`\)\]]|$)",
        protect_standalone_meta,
        text,
    )

    # 2f. Documentation placeholders: e.g. /absolute/path, /path/to/file, /path/to/dir
    def protect_doc_placeholders(m: re.Match[str]) -> str:
        placeholder = m.group(0)
        return save_protected(placeholder)

    text = re.sub(
        r"(?<![a-zA-Z0-9_.-])/(?:absolute/path|path/to(?:/[a-zA-Z0-9_.-]+)*|example/path)\b",
        protect_doc_placeholders,
        text,
    )

    # 2g. Clean unquoted API routes without file extensions or filesystem operation markers:
    # e.g. "Failed to reach /api/v1/documents", "Client error when calling /api/documents"
    fs_indicator_pattern = re.compile(
        r"\.(?:pdf|txt|bin|json|csv|sqlite3|db|index|faiss|tmp|log|tar|gz|zip|xml|html|py|sh|env)\b|"
        r"/(?:private|storage|secrets|internal|customer-data|runtime|cache|data)\b",
        re.IGNORECASE,
    )

    def protect_clean_api_route(m: re.Match[str]) -> str:
        prefix = m.group(1) or ""
        route_part = m.group(2)
        # If preceded by directory/file/folder or has filesystem indicators, do not protect
        if prefix or fs_indicator_pattern.search(route_part):
            return m.group(0)
        return save_protected(route_part)

    # Match /api/... or /v1/... not preceded by alphanumeric or filesystem words
    text = re.sub(
        r"(\b(?:directory|file|folder)\s+)?(?<![a-zA-Z0-9_.-])(/(?:api|v[0-9]+)/[a-zA-Z0-9_/-]+)",
        protect_clean_api_route,
        text,
        flags=re.IGNORECASE,
    )

    # 3. Quoted POSIX & Windows paths: e.g. '/path/...', "/path/...", `C:\...`
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
    # 2) Paths with filename (including spaces/Unicode/parens/#/[]/+) and extension
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

    # 7. Restore protected tokens and URLs
    for idx, token in enumerate(protected_tokens):
        text = text.replace(f"__PROTECTED_TOKEN_{idx}__", token)

    for idx, u in enumerate(urls):
        text = text.replace(f"__URL_TOKEN_{idx}__", u)

    return text
