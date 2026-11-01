"""ASGI middleware for request size enforcement and streaming boundary protection."""

from __future__ import annotations

import json
import logging
import os

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

# Default request-body size: 50MB (default file limit) + 5MB multipart/headers overhead = 55MB
DEFAULT_MAX_REQUEST_BODY_SIZE: int = 55 * 1024 * 1024


class RequestBodyTooLargeError(StarletteHTTPException):
    """Raised when an incoming HTTP request body exceeds the configured maximum size."""

    def __init__(self, detail: str) -> None:
        super().__init__(status_code=413, detail=detail)


class RequestBodyLimitMiddleware:
    """ASGI middleware enforcing a strict maximum request body size limit.

    Protects against denial-of-service, request-body amplification, and memory
    exhaustion attacks by:
    1. Inspecting Content-Length header early: Rejects oversized requests with HTTP 413
       before parsing or consuming any request body chunks.
    2. Enforcing cumulative byte limits on streaming/chunked requests: Wraps the ASGI
       `receive` callable to track bytes received across all incoming `http.request`
       messages. If the cumulative byte count exceeds `max_body_size`, immediately
       aborts processing and sends an HTTP 413 response without buffering or reading
       the remainder of the payload.
    """

    def __init__(
        self,
        app: ASGIApp,
        max_body_size: int | None = None,
    ) -> None:
        self.app = app
        if max_body_size is not None:
            self.max_body_size = max_body_size
        else:
            env_val = os.environ.get("RELIABILITY_LAB_MAX_REQUEST_BODY_SIZE")
            if env_val and env_val.strip().isdigit():
                self.max_body_size = int(env_val.strip())
            else:
                self.max_body_size = DEFAULT_MAX_REQUEST_BODY_SIZE

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        # 1. Early Content-Length header inspection
        headers = dict(scope.get("headers", []))
        content_length_raw = headers.get(b"content-length")
        if content_length_raw is not None:
            try:
                content_length = int(content_length_raw.decode("latin-1"))
                if content_length > self.max_body_size:
                    logger.warning(
                        "Rejecting request early: Content-Length %d exceeds max %d bytes",
                        content_length,
                        self.max_body_size,
                    )
                    await self._send_413(
                        send,
                        (
                            f"Request payload exceeds maximum size limit "
                            f"({content_length} bytes > {self.max_body_size} bytes limit)."
                        ),
                    )
                    return
            except (ValueError, UnicodeDecodeError):
                pass

        # 2. Streaming byte count enforcement via wrapped receive
        bytes_received = 0
        response_started = False

        async def limited_receive() -> Message:
            nonlocal bytes_received
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                bytes_received += len(body)
                if bytes_received > self.max_body_size:
                    logger.warning(
                        "Aborting request: cumulative body size %d exceeds max %d bytes",
                        bytes_received,
                        self.max_body_size,
                    )
                    raise RequestBodyTooLargeError(
                        f"Cumulative request body exceeded maximum size limit "
                        f"({bytes_received} bytes > {self.max_body_size} bytes limit)."
                    )
            return message

        async def monitored_send(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, limited_receive, monitored_send)
        except RequestBodyTooLargeError as exc:
            if not response_started:
                await self._send_413(send, exc.detail if hasattr(exc, "detail") else str(exc))

    @staticmethod
    async def _send_413(send: Send, detail: str) -> None:
        payload = json.dumps({"detail": detail}).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("latin-1")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": payload,
                "more_body": False,
            }
        )
