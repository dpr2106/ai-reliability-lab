"""FastAPI REST API adapter for AI Reliability Lab."""

from __future__ import annotations

from reliability_lab.api.main import app, create_app
from reliability_lab.api.middleware import (
    DEFAULT_MAX_REQUEST_BODY_SIZE,
    RequestBodyLimitMiddleware,
    RequestBodyTooLargeError,
)

__all__ = [
    "DEFAULT_MAX_REQUEST_BODY_SIZE",
    "RequestBodyLimitMiddleware",
    "RequestBodyTooLargeError",
    "app",
    "create_app",
]
