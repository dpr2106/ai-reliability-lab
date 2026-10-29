"""FastAPI REST API adapter for AI Reliability Lab."""

from __future__ import annotations

from reliability_lab.api.main import app, create_app

__all__ = ["app", "create_app"]
