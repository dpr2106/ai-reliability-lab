"""Command-line interface for AI Reliability Lab."""

from __future__ import annotations

from reliability_lab.cli.main import (
    EXIT_INCOMPLETE,
    EXIT_OPERATIONAL_ERROR,
    EXIT_PASS,
    EXIT_REGRESSION,
    main,
)

__all__ = [
    "EXIT_INCOMPLETE",
    "EXIT_OPERATIONAL_ERROR",
    "EXIT_PASS",
    "EXIT_REGRESSION",
    "main",
]
