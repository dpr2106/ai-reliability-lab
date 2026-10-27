"""Global pytest configuration and shared test fixtures."""

import pytest


@pytest.fixture
def sample_text() -> str:
    """Fixture providing sample document text for tests."""
    return "AI Reliability Lab provides deterministic evaluation for document AI."
