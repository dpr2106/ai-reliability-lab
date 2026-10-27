"""Smoke tests verifying test runner configuration and package metadata."""

import reliability_lab


def test_package_version():
    """Verify package version is defined."""
    assert reliability_lab.__version__ == "0.1.0"


def test_fixture_loading(sample_text: str):
    """Verify shared pytest fixtures load properly."""
    assert "AI Reliability Lab" in sample_text
