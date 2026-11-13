"""Regression test wrapping the hermetic demo smoke workflow (V2-006.2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from scripts.verify_sting_demo import run_hermetic_demo_smoke_test


@pytest.fixture
def demo_dir() -> Path:
    """Return path to data/demo directory."""
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "demo"


def test_hermetic_demo_smoke_workflow(tmp_path: Path, demo_dir: Path) -> None:
    """Execute complete 16-step hermetic demo smoke workflow in isolated temp directory."""
    result = run_hermetic_demo_smoke_test(demo_dir=demo_dir, temp_dir_path=tmp_path)

    assert result["status"] == "passed"
    assert len(result["documents"]) == 3
    assert result["query_count"] == 10
    assert result["evidence_anchors"] == 9
    assert result["abstentions"] == 1
    assert result["run_id"] != ""
    assert result["job_id"] != ""
    assert result["reproducibility_tier"] in ("exact", "partial")
