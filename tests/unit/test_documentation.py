"""Tests verifying canonical documentation presence and structural consistency."""

from pathlib import Path

REQUIRED_DOCS = [
    "GEMINI.md",
    "docs/PROJECT_BRIEF.md",
    "docs/ARCHITECTURE.md",
    "docs/STATUS.md",
    "docs/DECISIONS.md",
    "docs/roadmap/TASKS.md",
]


def test_canonical_documentation_files_exist():
    """Verify all 6 canonical project documents exist in the repository."""
    root = Path(__file__).resolve().parents[2]
    for rel_path in REQUIRED_DOCS:
        doc_path = root / rel_path
        assert doc_path.exists(), f"Missing required document: {rel_path}"
        assert doc_path.stat().st_size > 0, f"Document is empty: {rel_path}"


def test_tasks_file_contains_task_ids():
    """Verify docs/roadmap/TASKS.md defines expected milestone 0 tasks."""
    root = Path(__file__).resolve().parents[2]
    tasks_content = (root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8")
    for task_id in ["FOUND-001", "FOUND-002", "FOUND-003", "FOUND-004"]:
        assert f"[{task_id}]" in tasks_content, f"Missing task definition for {task_id}"


def test_status_file_references_active_task():
    """Verify docs/STATUS.md references active or next in-flight task."""
    root = Path(__file__).resolve().parents[2]
    status_content = (root / "docs/STATUS.md").read_text(encoding="utf-8")
    assert "FOUND-003" in status_content
    assert "FOUND-002" in status_content


def test_decisions_file_contains_adr_005():
    """Verify docs/DECISIONS.md records ADR-005 for pure standard library core."""
    root = Path(__file__).resolve().parents[2]
    decisions_content = (root / "docs/DECISIONS.md").read_text(encoding="utf-8")
    assert "ADR-005" in decisions_content
