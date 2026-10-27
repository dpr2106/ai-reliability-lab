"""Tests verifying canonical documentation presence and structural consistency."""

import re
from pathlib import Path

REQUIRED_DOCS = [
    "GEMINI.md",
    "docs/PROJECT_BRIEF.md",
    "docs/ARCHITECTURE.md",
    "docs/STATUS.md",
    "docs/DECISIONS.md",
    "docs/roadmap/TASKS.md",
]

VALID_STATUSES = {"BACKLOG", "READY", "IN_PROGRESS", "IN_REVIEW", "DONE", "BLOCKED"}
REQUIRED_TASK_SECTIONS = [
    "Objective",
    "Priority",
    "Dependencies",
    "Scope / Expected Files",
    "Acceptance Criteria",
    "Verification Command",
    "Status",
    "Verification Evidence",
]


def test_canonical_documentation_files_exist():
    """Verify all 6 canonical project documents exist in the repository."""
    root = Path(__file__).resolve().parents[2]
    for rel_path in REQUIRED_DOCS:
        doc_path = root / rel_path
        assert doc_path.exists(), f"Missing required document: {rel_path}"
        assert doc_path.stat().st_size > 0, f"Document is empty: {rel_path}"


def test_tasks_file_structure_and_invariants():
    """Verify docs/roadmap/TASKS.md adheres to task structure and evidence invariants."""
    root = Path(__file__).resolve().parents[2]
    tasks_content = (root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8")

    task_blocks = re.findall(
        r"### \[([A-Z]+-\d+)\][^\n]+(.*?)(?=\n### \[|\Z)", tasks_content, re.DOTALL
    )
    assert len(task_blocks) >= 4, f"Found only {len(task_blocks)} task blocks in TASKS.md"

    found_ids = set()
    for task_id, block in task_blocks:
        found_ids.add(task_id)
        for section in REQUIRED_TASK_SECTIONS:
            assert f"**{section}**" in block, f"Task {task_id} missing section: {section}"

        status_match = re.search(r"\*\*Status\*\*:\s*`([^`]+)`", block)
        assert status_match is not None, f"Task {task_id} missing backticked Status value"
        status_value = status_match.group(1).strip()
        assert status_value in VALID_STATUSES, (
            f"Task {task_id} has invalid status: '{status_value}'"
        )

        if status_value == "DONE":
            evidence_match = re.search(r"\*\*Verification Evidence\*\*:\s*([^\n]+)", block)
            assert evidence_match is not None, f"Task {task_id} marked DONE must record evidence"
            evidence_first_line = evidence_match.group(1).strip()
            assert evidence_first_line.lower() not in {"none", "pending"}, (
                f"Task {task_id} marked DONE cannot have empty/pending evidence"
            )

    for required_id in ["FOUND-001", "FOUND-002", "FOUND-003", "FOUND-004"]:
        assert required_id in found_ids, f"Required task {required_id} missing from TASKS.md"


def test_status_file_structural_invariants():
    """Verify docs/STATUS.md maintains a durable baseline without rotting branch state."""
    root = Path(__file__).resolve().parents[2]
    status_content = (root / "docs/STATUS.md").read_text(encoding="utf-8")
    tasks_content = (root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8")

    assert "## Verified Operational State" in status_content
    assert "## PR Review & Gating Architecture State" in status_content
    assert "## Next In-Flight Deliverable" in status_content

    assert "- **Base Branch**: `main`" in status_content
    assert not re.search(r"\*\*Active Branch\*\*:\s*`(feat|chore|fix)/", status_content), (
        "STATUS.md must not store ephemeral feature branch names that rot upon merge"
    )

    sha_match = re.search(r"Commit SHA:\s*`([0-9a-f]{40})`", status_content)
    assert sha_match is not None, "STATUS.md must record a valid 40-character commit SHA"

    next_task_match = re.search(r"- `([A-Z]+-\d+)`", status_content)
    assert next_task_match is not None, "STATUS.md must point to a specific next task ID"
    next_task_id = next_task_match.group(1)

    task_block_match = re.search(
        rf"### \[{next_task_id}\][^\n]+(.*?)(?=\n### \[|\Z)", tasks_content, re.DOTALL
    )
    assert task_block_match is not None, f"Next task {next_task_id} not found in TASKS.md"
    task_block = task_block_match.group(1)

    status_match = re.search(r"\*\*Status\*\*:\s*`([^`]+)`", task_block)
    assert status_match is not None
    assert status_match.group(1).strip() in {"READY", "IN_PROGRESS"}, (
        f"Next task {next_task_id} must have status READY or IN_PROGRESS in TASKS.md"
    )


def test_decisions_file_contains_approved_adrs():
    """Verify docs/DECISIONS.md records all active ADRs from ADR-001 through ADR-006."""
    root = Path(__file__).resolve().parents[2]
    decisions_content = (root / "docs/DECISIONS.md").read_text(encoding="utf-8")
    for i in range(1, 7):
        adr_id = f"ADR-{i:03d}"
        assert adr_id in decisions_content, f"Missing {adr_id} in docs/DECISIONS.md"
