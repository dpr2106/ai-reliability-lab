"""Tests verifying canonical documentation presence and structural consistency."""

import re
from pathlib import Path

import pytest

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
PLACEHOLDER_EVIDENCE = ("none", "pending", "tbd", "todo", "placeholder", "not yet", "to be added")


def _parse_tasks(tasks_content: str) -> dict[str, tuple[str, str]]:
    blocks = re.findall(
        r"^### \[([A-Z]+-\d+)\][^\n]*\n(.*?)(?=^### \[|\Z)",
        tasks_content,
        re.DOTALL | re.MULTILINE,
    )
    assert blocks, "No task blocks found in TASKS.md"

    tasks: dict[str, tuple[str, str]] = {}
    for task_id, block in blocks:
        assert task_id not in tasks, f"Duplicate task ID: {task_id}"
        for section in REQUIRED_TASK_SECTIONS:
            assert f"**{section}**" in block, f"Task {task_id} missing section: {section}"

        status_lines = re.findall(r"^\s*- \*\*Status\*\*:\s*`([^`]+)`\s*$", block, re.MULTILINE)
        assert len(status_lines) == 1, f"Task {task_id} must have exactly one Status field"
        status_value = status_lines[0].strip()
        assert status_value in VALID_STATUSES, (
            f"Task {task_id} has invalid status: '{status_value}'"
        )

        evidence_match = re.search(
            r"\*\*Verification Evidence\*\*:\s*(.*?)(?=\n---\s*$|\Z)",
            block,
            re.DOTALL | re.MULTILINE,
        )
        assert evidence_match is not None, f"Task {task_id} missing verification evidence"
        evidence = evidence_match.group(1).strip()
        if status_value == "DONE":
            assert evidence, f"Task {task_id} marked DONE must record evidence"
            lowered_evidence = evidence.lower()
            assert not any(
                placeholder in lowered_evidence for placeholder in PLACEHOLDER_EVIDENCE
            ), f"Task {task_id} marked DONE has placeholder evidence"
        tasks[task_id] = (status_value, evidence)
    return tasks


def _extract_pointer(status_content: str, label: str) -> str:
    matches = re.findall(
        rf"^\s*- \*\*{re.escape(label)}\*\*:\s*`([^`]+)`\s*$", status_content, re.MULTILINE
    )
    assert len(matches) == 1, f"STATUS.md must contain exactly one {label} pointer"
    return matches[0]


def _validate_status_snapshot(tasks_content: str, status_content: str) -> None:
    tasks = _parse_tasks(tasks_content)
    active_ids = re.findall(
        r"^\s*- \*\*Active Task\*\*:\s*`([^`]+)`\s*$", status_content, re.MULTILINE
    )
    assert len(active_ids) <= 1, "STATUS.md may contain at most one Active Task pointer"
    active_id = active_ids[0] if active_ids else None
    next_id = _extract_pointer(status_content, "Next READY Task")
    assert active_id is None or active_id != next_id, "Active and next task pointers conflict"

    pointers = [("Next READY Task", next_id)]
    if active_id is not None:
        pointers.append(("Active Task", active_id))
    for label, task_id in pointers:
        assert task_id in tasks, f"{label} {task_id} does not exist in TASKS.md"
        task_status = tasks[task_id][0]
        assert task_status != "DONE", f"Completed task {task_id} cannot be listed as {label}"

    if active_id is not None:
        assert tasks[active_id][0] in {"IN_PROGRESS", "IN_REVIEW"}, (
            f"Active task {active_id} has incompatible status {tasks[active_id][0]}"
        )
    assert tasks[next_id][0] == "READY", f"Next task {next_id} must be READY in TASKS.md"
    assert not re.search(r"Commit SHA:\s*`[0-9a-f]{40}`", status_content), (
        "STATUS.md must not embed a transient purported latest merge SHA"
    )


def test_canonical_documentation_files_exist():
    """Verify all canonical project documents exist and are non-empty."""
    root = Path(__file__).resolve().parents[2]
    for rel_path in REQUIRED_DOCS:
        doc_path = root / rel_path
        assert doc_path.exists(), f"Missing required document: {rel_path}"
        assert doc_path.stat().st_size > 0, f"Document is empty: {rel_path}"


def test_tasks_file_structure_and_invariants():
    """Verify all tasks have complete fields, valid statuses, and DONE evidence."""
    root = Path(__file__).resolve().parents[2]
    tasks = _parse_tasks((root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8"))
    for required_id in ["FOUND-001", "FOUND-002", "FOUND-003", "FOUND-004", "FOUND-005"]:
        assert required_id in tasks, f"Required task {required_id} missing from TASKS.md"


def test_status_file_structural_invariants():
    """Verify status pointers resolve to compatible canonical task states."""
    root = Path(__file__).resolve().parents[2]
    status_content = (root / "docs/STATUS.md").read_text(encoding="utf-8")
    tasks_content = (root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8")

    assert "## Verified Operational State" in status_content
    assert "## PR Review & Gating Architecture State" in status_content
    assert "## Task Pointers" in status_content
    assert "- **Base Branch**: `main`" in status_content
    assert (
        "empty webhooks API response does not establish that no GitHub Apps are installed"
        in status_content
    )
    assert "Copilot could not review PR #3 because its review quota was exhausted" in status_content
    _validate_status_snapshot(tasks_content, status_content)


def test_documentation_validator_rejects_invalid_references_and_status_conflicts():
    """Exercise pointer validation against unknown, duplicate, and completed tasks."""
    root = Path(__file__).resolve().parents[2]
    tasks = (root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8")
    status = (root / "docs/STATUS.md").read_text(encoding="utf-8")
    next_task = _extract_pointer(status, "Next READY Task")
    invalid_snapshots = [
        status.replace(f"Next READY Task**: `{next_task}`", "Next READY Task**: `FOUND-001`", 1),
        status.replace(f"Next READY Task**: `{next_task}`", "Next READY Task**: `FOUND-999`", 1),
        status + "\n- **Active Task**: `FOUND-001`\n",
        status + "\n- **Active Task**: `FOUND-999`\n",
        status + f"\n- **Active Task**: `{next_task}`\n- **Active Task**: `FOUND-004`\n",
    ]
    for invalid_status in invalid_snapshots:
        with pytest.raises(AssertionError):
            _validate_status_snapshot(tasks, invalid_status)


def test_documentation_validator_rejects_invalid_task_fields_and_done_evidence():
    """Exercise task parsing against missing fields, invalid states, and placeholder evidence."""
    root = Path(__file__).resolve().parents[2]
    tasks = (root / "docs/roadmap/TASKS.md").read_text(encoding="utf-8")
    task_005_match = re.search(
        r"^### \[FOUND-005\].*?(?=^### \[|\Z)", tasks, re.DOTALL | re.MULTILINE
    )
    assert task_005_match is not None
    task_005 = task_005_match.group(0)
    invalid_tasks = [
        tasks.replace("- **Objective**:", "- **Aim**:", 1),
        tasks.replace(
            task_005, task_005.replace("**Status**: `DONE`", "**Status**: `UNRECOGNIZED`", 1), 1
        ),
        tasks.replace(
            task_005,
            task_005.replace("**Status**: `DONE`", "**Status**: `READY`\n- **Status**: `DONE`", 1),
            1,
        ),
        tasks.replace(
            task_005,
            task_005.replace(
                "- **Verification Evidence**:\n  - Focused documentation tests",
                "- **Verification Evidence**: None",
                1,
            ),
            1,
        ),
        tasks + "\n" + task_005,
    ]
    for invalid_content in invalid_tasks:
        with pytest.raises(AssertionError):
            _parse_tasks(invalid_content)


def test_decisions_file_contains_approved_adrs():
    """Verify docs/DECISIONS.md records all active ADRs from ADR-001 through ADR-006."""
    root = Path(__file__).resolve().parents[2]
    decisions_content = (root / "docs/DECISIONS.md").read_text(encoding="utf-8")
    for i in range(1, 7):
        adr_id = f"ADR-{i:03d}"
        assert adr_id in decisions_content, f"Missing {adr_id} in docs/DECISIONS.md"
