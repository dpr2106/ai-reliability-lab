# AI Reliability Lab — Operating Guidelines & Conventions

## Mission & Identity
AI Reliability Lab is an industrial-grade engineering and evaluation platform designed to assess, audit, and benchmark document-based AI systems (RAG, vector retrieval, and LLM grounding). It is built with rigorous baseline-first principles, reproducible Cranfield IR metrics, and automated regression detection.

## Fundamental Operating Rules
1. **Understand Before Modifying**: Inspect existing code, schemas, and tests before editing.
2. **Preserve Working Interfaces**: Never refactor unrelated modules or break established public API contracts without an ADR in `docs/DECISIONS.md`.
3. **Evidence-or-Revert**: Never mark a task `DONE` or claim a test passed without running the command and recording the exact output, exit code, and commit hash.
4. **Honest Measurement**: Never fabricate benchmarks, metrics, or claim simulated systems are production-ready.
5. **No Direct Pushes to `main`**: All changes to `main` occur via task-specific feature branches, passing CI checks, and GitHub auto-merge (`gh pr merge --auto --rebase --delete-branch` per ADR-007).
6. **Atomic Status Updates**: Feature branches update their own task completion state in `docs/roadmap/TASKS.md` and `docs/STATUS.md` so that code, tests, and documentation merge into `main` atomically.
7. **Status Snapshot Invariants**: `docs/roadmap/TASKS.md` is the source of truth for task IDs and statuses. Every status pointer must reference exactly one task with a compatible non-`DONE` status; do not record transient merge SHAs in `docs/STATUS.md`. A task may be marked `DONE` only after it is merged and its evidence records the commit, exact verification results, and CI evidence. An empty webhooks response does not establish that no GitHub Apps are installed.
8. **Permanent UI/UX Standard**: Any presentation or visual interface must exhibit top-tier aesthetic polish, high information hierarchy, clear failure diagnostics, and deterministic data credibility.

## Git & Task Protocol
- **Branch Naming**: `feat/<TASK-ID>-<kebab-description>`, `fix/<TASK-ID>-<kebab-description>`, `chore/<TASK-ID>-<kebab-description>`
- **Commit Standards**: Conventional Commits linking Task ID: `<type>(<scope>): <summary> (<TASK-ID>)`. Commits on feature branches must be atomic, reviewable, and meaningful to preserve high-fidelity history upon rebase merge (ADR-007).
- **Verification Loop**:
  ```bash
  ruff check .
  ruff format --check .
  mypy reliability_lab
  pytest --cov=reliability_lab
  ```
- **PR Lifecycle**:
  ```bash
  git push -u origin <branch-name>
  gh pr create --title "<type>(<scope>): <description> (<TASK-ID>)" --body "<PR_BODY>"
  gh pr merge --auto --rebase --delete-branch
  ```

## Canonical Documentation Roles
- `GEMINI.md`: Operating rules, developer constraints, and AI pair-programming boundaries.
- `docs/PROJECT_BRIEF.md`: Vision, requirements, placement goals, and non-functional requirements.
- `docs/ARCHITECTURE.md`: Technical blueprint, domain models, protocol contracts, and data flows.
- `docs/STATUS.md`: Current verified operational state and active task pointer.
- `docs/DECISIONS.md`: Architecture Decision Records (ADRs) with trade-offs.
- `docs/roadmap/TASKS.md`: Single source of truth for the milestone backlog.
