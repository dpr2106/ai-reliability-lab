# Architecture Decision Records (ADRs)

## ADR-001: Baseline-First Evaluation Architecture
- **Status**: Accepted
- **Context**: RAG applications frequently add vector databases without measuring whether dense retrieval outperforms simple keyword search on domain-specific tokens or acronyms.
- **Decision**: Implement an in-memory Okapi BM25 sparse index as the primary baseline. All dense embedding models (e.g., SentenceTransformers + FAISS) must be measured against this baseline on identical evaluation splits.
- **Consequences**: Avoids unverified claims of "semantic superiority"; establishes reproducible Cranfield IR baselines.

## ADR-002: Protected `main` Branch with Native GitHub Auto-Merge
- **Status**: Accepted
- **Context**: Solo engineering projects often decay through undisciplined direct pushes to `main`, bypassing linters and tests.
- **Decision**: Protect `main` against direct pushes and force pushes. Require `test-and-lint` CI checks to pass. Enable GitHub native auto-merge (`gh pr merge --auto --squash`) without mandatory peer approval click to avoid author deadlocks.
- **Consequences**: Every change on `main` is backed by a verified, green pull request and linear commit history.

## ADR-003: Atomic In-Branch Task Status Updates
- **Status**: Accepted
- **Context**: If `main` is protected, attempting to update `docs/roadmap/TASKS.md` or `docs/STATUS.md` after a PR merges requires a direct push that GitHub will reject.
- **Decision**: Include verification evidence and mark tasks complete within the feature branch PR itself.
- **Consequences**: When auto-merge executes, code, tests, and documentation updates land on `main` in a single atomic squash commit.

## ADR-004: Ingestion Engine Selection (PyMuPDF / `fitz`)
- **Status**: Accepted
- **Context**: Citation verification requires attributing claims to exact page numbers and character bounding spans. Naive PDF text extractors strip layout coordinates and often misorder multi-column text.
- **Decision**: Use PyMuPDF (`fitz`) for document parsing due to its C-backed speed and explicit page/span coordinate preservation.
- **Consequences**: Enables deterministic citation auditing and span matching without cloud document processing APIs.

## ADR-005: Pure Python Standard Library Domain Models in Core
- **Status**: Accepted
- **Context**: `docs/ARCHITECTURE.md` declared `reliability_lab/core/` to possess zero external dependencies, but `FOUND-003`'s initial acceptance criteria referenced Pydantic models. Third-party dependencies in the core domain layer introduce coupling and potential dependency churn.
- **Decision**: Implement all core domain models (`Document`, `Chunk`, `Query`, `RetrievalResult`, `EvaluationReport`) and abstract interfaces using pure Python standard library (`dataclasses`, `typing`, `enum`, `abc`, `json`, `uuid`). Enforce domain constraints via `@dataclass` and `__post_init__` validation. Pydantic or schema serializers may be utilized at boundaries (e.g., dataset ingestion or API layers), but `core/` remains strictly zero-external-dependency standard library Python.
- **Consequences**: Ensures maximum portability, clean hexagonal architecture, zero vendor/library lock-in for core entities, and complete alignment between documentation and implementation.

## ADR-006: Squash-Only Merge Strategy & Linear History Enforcement
- **Status**: Superseded by ADR-007
- **Context**: Multiple merge strategies (merge commits, rebase merges, squash merges) create inconsistent Git graphs. Inadvertent merge commits introduce nonlinear branch clutter and obscure per-task atomic changes on `main`.
- **Decision**: Configure GitHub repository settings to exclusively allow squash merges (`allow_squash_merge: true`, `allow_merge_commit: false`, `allow_rebase_merge: false`). Enforce linear history in branch protection (`required_linear_history: true`).
- **Consequences**: Ensures a clean, 1-commit-per-task linear Git history on `main`, guarantees PR links in commit headlines, and eliminates accidental merge commits.

## ADR-007: Rebase-Only Merge Strategy & Commit Preservation on Main
- **Status**: Accepted
- **Context**: ADR-006 configured squash-only merging to guarantee a clean linear history. However, squash merging collapses all feature-branch commits into a single commit upon merge, erasing distinct, reviewable engineering milestones (such as separate test setup, core implementation, refactoring, and documentation commits). The project requires at least 300 genuine, meaningful engineering commits reachable from `main`. Continuing with squash-only merging obscures granular commit provenance and would require hundreds of micro-PRs.
- **Decision**: Configure GitHub repository settings to exclusively allow rebase merges (`allow_rebase_merge: true`, `allow_squash_merge: false`, `allow_merge_commit: false`). Preserve strict branch protection requiring linear history (`required_linear_history: true`), strict CI checks (`test-and-lint (3.11)` and `test-and-lint (3.12)`), and prohibition of direct and force pushes. Automate merging using `gh pr merge --auto --rebase --delete-branch`.
- **Consequences**:
  - Genuine, meaningful commits authored on feature branches are preserved in full on `main`, each retaining its individual author timestamp, commit message, and diff history.
  - Linear history is preserved without non-linear merge bubbles or merge commits.
  - Developers must maintain disciplined, reviewable commit hygiene on feature branches (no "fix typo" or "wip" noise; every commit must be coherent, atomic, and pass lint/tests).
  - Squash merging is disabled to prevent accidental squashing of multi-commit engineering tasks.

## ADR-008: Svelte/SvelteKit Frontend with FastAPI Adapter for Diagnostic Studio
- **Status**: Accepted
- **Context**: The original roadmap specified Streamlit for UI-002. However, Streamlit couples UI state to server-side script re-execution, lacks fine-grained reactive DOM control for dense diagnostics, prevents clean separation of API contracts from presentation, and cannot easily support modern component systems or responsive client-side interactions. The project requires a professional, industrial-grade diagnostic studio with custom layout hierarchy, fast client-side sorting/filtering, side-by-side run comparisons, and interactive citation audits.
- **Decision**: Build the frontend as a dedicated Svelte/SvelteKit application (`frontend/`) written in strict TypeScript. Expose the existing Python SQLite experiment repository, regression detection engine, and citation evaluation capabilities via a lightweight, typed FastAPI local backend (`reliability_lab/api/`). Keep `web` dependencies (`fastapi`, `uvicorn`, `httpx`) isolated in an optional dependency group in `pyproject.toml`.
- **Consequences**:
  - Delivers a high-performance, responsive, accessible web interface with full control over design system primitives, reactive state, and failure visualizations.
  - Maintains strict separation of concerns: Python domain logic (`reliability_lab/`) remains the authoritative source of truth for metrics, regression mathematics, and persistence, accessed via clean REST API contracts.
  - The browser never accesses the raw SQLite database directly; all operations pass through validated repository interfaces and safety checks.
  - CLI and library users remain unencumbered by web framework dependencies unless installing `.[web]`.
