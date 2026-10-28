# Implementation Backlog: AI Reliability Lab

This document is the canonical single source of truth for all engineering tasks.

**Status Legend**:
- `BACKLOG`: Defined and scoped, but blocked by unsatisfied dependencies.
- `READY`: All dependencies satisfied; available for immediate execution.
- `IN_PROGRESS`: Actively under development on a dedicated feature branch.
- `IN_REVIEW`: Code complete, tested, and under pull request review with CI running.
- `DONE`: Merged to `main` with verified test evidence recorded.
- `BLOCKED`: Development halted due to an explicit external blocker.

---

## Milestone 0: Foundations, Governance & Test Infrastructure

### [FOUND-001] Project Scaffolding & Virtual Environment Setup
- **Objective**: Establish standard Python packaging (`pyproject.toml`), pinning linting, type-checking, and test dependencies.
- **Priority**: P0 | **Dependencies**: None
- **Scope / Expected Files**: `pyproject.toml`, `.gitignore`, `README.md`, `reliability_lab/__init__.py`, `tests/conftest.py`, `tests/unit/test_smoke.py`, `.github/workflows/ci.yml`
- **Acceptance Criteria**: `pytest`, `ruff`, and `mypy` run cleanly with zero errors.
- **Verification Command**:
  ```bash
  ruff check .
  ruff format --check .
  mypy reliability_lab
  pytest --cov=reliability_lab
  ```
- **Status**: `DONE`
- **Verification Evidence**:
  - Commit SHA: `17115ca21d228e333347c00609f590cac32c8841` on `main`
  - GitHub Actions Run: [40491157361](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40491157361) (Success on Python 3.11 & 3.12)
  - Output: `All checks passed! 5 files already formatted. Success: no issues found in 1 source file. 2 passed in 0.10s, 100% coverage.`

---

### [FOUND-002] Canonical Documentation Suite Setup
- **Objective**: Commit the 6 persistent project documents establishing operating rules, architecture, status, decisions, and backlog.
- **Priority**: P0 | **Dependencies**: `FOUND-001`
- **Scope / Expected Files**: `GEMINI.md`, `docs/PROJECT_BRIEF.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/roadmap/TASKS.md`
- **Acceptance Criteria**:
  - All 6 documents present, non-overlapping, and cross-referenced.
  - Permanent UI/UX standards established as non-functional requirements.
  - Initial verified operational status accurately recorded.
- **Verification Command**: Documentation presence and link check; local test execution.
- **Status**: `DONE`
- **Verification Evidence**:
  - Squash Merge Commit: `f49752fdf3053f92503ee8f3c3f8a8c8da91d163` on `main` ([PR #1](https://github.com/dpr2106/ai-reliability-lab/pull/1))
  - GitHub Actions Run: [40491747319](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40491747319) (Success on Python 3.11 & 3.12)
  - Test Suite: 5 passed in 0.20s (`tests/unit/test_documentation.py`, `tests/unit/test_smoke.py`)
  - Canonical Files: `GEMINI.md`, `docs/PROJECT_BRIEF.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/roadmap/TASKS.md` all verified present and consistent.

---

### [FOUND-003] Core Domain Entities & Protocol Interfaces
- **Objective**: Implement pure domain models (`Document`, `Chunk`, `Query`, `RetrievalResult`, `EvaluationReport`) with zero external library dependencies (ADR-005).
- **Priority**: P0 | **Dependencies**: `FOUND-002`
- **Scope / Expected Files**: `reliability_lab/core/models.py`, `reliability_lab/core/interfaces.py`, `tests/unit/test_core_models.py`
- **Acceptance Criteria**:
  - Pure Python standard library dataclasses with validation and serialization helper methods (`to_dict`, `from_dict`, `to_json`).
  - Standard library abstract base classes (`BaseRetriever`, `BaseChunker`, `BaseMetric`).
  - Unit tests verify immutable properties, validation rules, and JSON round-trips without external third-party dependencies.
- **Verification Command**: `pytest tests/unit/test_core_models.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Test Suite: 69 passed in 0.74s (`tests/unit/test_core_models.py`, `tests/unit/test_documentation.py`, `tests/unit/test_smoke.py`)
  - Quality Checks: `ruff check .` (0 errors), `ruff format --check .` (0 errors), `mypy reliability_lab` (0 errors), `pytest --cov=reliability_lab` (93% package coverage).
  - Models Implemented: `Document`, `Chunk`, `Query`, `RetrievalResult`, `EvaluationReport` (pure Python stdlib dataclasses with validation and `to_dict`/`from_dict`/`to_json`/`from_json`).
  - Abstract Interfaces Implemented: `BaseRetriever`, `BaseChunker`, `BaseMetric`.

---

### [FOUND-004] Repository Governance, Branch Protection & CI Review Audit
- **Objective**: Audit GitHub repository configuration: branch protection on `main`, strict required CI checks, squash-only merge policy, linear history enforcement, and accurately document review evidence limits.
- **Priority**: P0 | **Dependencies**: `FOUND-002`
- **Scope / Expected Files**: `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/roadmap/TASKS.md`, `tests/unit/test_documentation.py`
- **Acceptance Criteria**:
  - GitHub branch protection verified via API: direct pushes and force pushes disallowed, required checks `["test-and-lint (3.11)", "test-and-lint (3.12)"]` with `strict: true`.
  - Repository settings enforce `allow_squash_merge: true`, `allow_merge_commit: false`, `allow_rebase_merge: false`, and `required_linear_history: true`.
  - Review automation findings are accurately scoped to the evidence inspected; an empty webhooks response is not treated as proof that no GitHub Apps are installed.
  - `docs/STATUS.md` adheres to the maintainable repository baseline rule without transient branch names that rot upon merge.
  - Unit tests in `tests/unit/test_documentation.py` assert structural integrity, required fields in `TASKS.md`, and valid task pointers in `STATUS.md`.
- **Verification Command**:
  ```bash
  gh api /repos/dpr2106/ai-reliability-lab/branches/main/protection
  gh api /repos/dpr2106/ai-reliability-lab --jq '{allow_squash_merge: .allow_squash_merge, allow_merge_commit: .allow_merge_commit, allow_rebase_merge: .allow_rebase_merge}'
  pytest tests/unit/test_documentation.py -v
  ```
- **Status**: `DONE`
- **Verification Evidence**:
  - Branch Protection API verified: `strict: true`, `contexts: ["test-and-lint (3.11)", "test-and-lint (3.12)"]`, `allow_force_pushes: false`, `allow_deletions: false`, `required_linear_history: true`.
  - Repository Settings API verified: `allow_auto_merge: true`, `allow_squash_merge: true`, `allow_merge_commit: false`, `allow_rebase_merge: false`, `delete_branch_on_merge: true`.
  - Webhooks API response recorded as `[]`; this reports an empty webhooks list only and does not establish whether GitHub Apps are installed.
  - Test Suite: Comprehensive assertions in `tests/unit/test_documentation.py` verifying documentation structural invariants.

---

### [FOUND-005] Status Snapshot Semantics & Documentation Invariants
- **Objective**: Make the project status snapshot durable and enforce consistency between status pointers and the canonical task backlog.
- **Priority**: P0 | **Dependencies**: `FOUND-004`
- **Scope / Expected Files**: `GEMINI.md`, `docs/STATUS.md`, `docs/roadmap/TASKS.md`, `tests/unit/test_documentation.py`
- **Acceptance Criteria**:
  - `docs/STATUS.md` records durable governance facts and active/next task pointers that resolve to exactly one task in `TASKS.md` with a compatible status.
  - The status snapshot does not assert a current merge SHA or infer that no GitHub Apps are installed from an empty webhooks response.
  - The known fact that Copilot could not review PR #3 because of quota remains documented.
  - Documentation tests reject unknown or duplicate task IDs, invalid/conflicting statuses, completed tasks referenced as active/next, missing task sections, and absent or placeholder evidence for `DONE` tasks.
  - Operating guidance defines these task and status rules durably.
- **Verification Command**:
  ```bash
  pytest tests/unit/test_documentation.py -v
  ruff check .
  ruff format --check .
  mypy reliability_lab
  pytest --cov=reliability_lab
  ```
- **Verification Evidence Policy**: A `DONE` task must have merged commit evidence, exact verification command outputs and exit codes, and successful required CI run evidence. Do not use placeholders (including `None`, `pending`, `TBD`, or `TODO`); record evidence only after it has been observed. A task under review remains `IN_REVIEW` until GitHub confirms merge.
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused documentation tests: `6 passed in 0.09s` (exit code 0).
  - `ruff check .`: `All checks passed!` (exit code 0).
  - `ruff format --check .`: `12 files already formatted` (exit code 0).
  - `mypy reliability_lab`: `Success: no issues found in 1 source file` (exit code 0; configuration note: unused `tests.*` override).
  - `pytest --cov=reliability_lab`: `8 passed in 0.12s`, 100% package coverage (exit code 0).
  - PR #4 was squash-merged to `main` as commit `cc3da1f608b3c0defd42465d3263346d0ed3fbcc`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40502440838/job/122183202675) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40502440838/job/122183202130); both completed successfully before merge.

---

### [GOV-006] Commit-Preserving Merge Workflow
- **Objective**: Establish and verify a GitHub merge workflow that preserves meaningful implementation commits on `main` to enable reaching the project target of at least 300 commits on the default branch.
- **Priority**: P0 | **Dependencies**: `FOUND-004`
- **Scope / Expected Files**: `GEMINI.md`, `docs/DECISIONS.md`, `docs/STATUS.md`, `docs/roadmap/TASKS.md`, `docs/roadmap/COMMIT_PROGRESS.md`, `tests/unit/test_documentation.py`
- **Acceptance Criteria**:
  - Rebase-based merging is configured on GitHub (`allow_rebase_merge: true`, `allow_squash_merge: false`, `allow_merge_commit: false`).
  - Strict CI checks (`test-and-lint (3.11)` and `test-and-lint (3.12)`), linear history enforcement, and direct/force push prohibitions are preserved.
  - Initial governance PR establishes ADR-007, COMMIT_PROGRESS.md, and updates task ledger under current allowed merge policy.
  - Follow-up test PR verifies that multiple feature branch commits land on `main` without being squashed.
  - If permissions are insufficient to alter repository settings, the task documents a clear BLOCKED state with required administrator actions.
- **Verification Command**:
  ```bash
  gh repo view dpr2106/ai-reliability-lab --json mergeCommitAllowed,rebaseMergeAllowed,squashMergeAllowed
  gh api repos/dpr2106/ai-reliability-lab/branches/main/protection
  git rev-list --count origin/main
  pytest tests/unit/test_documentation.py -v
  ```
- **Status**: `DONE`
- **Verification Evidence**:
  - Initial governance PR #19 established ADR-007 (superseding ADR-006) and `docs/roadmap/COMMIT_PROGRESS.md` (initial count 19), merged under final squash policy as commit `9de7e1951f28b499e74d1a4983fb08d98d633c77`.
  - Repository settings configured via GitHub API and verified: `rebaseMergeAllowed: true`, `squashMergeAllowed: false`, `mergeCommitAllowed: false`, `delete_branch_on_merge: true`, `allow_auto_merge: true`.
  - Branch protection verified via API: `required_linear_history: true`, strict checks `["test-and-lint (3.11)", "test-and-lint (3.12)"]`, `allow_force_pushes: false`, `allow_deletions: false`.
  - Rebase merge workflow verified via PR #20 preserving distinct feature-branch commits on `main`: commit `15e8096` (`docs(guidelines): update PR workflow in GEMINI.md for rebase merging`) and commit `4b192ad` (`docs(status): record verified rebase settings and complete GOV-006`).
  - Required CI verified on PR #20: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40636492854/job/122644547644) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40636492854/job/122644547846).
  - Main branch commit count verified via `git rev-list --count origin/main`: 22 commits reachable from `main`.
  - Operating guidelines updated in `GEMINI.md` to document `gh pr merge --auto --rebase --delete-branch`.
  - Documentation tests verified: `pytest tests/unit/test_documentation.py -v` (6 passed in 0.16s).

---

## Milestone 1: Document Ingestion & Citation-Aware Chunking

### [ING-001] PyMuPDF Document Parser with Page & Bounding Span Tracking
- **Objective**: Extract text per page while preserving line numbers, page indexes, and character offsets for citation audits.
- **Priority**: P0 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/ingestion/parser.py`, `tests/unit/test_parser.py`, `reliability_lab/ingestion/__init__.py`
- **Acceptance Criteria**: Extracts clean text, handles encrypted/corrupt PDFs gracefully, records bounding metadata, ensures exact substring slicing.
- **Verification Command**: `pytest tests/unit/test_parser.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused parser tests: `15 passed in 0.52s` (`pytest tests/unit/test_parser.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (19 files already formatted), `mypy reliability_lab` (Success: no issues found in 6 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (84 passed in 0.89s, 93% package coverage, exit code 0).
  - PR #7 was squash-merged to `main` as commit `b14a707959c2b97a29442b5377c0539ba5ba1ef5`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40508473499/job/122203815962) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40508473499/job/122203815908); both completed successfully before merge.
  - Character provenance invariants verified: `Document.content[char_start:char_end]` exactly reproduces the text for all pages, lines, and spans under half-open indexing `[start, end)`.
  - Deterministic separators: `\n\n` between pages, `\n` between lines.
  - Zero sensitive path or credential leakage: only basename stored in `source_file`, passwords never stored in metadata or error strings.

---

### [ING-002] Citation-Aware Sliding & Recursive Chunkers
- **Objective**: Chunk documents into token/character windows while preserving parent document IDs, page spans, and offsets.
- **Priority**: P0 | **Dependencies**: `ING-001`
- **Scope / Expected Files**: `reliability_lab/ingestion/chunker.py`, `tests/unit/test_chunker.py`
- **Acceptance Criteria**: Supports configurable chunk size and overlap; verifies zero metadata leakage across boundaries.
- **Verification Command**: `pytest tests/unit/test_chunker.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused chunker tests: `27 passed in 2.40s` (`pytest tests/unit/test_chunker.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (21 files already formatted), `mypy reliability_lab` (Success: no issues found in 7 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (111 passed in 3.84s, 93% package coverage, exit code 0).
  - PR #9 was squash-merged to `main` as commit `a77655e2c25d90c45ee27e9b50b475e18483a386`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40511262693/job/122213478417) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40511262693/job/122213477986); both completed successfully before merge.
  - Character provenance invariants verified: `Document.content[char_start:char_end]` exactly reproduces chunk content, line text, and span text for all elements under half-open indexing `[start, end)`.
  - Zero metadata leakage: only intersecting pages, lines, and spans are retained; partial spans are clipped without leaking outer text; page separators are never falsely attributed to page text.
  - Stable deterministic chunk IDs across repeated runs.

---

## Milestone 2: Baseline Lexical Retrieval (BM25)

### [RET-001] In-Memory BM25 Lexical Index
- **Objective**: Implement an in-memory Okapi BM25 sparse index as the foundational IR baseline.
- **Priority**: P0 | **Dependencies**: `ING-002`
- **Scope / Expected Files**: `reliability_lab/indexing/bm25_index.py`, `tests/unit/test_bm25_index.py`
- **Acceptance Criteria**: Constructs index from `List[Chunk]`, retrieves ranked results with scores, handles out-of-vocabulary terms.
- **Verification Command**: `pytest tests/unit/test_bm25_index.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused BM25 tests: `53 passed in 0.23s` (`pytest tests/unit/test_bm25_index.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (24 files already formatted), `mypy reliability_lab` (Success: no issues found in 9 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (164 passed in 3.47s, 94% package coverage, exit code 0).
  - PR #11 was squash-merged to `main` as commit `0825c39d87b2f2f1a2e778f6a5f9678c2e160ff3`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40513705919/job/122221964294) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40513705919/job/122221965217); both completed successfully before merge.
  - Mathematical correctness: Verified against hand-calculated scores down to 1e-6 relative tolerance for Robertson-Spärck Jones non-negative IDF, term frequency saturation, and document length normalization.
  - Ranking & stability: Deterministic tie-breaking preserves original corpus input order, 1-based ranks, and top_k bounds.
  - Zero external dependencies: pure standard-library implementation.

---

## Milestone 3: Manually Verified Golden Evaluation Dataset

### [DATA-001] Benchmark Schema & Golden Dataset Construction
- **Objective**: Build a verified benchmark dataset containing queries, multi-grade chunk relevances, and required citations.
- **Priority**: P0 | **Dependencies**: `ING-002`
- **Scope / Expected Files**: `data/benchmarks/schema.json`, `data/benchmarks/sample_corpus.jsonl`, `data/benchmarks/sample_benchmark.jsonl`, `tests/unit/test_dataset_schema.py`
- **Acceptance Criteria**: At least 15 verified queries with graded relevance (0=irrelevant, 1=relevant, 2=critical evidence), grounded in a deterministic 25-chunk corpus fixture (`sample_corpus.jsonl`) required for independent verification of source provenance and exact verbatim citations.
- **Verification Command**: `pytest tests/unit/test_dataset_schema.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused schema and dataset tests: `10 passed in 0.18s` (`pytest tests/unit/test_dataset_schema.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (25 files already formatted), `mypy reliability_lab` (Success: no issues found in 9 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (174 passed in 15.68s, 94% package coverage, exit code 0).
  - PR #13 was squash-merged to `main` as commit `eebfcc7d64b9a105aa985afbf30b00df4d7a9226`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40630175988/job/122624479419) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40630175988/job/122624479239); both completed successfully before merge.
  - JSON Schema Draft 2020-12 verified at `data/benchmarks/schema.json` with strict type bounds, enum grades `[0, 1, 2]`, rationale strings, and explicit citation mappings.
  - Deterministic 25-chunk source corpus across 5 synthetic documents created at `data/benchmarks/sample_corpus.jsonl` with full character span and page metadata.
  - 16 verified gold queries at `data/benchmarks/sample_benchmark.jsonl` spanning direct lookup, multi-evidence synthesis, lexical distractors, near-duplicate terms, and numeric units; all required citations verified verbatim against source chunk content.

---

## Milestone 4: Cranfield Retrieval Evaluation Metrics

### [EVAL-001] Set-Based Retrieval Metrics: Recall@K, Precision@K, HitRate@K
- **Objective**: Implement exact set-based retrieval diagnostics with hand-calculated unit tests.
- **Priority**: P0 | **Dependencies**: `FOUND-003`, `DATA-001`
- **Scope / Expected Files**: `reliability_lab/evaluation/retrieval_metrics.py`, `reliability_lab/evaluation/__init__.py`, `tests/unit/test_retrieval_metrics.py`
- **Acceptance Criteria**: Unit tests match hand calculations for empty results, $K > N$, and zero-ground-truth edge cases.
- **Verification Command**: `pytest tests/unit/test_retrieval_metrics.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `56 passed in 0.46s` (`pytest tests/unit/test_retrieval_metrics.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (28 files already formatted), `mypy reliability_lab` (Success: no issues found in 11 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (230 passed in 10.98s, 94% package coverage, 100% coverage on `reliability_lab/evaluation/retrieval_metrics.py`, exit code 0).
  - PR #15 was squash-merged to `main` as commit `d1594895169dd946f7eae7227dbfbe48da5f5caa`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40632547960/job/122632049436) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40632547960/job/122632049110); both completed successfully before merge.
  - Set-based metric suite implemented with standard-library-only interfaces adhering to ADR-005: `RecallAtK`, `PrecisionAtK`, and `HitRateAtK` implementing `BaseMetric` with configurable positive cutoff `k` (default 5).
  - Mathematical correctness & edge cases verified: empty ground-truth returns 0.0, empty retrieval returns 0.0, duplicate retrieved chunk IDs deduplicated, fewer than k results keeps denominator k in precision, and grade 0 overrides properly exclude non-relevant items.

---

### [EVAL-002] Positional Metrics: MRR@K and Graded NDCG@K
- **Objective**: Implement Mean Reciprocal Rank and Normalized Discounted Cumulative Gain using standard logarithmic discounting.
- **Priority**: P0 | **Dependencies**: `EVAL-001`
- **Scope / Expected Files**: `reliability_lab/evaluation/retrieval_metrics.py`, `reliability_lab/evaluation/__init__.py`, `tests/unit/test_retrieval_metrics.py`
- **Acceptance Criteria**: NDCG@K matches standard reference calculations for arbitrary graded gains.
- **Verification Command**: `pytest tests/unit/test_retrieval_metrics.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `112 passed in 0.55s` (`pytest tests/unit/test_retrieval_metrics.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (28 files already formatted), `mypy reliability_lab` (Success: no issues found in 11 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (286 passed in 11.41s, 94% package coverage, 99% coverage on `reliability_lab/evaluation/retrieval_metrics.py`, exit code 0).
  - PR #17 was squash-merged to `main` as commit `7f6c41e038790507c30b5c350d9124a48cbe8f24`.
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40634334439/job/122637700509) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40634334439/job/122637700836); both completed successfully before merge.
  - Positional metrics suite implemented with standard-library-only interfaces adhering to ADR-005: `MRRAtK` and `NDCGAtK` implementing `BaseMetric` with configurable positive cutoff `k` (default 5).
  - Mathematical correctness & edge cases verified: logarithmic discounting $\log_2(i + 1)$, exponential gain $2^{rel} - 1$, duplicate retrieved chunk IDs preserve rank discount with zero gain, and empty retrieval or zero ground truth safely returns 0.0 without division by zero.

---

### [EVAL-003] End-to-End Retrieval Benchmark Runner
- **Objective**: Execute an index against the golden benchmark and compile aggregate reports and query failure traces.
- **Priority**: P0 | **Dependencies**: `RET-001`, `EVAL-002`, `DATA-001`
- **Scope / Expected Files**: `reliability_lab/evaluation/benchmark_runner.py`, `tests/integration/test_benchmark_runner.py`
- **Acceptance Criteria**: Produces an `EvaluationReport` with mean Recall@5, MRR@5, NDCG@5, and latency percentiles.
- **Verification Command**: `pytest tests/integration/test_benchmark_runner.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused integration tests: `23 passed in 0.52s` (`pytest tests/integration/test_benchmark_runner.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (31 files already formatted), `mypy reliability_lab` (Success: no issues found in 12 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (309 passed in 5.13s, 93% package coverage, exit code 0).
  - Retrieval benchmark runner executed over 16 queries and 25 corpus chunks: produces complete `EvaluationReport` with mean Recall@5, Precision@5, HitRate@5, MRR@5, NDCG@5, latency percentiles (`mean_latency_ms`, `latency_p50_ms`, `latency_p90_ms`, `latency_p95_ms`, `latency_p99_ms`), failure diagnostics (`zero_hit`, `missed_relevant_chunk_ids`, `missed_critical_chunk_ids`, `first_relevant_rank`), and full dictionary / JSON round-trip serialization.
  - Rebase merge workflow verified via PR #22 preserving 5 distinct feature-branch commits on `main`: `7d16a87`, `1b2a1a3`, `6f23e22`, `c747e8a`, and `15c3cc5`.
  - Required CI verified on PR #22: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40638366380/job/122650576602) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40638366380/job/122650576749).
  - Main branch commit count verified via `git rev-list --count origin/main`: 28 commits reachable from `main`.

---

## Milestone 5: Dense Retrieval, FAISS & Hybrid Ranking

### [DENSE-001] Local Sentence-Transformers Embedding Pipeline
- **Objective**: Provide reproducible, offline vector generation using lightweight models (`all-MiniLM-L6-v2`).
- **Priority**: P1 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/indexing/embeddings.py`, `tests/unit/test_embeddings.py`
- **Acceptance Criteria**: Batch embedding generation with dimension checks, device fallback, and L2 normalization.
- **Verification Command**: `pytest tests/unit/test_embeddings.py -v`
- **Status**: `IN_REVIEW`
- **Verification Evidence**:
  - Focused unit tests: `31 passed in 0.71s` (`pytest tests/unit/test_embeddings.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (33 files already formatted), `mypy reliability_lab` (Success: no issues found in 13 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (340 passed, 93% package coverage, exit code 0).
  - `BaseEmbedder` interface and `SentenceTransformerEmbedder` adapter implemented with default `all-MiniLM-L6-v2` producing 384-dimensional vectors.
  - Verified L2 normalization: non-zero vectors normalized to unit length $\|v\|_2 = 1.0 \pm 10^{-4}$.
  - Numerical safety verified: zero-norm vectors handled without NaNs, non-finite (NaN/Inf) outputs rejected.
  - Device resolution: automatic CPU fallback when requested CUDA/MPS is unavailable.
  - Offline support: `local_files_only=True` prevents network access and raises actionable `FileNotFoundError` if weights are missing.
  - Optional dependency isolation: `reliability-lab[dense]` extras configured in `pyproject.toml` while keeping lightweight Tier 1 BM25 execution unaffected.

---

### [DENSE-002] FAISS Vector Index (IndexFlatIP & IndexHNSWFlat)
- **Objective**: Benchmark exact inner product against approximate graph search for latency vs recall trade-offs.
- **Priority**: P1 | **Dependencies**: `DENSE-001`
- **Scope / Expected Files**: `reliability_lab/indexing/faiss_index.py`, `tests/unit/test_faiss_index.py`
- **Acceptance Criteria**: Supports serialization to disk, add/search operations, and mapping integer IDs to chunk UUIDs.
- **Verification Command**: `pytest tests/unit/test_faiss_index.py -v`
- **Status**: `READY`
- **Verification Evidence**: None

---

### [DENSE-003] Hybrid Search with Reciprocal Rank Fusion (RRF)
- **Objective**: Combine sparse (BM25) and dense (FAISS) rankings without score scale discrepancies.
- **Priority**: P1 | **Dependencies**: `RET-001`, `DENSE-002`
- **Scope / Expected Files**: `reliability_lab/indexing/hybrid.py`, `tests/unit/test_hybrid.py`
- **Acceptance Criteria**: Implements $RRF(d) = \sum \frac{1}{k + r_i(d)}$; unit tests verify fused ranking matches math.
- **Verification Command**: `pytest tests/unit/test_hybrid.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 6: Groundedness & Citation Evaluation

### [GROUND-001] Citation Parser & Reference Link Validator
- **Objective**: Parse citation markers in generated text and verify whether cited chunks exist in the retrieved set.
- **Priority**: P1 | **Dependencies**: `ING-002`, `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/evaluation/citation_metrics.py`, `tests/unit/test_citation_metrics.py`
- **Acceptance Criteria**: Calculates Citation Precision and Citation Recall against ground truth citations.
- **Verification Command**: `pytest tests/unit/test_citation_metrics.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [GROUND-002] Deterministic Claim Span & Lexical Overlap Grounding
- **Objective**: Fast, 100% local metric evaluating token containment and longest common subsequence between answers and source chunks.
- **Priority**: P1 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/evaluation/groundedness.py`, `tests/unit/test_groundedness.py`
- **Acceptance Criteria**: Accurately flags completely unsupported sentences vs fully grounded sentences.
- **Verification Command**: `pytest tests/unit/test_groundedness.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [GROUND-003] Local NLI Entailment Classifier
- **Objective**: Classify chunk premise and answer hypothesis pairs into Entailment, Neutral, or Contradiction.
- **Priority**: P2 | **Dependencies**: `GROUND-002`
- **Scope / Expected Files**: `reliability_lab/evaluation/nli_evaluator.py`, `tests/unit/test_nli_evaluator.py`
- **Acceptance Criteria**: Returns probability distribution across NLI labels and computes aggregate Hallucination Score.
- **Verification Command**: `pytest tests/unit/test_nli_evaluator.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 7: Experiment Storage & Regression Detection

### [EXP-001] SQLite Schema & Experiment Run Repository
- **Objective**: Persist benchmark runs, index configurations, and query metric scores in relational storage.
- **Priority**: P1 | **Dependencies**: `EVAL-003`
- **Scope / Expected Files**: `reliability_lab/storage/database.py`, `reliability_lab/storage/repository.py`, `tests/unit/test_storage.py`
- **Acceptance Criteria**: Schema stores configuration hashes, run timestamps, and per-query score breakdowns.
- **Verification Command**: `pytest tests/unit/test_storage.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [EXP-002] Regression Detection Engine & Metric Diffing
- **Objective**: Compare Run B against Baseline Run A and identify queries experiencing ranking degradation.
- **Priority**: P1 | **Dependencies**: `EXP-001`
- **Scope / Expected Files**: `reliability_lab/storage/regression.py`, `tests/unit/test_regression.py`
- **Acceptance Criteria**: Computes $\Delta \text{MRR}$ and $\Delta \text{NDCG}$; flags queries exceeding regression tolerance.
- **Verification Command**: `pytest tests/unit/test_regression.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 8: CLI & Streamlit Diagnostic Studio

### [UI-001] Headless CLI for Benchmarking & Regression Gating
- **Objective**: Provide CI-ready commands: `reliability-lab run-benchmark` and `reliability-lab assert-regression`.
- **Priority**: P1 | **Dependencies**: `EXP-002`
- **Scope / Expected Files**: `reliability_lab/cli/main.py`, `tests/unit/test_cli.py`
- **Acceptance Criteria**: Returns non-zero exit code if regression threshold is violated.
- **Verification Command**: `pytest tests/unit/test_cli.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [UI-002] Streamlit Run Explorer & Failure Diagnostics Dashboard
- **Objective**: Visual diagnostic studio for side-by-side run comparisons, retrieval failure inspection, and citation auditing.
- **Priority**: P2 | **Dependencies**: `EXP-001`
- **Scope / Expected Files**: `app/app.py`, `app/pages/run_explorer.py`, `app/pages/citation_audit.py`
- **Acceptance Criteria**: Satisfies permanent UI/UX quality standards; runs locally without runtime errors.
- **Verification Command**: Headless smoke test of Streamlit pages.
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 9: Optional Gemini Integration & LLM-as-a-Judge

### [LLM-001] Abstract LLM Client & Gemini Flash Wrapper
- **Objective**: Pluggable interface for answer generation and evaluation with rate limiting and mock fallbacks.
- **Priority**: P2 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/llm/client.py`, `tests/unit/test_llm_client.py`
- **Acceptance Criteria**: Works with API key when present; falls back gracefully to deterministic mocks in tests.
- **Verification Command**: `pytest tests/unit/test_llm_client.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [LLM-002] G-Eval / Ragas-Style LLM Judge with Chain-of-Thought
- **Objective**: Structured prompt templates for evaluating faithfulness and citation support with reasoning chains.
- **Priority**: P2 | **Dependencies**: `LLM-001`, `GROUND-001`
- **Scope / Expected Files**: `reliability_lab/llm/judge.py`, `tests/unit/test_llm_judge.py`
- **Acceptance Criteria**: Schema-validated JSON output; calculates correlation between LLM judge and local NLI scores.
- **Verification Command**: `pytest tests/unit/test_llm_judge.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 10: Reproducibility, Packaging & Portfolio Presentation

### [PUB-001] Containerization & Reproducibility Suite
- **Objective**: Dockerfile and docker-compose configurations to run benchmarks and dashboard in a single command.
- **Priority**: P2 | **Dependencies**: `UI-001`, `UI-002`
- **Scope / Expected Files**: `Dockerfile`, `docker-compose.yml`, `scripts/run_benchmarks.sh`
- **Acceptance Criteria**: Docker image builds cleanly and runs the test suite and dashboard.
- **Verification Command**: Docker build and execution verification.
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [PUB-002] Architecture Case Study & Benchmark Report
- **Objective**: Formal technical report detailing IR trade-offs (BM25 vs FAISS vs Hybrid) and placement portfolio write-up.
- **Priority**: P2 | **Dependencies**: All prior tasks
- **Scope / Expected Files**: `docs/CASE_STUDY.md`, `README.md`
- **Acceptance Criteria**: Complete empirical findings, regression case studies, and architecture diagrams documented.
- **Verification Command**: Document review.
- **Status**: `BACKLOG`
- **Verification Evidence**: None
