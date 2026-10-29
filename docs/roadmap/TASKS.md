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
- **Status**: `DONE`
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
  - Rebase merge workflow verified via PR #24 preserving 4 distinct feature-branch commits on `main`: `be5be7f`, `6d99ddf`, `c0cce2a`, and `181cfb4`.
  - Required CI verified on PR #24: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40640232072/job/122656417671) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40640232072/job/122656417639).
  - Main branch commit count verified via `git rev-list --count origin/main`: 33 commits reachable from `main`.

---

### [DENSE-002] FAISS Vector Index (IndexFlatIP & IndexHNSWFlat)
- **Objective**: Benchmark exact inner product against approximate graph search for latency vs recall trade-offs.
- **Priority**: P1 | **Dependencies**: `DENSE-001`
- **Scope / Expected Files**: `reliability_lab/indexing/faiss_index.py`, `tests/unit/test_faiss_index.py`
- **Acceptance Criteria**: Supports serialization to disk, add/search operations, and mapping integer IDs to chunk UUIDs.
- **Verification Command**: `pytest tests/unit/test_faiss_index.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `23 passed in 0.84s` (`pytest tests/unit/test_faiss_index.py -v`, exit code 0).
  - BM25 regression tests: `53 passed in 0.37s` (`pytest tests/unit/test_bm25_index.py -v`, exit code 0).
  - Documentation tests: `6 passed in 0.10s` (`pytest tests/unit/test_documentation.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (35 files already formatted), `mypy reliability_lab` (Success: no issues found in 14 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (363 passed, 93% package coverage, exit code 0).
  - `FaissVectorRetriever` implemented supporting `IndexFlatIP` (exact) and `IndexHNSWFlat` (approximate) wrapped with `IndexIDMap2` to normalize integer ID assignment and guarantee stable Chunk provenance reconstruction.
  - Verified incremental chunk additions, corpus replacement (`index()`), atomic persistence (`save()` and `load()`), and post-reload incremental expansion.
  - Verified safe L2 normalization without zero-norm NaNs, rejection of non-finite vectors, and dimension matching against injected `BaseEmbedder`.
  - Controlled benchmark verified: HNSW achieved $\ge 90\%$ recall relative to exact Flat on synthetic vectors.
  - Optional dependency isolation verified: Tier 1 BM25 executes independently of FAISS, and missing runtime FAISS triggers actionable `ImportError`.
  - Rebase merge workflow verified via PR #26 preserving 4 distinct feature-branch commits on `main`: `adc6471`, `f852094`, `05b59e7`, and `602d3b7`.
  - Required CI verified on PR #26: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40643871110/job/122668044588) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40643871110/job/122668044742).
  - Main branch commit count verified via `git rev-list --count origin/main`: 38 commits reachable from `main`.

---

### [DENSE-003] Hybrid Search with Reciprocal Rank Fusion (RRF)
- **Objective**: Combine sparse (BM25) and dense (FAISS) rankings without score scale discrepancies.
- **Priority**: P1 | **Dependencies**: `RET-001`, `DENSE-002`
- **Scope / Expected Files**: `reliability_lab/indexing/hybrid.py`, `tests/unit/test_hybrid.py`
- **Acceptance Criteria**: Implements $RRF(d) = \sum \frac{1}{k + r_i(d)}$; unit tests verify fused ranking matches math.
- **Verification Command**: `pytest tests/unit/test_hybrid.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `25 passed in 0.57s` (`pytest tests/unit/test_hybrid.py -v`, exit code 0).
  - BM25 regression tests: `53 passed in 0.37s` (`pytest tests/unit/test_bm25_index.py -v`, exit code 0).
  - FAISS regression tests: `23 passed in 0.72s` (`pytest tests/unit/test_faiss_index.py -v`, exit code 0).
  - Documentation tests: `6 passed in 0.18s` (`pytest tests/unit/test_documentation.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (37 files already formatted), `mypy reliability_lab` (Success: no issues found in 15 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (388 passed in 13.41s, 97% coverage on `reliability_lab/indexing/hybrid.py`, 93% total package coverage, exit code 0).
  - `HybridRetriever` implemented supporting unweighted Reciprocal Rank Fusion $\operatorname{RRF}(d) = \sum \frac{1}{k + r_m(d)}$ with configurable $k$ (default 60), ranking solely on rank positions to eliminate score-scale discrepancies.
  - Mathematical correctness verified against hand calculations for multi-hit candidates ($1/61 + 1/62 = 0.032522...$) and single-hit candidates ($1/62$).
  - Strict deduplication by `chunk_id` with cross-retriever provenance conflict detection, preserving original `Chunk` domain entity and rich JSON-serializable diagnostic metadata.
  - Deterministic 4-level tie-breaking hierarchy (fused score -> retriever count consensus -> minimum individual rank -> chunk ID).
  - Lifecycle defense verified: unindexed retrieval raises `RuntimeError`, child indexing failures leave hybrid unindexed.
  - End-to-end integration verified with `RetrievalBenchmarkRunner` across golden benchmark and corpus.
  - Real-backend integration verified combining actual `BM25Retriever` and `FaissVectorRetriever`.
  - Rebase merge workflow verified via PR #28 preserving 3 distinct feature-branch commits on `main`: `88ef512`, `f834c52`, and `5ef4589`.
  - Required CI verified on PR #28: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40645878099/job/122674442819) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40645878099/job/122674443067).
  - Main branch commit count verified via `git rev-list --count origin/main`: 42 commits reachable from `main`.

---

## Milestone 6: Groundedness & Citation Evaluation

### [GROUND-001] Citation Parser & Reference Link Validator
- **Objective**: Parse citation markers in generated text and verify whether cited chunks exist in the retrieved set.
- **Priority**: P1 | **Dependencies**: `ING-002`, `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/evaluation/citation_metrics.py`, `tests/unit/test_citation_metrics.py`
- **Acceptance Criteria**: Calculates Citation Precision and Citation Recall against ground truth citations.
- **Verification Command**: `pytest tests/unit/test_citation_metrics.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `26 passed in 0.16s` (`pytest tests/unit/test_citation_metrics.py -v`, exit code 0).
  - Core model & dataset regression tests: `94 passed in 2.59s` (`pytest tests/unit/test_core_models.py tests/unit/test_dataset_schema.py tests/integration/test_benchmark_runner.py -v`, exit code 0).
  - Documentation tests: `6 passed in 0.10s` (`pytest tests/unit/test_documentation.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (39 files already formatted), `mypy reliability_lab` (Success: no issues found in 16 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (414 passed in 8.21s, 96% coverage on `reliability_lab/evaluation/citation_metrics.py`, 94% total package coverage, exit code 0).
  - Deterministic `CitationParser` extracts `[CIT-xxx]`, `[CHK-xxx]`, and composite `[CIT-001, CIT-002]` citation markers while ignoring unrelated bracketed prose (`[see above]`, `[123]`) and detecting malformed citation candidates (`[CIT-]`, `[citation: 1]`).
  - Reference resolution validates whether referenced chunk IDs exist within the actual retrieved candidate set (`Sequence[RetrievalResult]`).
  - Citation Precision and Citation Recall calculated with strict deduplication preventing marker repetition inflation, and deterministic zero-denominator contracts.
  - Backward-compatible citation schema preservation: `load_benchmark_dataset()` preserves structured `citation_records` and `citation_to_chunk_id` in `Query.metadata` and rejects conflicting mappings, while preserving `Query.required_citations: list[str]`.
  - Rebase merge workflow verified via PR #30 preserving 4 distinct feature-branch commits on `main`: `b679d9b`, `d539347`, `8ffa833`, and `a5d35e8`.
  - Required CI verified on PR #30: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40677368106/job/122779097224) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40677368106/job/122779096920).
  - Main branch commit count verified via `git rev-list --count origin/main`: 47 commits reachable from `main`.

---

### [GROUND-002] Deterministic Claim Span & Lexical Overlap Grounding
- **Objective**: Fast, 100% local metric evaluating token containment and longest common subsequence between answers and source chunks.
- **Priority**: P1 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/evaluation/groundedness.py`, `tests/unit/test_groundedness.py`
- **Acceptance Criteria**: Accurately flags completely unsupported sentences vs fully grounded sentences.
- **Verification Command**: `pytest tests/unit/test_groundedness.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `27 passed in 0.28s` (`pytest tests/unit/test_groundedness.py -v`, exit code 0).
  - Existing metric regression tests: `32 passed in 0.26s` (`pytest tests/unit/test_citation_metrics.py tests/unit/test_documentation.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (41 files already formatted), `mypy reliability_lab` (Success: no issues found in 17 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (441 passed in 5.73s, 98% coverage on `reliability_lab/evaluation/groundedness.py`, 94% total package coverage, exit code 0).
  - `GroundednessEvaluator`, `ClaimGroundingResult`, `GroundednessResult`, `LexicalGroundednessMetric`, and core calculation functions (`compute_token_containment`, `compute_lcs_length`, `compute_token_lcs_score`, `extract_claim_spans`, `tokenize`) implemented and exported from `reliability_lab.evaluation`.
  - Deterministic sentence-level claim segmentation extracts exact character offsets (`answer_text[start:end] == claim_text`) while protecting decimal numbers (`1.5`, `0.75`, `25.5ms`) and handling Unicode and line breaks.
  - Independent candidate chunk evaluation with deterministic tie-breaking (containment $\to$ LCS $\to$ lexicographical chunk ID) preventing cross-chunk synthetic concatenation.
  - Configurable dual-threshold decision rule (`containment >= 0.5`, `lcs >= 0.3`) and aggregate score bounded in `[0.0, 1.0]` with `has_claims` tracking.
  - Rebase merge workflow verified via PR #32 preserving 3 distinct feature-branch commits on `main`: `2aaca49`, `5b5be8f`, and `edf5cc5`.
  - Required CI verified on PR #32: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40679783719/job/122787249169) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40679783719/job/122787248638).
  - Main branch commit count verified via `git rev-list --count origin/main`: 51 commits reachable from `main`.

---

### [GROUND-003] Local NLI Entailment Classifier
- **Objective**: Classify chunk premise and answer hypothesis pairs into Entailment, Neutral, or Contradiction.
- **Priority**: P2 | **Dependencies**: `GROUND-002`
- **Scope / Expected Files**: `reliability_lab/evaluation/nli_evaluator.py`, `tests/unit/test_nli_evaluator.py`
- **Acceptance Criteria**: Returns probability distribution across NLI labels and computes aggregate Hallucination Score.
- **Verification Command**: `pytest tests/unit/test_nli_evaluator.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `28 passed in 0.29s` (`pytest tests/unit/test_nli_evaluator.py -v`, exit code 0).
  - Groundedness & citation regression tests: `87 passed in 0.55s` (`pytest tests/unit/test_nli_evaluator.py tests/unit/test_groundedness.py tests/unit/test_citation_metrics.py tests/unit/test_documentation.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (43 files already formatted), `mypy reliability_lab` (Success: no issues found in 18 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (469 passed in 5.44s, 83% coverage on `reliability_lab/evaluation/nli_evaluator.py`, 93% total package coverage, exit code 0).
  - `NLILabel` (`ENTAILMENT`, `NEUTRAL`, `CONTRADICTION`), `BaseNLIBackend`, `HuggingFaceNLIBackend`, `MockNLIBackend`, `ClaimNLIResult`, `NLIResult`, `NLIEntailmentEvaluator`, `NLIHallucinationMetric`, and `validate_nli_distribution` implemented and exported from `reliability_lab.evaluation`.
  - Hallucination Score defined transparently: $s_i = \max_{c \in E_i} P(\text{entailment} \mid c, \text{claim}_i)$, $h_i = 1 - s_i$, $H_{\text{answer}} = \frac{1}{N}\sum_{i=1}^N h_i$.
  - Empty-input contracts verified: no claims $\to$ $H_{\text{answer}}=0.0$ (`has_claims=False`); claims without evidence $\to$ $H_{\text{answer}}=1.0$ (`no_evidence_available=True`).
  - Deterministic tie-breaking on equal entailment probabilities using lexicographically smallest `chunk_id`.
  - Claim extraction reuses `extract_claim_spans()` preserving exact character spans and answer slice provenance.
  - Optional dependency isolation verified: `[nli]` extra added in `pyproject.toml` (`transformers>=4.30.0`, `torch>=2.0.0`), lazy import guarantees zero model downloads or torch/transformers initialization on import; deterministic tests run with zero network/disk weight dependencies via `MockNLIBackend`.
  - Rebase merge workflow verified via PR #34 preserving 3 distinct feature-branch commits on `main`: `135ef45`, `a699c81`, and `8abb889`.
  - Required CI verified on PR #34: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40689587840/job/122820677532) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40689587840/job/122820677258).
  - Main branch commit count verified via `git rev-list --count origin/main`: 55 commits reachable from `main`.

---

## Milestone 7: Experiment Storage & Regression Detection

### [EXP-001] SQLite Schema & Experiment Run Repository
- **Objective**: Persist benchmark runs, index configurations, and query metric scores in relational storage.
- **Priority**: P1 | **Dependencies**: `EVAL-003`
- **Scope / Expected Files**: `reliability_lab/storage/database.py`, `reliability_lab/storage/repository.py`, `tests/unit/test_storage.py`
- **Acceptance Criteria**: Schema stores configuration hashes, run timestamps, and per-query score breakdowns.
- **Verification Command**: `pytest tests/unit/test_storage.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `30 passed in 0.46s` (`pytest tests/unit/test_storage.py -v`, exit code 0).
  - Documentation and integration regression tests: `29 passed in 0.72s` (`pytest tests/unit/test_documentation.py tests/integration/test_benchmark_runner.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (47 files already formatted), `mypy reliability_lab` (Success: no issues found in 21 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (499 passed in 9.02s, 94% coverage on `database.py`, 91% coverage on `repository.py`, 93% total package coverage, exit code 0).
  - `Database`, `ExperimentRepository`, `StoredRun`, `StoredRunSummary`, `compute_config_hash`, `canonicalize_config`, and custom storage exceptions implemented and exported from `reliability_lab.storage`.
  - Normalized relational schema with foreign key cascades: `index_configurations`, `experiment_runs`, `run_metrics`, `query_results`, `query_metric_scores`, and indexes.
  - Zero external dependencies: pure standard-library `sqlite3` without ORM coupling; ADR-005 preserved with zero external dependencies in core.
  - Deterministic configuration hashing via canonical JSON serialization and SHA-256 digests.
  - Transactional ACID boundaries with idempotent replay on exact duplicates and explicit exception raising on conflicting duplicate runs or hash collisions.
  - Rebase merge workflow verified via PR #36 preserving 3 distinct feature-branch commits on `main`: `812a9b8`, `caf7640`, and `49aee42`.
  - Required CI verified on PR #36: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40695551765/job/122841239250) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40695551765/job/122841239695).
  - Main branch commit count verified via `git rev-list --count origin/main`: 59 commits reachable from `main`.

---

### [EXP-002] Regression Detection Engine & Metric Diffing
- **Objective**: Compare Run B against Baseline Run A and identify queries experiencing ranking degradation.
- **Priority**: P1 | **Dependencies**: `EXP-001`
- **Scope / Expected Files**: `reliability_lab/storage/regression.py`, `tests/unit/test_regression.py`
- **Acceptance Criteria**: Computes $\Delta \text{MRR}$ and $\Delta \text{NDCG}$; flags queries exceeding regression tolerance.
- **Verification Command**: `pytest tests/unit/test_regression.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Focused unit tests: `20 passed in 0.14s` (`pytest tests/unit/test_regression.py -v`, exit code 0).
  - Storage, documentation, and benchmark runner regression tests: `79 passed in 0.71s` (`pytest tests/unit/test_regression.py tests/unit/test_storage.py tests/unit/test_documentation.py tests/integration/test_benchmark_runner.py -v`, exit code 0).
  - Code quality checks: `ruff check .` (All checks passed!), `ruff format --check .` (49 files already formatted), `mypy reliability_lab` (Success: no issues found in 22 source files).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (519 passed in 8.65s, 94% coverage on `reliability_lab/storage/regression.py`, 93% total package coverage, exit code 0).
  - `RegressionDetector`, `RegressionResult`, `MetricDelta`, `QueryMetricDelta`, `QueryRegression`, `RegressionStatus`, `MetricDirection`, `RegressionError`, `IncomparableRunsError`, `InvalidToleranceError`, `normalize_metric_name`, `is_ranking_metric`, `is_degradation_regressed`, and `validate_tolerance` implemented and exported from `reliability_lab.storage`.
  - Signed delta ($\Delta_m = m_{\text{candidate}} - m_{\text{baseline}}$) and degradation magnitude ($D_m = m_{\text{baseline}} - m_{\text{candidate}} = -\Delta_m$) for ranking metrics.
  - Deterministic threshold check ($D_m > \tau_m$) with floating-point equality protection via `math.isclose` (equality is strictly non-regressed).
  - Configurable global tolerance (default $\tau = 0.02$) and per-metric overrides, validated against booleans, negative numbers, and non-finite floats.
  - Status classification (`PASS`, `REGRESSION`, `INCOMPLETE`) with fail-safe query-set mismatch handling (never zero-imputing missing queries).
  - Aggregate metric alias normalization and deduplication (`mean_mrr@5` -> `mrr@5`) with conflict detection (> 1e-5).
  - Rebase merge workflow verified via PR #38 preserving 3 distinct feature-branch commits on `main`: `e1d6c87`, `ac19eec`, and `536d7c8`.
  - Required CI verified on PR #38: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40699587103/job/122855440621) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40699587103/job/122855441208).
  - Main branch commit count verified via `git rev-list --count origin/main`: 63 commits reachable from `main`.

---

## Milestone 8: CLI & Streamlit Diagnostic Studio

### [UI-001] Headless CLI for Benchmarking & Regression Gating
- **Objective**: Provide CI-ready commands: `reliability-lab run-benchmark` and `reliability-lab assert-regression`.
- **Priority**: P1 | **Dependencies**: `EXP-002`
- **Scope / Expected Files**: `reliability_lab/cli/main.py`, `reliability_lab/cli/__init__.py`, `tests/unit/test_cli.py`, `pyproject.toml`, `README.md`
- **Acceptance Criteria**: Returns non-zero exit code if regression threshold is violated.
- **Verification Command**: `pytest tests/unit/test_cli.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Implementation PR #40 merged via rebase with commit SHA `d8c970a71fa59d3cf53cefcc1340064c28989770` on `main`.
  - Required CI verified on PR #40: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40858500355/job/123395448066) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40858500355/job/123395447862).
  - Focused test suite: `pytest tests/unit/test_cli.py -v` (27 passed in 1.64s, exit code 0).
  - Full test suite & coverage: `pytest --cov=reliability_lab` (546 passed in 14.37s, 93% package coverage, 88% on `reliability_lab/cli/main.py`, exit code 0).
  - Code quality: `ruff check .` clean, `ruff format --check .` clean, `mypy reliability_lab` clean.
  - Console entrypoint `reliability-lab` registered in `pyproject.toml` and verified via `pip install -e .`.
  - Commands verified: `reliability-lab run-benchmark` executes and stores runs in SQLite, `reliability-lab assert-regression` produces deterministic exit codes: PASS (0), REGRESSION (1), INCOMPLETE (2), and operational error (3).
  - Rebase merge workflow verified via PR #40 preserving 3 distinct feature commits on `main`: `4561979`, `db8ea75`, and `d8c970a`.
  - Main branch commit count verified via `git rev-list --count origin/main`: 67 commits reachable from `main`.

---

### [UI-002] Svelte Run Explorer & Failure Diagnostics Studio
- **Objective**: Modern visual diagnostic studio for side-by-side run comparisons, retrieval failure inspection, and citation auditing built with SvelteKit and FastAPI.
- **Priority**: P2 | **Dependencies**: `EXP-001`
- **Scope / Expected Files**: `frontend/`, `reliability_lab/api/`, `tests/unit/test_api.py`, `docs/DECISIONS.md`, `docs/ARCHITECTURE.md`
- **Acceptance Criteria**: Decoupled architecture with SvelteKit frontend (TypeScript, dark theme, runes) and FastAPI backend adapter; runs locally with full static check coverage and zero test failures.
- **Verification Command**:
  ```bash
  pytest tests/unit/test_api.py -v
  cd frontend && npm run check && npm run test && npm run build
  ```
- **Status**: `DONE`
- **Verification Evidence**:
  - `pytest tests/unit/test_api.py -v`: 15 passed in 0.24s (exit code 0).
  - Full Python suite: `pytest`: 561 passed in 11.33s (exit code 0).
  - `ruff check .`: `All checks passed!` (exit code 0).
  - `ruff format --check .`: `57 files already formatted` (exit code 0).
  - `mypy reliability_lab`: `Success: no issues found in 27 source files` (exit code 0).
  - Frontend typecheck: `npm run check`: `svelte-check found 0 errors and 0 warnings` (exit code 0).
  - Frontend unit tests: `npm run test`: `6 passed in 883ms` (exit code 0).
  - Frontend build: `npm run build`: `built in 647ms` (exit code 0).
  - PR #42 merged to `main` via rebase merge, preserving 5 commits (`71f7bc5508e1706629e2c5b5f1e25e593d37355f` through `f770e2baffb6aa8cc61805cb6c9849b20f635a72`).
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40866649511/job/123422980555) (37s) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40866649511/job/123422980814) (42s); both completed successfully before merge.


---

## Milestone 9: Optional Gemini Integration & LLM-as-a-Judge

### [LLM-001] Abstract LLM Client & Gemini Flash Wrapper
- **Objective**: Pluggable interface for answer generation and evaluation with rate limiting and mock fallbacks.
- **Priority**: P2 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/llm/client.py`, `reliability_lab/llm/__init__.py`, `reliability_lab/llm/models.py`, `reliability_lab/llm/rate_limiter.py`, `reliability_lab/llm/exceptions.py`, `tests/unit/test_llm_client.py`
- **Acceptance Criteria**: Works with API key when present; falls back gracefully to deterministic mocks in tests.
- **Verification Command**: `pytest tests/unit/test_llm_client.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - `pytest tests/unit/test_llm_client.py -v`: 27 passed in 2.18s (exit code 0).
  - Full Python suite: `pytest`: 588 passed in 26.98s (exit code 0).
  - `ruff check .`: `All checks passed!` (exit code 0).
  - `ruff format --check .`: `63 files already formatted` (exit code 0).
  - `mypy reliability_lab`: `Success: no issues found in 32 source files` (exit code 0).
  - Hermetic testing: Zero external network egress; validated with `MockLLMClient` and `httpx.MockTransport`.
  - PR #44 merged to `main` via rebase merge, preserving 5 commits (`f806dd5` through `0d0f88a`).
  - Required CI: [`test-and-lint (3.11)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40869554190/job/123432821038) (39s) and [`test-and-lint (3.12)` passed](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40869554190/job/123432820614) (47s); both completed successfully before merge.

---

### [LLM-002] G-Eval / Ragas-Style LLM Judge with Chain-of-Thought
- **Objective**: Structured prompt templates for evaluating faithfulness and citation support with reasoning chains.
- **Priority**: P2 | **Dependencies**: `LLM-001`, `GROUND-001`
- **Scope / Expected Files**: `reliability_lab/llm/judge.py`, `tests/unit/test_llm_judge.py`
- **Acceptance Criteria**: Schema-validated JSON output; calculates correlation between LLM judge and local NLI scores.
- **Verification Command**: `pytest tests/unit/test_llm_judge.py -v`
- **Status**: `DONE`
- **Verification Evidence**:
  - Rebase Merge Commits: `9deedab`, `7226646`, `9449e84` on `main` ([PR #46](https://github.com/dpr2106/ai-reliability-lab/pull/46))
  - GitHub Actions Run: [40874098043](https://github.com/dpr2106/ai-reliability-lab/actions/runs/40874098043) (Success on Python 3.11 & 3.12)
  - Test Suite: 609 passed, 1 warning in 20.13s, 93% total package coverage (21 unit tests in `tests/unit/test_llm_judge.py`)

---

## Milestone 10: Reproducibility, Packaging & Portfolio Presentation

### [PUB-001] Containerization & Reproducibility Suite
- **Objective**: Dockerfile and docker-compose configurations to run benchmarks and dashboard in a single command.
- **Priority**: P2 | **Dependencies**: `UI-001`, `UI-002`
- **Scope / Expected Files**: `Dockerfile`, `docker-compose.yml`, `scripts/run_benchmarks.sh`
- **Acceptance Criteria**: Docker image builds cleanly and runs the test suite and dashboard.
- **Verification Command**: Docker build and execution verification.
- **Status**: `READY`
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
