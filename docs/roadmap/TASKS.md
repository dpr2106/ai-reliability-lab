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
  - Test Suite: 5 passed in 0.20s (`tests/unit/test_documentation.py`, `tests/unit/test_smoke.py`)
  - Verification Command: `ruff check . && ruff format --check . && mypy reliability_lab && pytest --cov=reliability_lab -v`
  - Canonical Files: `GEMINI.md`, `docs/PROJECT_BRIEF.md`, `docs/ARCHITECTURE.md`, `docs/STATUS.md`, `docs/DECISIONS.md`, `docs/roadmap/TASKS.md` all verified present and consistent.

---

### [FOUND-003] Core Domain Entities & Protocol Interfaces
- **Objective**: Implement pure domain dataclasses (`Document`, `Chunk`, `Query`, `RetrievalResult`, `MetricResult`) with zero external library dependencies.
- **Priority**: P0 | **Dependencies**: `FOUND-002`
- **Scope / Expected Files**: `reliability_lab/core/models.py`, `reliability_lab/core/interfaces.py`, `tests/unit/test_core_models.py`
- **Acceptance Criteria**:
  - Pydantic models with validation and serialization methods.
  - Abstract base classes (`BaseRetriever`, `BaseChunker`, `BaseMetric`).
  - Unit tests verify immutable properties and JSON round-trips.
- **Verification Command**: `pytest tests/unit/test_core_models.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [FOUND-004] Enhanced CI Review & Branch Protection Audit
- **Objective**: Verify branch protection rules, required CI checks, and automated review annotations on GitHub.
- **Priority**: P0 | **Dependencies**: `FOUND-002`
- **Scope / Expected Files**: `.github/workflows/ci.yml`
- **Acceptance Criteria**:
  - Direct pushes to `main` are rejected.
  - Auto-merge completes only when `test-and-lint` passes.
- **Verification Command**: `gh pr checks` and branch protection query.
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 1: Document Ingestion & Citation-Aware Chunking

### [ING-001] PyMuPDF Document Parser with Page & Bounding Span Tracking
- **Objective**: Extract text per page while preserving line numbers, page indexes, and character offsets for citation audits.
- **Priority**: P0 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/ingestion/parser.py`, `tests/unit/test_parser.py`
- **Acceptance Criteria**: Extracts clean text, handles encrypted/corrupt PDFs gracefully, records bounding metadata.
- **Verification Command**: `pytest tests/unit/test_parser.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [ING-002] Citation-Aware Sliding & Recursive Chunkers
- **Objective**: Chunk documents into token/character windows while preserving parent document IDs, page spans, and offsets.
- **Priority**: P0 | **Dependencies**: `ING-001`
- **Scope / Expected Files**: `reliability_lab/ingestion/chunker.py`, `tests/unit/test_chunker.py`
- **Acceptance Criteria**: Supports configurable chunk size and overlap; verifies zero metadata leakage across boundaries.
- **Verification Command**: `pytest tests/unit/test_chunker.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 2: Baseline Lexical Retrieval (BM25)

### [RET-001] In-Memory BM25 Lexical Index
- **Objective**: Implement an in-memory Okapi BM25 sparse index as the foundational IR baseline.
- **Priority**: P0 | **Dependencies**: `ING-002`
- **Scope / Expected Files**: `reliability_lab/indexing/bm25_index.py`, `tests/unit/test_bm25_index.py`
- **Acceptance Criteria**: Constructs index from `List[Chunk]`, retrieves ranked results with scores, handles out-of-vocabulary terms.
- **Verification Command**: `pytest tests/unit/test_bm25_index.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 3: Manually Verified Golden Evaluation Dataset

### [DATA-001] Benchmark Schema & Golden Dataset Construction
- **Objective**: Build a verified benchmark dataset containing queries, multi-grade chunk relevances, and required citations.
- **Priority**: P0 | **Dependencies**: `ING-002`
- **Scope / Expected Files**: `data/benchmarks/schema.json`, `data/benchmarks/sample_benchmark.jsonl`, `tests/unit/test_dataset_schema.py`
- **Acceptance Criteria**: At least 15 verified queries with graded relevance (0=irrelevant, 1=relevant, 2=critical evidence).
- **Verification Command**: `pytest tests/unit/test_dataset_schema.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 4: Cranfield Retrieval Evaluation Metrics

### [EVAL-001] Set-Based Retrieval Metrics: Recall@K, Precision@K, HitRate@K
- **Objective**: Implement exact set-based retrieval diagnostics with hand-calculated unit tests.
- **Priority**: P0 | **Dependencies**: `FOUND-003`, `DATA-001`
- **Scope / Expected Files**: `reliability_lab/evaluation/retrieval_metrics.py`, `tests/unit/test_retrieval_metrics.py`
- **Acceptance Criteria**: Unit tests match hand calculations for empty results, $K > N$, and zero-ground-truth edge cases.
- **Verification Command**: `pytest tests/unit/test_retrieval_metrics.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [EVAL-002] Positional Metrics: MRR@K and Graded NDCG@K
- **Objective**: Implement Mean Reciprocal Rank and Normalized Discounted Cumulative Gain using standard logarithmic discounting.
- **Priority**: P0 | **Dependencies**: `EVAL-001`
- **Scope / Expected Files**: `reliability_lab/evaluation/retrieval_metrics.py`
- **Acceptance Criteria**: NDCG@K matches standard reference calculations for arbitrary graded gains.
- **Verification Command**: `pytest tests/unit/test_retrieval_metrics.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [EVAL-003] End-to-End Retrieval Benchmark Runner
- **Objective**: Execute an index against the golden benchmark and compile aggregate reports and query failure traces.
- **Priority**: P0 | **Dependencies**: `RET-001`, `EVAL-002`, `DATA-001`
- **Scope / Expected Files**: `reliability_lab/evaluation/benchmark_runner.py`, `tests/integration/test_benchmark_runner.py`
- **Acceptance Criteria**: Produces an `EvaluationReport` with mean Recall@5, MRR@5, NDCG@5, and latency percentiles.
- **Verification Command**: `pytest tests/integration/test_benchmark_runner.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

## Milestone 5: Dense Retrieval, FAISS & Hybrid Ranking

### [DENSE-001] Local Sentence-Transformers Embedding Pipeline
- **Objective**: Provide reproducible, offline vector generation using lightweight models (`all-MiniLM-L6-v2`).
- **Priority**: P1 | **Dependencies**: `FOUND-003`
- **Scope / Expected Files**: `reliability_lab/indexing/embeddings.py`, `tests/unit/test_embeddings.py`
- **Acceptance Criteria**: Batch embedding generation with dimension checks, device fallback, and L2 normalization.
- **Verification Command**: `pytest tests/unit/test_embeddings.py -v`
- **Status**: `BACKLOG`
- **Verification Evidence**: None

---

### [DENSE-002] FAISS Vector Index (IndexFlatIP & IndexHNSWFlat)
- **Objective**: Benchmark exact inner product against approximate graph search for latency vs recall trade-offs.
- **Priority**: P1 | **Dependencies**: `DENSE-001`
- **Scope / Expected Files**: `reliability_lab/indexing/faiss_index.py`, `tests/unit/test_faiss_index.py`
- **Acceptance Criteria**: Supports serialization to disk, add/search operations, and mapping integer IDs to chunk UUIDs.
- **Verification Command**: `pytest tests/unit/test_faiss_index.py -v`
- **Status**: `BACKLOG`
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
