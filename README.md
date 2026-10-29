# AI Reliability Lab

Industrial-grade reliability, evaluation, and regression testing platform for document-based AI and RAG architectures.

## Overview
AI Reliability Lab evaluates retrieval quality, verifies groundedness against source documents, validates citations, and detects regressions across model and configuration changes.

## Engineering Standards
- **Baseline-First IR**: Exact lexical baselines (BM25) before vector and hybrid approaches.
- **Deterministic Metrics**: Hand-calculated Cranfield ranking metrics (Recall@K, MRR@K, NDCG@K) and claim-level verification.
- **Protected CI/CD & Auto-Merge**: Automated regression gating on every pull request.
- **Modular Architecture**: Decoupled domain models, parsers, retrievers, metrics, and persistence layers.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dpr2106/ai-reliability-lab.git
cd ai-reliability-lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Run static checks and tests
ruff check .
mypy reliability_lab
pytest
```

## Command-Line Interface (CLI)

AI Reliability Lab provides a headless CLI (`reliability-lab`) for automated benchmark execution and CI/CD regression gating.

### 1. Run Retrieval Benchmark (`run-benchmark`)

Executes BM25 retrieval over a corpus and benchmark dataset, computes Cranfield IR metrics and latency percentiles, and persists the result to SQLite:

```bash
# Run benchmark with human-readable summary output
reliability-lab run-benchmark \
  --corpus data/benchmarks/sample_corpus.jsonl \
  --dataset data/benchmarks/sample_benchmark.jsonl \
  --database .reliability-lab/experiments.sqlite3 \
  --top-k 5 \
  --experiment-name bm25-baseline

# Or emit machine-readable JSON for automated tooling
reliability-lab run-benchmark --output json
```

Key options:
- `--corpus`: Path to JSONL corpus chunks (default: `data/benchmarks/sample_corpus.jsonl`).
- `--dataset`: Path to JSONL query dataset (default: `data/benchmarks/sample_benchmark.jsonl`).
- `--database`: Path to SQLite database destination (default: `.reliability-lab/experiments.sqlite3`).
- `--top-k`: Ranking cutoff integer (default: `5`).
- `--output`: Output format (`text` or `json`).

### 2. Assert Regression Gating (`assert-regression`)

Compares a candidate run against a baseline run stored in the same database, detecting metric degradations beyond configured tolerances:

```bash
# Compare candidate run against baseline run
reliability-lab assert-regression \
  --baseline <baseline-run-id> \
  --candidate <candidate-run-id> \
  --database .reliability-lab/experiments.sqlite3 \
  --tolerance 0.02
```

Configure per-metric overrides when tighter or looser bounds are required:
```bash
reliability-lab assert-regression \
  --baseline <baseline-run-id> \
  --candidate <candidate-run-id> \
  --tolerance 0.02 \
  --metric-tolerance mrr@5=0.01 \
  --metric-tolerance recall@5=0.05
```

### Exit Codes & CI Regression Gating

`assert-regression` returns deterministic exit codes designed for continuous integration pipelines:

| Exit Code | Status | Meaning |
|---|---|---|
| `0` | `PASS` | Comparison completed; no metrics or queries regressed beyond tolerance. |
| `1` | `REGRESSION` | At least one aggregate or per-query metric degraded beyond tolerance. |
| `2` | `INCOMPLETE` | Query-set mismatch or missing data prevents a clean comparison. |
| `3` | Operational Error | Run ID not found, database missing, invalid arguments, or non-finite tolerance. |

### Reusing Run IDs & Database Hygiene
- When running `run-benchmark`, the CLI outputs a unique `run_id` (UUID format). Save this ID to track baseline and candidate runs.
- Baseline and candidate runs must reside in the same `--database` path for comparison. Attempting to compare runs stored across different databases returns exit code `3` (`RunNotFoundError`).

