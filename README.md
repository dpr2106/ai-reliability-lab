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

---

## Diagnostics Studio (SvelteKit + FastAPI)

AI Reliability Lab provides an interactive, dark-themed diagnostics studio for deep failure analysis, side-by-side run diffing, and citation auditing.

### 1. Start the API Adapter
```bash
# Set path to experiments database (defaults to .reliability-lab/experiments.sqlite3)
export RELIABILITY_LAB_DB=".reliability-lab/experiments.sqlite3"

# Start the FastAPI service
uvicorn reliability_lab.api.main:app --reload --port 8000
```

### 2. Start the SvelteKit Frontend
```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173` in your browser to access:
- **Run Explorer** (`/`): Search, filter, and inspect benchmark runs.
- **Run Diagnostics** (`/runs/[run_id]`): Detailed aggregate metrics and per-query failure trace drill-downs.
- **Run Comparison** (`/compare`): Side-by-side regression diffing with tolerance slider and gate status banner.
- **Citation Audit** (`/citation-audit`): Verify inline citations, resolution maps, precision, and recall across answers.

---

## Docker & Containerized Quickstart

AI Reliability Lab includes production-grade containerization and a Docker Compose suite to build, launch, test, and benchmark the entire platform in isolated containers without configuring Python or Node.js locally.

### Prerequisites
- [Docker Engine](https://docs.docker.com/engine/) (v20.10+)
- [Docker Compose](https://docs.docker.com/compose/) (v2.0+)

### 1. Start the Full Application (Backend + Frontend)
Launch the FastAPI backend adapter and SvelteKit diagnostics studio with persistent SQLite storage:

```bash
docker compose up --build -d
```

Verify service status and access endpoints:
- **Diagnostic Studio (Frontend)**: [http://localhost:5173](http://localhost:5173)
- **REST API & Interactive Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Backend Health Check**: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### 2. Run Automated Verification Tests
Execute the full test suite, linting, formatting, and static typing inside the isolated container test environment:

```bash
docker compose run --rm tests
```

To run frontend tests and static checks independently:
```bash
cd frontend
npm run check
npm test
npm run build
```

### 3. Run Reproducible Benchmarks
Execute the retrieval benchmark and end-to-end RAG evaluation script, storing results in the shared persistent SQLite volume:

```bash
# Execute standard reproducible benchmark run (offline mock provider)
docker compose run --rm benchmark
```

To customize benchmark parameters via environment variables:
```bash
docker compose run --rm \
  -e TOP_K=10 \
  -e EXPERIMENT_NAME=top10-benchmark \
  benchmark
```

### 4. LLM Provenance: Mock vs. Live Gemini Execution
- **Default (Offline Mock)**: Out-of-the-box, the benchmark runner executes using the deterministic mock LLM provider (`LLM_PROVIDER=mock`). Results are labeled with `is_mock=True` and a prominent provenance notice is displayed.
- **Live Gemini Inference**: To execute live evaluations against Google Gemini models:
  ```bash
  docker compose run --rm \
    -e LLM_PROVIDER=gemini \
    -e MODEL=gemini-2.5-flash \
    -e GEMINI_API_KEY="your-api-key" \
    benchmark
  ```
  *Security Note: Never commit API keys or bake them into container images.*

### 5. Persistent Storage & Lifecycle Management
- **Persistent Volume**: Experiments are saved to an SQLite database located at `/data/experiments.sqlite3` inside the named volume `reliability-lab-data`.
- **Preserving Data**: Stopping containers with `docker compose down` or `docker compose stop` preserves all experiment history.
- **Resetting Data**: To deliberately wipe the experiment database and start fresh:
  ```bash
  docker compose down -v
  ```

### 6. Troubleshooting
- **Port Conflicts (`8000` or `5173` already in use)**: Check if local development servers are running. Stop them or adjust port mappings in `docker-compose.yml`.
- **CORS Errors**: The FastAPI backend permits origins matching `http://localhost:5173` and `http://127.0.0.1:5173`. Ensure browser requests target the published host address (`http://127.0.0.1:8000`).
- **Missing Credentials**: When using `LLM_PROVIDER=gemini`, the system fails fast if `GEMINI_API_KEY` is not provided. Set the variable in your shell or use `LLM_PROVIDER=mock`.



