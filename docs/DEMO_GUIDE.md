# STING Reproducible Demonstration Guide (V2-006.2)

This guide provides end-to-end instructions to clone, configure, run, verify, and demonstrate **STING** (Evidence-First RAG Reliability Workbench) from a genuinely clean checkout using the bundled **Northstar Technical Demonstration Suite** and the modern **User-First Workspace & Developer Mode (V2-007)**.

---

## 1. Prerequisites & System Requirements

### Hardware Considerations
- **Memory**: Minimum 4 GB RAM (8 GB+ recommended for local dense embedding inference).
- **Disk Space**: ~2 GB free disk space for Python virtual environment, dependencies, FAISS indexes, and embedding weights.
- **CPU / GPU**: Standard multi-core x86_64 or ARM64 CPU. GPU acceleration is optional.

### Software Prerequisites
- **Python**: Supported versions: `3.10`, `3.11`, `3.12`, or `3.13` (Python 3.10 and 3.11 verified in clean-checkout testing).
- **Node.js**: Node 18+ or 20+ (Node v20.x or v24.x tested with `npm 10+` or `11+`).
- **Git**: Modern git client.
- **Optional Local LLM (Ollama)**: For 100% offline neural generation, install [Ollama](https://ollama.com). (The platform includes a built-in deterministic mock provider for zero-dependency hermetic testing).

---

## 2. Configuration & Environment Variables

STING is designed with a **secure local-first default posture**. All core capabilities—document ingestion, BM25/FAISS retrieval, conversational document chat with citations, cross-document evidence investigations, and golden benchmark evaluations—run completely offline without external API keys or cloud credentials.

| Variable | Purpose | Required / Optional | Default Value | Safe Example Value |
|---|---|---|---|---|
| `RELIABILITY_LAB_DB` | Path to persistent SQLite database | Optional | `.reliability-lab/experiments.sqlite3` | `.reliability-lab/experiments.sqlite3` |
| `STING_API_URL` | Base URL of FastAPI backend adapter | Optional | `http://127.0.0.1:8000` | `http://127.0.0.1:8000` |
| `OLLAMA_BASE_URL` | Local Ollama daemon HTTP endpoint | Optional | `http://localhost:11434` | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model tag for Ollama | Optional | `llama3.2` | `llama3.2` |
| `GEMINI_API_KEY` | Google Gemini API key | Optional (cloud only) | None | `AIzaSy...` (do not commit) |
| `PORT` | Local service port for backend | Optional | `8000` | `8000` |

### Credential-Free Capabilities
The following features require **zero external API keys or network calls**:
- Document ingestion and citation-aware chunking (PDF, Markdown, Plain Text).
- Okapi BM25 and dense FAISS vector index generation (`all-MiniLM-L6-v2` / hermetic embedders).
- User Mode document chat with citation parsing and Evidence Sources drawer.
- Cross-document evidence investigations with structured findings matrices and Markdown export.
- Benchmark dataset management, golden question validation, and JSONL import/export.
- Deterministic RAG evaluation execution via the built-in `mock` provider.
- Automated regression gating and multi-tier reproducibility manifest auditing.

---

## 3. Clean-Checkout Setup

Follow these exact steps from a newly cloned repository or separate disposable directory:

### Step 1: Clone Repository
```bash
git clone https://github.com/dpr2106/ai-reliability-lab.git
cd ai-reliability-lab
```

### Step 2: Create and Activate Virtual Environment
**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**On Windows PowerShell:**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies
```bash
# Upgrade pip to latest version
python -m pip install --upgrade pip

# Install STING in editable mode with development, indexing, and web packages
pip install -e ".[dev,faiss,web]"
```

### Step 4: Install Frontend Dependencies
```bash
cd frontend
npm ci
cd ..
```

### Step 5: (Optional) Local Ollama LLM Setup
If demonstrating authentic offline neural generation:
```bash
# Start Ollama daemon (in a separate terminal)
ollama serve

# Pull default compact model (3B parameters)
ollama pull llama3.2
```

---

## 4. Starting the Platform

Open two terminal sessions:

### Terminal 1: Backend Server (FastAPI)
**On Linux / macOS:**
```bash
source .venv/bin/activate
export RELIABILITY_LAB_DB=".reliability-lab/experiments.sqlite3"
uvicorn reliability_lab.api.main:app --port 8000 --host 127.0.0.1
```

**On Windows PowerShell:**
```powershell
.\.venv\Scripts\Activate.ps1
$env:RELIABILITY_LAB_DB = ".reliability-lab\experiments.sqlite3"
uvicorn reliability_lab.api.main:app --port 8000 --host 127.0.0.1
```

- **Health Check**: [http://127.0.0.1:8000/api/health](http://127.0.0.1:8000/api/health)
- **Interactive OpenAPI Docs**: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Terminal 2: Frontend Diagnostic Studio (SvelteKit)
```bash
cd frontend
npm run dev -- --host 127.0.0.1 --port 5173
```

- **STING Application**: [http://127.0.0.1:5173](http://127.0.0.1:5173)

---

## 5. Preparing the Bundled Demo Dataset

Execute the preparation workflow to ingest the Northstar demonstration corpus and establish the golden benchmark dataset:

### Option A: Against Running Backend (API Mode)
```bash
python scripts/prepare_sting_demo.py --api-url http://127.0.0.1:8000 --receipt-json data/demo/receipt.json
```

### Option B: Local Service Mode (Direct SQLite / No Server Needed)
```bash
python scripts/prepare_sting_demo.py --local --receipt-json data/demo/receipt.json
```

### What This Workflow Does:
1. **Validates Demo Pack**: Validates `data/demo/manifest.json`, the 3 Markdown documents, and 10 golden benchmark questions.
2. **Ingests Documents**: Parses and indexes `architecture.md`, `benchmark_report.md`, and `incident_report.md`.
3. **Builds Dual Indexes**: Generates lexical BM25 chunks and FAISS dense vector embeddings.
4. **Resolves Evidence Anchors**: Resolves manually verified textual evidence anchors to real runtime chunk IDs fail-closed.
5. **Creates Benchmark Dataset**: Stores `Northstar Golden Benchmark` in SQLite with explicit document scopes and citation mappings.
6. **Guarantees Idempotency**: Re-running the preparation script detects existing ingested documents and datasets, preventing duplicate data.

### Actual Execution Receipt Output:
```
======================================================================
  STING DEMO PACK PREPARATION RECEIPT
======================================================================
Execution Mode:             API
Benchmark Dataset:          Northstar Golden Benchmark
Dataset Identifier:         ds_8138dee2cb55
Total Golden Queries:       10
  * Evidence-Anchored:      9 (resolved to unique chunks: 9)
  * Safe Abstention:        1 (intentional zero-evidence ground truth)
----------------------------------------------------------------------
Ingested Documents:
  * architecture.md           -> Doc ID: doc_323cee5d40fe4b70 (7 chunks)
  * benchmark_report.md       -> Doc ID: doc_e151dadb04f14716 (6 chunks)
  * incident_report.md        -> Doc ID: doc_423e1e9d71c442cb (7 chunks)
Total Ingested Chunks:      20
----------------------------------------------------------------------
Quick Shell Variables (Bash / Zsh):
  export DOC_ARCH="doc_323cee5d40fe4b70"
  export DOC_BENCH="doc_e151dadb04f14716"
  export DOC_INCIDENT="doc_423e1e9d71c442cb"
  export DEMO_DATASET_ID="ds_8138dee2cb55"

Quick PowerShell Variables:
  $env:DOC_ARCH = "doc_323cee5d40fe4b70"
  $env:DOC_BENCH = "doc_e151dadb04f14716"
  $env:DOC_INCIDENT = "doc_423e1e9d71c442cb"
  $env:DEMO_DATASET_ID = "ds_8138dee2cb55"
======================================================================
```

---

## 6. End-to-End User Experience Walkthrough

STING provides two distinct operational modes: **User Mode** (default) and **Developer Mode** (optional diagnostic workbench).

### Phase 1: User Mode Walkthrough (Default Workspace)

1. **Open Workspace**:
   - Navigate to [http://127.0.0.1:5173/](http://127.0.0.1:5173/).
   - Observe that **Developer Mode is OFF by default** on fresh visits.
   - The headline reads: *"Your documents. Understood."*
   - Technical diagnostic links (Evaluation Studio, Run Explorer, Citation Audit) are cleanly hidden.
2. **Inspect Recently Uploaded Documents**:
   - The 3 Northstar demonstration documents (`architecture.md`, `benchmark_report.md`, `incident_report.md`) appear in the document collection grid.
   - Metadata badges show file size, chunk counts (7 chunks, 6 chunks, 7 chunks), and `READY` status.
3. **Upload a New Document**:
   - Drag and drop a supported file (`.pdf`, `.md`, or `.txt`) into the upload dropzone, or click **Choose Document**.
   - Observe the real-time stage progression:
     $$\text{UPLOAD} \longrightarrow \text{EXTRACTION} \longrightarrow \text{CHUNKING} \longrightarrow \text{INDEXING} \longrightarrow \text{READY}$$
   - Notice the glowing cyan progress bar (20% -> 42% -> 68% -> 88% -> 100%) and active stage diagnostic descriptions.
4. **Chat with Document**:
   - Click **Chat with Document &rarr;** on `architecture.md`.
   - The document workspace displays the document filename, format badge, chunk count, and readiness tag.
   - Ask a question:
     > *"What target chunk size and overlap are specified for Northstar document ingestion?"*
   - Inspect the generated answer:
     > *"According to the Northstar architectural specification [CIT-1], document ingestion uses a target chunk size of 500 characters and 100 character overlap."*
5. **Inspect Citations & Evidence Sources**:
   - Click the interactive `[CIT-1]` pill to highlight the verbatim grounded quotation.
   - Click **Evidence Sources** in the top-right header to toggle the slide-over drawer.
   - Inspect the verified chunk ID, source filename, page number, effective strategy (`hybrid`), and latency.
6. **Return to Document Library**:
   - Click **&larr; All Documents** to return to the root landing hub.

---

### Phase 2: Developer Mode Walkthrough (Diagnostic Workbench)

1. **Activate Developer Mode**:
   - Click the **Developer Mode** toggle switch in the application header.
   - The header displays the **DEV** badge, and the **DEVELOPER WORKBENCH ACTIVE** ribbon appears.
   - Diagnostic navigation links are revealed:
     - **Evaluation Studio** (`/evaluation`)
     - **Investigations** (`/investigations`)
     - **Run Explorer** (`/runs`)
     - **Run Comparison** (`/compare`)
     - **Citation Audit** (`/citation-audit`)
2. **Explore Evaluation Studio (`/evaluation`)**:
   - Click **Evaluation Studio**.
   - Confirm the page loads cleanly with zero console exceptions (the undefined filter defect is resolved).
   - Under the **Datasets Studio** tab, inspect the `Northstar Golden Benchmark` (10 queries, 3 scoped documents).
   - Under the **Experiment Launcher** tab:
     - Dataset: `Northstar Golden Benchmark`
     - Retriever: `bm25` (or `hybrid`)
     - Provider: `mock` (or `ollama`)
     - Model: `mock-pipeline` (or `llama3.2`)
     - Top-K: `5`
   - Click **Launch Evaluation Experiment**.
   - Watch the incremental progress indicator (`1/10` -> `5/10` -> `10/10`).
3. **Inspect Failure Explorer & Reproducibility Manifest**:
   - Open the **Failure Explorer** tab to inspect query-level diagnostic outcomes (`passed`, `retrieval_miss`, `citation_hallucination`, etc.).
   - Click **View Reproducibility Manifest** on any completed experiment.
   - Inspect the verified **Reproducibility Tier** (`exact`), dataset SHA-256 digest, and corpus chunk fingerprints.
4. **Cross-Document Evidence Investigations (`/investigations`)**:
   - Navigate to `/investigations`.
   - Create an investigation comparing `architecture.md`, `benchmark_report.md`, and `incident_report.md`.
   - Inspect the aspect comparison matrix and export a publication-grade Markdown report.
5. **Verify State Persistence**:
   - Refresh the browser (`F5` or `Ctrl+R`).
   - Developer Mode remains **ACTIVE** (persisted in `localStorage`).
   - Click the toggle switch again to return to User Mode.

---

## 7. Deterministic Benchmark Reproducibility Evidence

To prove deterministic reproducibility, execute `scripts/record_demo_evaluation.py` against the clean-checkout platform. This script dispatches the `Northstar Golden Benchmark` (10 queries) twice in succession using Okapi BM25 retrieval and the deterministic mock provider:

```bash
python scripts/record_demo_evaluation.py
```

### Actual Measured Comparison: Run 1 vs. Run 2

| Metric | Run 1 | Run 2 | Absolute Delta | Expected Determinism |
|:---|:---:|:---:|:---:|:---|
| **Hit Rate@5** | `0.8000` | `0.8000` | **`0.0000`** | Deterministic IR Metric |
| **Recall@5** | `0.8000` | `0.8000` | **`0.0000`** | Deterministic IR Metric |
| **Precision@5** | `0.1600` | `0.1600` | **`0.0000`** | Deterministic IR Metric |
| **MRR@5** | `0.7000` | `0.7000` | **`0.0000`** | Deterministic IR Metric |
| **NDCG@5** | `0.7262` | `0.7262` | **`0.0000`** | Deterministic IR Metric |
| **Lexical Groundedness** | `0.1000` | `0.1000` | **`0.0000`** | Deterministic Containment/LCS |
| **Total Prompt Tokens** | `4,406` | `4,406` | **`0`** | Deterministic Tokenizer Count |
| **Total Completion Tokens**| `30` | `30` | **`0`** | Deterministic Generator Count |
| **Total Evaluated Queries** | `10` | `10` | **`0`** | Exact Dataset Size |
| **Mean Latency (ms)** | `1.2886` | `1.3301` | `+0.0415` | Wall-clock execution (expected variance) |
| **Reproducibility Tier** | `exact` | `exact` | **Match** | Verified SHA-256 Provenance |

> **Evaluation Honesty Note**: Metrics and token counts are **100% identical across repeat executions (delta: 0.0000)**. Physical latency exhibits natural microsecond wall-clock variance (~1.28 ms vs ~1.33 ms, delta: ~0.04 ms). Synthetic mock output verifies pipeline contracts and metric computation; it does not claim to represent neural model reasoning quality.

---

## 8. Troubleshooting & Common Issues

| Issue / Symptom | Root Cause | Resolution |
|---|---|---|
| `ModuleNotFoundError: No module named 'pymupdf'` | Python virtual environment created without project dependencies installed | Activate venv and run `pip install -e ".[dev,web]"` or `pip install pymupdf`. |
| `No module named 'faiss.swigfaiss_avx2'` | Informational warning emitted on CPUs without AVX2 extensions | Safe to ignore. The library automatically falls back to standard generic instructions (`Successfully loaded faiss`). |
| `ERR_CONNECTION_REFUSED` on frontend startup | Backend FastAPI server is not running on port 8000 | Ensure Terminal 1 is running `uvicorn reliability_lab.api.main:app --port 8000`. |
| `npm ci` fails with lockfile mismatch | Node version or global npm environment mismatch | Use Node 20+ with npm 10+. Run `npm ci` without `--force`. |
| Ollama generation times out on CPU | Local LLM inference on CPU without GPU takes 15-25s per turn | Use the deterministic `mock` provider for rapid evaluations, or extend HTTP timeouts when running `llama3.2` on CPU. |
| Missing Developer Mode options | User Mode is active (default) | Click the **Developer Mode** toggle switch in the application header. |

---

## 9. Verification & Quality Commands

Run the full verification suite before submitting pull requests:

```bash
# Backend Quality Checks
ruff check .
ruff format --check .
mypy reliability_lab
pytest --cov=reliability_lab

# Frontend Quality Checks
cd frontend
npm run test     # Vitest unit tests (29/29 passed)
npm run check    # Svelte TypeScript & A11y diagnostics (0 errors, 0 warnings)
npm run build    # Production build with @sveltejs/adapter-node
cd ..

# Hermetic Demo Integration Test
python scripts/verify_sting_demo.py
```

---

## 10. Known Limitations

- **Single-File Ingestion Drop**: The dropzone processes files one at a time to ensure deterministic lifecycle tracking and honest progress reporting.
- **Cold-Start Vector Cache**: The first dense vector embedding execution downloads `all-MiniLM-L6-v2` (~90 MB) into the local HuggingFace cache. Subsequent executions run entirely offline.
- **CPU Ollama Latency**: Running local neural models on CPU without dedicated GPU compute exhibits longer generation times (~15-25 seconds per turn).
