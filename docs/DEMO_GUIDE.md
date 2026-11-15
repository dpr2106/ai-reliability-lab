# STING Reproducible Demonstration Guide (V2-006.2)

This guide provides end-to-end instructions to clone, configure, run, verify, and demonstrate **STING** (Evidence-First RAG Reliability Workbench) from a genuinely clean checkout using the bundled **Northstar Technical Demonstration Suite** and the modern **User-First Workspace & Developer Mode (V2-007)**.

---

## 1. Prerequisites & System Requirements

### Hardware Considerations
- **Memory**: Minimum 4 GB RAM (8 GB+ recommended for local dense embedding inference).
- **Disk Space**: ~2 GB free disk space for Python virtual environment, dependencies, FAISS indexes, and embedding weights.
- **CPU / GPU**: Standard multi-core x86_64 or ARM64 CPU. GPU acceleration is optional.

### Software Prerequisites
- **Python**: Supported versions: `Python >= 3.10` per `pyproject.toml`.
  - **Clean-checkout tested**: `Python 3.10.11` (Windows 11 x86_64, isolated virtual environment).
  - **CI matrix verified**: `Python 3.11` and `Python 3.12` (GitHub Actions Ubuntu runners).
  - **Local development**: `Python 3.13.3` (used for development and type-checking, but not certified in CI).
- **Node.js**: Required runtime: **`Node.js >= 22.12.0`** (Active LTS Node 22, verified with `Node.js v22.14.0` and `npm 10.9.2`).
  - *Dependency requirement note*: Modern frontend dependencies (SvelteKit 3 / `@sveltejs/kit: ^3.0.0`, Vite 8 / `vite: ^8.3.0`, and Puppeteer 25 / `puppeteer: ^25.17.0`) strictly declare `node: ">=22.12.0"`. Node 18 (end-of-life) and older Node 20 releases (`< 22.12.0`) are not supported.
- **Git**: Modern git client.
- **Optional Local LLM (Ollama)**: For 100% offline neural generation, install [Ollama](https://ollama.com). (The platform includes a built-in deterministic mock provider for zero-dependency hermetic testing).

---

## 2. Configuration & Network Boundary Separation

STING enforces a **secure local-first posture**. To understand how STING operates, distinguish clearly between installation-time dependencies, credential-free execution, and offline runtime:

1. **Installation-Time Network Access (One-Time Setup)**:
   - Initial Python dependency installation (`pip install -e ".[dev,faiss,web]"`) and Node module fetching (`npm ci`) require internet access.
   - The first invocation of dense neural indexing downloads Sentence Transformers embedding weights (`all-MiniLM-L6-v2`, ~80 MB from HuggingFace) into the local cache (`~/.cache/huggingface/hub/`).
2. **Credential-Free Operation (Zero Cloud API Keys)**:
   - STING never requires external API keys, user logins, credit cards, or cloud accounts for any core feature.
   - Retrieval (BM25, FAISS, Hybrid RRF), document chat, cross-document investigations, and golden evaluations function fully without external credentials.
3. **100% Offline Runtime (Air-Gapped Operation)**:
   - Once packages and embedding weights are locally cached, STING runs entirely offline with zero external network egress.
   - Local LLM inference runs via Ollama on `localhost:11434` or the hermetic deterministic mock provider.

| Variable | Purpose | Required / Optional | Default Value | Safe Example Value |
|---|---|---|---|---|
| `RELIABILITY_LAB_DB` | Path to persistent SQLite database | Optional | `.reliability-lab/experiments.sqlite3` | `.reliability-lab/experiments.sqlite3` |
| `STING_API_URL` | Base URL of FastAPI backend adapter | Optional | `http://127.0.0.1:8000` | `http://127.0.0.1:8000` |
| `OLLAMA_BASE_URL` | Local Ollama daemon HTTP endpoint | Optional | `http://localhost:11434` | `http://localhost:11434` |
| `OLLAMA_MODEL` | Default model tag for Ollama | Optional | `llama3.2` | `llama3.2` |
| `GEMINI_API_KEY` | Google Gemini API key | Optional (cloud only) | None | `AIzaSy...` (do not commit) |
| `PORT` | Local service port for backend | Optional | `8000` | `8000` |

### Credential-Free Capabilities
The following features require **zero external API keys or cloud credentials**:
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

To prove deterministic reproducibility, execute `scripts/record_demo_evaluation.py` against the clean-checkout platform. This script dispatches the `Northstar Golden Benchmark` (10 queries, dataset `ds_8138dee2cb55`) twice in succession using Okapi BM25 retrieval and the deterministic mock provider, verifies the exact dataset SHA-256 digest, checks for complete metric coverage, and outputs structured provenance into `docs/reproducibility_evidence.json`:

```bash
python scripts/record_demo_evaluation.py
```

### Verified Dataset & Run Provenance
- **Dataset Name**: `Northstar Golden Benchmark` (`ds_8138dee2cb55`)
- **Verified Dataset SHA-256**: `88ee2c58555b5d86f789dd660f30365659acc59375809c6b5df528780280425e`
- **Execution Config Hash**: `f6fa8444f4c72e2b8e97cb309644d0ee8b802f8ed7425d30687af02c7b0350b5`
- **Retriever & Parameters**: `bm25` (top-K: 5, RSJ IDF, deterministic tie-breaking)
- **Generator Provider / Model**: `mock` / `mock-pipeline` (Deterministic rule-based mock)
- **Evaluation Scoped Corpus**: 3 documents (`architecture.md`, `benchmark_report.md`, `incident_report.md` — 20 chunks total)
- **Reproducibility Tier**: `exact` (Run 1 and Run 2)

### Actual Measured Comparison: Run 1 vs. Run 2

| Metric | Run 1 | Run 2 | Absolute Delta | Metric Status / Interpretation |
|:---|:---:|:---:|:---:|:---|
| **Hit Rate@5** | `0.8000` | `0.8000` | **`0.0000`** | Deterministic IR Metric (exact match) |
| **Recall@5** | `0.8000` | `0.8000` | **`0.0000`** | Deterministic IR Metric (exact match) |
| **Precision@5** | `0.1600` | `0.1600` | **`0.0000`** | Deterministic IR Metric (exact match) |
| **MRR@5** | `0.7000` | `0.7000` | **`0.0000`** | Deterministic IR Metric (exact match) |
| **NDCG@5** | `0.7262` | `0.7262` | **`0.0000`** | Deterministic IR Metric (exact match) |
| **Citation Precision** | `0.0000` | `0.0000` | **`0.0000`** | Failing: Mock generator emits no citations |
| **Citation Recall** | `0.0000` | `0.0000` | **`0.0000`** | Failing: Mock generator emits no citations |
| **Lexical Groundedness** | `0.1000` | `0.1000` | **`0.0000`** | Low: Synthetic template token overlap |
| **Total Prompt Tokens** | `4,406` | `4,406` | **`0`** | Deterministic Tokenizer Count |
| **Total Completion Tokens**| `30` | `30` | **`0`** | Deterministic Generator Count |
| **Total Evaluated Queries** | `10` | `10` | **`0`** | Exact Dataset Size (10 queries) |
| **Mean Latency (ms)** | `1.2886` | `1.3301` | `+0.0415` | Wall-clock execution (expected ~40µs CPU jitter) |
| **Reproducibility Tier** | `exact` | `exact` | **Match** | Verified SHA-256 Provenance |

> [!IMPORTANT]
> **Honest Evaluation Disclosure — Mock vs. Neural Generation**:
> - **Deterministic Repeatability**: Ranking metrics (Hit Rate, MRR, NDCG, Recall, Precision) and token counts match **identically (0.0000 delta)** across runs. This proves algorithmic determinism and mathematical evaluation correctness under identical inputs.
> - **Generative Quality Disclaimer**: Exact reproducibility does **not** establish high generative quality. Because the evaluation uses the deterministic `mock` provider (`mock-pipeline`), the generated answers are rule-based synthetic strings that contain no citation markers and minimal lexical overlap.
> - **Transparent Failure Reporting**: Citation Precision (`0.0000`), Citation Recall (`0.0000`), and Lexical Groundedness (`0.1000`) reflect genuine generation failures for the mock provider. These metrics are intentionally preserved and disclosed, never zero-filled or disguised. To evaluate authentic neural citation placement and groundedness, configure local Ollama (`llama3.2`).

---

## 8. Troubleshooting & Common Issues

| Issue / Symptom | Root Cause | Resolution |
|---|---|---|
| `ModuleNotFoundError: No module named 'pymupdf'` | Python virtual environment created without project dependencies installed | Activate venv and run `pip install -e ".[dev,web]"` or `pip install pymupdf`. |
| `No module named 'faiss.swigfaiss_avx2'` | Informational warning emitted on CPUs without AVX2 extensions | Safe to ignore. The library automatically falls back to standard generic instructions (`Successfully loaded faiss`). |
| `ERR_CONNECTION_REFUSED` on frontend startup | Backend FastAPI server is not running on port 8000 | Ensure Terminal 1 is running `uvicorn reliability_lab.api.main:app --port 8000`. |
| `npm ci` fails with engine or lockfile mismatch | Node.js version is below modern SvelteKit/Puppeteer requirements | Install **Node.js >= 22.12.0** (Active LTS Node 22, e.g. v22.14.0) with npm 10+. Run `npm ci` without `--force`. |
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
npm run test     # Vitest unit tests (29/29 passed across api.spec.ts and workspace.spec.ts)
npm run check    # Svelte TypeScript & A11y diagnostics (0 errors, 0 warnings)
npm run build    # Production build with @sveltejs/adapter-node
cd ..

# Hermetic Demo Integration Test (16/16 checkpoints)
python scripts/verify_sting_demo.py
```

---

## 10. Known Limitations

- **Single-File Ingestion Drop**: The dropzone processes files one at a time to ensure deterministic lifecycle tracking and honest progress reporting.
- **Cold-Start Vector Cache**: The first dense vector embedding execution downloads `all-MiniLM-L6-v2` (~90 MB) into the local HuggingFace cache. Subsequent executions run entirely offline.
- **CPU Ollama Latency**: Running local neural models on CPU without dedicated GPU compute exhibits longer generation times (~15-25 seconds per turn).
