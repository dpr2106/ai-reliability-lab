# STING Reproducible Demonstration Guide (V2-006.1)

This guide provides end-to-end instructions to run, verify, and demonstrate **STING** (Evidence-First RAG Reliability Workbench) on a clean machine using the bundled **Northstar Technical Demonstration Suite**.

---

## 1. Prerequisites & System Requirements

### Hardware Considerations
- **Memory**: Minimum 4 GB RAM (8 GB+ recommended for local dense embedding inference).
- **Disk Space**: ~2 GB free disk space for Python virtual environment, dependencies, FAISS indexes, and embedding weights.
- **CPU / GPU**: Standard multi-core x86_64 or ARM64 CPU. GPU acceleration is optional.

### Software Prerequisites
- **Python**: 3.11 or 3.12 recommended (3.10 and 3.13 also supported).
- **Node.js**: Node 18+ or 20+ (with `npm`).
- **Git**: Modern git client.
- **Optional Local LLM (Ollama)**: For 100% offline neural generation, install [Ollama](https://ollama.com). (The workbench also includes a built-in deterministic mock provider for zero-dependency testing).

---

## 2. Clean-Checkout Setup

### Step 1: Clone Repository & Create Virtual Environment
```bash
git clone https://github.com/dpr2106/ai-reliability-lab.git
cd ai-reliability-lab

# Create and activate virtual environment
python -m venv .venv

# On Linux / macOS:
source .venv/bin/activate
# On Windows PowerShell:
.\.venv\Scripts\Activate.ps1
```

### Step 2: Install Python Dependencies
```bash
# Upgrade pip and install STING with development, indexing, and web packages
pip install --upgrade pip
pip install -e .[dev,faiss,web]
```

### Step 3: Install Frontend Dependencies
```bash
cd frontend
npm ci
cd ..
```

### Step 4: (Optional) Setup Local Ollama Model
If running live offline neural generation:
```bash
# Start Ollama daemon (in a separate terminal)
ollama serve

# Pull default compact model
ollama pull llama3.2
```

---

## 3. Starting the Platform

### Terminal 1: Backend Server (FastAPI)
```bash
# Activate virtual environment
source .venv/bin/activate  # Or Windows: .\.venv\Scripts\Activate.ps1

# Set persistent SQLite database path
export RELIABILITY_LAB_DB=".reliability-lab/experiments.sqlite3"
# On Windows PowerShell: $env:RELIABILITY_LAB_DB = ".reliability-lab/experiments.sqlite3"

# Start FastAPI application
uvicorn reliability_lab.api.main:app --reload --port 8000
```
- Interactive Swagger API Docs: [http://localhost:8000/docs](http://localhost:8000/docs)
- Health Check: [http://localhost:8000/api/health](http://localhost:8000/api/health)

### Terminal 2: Frontend Diagnostic Studio (SvelteKit)
```bash
cd frontend
npm run dev -- --port 5173
```
- Interactive Workbench Studio: [http://localhost:5173](http://localhost:5173)

---

## 4. Preparing the Demo Dataset

Execute the preparation workflow to ingest the bundled Northstar corpus and create the verified golden benchmark dataset:

```bash
# Run against running backend server:
python scripts/prepare_sting_demo.py --api-url http://127.0.0.1:8000

# Or run directly in local service mode (no server needed):
python scripts/prepare_sting_demo.py --local
```

### What This Workflow Does:
1. **Validates Demo Pack**: Confirms integrity of `data/demo/manifest.json`, the 3 Markdown documents, and 10 golden benchmark questions.
2. **Ingests Documents**: Parses and indexes `architecture.md`, `benchmark_report.md`, and `incident_report.md` via `DocumentIngestionService`.
3. **Builds Dual Indexes**: Generates BM25 term matrices and FAISS dense vector embeddings (`all-MiniLM-L6-v2`).
4. **Resolves Dynamic Evidence Anchors**: Resolves manually verified textual evidence anchors to real runtime chunk IDs fail-closed.
5. **Creates Benchmark Dataset**: Stores `Northstar Golden Benchmark` in SQLite with explicit document scopes and citation mappings.

Example output receipt:
```
======================================================================
  STING DEMO PACK PREPARATION RECEIPT
======================================================================
Execution Mode:       API
Benchmark Dataset:    Northstar Golden Benchmark
Dataset Identifier:   ds_2ce1e51bb4ec
Total Golden Queries: 10 (100% verified anchors)
----------------------------------------------------------------------
Ingested Documents:
  * architecture.md           -> Doc ID: doc_447de7d4db4347ed (7 chunks)
  * benchmark_report.md       -> Doc ID: doc_062d269988764857 (6 chunks)
  * incident_report.md        -> Doc ID: doc_e8e99352d25b4da9 (7 chunks)
Total Ingested Chunks: 20
----------------------------------------------------------------------
Ready for STING demonstration:
  1. Evidence Queries:      POST /api/query or /api/query/stream
  2. Cross-Document Invest: POST /api/investigations
  3. Evaluation Studio:     POST /api/evaluation/run with dataset 'ds_2ce1e51bb4ec'
======================================================================
```

---

## 5. Walkthrough: Demonstrating STING's Core Capabilities

### Workflow 1: Evidence-First Query & Citation Verification
Submit a document-grounded query with verified citations:
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What target chunk size and overlap are specified for Northstar document ingestion?",
    "document_ids": ["doc_447de7d4db4347ed"],
    "retriever_type": "hybrid",
    "top_k": 5
  }'
```
- **Inspect Citation Tags**: Generated answers cite exact chunks `[CIT-xxx]`.
- **Verify Provenance**: Citation resolver cross-checks chunk IDs against retrieved evidence candidates.
- **Inspect Retrieval Trace**: The response metadata contains the full query trace: candidate chunk ranks, BM25 scores, FAISS cosine distances, and RRF fusion contributions.

For genuine token streaming, query the Server-Sent Events endpoint:
```bash
curl -N "http://localhost:8000/api/query/stream" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What target chunk size and overlap are specified for Northstar document ingestion?",
    "document_ids": ["doc_447de7d4db4347ed"],
    "retriever_type": "hybrid",
    "top_k": 5
  }'
```

### Workflow 2: Cross-Document Evidence Investigation
Launch an investigation across the three documents:
```bash
curl -X POST "http://localhost:8000/api/investigations" \
  -H "Content-Type: application/json" \
  -d '{
    "title": "Hybrid Retrieval Performance on Alphanumeric Identifiers",
    "query": "Does Hybrid RRF perform reliably on alphanumeric model codes compared to standard benchmarks?",
    "document_ids": ["doc_447de7d4db4347ed", "doc_062d269988764857", "doc_e8e99352d25b4da9"]
  }'
```
- **Inspect Findings Matrix**: In the UI (`/investigations`), observe the comparison table across documents.
- **Synthesize Disagreement**: Notice that `benchmark_report.md` reports Hybrid RRF as universally superior for natural language, while `incident_report.md` proves it failed on exact SKU searches due to RRF rank dilution.
- **Export Publication Report**: Download a formatted report via `GET /api/investigations/{id}/export`.

### Workflow 3: RAG Evaluation & Regression Studio
1. Navigate to the **Evaluation Studio** in the UI ([http://localhost:5173/evaluation](http://localhost:5173/evaluation)).
2. Under the **Datasets Studio** tab, inspect the imported `Northstar Golden Benchmark` (10 queries).
3. Under the **Experiment Launcher** tab, configure a run:
   - Dataset: `Northstar Golden Benchmark`
   - Retriever: `hybrid` (or compare `bm25` vs `dense`)
   - Provider: `mock` (or `ollama` with `llama3.2`)
4. Click **Launch Evaluation Experiment**:
   - Observe live thread-safe incremental progress (`queries_completed` / `total_queries`).
5. Open the **Failure Explorer** tab:
   - Filter query results by diagnostic failure mode: `retrieval_miss`, `partial_retrieval`, `citation_hallucination`, `citation_omission`, `ungrounded_generation`, or `passed`.
6. Open the **Reproducibility Manifest** modal:
   - Inspect the verified reproducibility tier (`exact`, `partial`, `impossible`), dataset SHA-256 content hash, and corpus chunk fingerprints.

### Workflow 4: Safe Abstention on Missing Evidence
Submit a query about an unmentioned topic:
```bash
curl -X POST "http://localhost:8000/api/query" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What reinforcement learning from human feedback (RLHF) reward model and PPO hyperparameters are used in Northstar?",
    "document_ids": ["doc_447de7d4db4347ed", "doc_062d269988764857", "doc_e8e99352d25b4da9"],
    "retriever_type": "hybrid",
    "top_k": 5
  }'
```
- **Abstention Behavior**: Because retrieval returns no evidence meeting relevance criteria, STING triggers Tier-1 or Tier-2 safe abstention, refusing to fabricate answers.

---

## 6. Provider Transparency & Hardware Boundaries

STING supports three distinct LLM execution paths:

| Provider | Purpose | Network Requirement | Hardware Requirements | Reproducibility |
|---|---|---|---|---|
| **Deterministic Mock (`mock`)** | Hermetic CI, unit testing, deterministic validation | Zero network calls | Minimal CPU (< 100 MB RAM) | 100% deterministic rule-based output |
| **Local Ollama (`ollama`)** | Authentic local-first neural generation | 100% offline (local daemon) | 4-8 GB RAM, modern CPU | Genuine local LLM reasoning (`llama3.2`) |
| **Cloud Gemini (`gemini`)** | Comparison baseline against hosted models | Requires internet & `GEMINI_API_KEY` | Minimal client overhead | API response latency & quota dependent |

> **Transparency Note**: Synthetic mock execution should never be presented as real neural model performance. Local inference requires downloading model weights via `ollama pull` and consumes local system memory during execution.

