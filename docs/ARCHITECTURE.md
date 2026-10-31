# Architecture Blueprint: AI Reliability Lab

## Architectural Principles
1. **Separation of Concerns**: Pure domain models (`core/`) have zero external dependencies.
2. **Pluggable Protocols**: Retrievers, chunkers, and evaluation metrics conform to strict abstract base classes.
3. **Dual Execution Tiers**:
   - *Tier 1 (100% Local / Zero Download)*: Ingestion, BM25 indexing, exact ranking metrics, citation regex validation, SQLite persistence.
   - *Tier 2 (Local Weights / CPU Friendly)*: Dense vector embeddings (`all-MiniLM-L6-v2`), FAISS vector indices, Cross-Encoder NLI.
   - *Tier 3 (External API / Optional)*: Gemini API adapter for answer generation and comparative LLM-as-a-judge studies.

## System Topology & Data Flow

```
[Raw Documents: PDF / Markdown]
             │
             ▼
[Ingestion: PyMuPDF Parser] ──► Preserves exact page numbers and character bounding offsets
             │
             ▼
[Chunking Engine] ────────────► Produces List[Chunk] with parent document provenance
             │
             ├──────────────────────────────┐
             ▼                              ▼
[Sparse Index: BM25]             [Dense Index: FAISS + MiniLM]
             │                              │
             └──────────────┬───────────────┘
                            ▼
              [Hybrid Search: RRF Fusion]
                            │
                            ▼
               [Ranked Retrieval Results]
                            │
       ┌────────────────────┴────────────────────┐
       ▼                                         ▼
[Retrieval Evaluator]                  [Groundedness Evaluator]
• Recall@K, Precision@K                • Claim extraction & span match
• MRR@K, NDCG@K                        • NLI entailment (Premise vs Hypothesis)
• Latency & Memory profiling           • Citation Precision & Recall
       │                                         │
       └────────────────────┬────────────────────┘
                            ▼
       [Storage & Regression Engine: SQLite]
       • Run configuration hashing
       • Delta comparison (Run A vs Run B)
       • CI regression threshold gating
```

## Core Domain Models (Pure Python Standard Library, Zero External Dependencies)
Per `ADR-005`, the domain model layer has zero third-party dependencies and relies exclusively on standard library `dataclasses`, `typing`, `abc`, `enum`, and `json`:
- `Document`: Raw parsed text, metadata dictionary, and page collection.
- `Chunk`: Text content, `doc_id`, `chunk_id`, `page_number`, `char_start`, `char_end`.
- `Query`: Unique `query_id`, query text, and evaluation metadata.
- `RetrievalResult`: Reference to `Chunk`, retrieval rank, score, and retriever source metadata.
- `EvaluationReport`: Aggregate metrics, per-query score breakdowns, and execution latency.
- `ExperimentRun`: Hash of run configuration, timestamp, dataset reference, and metric results.

## SQLite Experiment Storage & Repository (`reliability_lab/storage/`)
Built with standard library `sqlite3` without external ORMs, providing normalized relational storage for benchmark reports, index reproducibility, and regression audits:

1. **Relational Schema**:
   - `index_configurations`: Deduplicated index/run configurations indexed by 64-char SHA-256 `config_hash`.
   - `experiment_runs`: Run headers (`run_id`, `experiment_name`, `run_timestamp`, `config_hash`, `dataset_reference`, `metadata_json`, `created_at`).
   - `run_metrics`: Normalized aggregate metric scores (`run_id`, `metric_name`, `score`), foreign-keyed with cascade delete.
   - `query_results`: Granular per-query diagnostic payloads (`run_id`, `query_id`, `latency_ms`, `result_json`).
   - `query_metric_scores`: Normalized per-query metric values (`run_id`, `query_id`, `metric_name`, `score`).

2. **Deterministic Configuration Hashing**:
   - Configuration maps are canonically sorted (`sort_keys=True`), formatted with compact separators (`","`, `":"`), validated against non-finite floats (NaN/Inf), and hashed using SHA-256 (`compute_config_hash()`).
   - *Limitation*: The hash reflects only explicitly passed configuration keys; non-deterministic runtime attributes (e.g. random seeds unless configured) must be recorded in config for full reproducibility.

3. **Transaction Semantics & Idempotency**:
   - All writes are coordinated via `Database.transaction()` (`BEGIN IMMEDIATE` / `COMMIT` / `ROLLBACK`).
   - Saving an identical run ID with identical contents is idempotent.
   - Conflicting duplicate run IDs raise `DuplicateRunError`; hash collisions with divergent JSON raise `ConfigurationConflictError`.

4. **Usage**:
   ```python
   from reliability_lab.storage import Database, ExperimentRepository

   # In-memory for hermetic testing or file-backed for durable persistence:
   repo = ExperimentRepository(":memory:")  # or ExperimentRepository("experiments.db")
   repo.save_report(report, dataset_reference="sample_benchmark.jsonl")
   stored_run = repo.get_run(report.report_id)
   reconstructed_report = stored_run.to_report()
   ```

## Regression Detection Engine & Metric Diffing (`reliability_lab/storage/regression.py`)
Deterministic, rule-based regression detection comparing a baseline experiment run against a candidate run to detect ranking degradation across aggregate metrics and per-query results:

1. **Signed Metric Delta & Degradation Formulation**:
   - For any metric $m$: $\Delta_m = m_{\text{candidate}} - m_{\text{baseline}}$.
   - For higher-is-better retrieval metrics (Recall, MRR, NDCG, Precision, Hit Rate, MAP):
     Degradation magnitude: $D_m = m_{\text{baseline}} - m_{\text{candidate}} = -\Delta_m$.
   - A metric is classified as regressed when degradation strictly exceeds the configured tolerance: $D_m > \tau_m$.
   - Equality ($D_m = \tau_m$) is protected against floating-point inaccuracies (`math.isclose`) and is never reported as a regression.

2. **Tolerance Configuration**:
   - Global default tolerance: $\tau = 0.02$ (`DEFAULT_TOLERANCE`).
   - Per-metric overrides: e.g. `tolerance_overrides={"mrr@5": 0.01}`. Overrides are validated against non-finite floats, booleans, and negative values (`validate_tolerance()`).

3. **Status Classification Contract**:
   - `PASS`: The comparison is complete (all baseline queries evaluated) and no supported aggregate or per-query metric regressed beyond tolerance.
   - `REGRESSION`: At least one aggregate or per-query metric degradation strictly exceeds tolerance.
   - `INCOMPLETE`: Baseline and candidate query sets diverge (e.g. baseline-only or candidate-only queries), or required metrics are missing. Missing queries are never zero-imputed. If any observed metric regressed, `REGRESSION` takes precedence; otherwise, status remains `INCOMPLETE` with actionable comparability warnings.

4. **Aggregate Metric Aliasing & Deduplication**:
   - Canonical metric normalization strips aggregate prefixes (`mean_mrr@5` -> `mrr@5`).
   - Aliases representing the same metric are deduplicated. Conflicting scores between aliases (> 1e-5) raise `IncomparableRunsError`.

5. **Per-Query Diagnostics & Deterministic Ordering**:
   - Queries are evaluated across common canonical ranking metrics.
   - Per-query metric deltas record baseline, candidate, signed delta, degradation, tolerance, and regression status.
   - Worst degradations are surfaced deterministically, ordered descending by maximum degradation magnitude, then by `query_id`.

6. **Usage**:
   ```python
   from reliability_lab.storage import ExperimentRepository, RegressionDetector, RegressionStatus

   repo = ExperimentRepository("experiments.db")
   detector = RegressionDetector(repo)

   result = detector.compare_runs(
       baseline_run_id="run-baseline-001",
       candidate_run_id="run-candidate-002",
       tolerance=0.02,
       tolerance_overrides={"mrr@5": 0.01},
   )

   if result.status == RegressionStatus.REGRESSION:
       print("Regressed aggregates:", result.regressed_aggregate_metrics)
       for reg in result.worst_query_regressions:
           print(f"Query {reg.query_id} regressed metrics: {reg.regressed_metrics}")
   ```

7. **Limitations**:
   - Comparisons across different metric cutoffs (e.g. `mrr@5` vs `mrr@10`) or dataset references raise comparability warnings or are omitted from common metric sets.
   - Initial regression detection focuses on retrieval ranking metrics; latency and cost metrics are preserved in query diagnostics but not evaluated under ranking directionality.

---

## Diagnostics Studio & Web Interface (UI-002)

Per ADR-008, the diagnostic web interface is built using a modern decoupled architecture:
- **Frontend**: SvelteKit (Svelte 5 runes mode) with TypeScript, dark theme, and Vite.
  - `Run Explorer` (`/`): Paginated experiment list with filter, sort, and dual-run comparison selection.
  - `Run Details & Diagnostics` (`/runs/[run_id]`): Detailed aggregate metrics grid, configuration JSON viewer, and per-query failure trace drill-down.
  - `Run Comparison & Gating` (`/compare`): Side-by-side metric diffing, tolerance adjustment, gate status banners (`PASS`, `REGRESSION`, `INCOMPLETE`), and worst degradations list.
  - `Citation Audit Studio` (`/citation-audit`): Answer text inspection, inline citation mapping, precision/recall metrics, and evidence target resolution.
- **Backend Adapter**: Lightweight FastAPI service (`reliability_lab.api`) running via Uvicorn. Exposes REST endpoints reading from SQLite (`ExperimentRepository`), calling `RegressionDetector` for diffing, and using `CitationEvaluator` for citation auditing.

---

## Abstract LLM Client & Gemini Flash Integration (LLM-001)

Per ADR-009, generative answer generation and LLM-as-a-judge capabilities are built on a provider-independent contract:
- **Core Interface** (`reliability_lab.llm.client.BaseLLMClient`): Defines `generate()` and `generate_json()` contracts taking `LLMMessage` turns and `LLMGenerationConfig`, returning structured `LLMResponse` containers with latency metrics and `TokenUsage`.
- **Deterministic Mock Provider** (`reliability_lab.llm.client.MockLLMClient`): 100% offline, zero-network mock client supporting canned regex/substring responses, sequential response queues, error simulation, and call recording for hermetic testing.
- **Gemini Flash Adapter** (`reliability_lab.llm.client.GeminiLLMClient`): Direct integration with Google's Generative Language REST API using `httpx`, bounded exponential backoff retries on transient 429/503 errors, token-bucket/sliding-window rate limiting (`RateLimiter`), structured JSON mode, and safe mock fallback when credentials are not present. Defaults to `gemini-2.5-flash`.
- **Factory** (`reliability_lab.llm.client.get_llm_client`): Unified factory instantiation supporting `'auto'`, `'gemini'`, and `'mock'` provider modes.

---

## Structured LLM Judge, Faithfulness Evaluation & NLI Correlation (LLM-002)

Per ADR-010, the evaluation subsystem provides automated LLM-as-a-judge capabilities coupled with statistical correlation analysis against local NLI models:
- **Judge Engine** (`reliability_lab.llm.judge.LLMJudge`):
  - Deterministically extracts sentence-level claim spans using `extract_claim_spans()` from `reliability_lab.evaluation.groundedness`.
  - Normalizes evidence chunks into isolated XML delimiters (`<evidence id="...">`) with prompt injection defenses (escapes closing tags and instructs the model to ignore instructions within evidence).
  - Enforces schema-validated JSON outputs specifying categorical verdicts (`JudgeVerdict`: `SUPPORTED`, `PARTIALLY_SUPPORTED`, `UNSUPPORTED`, `CONTRADICTED`, `INSUFFICIENT_EVIDENCE`), calibrated faithfulness scores in `[0.0, 1.0]`, and cited chunk IDs.
  - Strict evidence verification: strips hallucinated or unprovided chunk citations; demotes claims to `UNSUPPORTED` when citation provenance fails.
  - Batch evaluation (`evaluate_batch`) and joint citation evaluation (`evaluate_with_citations`) interfacing cleanly with `CitationEvaluator`.
- **Directional NLI Correlation** (`calculate_llm_nli_correlation`):
  - Aligns LLM unsupportedness risk ($1.0 - \text{faithfulness\_score}$) with local NLI unsupportedness risk (`unsupported_risk` from `ClaimNLIResult`).
  - Calculates Pearson correlation ($r$) and Spearman rank correlation ($\rho$) with fractional average ranks for tied values.
  - Returns `None` with actionable warnings for zero-variance degenerate distributions and sample sizes $N < 3$ without fabricating misleading correlation coefficients.

---

## End-to-End RAG Evaluation Pipeline Orchestration (`reliability_lab/pipeline/`)

Per ADR-011, the pipeline subsystem unifies retrieval, answer generation, multi-stage evaluation, statistical alignment, and persistent experiment storage into a single cohesive runner:

1. **Pipeline Execution Flow**:
   ```
   [Corpus Chunks + Benchmark Queries]
                    │
                    ▼
          [Stage A: Retrieval]
          • BaseRetriever (BM25, FAISS, Hybrid)
          • Cranfield Metrics (Recall, Precision, HitRate, MRR, NDCG)
                    │
                    ▼
      [Stage B: Answer Generation]
      • BaseLLMClient (Gemini Flash or Deterministic Mock)
      • Anti-Injection Prompt Sanitization (XML Delimiter Escaping)
      • Zero Ground-Truth Answer Leakage
      • Bracketed Citation Directive ([CHK-xxx] / [CIT-xxx])
                    │
                    ▼
     [Stage C: Multi-Faceted Evaluation]
     • Lexical Groundedness (GroundednessEvaluator)
     • Citation Precision & Recall (CitationEvaluator)
     • Local NLI Entailment (NLIEntailmentEvaluator)
     • Structured LLM Judge (LLMJudge)
                    │
                    ▼
     [Stage D: Statistical Alignment]
     • calculate_llm_nli_correlation() (Pearson r, Spearman rho)
     • Sample size & zero-variance guards
                    │
                    ▼
     [Stage E: Result Aggregation & Persistence]
     • Canonical EvaluationReport
     • Granular per-query diagnostic payload
     • Explicit mock/live provenance tracking (is_mock)
     • Transactional SQLite write via ExperimentRepository
   ```

2. **Core Abstractions**:
   - `RAGEvaluationRunner` (`reliability_lab.pipeline.runner`): Principal orchestrator injecting retriever, LLM client generator, evaluators, and configuration.
   - `RAGPipelineConfig` (`reliability_lab.pipeline.models`): Hyperparameter and stage execution toggles (`top_k`, `temperature`, `prompt_version`, `enable_groundedness`, `enable_citations`, `enable_nli`, `enable_judge`).
   - `PerQueryRAGResult` (`reliability_lab.pipeline.models`): Complete per-query vertical slice containing retrieved candidates, generated answer, generation metadata, and diagnostic records from each active evaluation stage.

3. **LLM Provenance & Mock Safety Invariant**:
   - Provenance is explicitly tracked across answer generation and judging.
   - If either answer generation or the LLM judge uses a mock client, the overall pipeline result is stamped with `is_mock = True`.
   - The CLI emits prominent visual warnings whenever synthetic mock inference is utilized, preventing simulated runs from being misrepresented as live Gemini benchmarks.

---

## 10. Containerization & Deployment Architecture (ADR-012)

1. **Topology Overview**:
   ```
   ┌───────────────────────────────────────────────────────────────┐
   │                        Host Network                           │
   │                                                               │
   │   Browser Client                   Host Shell                 │
   │         │                               │                     │
   │         │ http://localhost:5173         │ docker compose ...  │
   │         ▼                               ▼                     │
   │   ┌──────────────┐              ┌──────────────┐              │
   │   │   frontend   │              │     api      │              │
   │   │  (:5173)     │              │   (:8000)    │              │
   │   └──────┬───────┘              └──────┬───────┘              │
   │          │ http://127.0.0.1:8000       │                      │
   │          └─────────────────────────────┤                      │
   │                                        │                      │
   │   ┌──────────────┐              ┌──────▼───────┐              │
   │   │  benchmark   │─────────────►│ Named Volume │              │
   │   │  (one-shot)  │              │ /data/       │              │
   │   └──────────────┘              │ *.sqlite3    │              │
   │   ┌──────────────┐              └──────────────┘              │
   │   │    tests     │                                            │
   │   │  (one-shot)  │                                            │
   │   └──────────────┘                                            │
   └───────────────────────────────────────────────────────────────┘
   ```

2. **Multi-Stage Container Specifications**:
   - **Backend (`Dockerfile`)**:
     - `base`: `python:3.11-slim`, non-root execution (`appuser:appgroup`), no wheel caching.
     - `test`: Installs `.[dev]`, executes static quality checks (`ruff`, `mypy`) and `pytest`.
     - `runtime`: Minimal footprint with `.[web,llm]`, mounting `/data` for SQLite persistence.
   - **Frontend (`frontend/Dockerfile`)**:
     - `builder`: `node:22-alpine`, reproducible installation via `npm ci`, runs `svelte-check`, unit tests, and production build using `@sveltejs/adapter-node`.
     - `runtime`: Lean Alpine container running as unprivileged `node` user, serving standalone `build/index.js` on `0.0.0.0:5173`.

3. **Data Durability & Volume Contract**:
   - The SQLite experiment database is persisted in a named Docker volume (`reliability-lab-data`) mounted at `/data`.
   - Standard container lifecycle events (`docker compose stop`, `docker compose down`, `docker compose up --build -d`) preserve the named volume and all historic benchmark runs.






