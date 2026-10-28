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
