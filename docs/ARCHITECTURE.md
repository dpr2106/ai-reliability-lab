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

## Core Domain Models (Pure Python)
- `Document`: Raw parsed text, metadata dictionary, and page collection.
- `Chunk`: Text content, `doc_id`, `chunk_id`, `page_number`, `char_start`, `char_end`.
- `Query`: Unique `query_id`, query text, and evaluation metadata.
- `RetrievalResult`: Reference to `Chunk`, retrieval rank, score, and retriever source metadata.
- `EvaluationReport`: Aggregate metrics, per-query score breakdowns, and execution latency.
- `ExperimentRun`: Hash of run configuration, timestamp, dataset reference, and metric results.
