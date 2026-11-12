# Northstar System Architecture & Indexing Topology

Document ID: NORTHSTAR-ARCH-2026-V2
Version: 2.2.0
Publication Date: 2026-08-15
Author: Northstar Core Platform Engineering

## 1. Executive Summary

Northstar is a distributed, evidence-first document retrieval and citation verification engine. The platform ingests heterogeneous technical manuals, specifications, and post-mortem incident logs, producing durable indexes for hybrid semantic-lexical search and evidence-grounded generation.

## 2. Ingestion & Chunking Specification

Incoming documents undergo format-specific parsing (PDF, Markdown, and plaintext) followed by citation-aware character segmentation:
- Target Chunk Size: 500 characters
- Chunk Overlap: 100 characters
- Segmentation Boundary: Sentence and newline boundary preservation

Every chunk record retains provenance coordinates including document identifier, page number, byte offset, and character start/end offsets.

## 3. Dual Persistent Indexing Architecture

Northstar maintains two parallel retrieval indexes for every ingested document:

1. **Okapi BM25 Sparse Index**:
   - Tokenization: Alphanumeric lowercased terms with punctuation stripping.
   - Scoring Formula: Robertson-Spärck Jones BM25 with k1=1.5 and b=0.75.
   - Primary Role: Exact term matching, technical error codes, and alphanumeric model identifiers.

2. **FAISS Dense Vector Index**:
   - Model: 384-dimensional dense sentence embeddings using `all-MiniLM-L6-v2`.
   - Index Structure: `IndexFlatIP` (Exact Inner Product on normalized unit vectors).
   - Primary Role: Conceptual similarity, paraphrased user queries, and cross-lingual synonym matching.

## 4. Hybrid Ranking & Fusion Topology

To combine sparse and dense candidate lists, Northstar utilizes Reciprocal Rank Fusion (RRF) with a standard smoothing constant of k=60:
`RRF_Score(d) = sum(1.0 / (60 + Rank_i(d)))`

Candidates from BM25 and FAISS are merged without score normalization, ensuring robust rank aggregation across disparate metric scales.

## 5. Evidence-Bounded Citation Resolution

All answer generation pipelines operate under strict evidence boundaries:
- The language model receives only the top-K retrieved candidate chunks.
- The generation prompt enforces bracketed inline citation tags `[CIT-xxx]` corresponding to candidate chunk identifiers.
- A citation resolver verifies each generated tag against candidate chunks before presenting the response to the user.
- Any citation tag referencing a chunk outside the retrieved evidence set is flagged as an invalid citation.

