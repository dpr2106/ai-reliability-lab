# AI Reliability Lab: Empirical Architecture Case Study & Retrieval Benchmark Report

---

## 1. Executive Summary

Document-based Generative AI and Retrieval-Augmented Generation (RAG) systems frequently fail in silent, unmonitored ways: retrieval misses relevant passages due to vocabulary mismatch, language models generate unsupported claims (hallucinations), citation numbers point to irrelevant chunks, and subtle pipeline refactors degrade retrieval quality undetected.

**AI Reliability Lab** is an industrial-grade evaluation, diagnostics, and regression-testing platform built to systematically diagnose and prevent these failure modes. This case study presents the system architecture, mathematical foundations, and empirical benchmark findings across sparse (Okapi BM25), dense (Sentence Transformers + FAISS), and hybrid (Reciprocal Rank Fusion) retrieval strategies, as well as multi-faceted post-generation auditing (lexical groundedness, citation precision/recall, neural NLI entailment, and structured LLM-as-a-judge).

### Headline Empirical Findings
Under identical evaluation protocols on the verified 25-chunk, 16-query benchmark dataset at top-$K=5$:
- **Hybrid RRF** achieved the highest overall coverage (**Recall@5 = 0.9688**, **Precision@5 = 0.4000**), successfully bridging the semantic gaps where BM25 missed paraphrased relevant passages and dense retrieval missed technical terminology.
- **Dense FAISS (`all-MiniLM-L6-v2`)** demonstrated strong semantic generalization (**Recall@5 = 0.9479**, **MRR@5 = 1.0000**, **NDCG@5 = 0.9548**), placing a relevant document at rank 1 across all 16 queries.
- **Okapi BM25** provided sub-millisecond execution (**0.0951 ms mean latency**, ~360× faster than CPU dense embedding inference) with strong exact-match ranking (**NDCG@5 = 0.9113**), but suffered from vocabulary mismatch on paraphrased queries (e.g., sliding window loop invariants).
- **Latency / Quality Trade-off**: Dense encoding on CPU incurred a ~34.5 ms per-query latency overhead, illustrating that hybrid retrieval in production requires caching or GPU acceleration to maintain interactive SLAs.

---

## 2. Problem Statement

Modern RAG architectures commonly suffer from four compounding reliability bottlenecks:

1. **Retrieval Blindspots**: Lexical search fails when user queries use synonyms or abstractions not present in the document index. Conversely, dense embedding search fails when exact identifiers, alphanumeric error codes, or product codes are blurred across high-dimensional vector space.
2. **Unsupported Claims (Hallucinations)**: Generative models inject plausible but ungrounded assertions into responses, often extrapolating beyond the retrieved evidence.
3. **Citation Degradation**: Inline citations (`[CHK-xxx]`) often cite irrelevant source chunks or fail to reference the specific chunks containing the supporting evidence.
4. **Silent Regressions**: Refactors to chunking strategies, embedding models, prompt versions, or ranking weights frequently introduce regressions that pass standard unit tests but silently degrade end-to-end question-answering accuracy.

---

## 3. Why AI Reliability Lab Exists

Traditional LLM evaluation frameworks either rely exclusively on non-deterministic LLM-as-a-judge metrics (which suffer from prompt sensitivity, vendor lock-in, and high latency/cost) or reduce system health to a single aggregate score that obscures root causes.

AI Reliability Lab was designed from first principles with:
- **Baseline-First IR**: Mathematically rigorous, reproducible Cranfield IR metrics (Recall@K, Precision@K, Hit Rate@K, MRR@K, NDCG@K).
- **Zero Ground-Truth Answer Leakage**: Prompts strictly isolate retrieved evidence chunks from ground-truth references, preventing circular evaluation contamination.
- **Multi-Faceted Decoupled Auditing**: Independent measurement of retrieval recall, lexical groundedness (token containment and longest common subsequence), citation validity, neural NLI entailment, and structured factual judging.
- **Automated CI/CD Regression Gating**: Signed delta comparison against historic SQLite experiment baselines with deterministic exit codes for automated build pipelines.
- **100% Offline Reproducibility**: Pluggable provider-independent LLM abstractions with deterministic mock fallbacks and reproducible container topologies.

---

## 4. System Architecture

AI Reliability Lab is engineered around a clean layered architecture with pure Python standard-library domain models at its center (ADR-005), fully decoupled from external storage, presentation, and machine learning runtimes.

```mermaid
graph TD
    subgraph Core ["1. Core Domain Layer (Pure Standard Library - ADR-005)"]
        Models["Domain Models<br/>(Document, Chunk, Query, RetrievalResult, EvaluationReport)"]
        Interfaces["Protocol Interfaces<br/>(BaseRetriever, BaseChunker, BaseMetric, BaseLLMClient)"]
    end

    subgraph Ingestion ["2. Ingestion & Indexing Engine"]
        Parser["Document Parser (PDF / TXT / JSONL)"]
        Chunker["Citation-Aware Chunkers<br/>(Fixed, Sentence, SlidingWindow, Recursive)"]
        BM25["BM25 Lexical Index (k1=1.5, b=0.75)"]
        FAISS["FAISS Vector Index (IndexFlatIP, all-MiniLM-L6-v2)"]
        Hybrid["Hybrid RRF Index (rrf_k=60)"]
    end

    subgraph Pipeline ["3. End-to-End Orchestration (PIPE-001 / ADR-011)"]
        Runner["RAGEvaluationRunner"]
        PromptBuilder["Anti-Injection Prompt Builder<br/>(XML Delimiter Escaping, Zero Leakage)"]
        LLMClient["BaseLLMClient (Gemini Flash / MockClient)"]
    end

    subgraph Evaluation ["4. Multi-Faceted Evaluation Engine"]
        Cranfield["Cranfield IR Metrics (Recall, Prec, MRR, NDCG)"]
        Groundedness["Lexical Groundedness (Containment + LCS)"]
        Citation["Citation Evaluator (Parser & Resolution)"]
        NLI["Local NLI Entailment (RoBERTa / Cross-Encoder)"]
        Judge["Structured LLM Judge (Faithfulness & Risk)"]
        Corr["LLM-NLI Directional Correlation"]
    end

    subgraph Persistence ["5. Storage & Persistence Layer (EXP-001)"]
        SQLite["SQLite Experiment Database (.sqlite3)"]
        Repo["ExperimentRepository"]
        Detector["RegressionDetector (Tolerance Gating)"]
    end

    subgraph Presentation ["6. Presentation & Delivery"]
        CLI["Headless CLI (reliability-lab)"]
        FastAPI["Diagnostic REST API Adapter (:8000)"]
        Svelte["SvelteKit Diagnostic Studio (:5173 - ADR-008)"]
        Docker["Docker Compose Topology (PUB-001 / ADR-012)"]
    end

    Parser --> Chunker --> Core
    Core --> BM25
    Core --> FAISS
    BM25 & FAISS --> Hybrid
    Hybrid --> Runner
    Runner --> PromptBuilder --> LLMClient
    Runner --> Cranfield & Groundedness & Citation & NLI & Judge
    Judge & NLI --> Corr
    Evaluation --> Repo --> SQLite
    Repo --> Detector
    Repo --> FastAPI --> Svelte
    CLI --> Runner & Detector
```

---

## 5. Retrieval System Design

To benchmark IR performance objectively, the platform implements three distinct retrieval paradigms against identical document corpora and queries:

```mermaid
graph LR
    Query["User Query"] --> BM25["Sparse BM25<br/>Robertson-Spärck Jones IDF<br/>Token Matching"]
    Query --> Dense["Dense FAISS<br/>all-MiniLM-L6-v2<br/>L2 Normalized Inner Product"]

    BM25 --> R_Sparse["Sparse Ranked Candidates<br/>(rank_s = 1, 2, ..., N)"]
    Dense --> R_Dense["Dense Ranked Candidates<br/>(rank_d = 1, 2, ..., N)"]

    R_Sparse --> RRF["Reciprocal Rank Fusion<br/>Score = Σ 1 / (60 + rank)"]
    R_Dense --> RRF

    RRF --> Final["Final Fused Top-K Results"]
```

### Retrieval Paradigm Comparison
1. **Okapi BM25 (`BM25Retriever`)**: Lexical frequency-weighted term matching based on probabilistic relevance.
2. **Dense Vector Retrieval (`FaissVectorRetriever`)**: Bi-encoder transformer embeddings searching over semantic vector spaces using FAISS `IndexFlatIP`.
3. **Hybrid Reciprocal Rank Fusion (`HybridRetriever`)**: Score-invariant rank combination merging sparse and dense rankings.

---

## 6. BM25 Implementation Details

Implemented in `reliability_lab/indexing/bm25_index.py`:
- **Hyperparameters**: $k_1 = 1.5$ (term frequency saturation), $b = 0.75$ (document length penalization).
- **Inverse Document Frequency (IDF)**: Robertson-Spärck Jones formulation with lower-bound clipping at zero to prevent negative weights on common terms:
  $$\text{IDF}(q_i) = \max\left(0.0, \ln\left(\frac{N - n(q_i) + 0.5}{n(q_i) + 0.5}\right)\right)$$
- **Score Function**:
  $$\text{Score}(D, Q) = \sum_{q_i \in Q} \text{IDF}(q_i) \cdot \frac{f(q_i, D) \cdot (k_1 + 1)}{f(q_i, D) + k_1 \cdot \left(1 - b + b \cdot \frac{|D|}{\text{avgdl}}\right)}$$
- **Deterministic Tie-Breaking**: When chunk scores are equal, ordering defaults to ascending lexicographical chunk ID to ensure 100% reproducible benchmark execution across test runs.
- **Strengths**: Exact matching on technical identifiers (e.g., `IndexFlatIP`, `k1`, `SlidingWindowChunker`), near-instant index build, sub-millisecond query evaluation.
- **Weaknesses**: Vulnerable to vocabulary mismatch; fails when queries use paraphrasing or conceptual synonyms without literal term overlap.

---

## 7. Dense Retrieval with FAISS Implementation Details

Implemented in `reliability_lab/indexing/faiss_index.py` and `reliability_lab/indexing/embeddings.py`:
- **Underlying Embedding Model**: `sentence-transformers/all-MiniLM-L6-v2` (384-dimensional dense vectors).
- **Index Type**: `faiss.IndexFlatIP` (exact brute-force inner product search wrapped with `IndexIDMap2` for chunk tracking).
- **L2 Normalization**: Vectors are normalized to unit Euclidean norm ($\|v\|_2 = 1.0$), ensuring that inner product strictly equates to cosine similarity:
  $$\langle \mathbf{u}, \mathbf{v} \rangle = \frac{\mathbf{u} \cdot \mathbf{v}}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2} = \cos(\theta)$$
- **Memory & Storage**: In-memory dense array of float32 vectors ($25 \times 384 \times 4 \text{ bytes} \approx 38.4 \text{ KB}$).
- **Strengths**: Robust semantic recall on conceptual queries; handles paraphrasing and cross-lingual synonymy.
- **Weaknesses**: Higher computational cost during query embedding on CPU; potential degradation on rare alphanumeric product codes or formula symbols.

---

## 8. Hybrid Retrieval with Reciprocal Rank Fusion (RRF)

Implemented in `reliability_lab/indexing/hybrid.py`:
- **Rank-Based Merging**: Avoids fragile min-max normalization or arbitrary score scaling between disparate distributions (unbounded BM25 scores vs $[-1.0, 1.0]$ cosine similarities).
- **RRF Formula**:
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k_{\text{rrf}} + r_m(d)}$$
  where $k_{\text{rrf}} = 60$ (standard Cormack et al. constant) and $r_m(d) \in \{1, 2, \dots\}$ represents the 1-based rank of document $d$ in retriever $m$.
- **Deep Candidate Pooling**: Retrieves up to $K_{\text{pool}} = \max(20, 2K)$ candidates from each individual retriever before fusing and selecting the final top-$K$.
- **Synergistic Effect**: If a document is retrieved highly by both sparse and dense retrievers, its fused score accumulates aggressively. If a document is only captured by one retriever, it is still preserved above irrelevant items that neither retriever found.

---

## 9. Benchmark Dataset and Experimental Methodology

### Dataset Composition
- **Corpus**: `data/benchmarks/sample_corpus.jsonl`
  - 25 validated chunks derived from technical RAG documentation, chunking specifications, IR metric definitions, and safety guidelines.
  - Average chunk length: 118 tokens.
- **Query Set**: `data/benchmarks/sample_benchmark.jsonl`
  - 16 canonical queries with manually labeled binary and graded relevance judgments.
  - Each query is mapped to between 1 and 3 relevant chunks.
- **Evaluation Cutoff**: Top-$K = 5$ (standard RAG prompt context window constraint).

### Experimental Protocol
1. Exact same corpus chunks and queries loaded into each retriever.
2. Index construction time measured in isolation using high-resolution monotonic clocks (`time.perf_counter()`).
3. Single evaluation pass to compute canonical Cranfield ranking metrics.
4. Dedicated timing benchmark with 3 warm-up queries followed by 5 repetitions across all 16 queries (80 query executions per retriever) to obtain stable latency distributions (P50, P90, P95, P99).
5. All executions conducted on the identical local hardware and runtime environment.

---

## 10. Environment and Reproducibility

- **Operating System**: Windows 11 Enterprise (AMD64)
- **Processor**: AMD64 Family 25 Model 104 Stepping 2 (~3.29 GHz, multi-core)
- **Python Version**: 3.13.3 (x86_64)
- **Core Dependencies**:
  - `numpy`: 2.5.4
  - `faiss-cpu`: 1.15.1
  - `torch`: 2.15.0+cpu
  - `sentence-transformers`: 6.2.0
  - `transformers`: 5.22.1
- **Embedding Device**: CPU execution (`device="cpu"`)
- **Random Seed / Determinism**: Seed 42, deterministic tie-breaking.
- **Machine-Readable Results**: Persisted to `data/benchmarks/comparative_results.json`.

---

## 11. Comparative Results

The table below presents the verified empirical measurements collected across all 16 queries using `scripts/compare_retrievers.py`:

| Metric / Dimension | Okapi BM25 | Dense FAISS (Flat IP) | Hybrid RRF ($k=60$) | Delta (Hybrid vs BM25) | Delta (Hybrid vs Dense) |
|---|---|---|---|---|---|
| **Recall@5** | 0.8542 | 0.9479 | **0.9688** | **+0.1146 (+13.4%)** | **+0.0209 (+2.2%)** |
| **Precision@5** | 0.3500 | 0.3875 | **0.4000** | **+0.0500 (+14.3%)** | **+0.0125 (+3.2%)** |
| **Hit Rate@5** | 1.0000 | 1.0000 | 1.0000 | 0.0000 (parity) | 0.0000 (parity) |
| **MRR@5** | 0.9688 | **1.0000** | 0.9688 | 0.0000 (parity) | -0.0312 (-3.1%) |
| **NDCG@5** | 0.9113 | **0.9548** | 0.9494 | **+0.0381 (+4.2%)** | -0.0054 (-0.6%) |
| **Mean Latency** | **0.0951 ms** | 34.4916 ms | 35.6181 ms | +35.5230 ms | +1.1265 ms |
| **P50 Latency** | **0.0921 ms** | 34.2345 ms | 35.2590 ms | +35.1669 ms | +1.0245 ms |
| **P90 Latency** | **0.1174 ms** | 37.8920 ms | 38.4110 ms | +38.2936 ms | +0.5190 ms |
| **P95 Latency** | **0.1245 ms** | 38.7498 ms | 39.3691 ms | +39.2446 ms | +0.6193 ms |
| **P99 Latency** | **0.1346 ms** | 39.4144 ms | 44.9526 ms | +44.8180 ms | +5.5382 ms |
| **Index Build Time** | **3.33 ms** | 848.90 ms | 743.68 ms | +740.35 ms | -105.22 ms |

---

## 12. Latency and Indexing Cost Analysis

```
Retrieval Latency (P50 Log Scale, ms):
BM25:        [0.09 ms]   |
Dense FAISS: [34.23 ms]  |████████████████████████████████████
Hybrid RRF:  [35.26 ms]  |█████████████████████████████████████
```

### Analysis of Computational Trade-offs
1. **Query Encoding Bottleneck**: In dense retrieval, 99.8% of query latency (~34.2 ms) is consumed by the bi-encoder transformer forward pass generating the 384-dimensional query vector on CPU. The actual FAISS vector search across 25 chunks took less than 0.05 ms.
2. **Hybrid Overhead**: Hybrid RRF combines BM25 term search (0.09 ms), dense search (34.49 ms), candidate pooling, and rank reciprocal summation. The marginal overhead of RRF over pure dense retrieval is only ~1.12 ms.
3. **Index Construction**: BM25 indexed 25 chunks in **3.33 ms** (7,500 chunks/sec). Dense FAISS required **848.90 ms** due to computing 25 embedding vectors via PyTorch on CPU (~34 ms/chunk).

---

## 13. Query-Level Failure Analysis

Examining individual query traces revealed why hybrid retrieval out-performed individual approaches:

### Case A — Lexical Exact-Match Success (`Q-001`)
- **Query Text**: *"What are the default parameter values for k1 and b in the Okapi BM25 implementation?"*
- **Ground Truth**: `CHK-HYB-001`, `CHK-HYB-002`
- **BM25 Retrieval**: Rank 1: `CHK-HYB-001`, Rank 3: `CHK-HYB-002` (Recall@5 = 1.00, Latency = 0.08 ms).
- **Explanation**: Exact alphanumeric tokens (`k1`, `b`, `BM25`) appeared directly in the chunks, allowing BM25 to score highest with zero semantic inference required.

### Case B — Vocabulary Mismatch & Semantic Paraphrasing (`Q-002`)
- **Query Text**: *"How does SlidingWindowChunker guarantee that chunk generation will not encounter an infinite loop?"*
- **Ground Truth**: `CHK-ING-003`, `CHK-ING-004`
- **BM25 Output**: Retrieved `CHK-ING-003` at rank 1, but ranked `CHK-ING-004` outside top-5 (Recall@5 = **0.50**).
- **Dense FAISS Output**: Retrieved `CHK-ING-003` at rank 1, and `CHK-ING-004` at rank 4 (Recall@5 = **1.00**).
- **Explanation**: `CHK-ING-004` discusses progress assertions and step size guarantees without using the literal words *"infinite loop"*. BM25 missed the document entirely due to vocabulary mismatch, whereas Sentence Transformers mapped the semantic concept of loop safety accurately.

### Case C — Hybrid Synergistic Rank Fusion (`Q-003`)
- **Query Text**: *"What mathematical formula is used to calculate Discounted Cumulative Gain (DCG@K) for graded relevance?"*
- **Ground Truth**: `CHK-CRAN-003`, `CHK-CRAN-001`, `CHK-CRAN-002` (3 relevant chunks).
- **BM25 Output**: Retrieved `CHK-CRAN-003` and `CHK-CRAN-001`, missed `CHK-CRAN-002` (Recall@5 = **0.67**).
- **Dense FAISS Output**: Retrieved `CHK-CRAN-003` and `CHK-CRAN-002`, missed `CHK-CRAN-001` (Recall@5 = **0.67**).
- **Hybrid RRF Output**: Retrieved all three! Rank 1: `CHK-CRAN-003`, Rank 4: `CHK-CRAN-001`, Rank 5: `CHK-CRAN-002` (Recall@5 = **1.00**).
- **Engineering Implication**: BM25 and FAISS had complementary blind spots. RRF fused both sets of evidence into top-5 without diluting top-ranked items.

### Case D — Multi-Retriever Top-K Blindspot (`Q-014`)
- **Query Text**: *"What vector dimension and normalization are used by the local dense embedding model?"*
- **Ground Truth**: `CHK-HYB-003`, `CHK-HYB-004`
- **All Retrievers**: Captured `CHK-HYB-003` at rank 1 or 2, but all three failed to place `CHK-HYB-004` in the top 5 (Recall@5 = **0.50**).
- **Root Cause Analysis**: `CHK-HYB-004` focuses on FAISS index implementations (`IndexFlatIP` vs `IndexHNSWFlat`), whereas distractor chunks with heavy "vector" and "model" keyword density crowded out `CHK-HYB-004` at $K=5$.
- **Recommendation**: For multi-topic benchmark queries, increasing cutoff to $K=10$ or implementing reciprocal reranking over a larger context window resolves this boundary clipping.

---

## 14. RAG Groundedness and Citation Evaluation

Beyond retrieval ranking, RAG reliability requires auditing the generated response. AI Reliability Lab implements a complete evaluation pipeline (PIPE-001 / ADR-011):

```mermaid
sequenceDiagram
    autonumber
    actor Caller as Client / CLI
    participant Pipe as RAGEvaluationRunner
    participant Ret as BaseRetriever
    participant Prompt as PromptGenerator
    participant LLM as BaseLLMClient
    participant Ground as GroundednessEvaluator
    participant Cite as CitationEvaluator
    participant Judge as LLMJudge
    participant Repo as ExperimentRepository

    Caller->>Pipe: run_evaluation(query, corpus)
    Pipe->>Ret: retrieve(query.text, top_k=5)
    Ret-->>Pipe: candidate_chunks
    Pipe->>Prompt: format_prompt(query, candidate_chunks)
    Note over Prompt: Delimiters Escaped (</evidence>)<br/>Zero Ground-Truth Leakage
    Prompt-->>Pipe: secure_prompt
    Pipe->>LLM: generate(secure_prompt)
    LLM-->>Pipe: generated_answer
    par Lexical Evaluation
        Pipe->>Ground: evaluate(generated_answer, candidate_chunks)
        Ground-->>Pipe: containment & LCS scores
    and Citation Evaluation
        Pipe->>Cite: evaluate(generated_answer, candidate_chunks)
        Cite-->>Pipe: precision & recall
    and LLM Judge
        Pipe->>Judge: evaluate_faithfulness(generated_answer, candidate_chunks)
        Judge-->>Pipe: structured verdicts & justifications
    end
    Pipe->>Repo: save_report(evaluation_report)
    Repo-->>Caller: StoredRun (run_id, provenance, metrics)
```

### Metrics Defined
1. **Lexical Groundedness (`GroundednessEvaluator`)**: Extracts individual sentence claims and calculates token containment and Longest Common Subsequence (LCS) ratios against source chunks without calling external APIs.
2. **Citation Precision & Recall (`CitationEvaluator`)**: Extracts inline citation tags (`[CHK-xxx]`, `[CIT-xxx]`), validates whether cited chunks exist in the retrieved candidate set (Precision), and checks whether required reference sources were cited (Recall).
3. **Structured LLM Judge (`LLMJudge`)**: Enforces JSON-schema completions validating whether generated claims are supported by evidence chunks, penalizing unsupported statements.

---

## 15. LLM Judge and NLI Correlation

To avoid single-evaluator bias, AI Reliability Lab correlates structured LLM judge scores against local Natural Language Inference (NLI) cross-encoders:

- **Mathematical Alignment**: Unsupportedness risk from LLM judging ($1.0 - \text{faithfulness\_score}$) is directionally aligned with NLI contradiction/neutral probability (`unsupported_risk`).
- **Statistical Measures**: Pearson $r$ (linear correlation) and Spearman $\rho$ (monotonic rank correlation) with fractional rank tie-averaging.
- **Zero-Variance & Sample Guards**: If fewer than 3 aligned query pairs exist, or if metric variance is zero, the calculation returns `None` with actionable warnings rather than fabricating misleading correlation coefficients.

---

## 16. Regression Detection Case Study

Implemented in `reliability_lab/storage/regression.py` (EXP-002):
The platform provides automated CI/CD gating by comparing candidate experiment runs against historic baselines.

```mermaid
graph TD
    Baseline["Baseline Experiment Run<br/>(e.g., BM25 Baseline: Recall=0.8542)"] --> Compare["RegressionDetector<br/>(Tolerance = 0.05)"]
    Candidate["Candidate Experiment Run<br/>(e.g., Modified Tokenizer: Recall=0.7500)"] --> Compare

    Compare --> Calc["Compute Signed Metric Deltas<br/>Δ = Candidate - Baseline"]
    Calc --> Eval{"Is Higher-is-Better Metric<br/>degraded beyond tolerance?<br/>(Δ < -tolerance)"}

    Eval -- Yes --> Regressed["Status: REGRESSION<br/>Exit Code: 1 (CI Gate Fails)"]
    Eval -- No --> Pass["Status: PASS<br/>Exit Code: 0 (CI Gate Passes)"]
```

### Controlled Regression Demonstration
- **Baseline Run**: Recall@5 = 0.8542, NDCG@5 = 0.9113.
- **Candidate Run (Controlled Degradation)**: Query `Q-001` recall dropped from 1.0 to 0.0; aggregate Recall@5 dropped to 0.7500 ($\Delta = -0.1042$).
- **Detection Result**: `RegressionStatus.REGRESSION`, `has_regression = True`, flags `recall@5` and `ndcg@5` as regressed beyond the 0.05 tolerance threshold.

---

## 17. Containerized Deployment Architecture

Packaged in `Dockerfile`, `frontend/Dockerfile`, and `docker-compose.yml` (ADR-012 / PUB-001):

```mermaid
graph TD
    subgraph Host ["Host System"]
        Browser["Browser Client (:5173)"]
        CLI_User["Host Terminal"]
    end

    subgraph Compose ["Docker Compose Environment"]
        API["FastAPI Container (reliability-lab-api:8000)<br/>Python 3.11-slim, Non-Root (appuser)"]
        Frontend["SvelteKit Container (reliability-lab-frontend:5173)<br/>Node 22 Alpine, adapter-node, Non-Root (node)"]
        Benchmark["Benchmark Runner Container (one-shot)<br/>scripts/run_benchmarks.sh"]
        Volume[("Named Volume: reliability-lab-data<br/>/data/experiments.sqlite3")]
    end

    Browser -->|http://localhost:5173| Frontend
    Frontend -->|http://127.0.0.1:8000| API
    API --> Volume
    Benchmark --> Volume
    CLI_User -->|docker compose up -d| Compose
```

---

## 18. Limitations and Threats to Validity

1. **Benchmark Dataset Scale**: The sample benchmark consists of 25 chunks and 16 queries. While sufficient to demonstrate algorithmic mechanics and differential ranking behavior, findings cannot be extrapolated as universal claims across multi-million-document corpora.
2. **CPU Inference Latency**: Sentence Transformers encoding was benchmarked on CPU. In GPU-accelerated production environments, embedding latency typically drops from ~35 ms to < 2 ms.
3. **Environment Caveat**: Container runtime daemon was stopped on the local development host; configuration and static specifications were validated via `docker compose config` and native tests.

---

## 19. Architectural Decisions & Engineering Trade-offs

| ADR | Topic | Architectural Decision | Key Trade-off / Consequence |
|---|---|---|---|
| **ADR-005** | Domain Layer | Pure standard-library dataclasses in `core/` | Zero external dependencies; prevents Pydantic/framework churn. |
| **ADR-007** | Git Policy | Rebase-only merge policy with linear history | Preserves granular, atomic commits on `main` toward the 300-commit requirement. |
| **ADR-008** | Frontend Architecture | Decoupled SvelteKit (runes, TS) + FastAPI | Avoids Streamlit re-execution overhead; enables responsive dark-themed UI. |
| **ADR-009** | LLM Client | Provider-agnostic abstract client + Gemini wrapper | Hermetic testing with deterministic mocks; live cloud inference when configured. |
| **ADR-010** | LLM Judge | Schema-validated JSON judging + NLI alignment | Strict citation validation; directional alignment between LLM and cross-encoders. |
| **ADR-011** | Orchestration | End-to-end RAG runner (`PIPE-001`) | Prevents ground-truth leakage; enforces prominent mock provenance banners. |
| **ADR-012** | Deployment | Multi-stage Docker + `@sveltejs/adapter-node` | Zero host dependency requirements; persistent SQLite volume durability. |

---

## 20. Future Work

1. **BEIR / MS MARCO Ingestion**: Add loader utilities for large-scale public evaluation benchmarks (e.g. TREC-COVID, FiQA, SciFact).
2. **Learned Sparse Retrievers**: Integrate SPLADE or BM25 with learned term expansions.
3. **Cross-Encoder Reranking**: Benchmark monoT5 or BGE-Reranker stages over the hybrid candidate pool.
4. **Quantized Vector Indexing**: Benchmark FAISS `IndexIVFPQ` and HNSW-32 memory and recall trade-offs as corpus scale scales to $10^6$ chunks.

---

## 21. Conclusion

AI Reliability Lab demonstrates that building reliable RAG systems requires moving beyond ad-hoc experimentation toward systematic, baseline-first engineering. By combining Cranfield IR evaluation, multi-faceted post-generation auditing, and automated regression gating, the platform ensures that improvements in one component do not quietly degrade overall system reliability.
