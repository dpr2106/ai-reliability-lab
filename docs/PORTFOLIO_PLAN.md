# STING Portfolio Evidence Plan

This document keeps the project tied to demonstrable engineering outcomes instead of presentation-only claims. It is a plan, not proof that unfinished milestones have shipped.

## 1. The story

**One-line positioning (use after the core workbench is demonstrable):**

> Built STING, a local-first RAG reliability workbench that exposes hybrid retrieval traces, verifies citation provenance, inspects claim-level evidence, and benchmarks retrieval and local-model changes using reproducible evaluations.

**What makes the story credible:**

- A real implementation of BM25, FAISS, and hybrid RRF—not only API calls to a hosted RAG service.
- Citation-aware ingestion with document/page/span provenance.
- A trace showing what was retrieved and what was sent to generation.
- Citation validation that resolves to original chunks and source locations.
- Explicit abstention for insufficient evidence.
- Reproducible comparisons and regression detection.
- A local-first execution path using Ollama, with model/provider provenance.

Do not imply that the project created a foundation model, invented RAG, guarantees factual truth, or is more accurate than alternatives without a controlled and reproducible comparison.

## 2. Required proof assets

| Proof asset | What it should demonstrate | Evidence to capture |
|---|---|---|
| Architecture diagram | Ingestion → chunking → indexes → retrieval trace → generation → citation resolution → evaluation | Versioned diagram and linked source modules |
| Retrieval comparison | BM25 vs dense vs Hybrid RRF behavior | Dataset, queries, relevance labels, top-K results, metrics |
| Citation walkthrough | A cited claim resolves to an actual source passage | Query, answer, chunk ID, document/page/span |
| Abstention case | System refuses to invent an answer when evidence is absent | Input document set, question, retrieved candidates, response |
| Regression demo | A change degrades a measured result and the gate detects it | Baseline/candidate config hashes, failing query, regression output |
| Local deployment | User can run without sending routine document contents to a cloud API | Setup transcript, provider provenance, network/privacy notes |
| Test evidence | Correctness and maintainability | Commit SHA, CI run, test count, coverage, lint/typecheck output |
| Product demo | Usable experience end-to-end | Short unedited or clearly narrated screen recording |

## 3. Benchmark discipline

Every published result must include:

- Dataset identity, licensing/source, corpus size, query count, and relevance-label methodology.
- Retriever, embedding model, LLM, prompt version, chunking configuration, top-K, and hardware/software environment.
- Metric definitions and aggregation method.
- Whether the result was measured locally, mocked, or obtained through an external provider.
- Repeated-run methodology for latency and throughput.
- Known limitations and failure cases.

The existing case-study dataset (25 chunks and 16 queries) is suitable for demonstrating implementation and controlled examples, not for broad generalization. Build a larger, diverse, manually checked evaluation set before making comparative quality claims.

Do not publish a single "accuracy" percentage unless the target label, formula, dataset, and evaluation method are clearly defined.

## 4. Resume bullets — future templates only

Use only after the corresponding work is merged and verified. Replace all bracketed fields with actual measurements.

- Built a local-first RAG reliability workbench integrating BM25, FAISS, and Reciprocal Rank Fusion, with per-query retrieval traces and document-level citation provenance.
- Implemented citation validation and claim-level evidence diagnostics, including explicit insufficient-evidence abstention and prompt-injection protections.
- Developed a reproducible evaluation pipeline comparing [N] retrieval/model configurations across [N] manually verified queries; measured [named metrics] and added baseline regression detection.
- Delivered a local document-intelligence workflow using [actual model/provider], SQLite persistence, FastAPI, and [actual UI], with [verified test/CI evidence].

## 5. LinkedIn narrative — draft for a later milestone

Do not post this as a completed-project announcement until the core workflow is working and the claims are verified.

> I started by thinking I should build another offline AI chatbot. Then I realised that local chat and PDF Q&A were already well served by existing tools.
>
> The more interesting engineering problem was what happens after a RAG system answers: Which passages did it retrieve? Why did hybrid search rank one passage above another? Does the citation actually support the claim? What happens when the evidence is missing? And how do we know a model or retriever update did not make things worse?
>
> That became STING — an evidence-first, local-first RAG reliability workbench built on my AI Reliability Lab foundation. The focus is not just generating an answer, but making the retrieval and evidence behind it inspectable and measurable.
>
> [Insert a real retrieval trace screenshot, citation walkthrough, benchmark result, and repository/demo links here.]
>
> Still building, still finding edge cases. The goal is to make the system explainable enough to debug—not to pretend an LLM is always right.

## 6. Portfolio release checklist

- [ ] V2-003 query trace and citation resolution merged.
- [ ] V2-004 workbench UI and source navigation demonstrated.
- [ ] V2-005 benchmark comparison and regression workflow reproduced.
- [ ] Sample data and benchmark questions are redistributable and documented.
- [ ] Setup works from a clean checkout.
- [ ] Screenshots and demo show real current behavior.
- [ ] Published metrics include dataset/method/environment context.
- [ ] README clearly distinguishes Ollama/local inference from STING's value.
- [ ] Known limitations, hardware requirements, and privacy boundaries are disclosed.
- [ ] Resume and LinkedIn wording matches merged code, tests, and demo evidence.
