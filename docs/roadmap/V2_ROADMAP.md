# STING V2: Evidence-First RAG Reliability Workbench Roadmap

## Strategic Vision

AI Reliability Lab provides the reusable retrieval, evaluation, and reliability engineering core. STING is the local-first workbench built on it.

STING is **not another generic offline chatbot and is not an Ollama replacement**. It uses a local inference runtime such as Ollama, while differentiating through retrieval observability, citation provenance, claim-level evidence inspection, safe abstention, and reproducible evaluation.

The product should let a user ask a question about local documents and then inspect how the system arrived at its answer: which passages each retriever found, how RRF ranked them, which evidence reached the model, which claims cite which chunks, where support is missing, and how configuration changes affect measured quality.

## Milestone Sequence

```
[V2-001] Local inference foundation (Ollama)                  COMPLETE
        |
[V2-002] Document ingestion and persistent indexes           IN REVIEW
        |
[V2-003] Evidence-first query and trace orchestrator          NEXT
        |
[V2-004] STING workbench UI and source evidence inspector     BACKLOG
        |
[V2-005] RAG evaluation, comparison and regression studio     BACKLOG
        |
[V2-006] Reproducible demo and portfolio release              BACKLOG
```

## V2-001 — Local LLM Inference Provider (Ollama)

**Status:** `COMPLETED`

The provider-independent LLM contract supports Ollama over its native HTTP API without introducing an Ollama-specific SDK. It records provider/model provenance, token usage and latency, supports structured JSON responses, and reports actionable daemon/model errors. Existing non-streaming generation callers must remain compatible when streaming is added in later work.

## V2-002 — Document Ingestion and Persistent Indexing

**Status:** `IN_REVIEW` (core ingestion implementation exists; follow-up hardening is in PR #59)

The ingestion layer supports PDF, TXT, and Markdown, citation-aware chunks, source provenance, SQLite persistence, BM25 and FAISS indexes, lifecycle states, and safe deletion. PR #59 adds upload-body limits, freshness/lifecycle protection, stale-index invalidation, dimension validation, cleanup resilience, and host-path privacy. Do not mark the hardening follow-up complete until PR #59 is actually merged and verified.

## V2-003 — Evidence-First Query and Trace Orchestrator

**Status:** `BACKLOG` — next implementation milestone after V2-002 hardening is merged.

**Objective:** Build the application-level query service that connects selected documents, retrieval, local generation, citation validation, and an inspectable query trace. Conversational sessions are supported, but a generic chat interface is not the acceptance criterion.

**Key deliverables:**

1. A dedicated query/application orchestrator. Do not turn benchmark-oriented `RAGEvaluationRunner` into a conversation service. Reuse its lower-level interfaces and evaluation components where appropriate.
2. Document-scoped retrieval using BM25, FAISS, and Hybrid RRF, including deterministic deduplication and a safe BM25 fallback for partially indexed documents.
3. Query trace records containing normalized query, selected document scope, candidate chunks, retriever identity, original ranks/scores, RRF rank/contribution where applicable, final evidence set, and retrieval timing.
4. Evidence-bounded generation through the provider-independent LLM client, with Ollama as the initial local provider. Preserve prompt-injection protections and do not pass conversation history as verified evidence.
5. Citation resolution against chunks retrieved for that exact query. Each citation must resolve to document ID, filename, page, chunk ID, and available character/span coordinates. Reject or visibly flag invalid IDs.
6. A clear abstention path when retrieval is empty or evidence is insufficient. Do not invent a confidence or truth score.
7. Session/message persistence separated from benchmark experiment storage. Bounded history and context-aware follow-up handling must not leak between sessions.
8. Genuine Ollama streaming and an SSE API, added without breaking existing synchronous generation behavior.
9. Structured response metadata containing provider/model, token usage, generation latency, retrieval latency, citations, and the query trace.

**Acceptance criteria:**

- A deterministic integration test can submit a query against selected documents and inspect its retrieved candidates and resolved citations.
- No citation can point to a chunk outside the evidence set supplied to generation.
- A document outside the selected scope cannot influence the answer.
- A document in a reprocessing or failed state cannot expose stale evidence.
- The service abstains when evidence is inadequate.
- Conversation state persists and is isolated by session.
- Streaming is genuinely incremental from Ollama, not simulated after a complete response.
- The backend can return the same trace from a saved query/configuration sufficiently to reproduce retrieval ordering.

**Out of scope:** full chat UI, embedded PDF viewer, multi-model benchmark dashboards, LoRA/fine-tuning, cloud document processing, and unrelated refactors.

## V2-004 — STING Workbench UI and Source Evidence Inspector

**Status:** `BACKLOG`

**Objective:** Build the polished user interface that makes STING's differentiating engineering visible.

**Key deliverables:**

- Local workspace and document collection management with ingestion progress and failure states.
- Query/chat workspace with model and retrieval configuration visible.
- Retrieval inspector comparing BM25 candidates, dense candidates, and RRF results. Clearly distinguish ranks from raw scores.
- Evidence panel showing exactly what reached the model.
- Inline citations that open the correct document and source page; highlight text spans where reliable provenance exists.
- Claim/evidence inspection that shows supporting passages and separate diagnostic signals (lexical, NLI, judge) without reducing them to a fake universal score.
- Visible abstention and insufficient-evidence states.
- Responsive, accessible design with polished loading, empty, error, and partial-index states.

**Acceptance criteria:** A demo user can upload a PDF, ask a question, inspect the retrieval trace, click a citation, and verify the cited passage without reading backend logs.

## V2-005 — RAG Evaluation, Comparison and Regression Studio

**Status:** `BACKLOG`

**Objective:** Make reliability measurement a first-class user workflow, not merely a collection of backend modules.

**Key deliverables:**

- User-editable or bundled golden question sets with documented relevance labels and evidence references.
- Side-by-side comparison of BM25, dense, and Hybrid RRF retrieval.
- Configurable comparison of local LLMs, embedding models, prompts, chunking parameters, and top-K settings where supported.
- Retrieval metrics: Recall@K, Precision@K, Hit Rate@K, MRR@K, NDCG@K, and latency percentiles.
- Citation precision/recall, claim-groundedness diagnostics, abstention behavior, generation latency, and token throughput where measurable.
- Per-query failure inspection and regression comparison against a saved baseline.
- Reproducible experiment records containing dataset identity, configuration hash, provider/model provenance, software versions, and run metadata.
- Honest limitations and dataset-size disclosure beside charts and summary metrics.

**Acceptance criteria:** A user can reproduce a baseline, change a retrieval or generation setting, compare the results, inspect the worst query-level changes, and trigger a deterministic regression result.

## V2-006 — Reproducible Demo and Portfolio Release

**Status:** `BACKLOG`

**Objective:** Turn completed engineering into independently verifiable portfolio evidence.

**Key deliverables:**

- A small, redistributable sample document collection and manually verified benchmark questions.
- Clean-checkout setup instructions, local-model setup, data/privacy explanation, and resource requirements.
- Reproducible commands for ingestion, query traces, evaluation, and regression tests.
- Architecture diagram and concise engineering case study explaining design trade-offs and limitations.
- Screenshots and a short demo recording that show ingestion, retrieval inspection, citation verification, and evaluation comparison.
- CI status, test totals, coverage, and benchmark outputs captured from actual runs with date, commit, environment, and dataset size.
- Portfolio/resume/LinkedIn wording that only describes merged and demonstrated capabilities.

**Acceptance criteria:** A reviewer can follow the README from a clean checkout, reproduce the demo, inspect the evidence trace, and verify that published claims match the repository.

## Future Extensions

After the core workbench is reliable:

- Additional document formats such as DOCX, PPTX, XLSX, and CSV with format-specific provenance.
- OCR and vision-language ingestion for scanned documents, tables, and diagrams.
- Additional local inference providers and model quantization comparisons.
- User-approved saved preferences or retrieval configuration presets. No silent training on private conversations or documents.

## Engineering and Governance Rules

- Preserve branch protection and the repository's established branch/PR conventions.
- Never push directly to `main`; use task-specific branches and PRs.
- Do not merge or enable auto-merge unless explicitly authorized.
- Keep V1 historical milestones and benchmark evidence intact; V2 status is tracked separately here.
- Do not mark a task complete until the implementation is merged and verification evidence is recorded.
- Use targeted checks during development and the full backend/frontend/CI suite at milestone boundaries.
- Do not invent performance claims, coverage, latency, or accuracy figures.
- Prioritize a small, demonstrable vertical slice over broad, unfinished feature lists.
