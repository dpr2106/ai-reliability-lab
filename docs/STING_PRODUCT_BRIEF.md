# STING — Product Brief

**Product category:** Evidence-first, local-first RAG reliability workbench
**Built on:** AI Reliability Lab
**Tagline:** Ask. Inspect. Verify. Benchmark.

## 1. Product identity

STING is the user-facing workbench built on the engineering foundation in AI Reliability Lab. It helps developers and technically curious users inspect how document-grounded AI answers are produced, verify whether citations point to supporting evidence, and measure how retrieval and generation configurations behave.

STING is **not positioned as another general-purpose offline chatbot** and is not intended to replace Ollama, Open WebUI, AnythingLLM, or other local model interfaces. It uses local runtimes such as Ollama as inference infrastructure. Its value is the observability and evaluation layer around document-grounded answers.

The product should make the internal RAG pipeline understandable:

- What did the user ask?
- Which documents were in scope?
- Which passages did BM25 retrieve?
- Which passages did dense retrieval retrieve?
- How did hybrid fusion rank the candidates?
- Which passages were supplied to the model?
- Which answer claims cite those passages?
- Can each citation be resolved to the original source location?
- Where was evidence insufficient, and did the system abstain?
- Did a change to the model, prompt, chunker, or retriever improve or degrade measured quality?

## 2. Problem

A RAG answer can sound convincing and still be wrong. Common failure modes include retrieval blindspots, irrelevant or fabricated citations, unsupported claims, incorrect page references, prompt injection inside source documents, and regressions introduced by changing a retriever or model.

A standard chat interface hides most of these failure modes. STING should expose them in a form that can be inspected, reproduced, and evaluated.

## 3. Target users

- Students and engineers learning how RAG systems behave beyond a happy-path demo.
- Developers building local or private document-Q&A applications.
- ML/AI engineers comparing retrievers, embedding models, prompts, and local LLMs.
- Researchers or technical teams who need to inspect evidence provenance and compare experiments.

## 4. Product promise

For a document-grounded query, STING should provide both an answer and an inspectable trace of how that answer was produced. It should make evidence and uncertainty visible instead of presenting a fluent response as proof of correctness.

The product must not claim that a heuristic score, NLI classifier, or LLM judge proves truth. These are diagnostic signals with known limitations. Users must be able to inspect the underlying passages and understand how every displayed score was computed.

## 5. Core user workflow

1. Create a local workspace or knowledge collection.
2. Add PDF, TXT, or Markdown documents.
3. Inspect document processing and indexing status.
4. Ask a question about the selected documents.
5. Receive a locally generated answer with citations.
6. Inspect retrieved candidates, ranks, retrieval source, and fusion behavior.
7. Open the cited passage in its source document.
8. Inspect unsupported or weakly supported claims and any abstention decision.
9. Save the query as an evaluation case or add it to a small golden question set.
10. Compare a different retrieval/model configuration and inspect regressions.

## 6. Differentiating capabilities

### A. Retrieval observability

Show candidate lists from BM25 and dense retrieval separately, followed by hybrid RRF ranking. Where available, expose original ranks, scores, fusion contribution, selected top-K evidence, and deduplication decisions. Keep score semantics explicit; do not compare scores from unrelated retrievers as though they shared a common scale.

### B. Citation provenance

Resolve each citation against a chunk that was actually retrieved for the current answer. Return the document ID, original filename, page number, chunk ID, character offsets or available span coordinates, and a relevant source snippet. A citation that cannot be resolved must be rejected or visibly marked invalid.

### C. Claim-level evidence inspection

Where feasible, split the response into answer claims and associate each claim with cited evidence. Display lexical grounding, NLI, or judge outputs as separate diagnostic signals. Do not collapse them into a fabricated universal "truth score".

### D. Safe abstention

When retrieval returns no usable evidence or the evidence does not support an answer, the product must say so. An empty or weak evidence set must not trigger confident speculation.

### E. Reproducible evaluation

Allow a user to compare retrieval and generation configurations against a documented dataset of queries and relevance labels. Report the dataset size and methodology beside every metric. Preserve baselines and detect regressions instead of presenting one unqualified accuracy number.

### F. Local-first operation

Use Ollama for local inference initially, local embeddings/retrieval, and SQLite/file-backed persistence. Clearly disclose any optional component that downloads model weights. No routine document content should be sent to a cloud service.

## 7. Architecture boundaries

AI Reliability Lab remains the reusable engineering and evaluation core. STING is the interactive workbench built on that core.

Reuse existing components where appropriate:

- PyMuPDF and text adapters for ingestion and provenance.
- Citation-aware chunkers and persisted document chunks.
- BM25, FAISS, and Hybrid RRF retrieval.
- Ollama provider through the provider-independent LLM interface.
- Existing citation, groundedness, NLI, judge, benchmark, persistence, and regression modules.
- FastAPI for application APIs and SvelteKit for the workbench UI.

Build a dedicated query/application orchestration layer for interactive requests. Do not force the benchmark-oriented `RAGEvaluationRunner` to become a chat service. Reuse its lower-level interfaces and evaluation components, but keep interactive conversation state separate from benchmark experiment records.

## 8. Non-goals

- Reimplementing a general-purpose local model runtime or model manager.
- Competing with Ollama on model downloads, hardware support, or general chat UX.
- Building a generic assistant for unrelated productivity tasks.
- Claiming that citations guarantee correctness.
- Treating an LLM judge or NLI score as ground truth.
- Silent training on user documents or conversation history.
- Adding every possible file format before the core PDF/TXT/Markdown workflow is reliable.
- Building a complex multi-user cloud service before the local single-user workflow is strong.

## 9. Product quality bar

- Every visible metric has a documented definition and data source.
- Every citation is resolvable to actual retrieved evidence.
- Retrieval traces can be reproduced from saved configuration and inputs.
- Errors, partial indexing, missing model weights, empty states, and insufficient evidence are explicit.
- Local inference and cloud inference are never misrepresented; provider/model provenance is visible.
- UI quality is deliberate: dense but readable, technically credible, keyboard accessible, responsive, and complete across loading/error/empty states.
- Tests cover the failure path, not only the happy path.

## 10. Portfolio positioning

The project should be presented as an engineering workbench for inspecting and evaluating local RAG systems, not as a novel LLM or a generic chatbot. Portfolio claims must be backed by code, reproducible test results, a documented dataset, and a working demonstration.

A concise positioning statement, to use once the relevant capabilities are implemented and verified:

> Built STING, a local-first RAG reliability workbench that exposes hybrid retrieval traces, verifies citation provenance, inspects claim-level evidence, and benchmarks retrieval and local-model changes using reproducible evaluations.

Do not use this as a claim of completed functionality until the corresponding milestones are merged and demonstrable.
