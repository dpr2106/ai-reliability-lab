# Project Brief: AI Reliability Lab & STING

## Executive Summary

**AI Reliability Lab** is the reusable engineering and evaluation core for document-based AI systems. **STING** is the user-facing, local-first RAG reliability workbench built on that foundation.

The project's purpose is not to create another generic offline chatbot. Local inference runtimes and local document-chat applications already exist. STING uses infrastructure such as Ollama and focuses on the harder engineering questions around document-grounded answers: what was retrieved, why it was ranked, whether citations resolve to real source passages, where claims lack evidence, and whether system changes improve or regress measured quality.

## Product identity

- **Product:** STING
- **Category:** Evidence-first, local-first RAG reliability workbench
- **Tagline:** Ask. Inspect. Verify. Benchmark.
- **Engineering foundation:** AI Reliability Lab
- **Initial inference runtime:** Ollama (pluggable provider architecture retained)
- **Initial document formats:** PDF, TXT, Markdown

See [`STING_PRODUCT_BRIEF.md`](STING_PRODUCT_BRIEF.md) for the detailed product definition and [`roadmap/V2_ROADMAP.md`](roadmap/V2_ROADMAP.md) for milestone sequencing.

## Core Problem Statement

Document-based RAG systems can fail in ways hidden by a polished chat response:

1. **Retrieval blindspots:** The relevant passage is missing from the selected top-K evidence.
2. **Citation misattribution:** A citation is plausible-looking but does not support the claim, or refers to a chunk that was never retrieved.
3. **Unsupported generation:** The model uses parametric knowledge or inference where the selected documents provide insufficient evidence.
4. **Prompt injection in source material:** Instructions embedded in a document attempt to influence the model.
5. **Silent regressions:** A model, prompt, chunker, or retriever change reduces retrieval or grounding quality without breaking ordinary tests.
6. **Opaque decisions:** Users cannot see why a passage was selected or why an answer was produced.

## Product Promise

For every document-grounded query, STING should make the answer and its production trace inspectable. The user should be able to see retrieval candidates, hybrid ranking, the evidence supplied to generation, resolvable citations, and diagnostic signals for unsupported claims. When evidence is inadequate, STING should abstain transparently.

Scores from lexical overlap, NLI classifiers, or LLM judges are diagnostic signals—not proof of factual truth. Their meaning and limitations must be displayed honestly.

## Existing Engineering Foundation

The original AI Reliability Lab work remains valuable and should be preserved:

- Pure Python domain models and abstract interfaces.
- PyMuPDF parsing and provenance-aware chunking.
- BM25, Sentence Transformer embeddings, FAISS, and hybrid Reciprocal Rank Fusion.
- Cranfield-style retrieval metrics including Recall@K, Precision@K, Hit Rate@K, MRR@K, and NDCG@K.
- Citation parsing and precision/recall evaluation.
- Lexical groundedness, local NLI, structured LLM judging, and correlation analysis.
- SQLite experiment persistence and regression detection.
- Provider-independent LLM interface, Ollama integration, FastAPI adapter, SvelteKit diagnostic studio, and Docker packaging.

The existing benchmark case study uses a small canonical dataset (25 corpus chunks and 16 queries). It demonstrates implementation and trade-offs, but must not be presented as proof of universal performance. Future claims require broader, documented evaluation sets.

## Target Users and Portfolio Direction

Target users include students and engineers learning RAG internals, developers building private document-Q&A systems, and AI/ML engineers comparing local retrieval and generation configurations.

For a 2029 engineering portfolio, the project should demonstrate:

- Information retrieval fundamentals and practical trade-offs.
- Evidence provenance and citation correctness.
- Safe handling of untrusted document content.
- Local inference integration and resource-aware engineering.
- Reliable API, persistence, and background processing.
- Reproducible experiments, metrics, and regression gates.
- A polished interface that explains the system rather than merely wrapping a model.

## Technical and Product Requirements

- **Local-first:** Document processing, retrieval, persistence, and initial inference remain local. Any optional weight downloads are disclosed.
- **Inspectable retrieval:** Expose BM25, dense, and fused candidate results with clear score/rank semantics.
- **Citation integrity:** Only cite retrieved chunks; resolve each citation to the original document and source location.
- **Claim evidence:** Separate lexical overlap, NLI, and LLM-judge signals; do not invent a universal truth score.
- **Abstention:** Explicitly state when evidence is insufficient.
- **Reproducibility:** Save experiment configuration, dataset identity, provenance, and results.
- **Regression safety:** Compare candidates with baselines and surface query-level degradations.
- **Privacy:** Keep local paths private and avoid silent training on user data.
- **Premium UI/UX:** Strong hierarchy, retrieval trace visibility, usable evidence panels, responsive layouts, and complete loading/error/empty states.

## Scope Boundaries

STING is not a replacement for Ollama or a general-purpose model runtime. It is a workbench around RAG retrieval and reliability. The conversational interface is an entry point to the evidence inspector and evaluator—not the project's primary differentiator.

The project will not silently train on documents or conversation history. Any future personalization or adaptation must be explicit, user-controlled, and independently evaluated.

## Success Criteria

The core product is portfolio-ready when a user can:

1. Add a local document and observe its ingestion/indexing lifecycle.
2. Ask a question and receive a local answer with resolvable citations.
3. Inspect sparse, dense, and hybrid retrieval traces for that query.
4. Open the exact source passage behind a citation.
5. See unsupported claims or insufficient evidence clearly identified.
6. Run a documented benchmark comparing at least two retrieval or generation configurations.
7. Reproduce a baseline and detect a deliberately introduced regression.
8. Run the documented demo locally from a clean checkout.
