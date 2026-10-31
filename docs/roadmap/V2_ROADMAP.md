# AI Reliability Lab v2: Offline Document Intelligence Roadmap

## Strategic Vision

AI Reliability Lab v2 evolves the v1 evaluation and benchmarking platform into an extensible, privacy-preserving, local-first **Document Intelligence application**.

Where v1 established scientific rigor, baseline-first retrieval benchmarking, citation precision/recall metrics, local NLI entailment evaluation, and regression gating, v2 bridges these capabilities into an end-user document intelligence experience. Users can upload documents (starting with PDF and text), ask questions, and receive AI-generated answers with verified inline page/span citations—completely offline without data leaving local infrastructure.

---

## Milestone Sequence & Task Backlog

```
   [V2-001] Local LLM Inference Provider (Ollama)
          │
          ▼
   [V2-002] Document Ingestion Pipeline & File Upload
          │
          ▼
   [V2-003] Document Q&A / Chat Orchestrator (Conversational RAG)
          │
          ▼
   [V2-004] Interactive Document Viewer & Citation Highlighting UI
          │
          ▼
   [V2-005] Offline Evaluation & Reliability Benchmarking for Local Models
          │
          ▼
   [Future] Multi-Format Adapters & Multi-Modal Ingestion
```

---

### [V2-001] Local LLM Inference Provider (Ollama)
- **Status**: `COMPLETED` (in branch `feat/v2-001-ollama-provider`)
- **Objective**: Implement a native local inference provider using Ollama's HTTP REST API without third-party vendor SDK dependencies.
- **Key Deliverables**:
  - `OllamaLLMClient` implementing `BaseLLMClient` protocol in `reliability_lab/llm/client.py`.
  - Native communication with Ollama `POST /api/chat` with `stream: false`.
  - Structured parameter translation (`temperature`, `max_output_tokens` -> `num_predict`, `top_p`, `top_k`, `stop`).
  - Native JSON output extraction via `"format": "json"`.
  - Token evaluation accounting (`prompt_eval_count`, `eval_count`) and latency tracking.
  - Fail-fast error handling with actionable diagnostic messages for daemon unavailability and missing models (`ollama pull <model>`).
  - Provider factory integration in `get_llm_client(provider="ollama")`.
  - Distinct provenance tracking (`provider="ollama"`, `is_mock=False`) across storage, pipeline runner, and CLI.
  - Comprehensive hermetic unit tests with `httpx.MockTransport`.
  - Architecture Decision Record `ADR-013`.

---

### [V2-002] Document Ingestion Pipeline & File Upload
- **Status**: `BACKLOG`
- **Objective**: Provide robust file upload and storage management for user documents, supporting PDF and raw text ingestion with citation-aware chunking and persistent index generation.
- **Key Deliverables**:
  - Multi-part document upload endpoints in FastAPI backend (`/api/documents/upload`).
  - Local document repository managing raw files, extracted text, and document metadata in SQLite.
  - Automatic background indexing utilizing PyMuPDF parser and citation-aware chunkers (Fixed, Sentence, Sliding Window, Recursive).
  - Dual sparse (BM25) and dense (FAISS) persistent vector index creation per document collection.
  - Progress reporting and error validation for malformed, password-protected, or corrupted files.

---

### [V2-003] Document Q&A / Chat Orchestrator (Conversational RAG)
- **Status**: `BACKLOG`
- **Objective**: Build a conversational document Q&A orchestrator that synthesizes multi-turn dialogue with retrieved document context and strict source-page citations.
- **Key Deliverables**:
  - Conversational session manager maintaining query history and context-aware query reformulation.
  - Integration with `RAGEvaluationRunner` retrieval mechanisms (BM25, FAISS, and Hybrid RRF).
  - Prompt construction enforcing evidence boundaries, XML escaping, and strict inline citation tagging (`[CHK-xxx]`, `[p. X]`).
  - Multi-turn generation support via `OllamaLLMClient` with prompt caching and latency optimization.
  - Real-time streaming response protocol (Server-Sent Events) for progressive generation rendering.

---

### [V2-004] Interactive Document Viewer & Citation Highlighting UI
- **Status**: `BACKLOG`
- **Objective**: Implement a rich, side-by-side document reader and conversational interface in the SvelteKit frontend, providing interactive visual citation verification.
- **Key Deliverables**:
  - Embedded PDF viewer with page navigation, zoom, and text-layer selection.
  - Bidirectional citation linking: clicking an inline citation tag in the chat jumps to and highlights the corresponding bounding box/span on the PDF page.
  - Confidence scoring overlays and lexical containment indicators directly on evidence passages.
  - Drag-and-drop document upload tray with indexing progress indicators.
  - Chat interface supporting model selection (`llama3.2`, `mistral`, `deepseek-r1`, etc.) and generation parameter controls.

---

### [V2-005] Offline Evaluation & Reliability Benchmarking for Local Models
- **Status**: `BACKLOG`
- **Objective**: Extend the platform's reliability metrics to systematically benchmark local LLM models (e.g., Llama 3.2, Mistral 7B, Qwen 2.5, Phi-3) against cloud baselines for citation fidelity and groundedness.
- **Key Deliverables**:
  - Comparative benchmark harness evaluating local models on Cranfield datasets, citation recall/precision, and NLI entailment.
  - Quantization impact analysis (measuring faithfulness degradation across Q4_K_M, Q5_K_M, Q8_0, and FP16).
  - Automated offline regression testing to detect hallucination rate shifts between local model releases.
  - Comparative diagnostic reports in SvelteKit Studio charting inference throughput (tokens/sec) vs. citation accuracy.

---

## Future Extensions & Format Adapters

Following the completion of the core v2 roadmap, the architecture is designed to support modular format adapters and advanced document intelligence features:

1. **Enterprise Format Adapters**:
   - Microsoft Word (`.docx`) ingestion adapter extracting paragraph styles, tables, and footnote citations.
   - Microsoft PowerPoint (`.pptx`) adapter indexing slide text, presenter notes, and visual layout order.
   - Spreadsheet (`.xlsx`, `.csv`) adapter supporting structured tabular retrieval and row/column citation references.
2. **Multi-Modal Document Intelligence**:
   - Vision-language model integration via Ollama (e.g., `llava`, `llama3.2-vision`) for extracting tables, charts, diagrams, and figures from scanned PDFs.
   - OCR fallback pipeline (Tesseract / EasyOCR) for non-searchable or scanned document collections.
3. **Local Fine-Tuning & Distillation Evaluation**:
   - Evaluation frameworks for measuring custom LoRA adapters and domain-adapted quantized models against generic foundation models.
