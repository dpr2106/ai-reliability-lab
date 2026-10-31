# Architecture Decision Records (ADRs)

## ADR-001: Baseline-First Evaluation Architecture
- **Status**: Accepted
- **Context**: RAG applications frequently add vector databases without measuring whether dense retrieval outperforms simple keyword search on domain-specific tokens or acronyms.
- **Decision**: Implement an in-memory Okapi BM25 sparse index as the primary baseline. All dense embedding models (e.g., SentenceTransformers + FAISS) must be measured against this baseline on identical evaluation splits.
- **Consequences**: Avoids unverified claims of "semantic superiority"; establishes reproducible Cranfield IR baselines.

## ADR-002: Protected `main` Branch with Native GitHub Auto-Merge
- **Status**: Accepted
- **Context**: Solo engineering projects often decay through undisciplined direct pushes to `main`, bypassing linters and tests.
- **Decision**: Protect `main` against direct pushes and force pushes. Require `test-and-lint` CI checks to pass. Enable GitHub native auto-merge (`gh pr merge --auto --squash`) without mandatory peer approval click to avoid author deadlocks.
- **Consequences**: Every change on `main` is backed by a verified, green pull request and linear commit history.

## ADR-003: Atomic In-Branch Task Status Updates
- **Status**: Accepted
- **Context**: If `main` is protected, attempting to update `docs/roadmap/TASKS.md` or `docs/STATUS.md` after a PR merges requires a direct push that GitHub will reject.
- **Decision**: Include verification evidence and mark tasks complete within the feature branch PR itself.
- **Consequences**: When auto-merge executes, code, tests, and documentation updates land on `main` in a single atomic squash commit.

## ADR-004: Ingestion Engine Selection (PyMuPDF / `fitz`)
- **Status**: Accepted
- **Context**: Citation verification requires attributing claims to exact page numbers and character bounding spans. Naive PDF text extractors strip layout coordinates and often misorder multi-column text.
- **Decision**: Use PyMuPDF (`fitz`) for document parsing due to its C-backed speed and explicit page/span coordinate preservation.
- **Consequences**: Enables deterministic citation auditing and span matching without cloud document processing APIs.

## ADR-005: Pure Python Standard Library Domain Models in Core
- **Status**: Accepted
- **Context**: `docs/ARCHITECTURE.md` declared `reliability_lab/core/` to possess zero external dependencies, but `FOUND-003`'s initial acceptance criteria referenced Pydantic models. Third-party dependencies in the core domain layer introduce coupling and potential dependency churn.
- **Decision**: Implement all core domain models (`Document`, `Chunk`, `Query`, `RetrievalResult`, `EvaluationReport`) and abstract interfaces using pure Python standard library (`dataclasses`, `typing`, `enum`, `abc`, `json`, `uuid`). Enforce domain constraints via `@dataclass` and `__post_init__` validation. Pydantic or schema serializers may be utilized at boundaries (e.g., dataset ingestion or API layers), but `core/` remains strictly zero-external-dependency standard library Python.
- **Consequences**: Ensures maximum portability, clean hexagonal architecture, zero vendor/library lock-in for core entities, and complete alignment between documentation and implementation.

## ADR-006: Squash-Only Merge Strategy & Linear History Enforcement
- **Status**: Superseded by ADR-007
- **Context**: Multiple merge strategies (merge commits, rebase merges, squash merges) create inconsistent Git graphs. Inadvertent merge commits introduce nonlinear branch clutter and obscure per-task atomic changes on `main`.
- **Decision**: Configure GitHub repository settings to exclusively allow squash merges (`allow_squash_merge: true`, `allow_merge_commit: false`, `allow_rebase_merge: false`). Enforce linear history in branch protection (`required_linear_history: true`).
- **Consequences**: Ensures a clean, 1-commit-per-task linear Git history on `main`, guarantees PR links in commit headlines, and eliminates accidental merge commits.

## ADR-007: Rebase-Only Merge Strategy & Commit Preservation on Main
- **Status**: Accepted
- **Context**: ADR-006 configured squash-only merging to guarantee a clean linear history. However, squash merging collapses all feature-branch commits into a single commit upon merge, erasing distinct, reviewable engineering milestones (such as separate test setup, core implementation, refactoring, and documentation commits). The project requires at least 300 genuine, meaningful engineering commits reachable from `main`. Continuing with squash-only merging obscures granular commit provenance and would require hundreds of micro-PRs.
- **Decision**: Configure GitHub repository settings to exclusively allow rebase merges (`allow_rebase_merge: true`, `allow_squash_merge: false`, `allow_merge_commit: false`). Preserve strict branch protection requiring linear history (`required_linear_history: true`), strict CI checks (`test-and-lint (3.11)` and `test-and-lint (3.12)`), and prohibition of direct and force pushes. Automate merging using `gh pr merge --auto --rebase --delete-branch`.
- **Consequences**:
  - Genuine, meaningful commits authored on feature branches are preserved in full on `main`, each retaining its individual author timestamp, commit message, and diff history.
  - Linear history is preserved without non-linear merge bubbles or merge commits.
  - Developers must maintain disciplined, reviewable commit hygiene on feature branches (no "fix typo" or "wip" noise; every commit must be coherent, atomic, and pass lint/tests).
  - Squash merging is disabled to prevent accidental squashing of multi-commit engineering tasks.

## ADR-008: Svelte/SvelteKit Frontend with FastAPI Adapter for Diagnostic Studio
- **Status**: Accepted
- **Context**: The original roadmap specified Streamlit for UI-002. However, Streamlit couples UI state to server-side script re-execution, lacks fine-grained reactive DOM control for dense diagnostics, prevents clean separation of API contracts from presentation, and cannot easily support modern component systems or responsive client-side interactions. The project requires a professional, industrial-grade diagnostic studio with custom layout hierarchy, fast client-side sorting/filtering, side-by-side run comparisons, and interactive citation audits.
- **Decision**: Build the frontend as a dedicated Svelte/SvelteKit application (`frontend/`) written in strict TypeScript. Expose the existing Python SQLite experiment repository, regression detection engine, and citation evaluation capabilities via a lightweight, typed FastAPI local backend (`reliability_lab/api/`). Keep `web` dependencies (`fastapi`, `uvicorn`, `httpx`) isolated in an optional dependency group in `pyproject.toml`.
- **Consequences**:
  - Delivers a high-performance, responsive, accessible web interface with full control over design system primitives, reactive state, and failure visualizations.
  - Maintains strict separation of concerns: Python domain logic (`reliability_lab/`) remains the authoritative source of truth for metrics, regression mathematics, and persistence, accessed via clean REST API contracts.
  - The browser never accesses the raw SQLite database directly; all operations pass through validated repository interfaces and safety checks.
  - CLI and library users remain unencumbered by web framework dependencies unless installing `.[web]`.

## ADR-009: Provider-Independent Abstract LLM Client Architecture with Gemini Integration and Deterministic Mock Fallbacks
- **Status**: Accepted
- **Context**: Milestone 9 introduces answer generation, prompt-based evaluation, and comparative LLM-as-a-judge studies (LLM-002). Directly coupling evaluation pipelines to proprietary vendor SDKs introduces API version churn, requires live external credentials in CI, risks flaky test execution due to network latency and upstream rate limits, and incurs unpredictable inference costs. The architecture requires a unified, provider-agnostic interface that behaves identically whether backed by real cloud APIs or local deterministic mocks.
- **Decision**: Define a provider-independent abstract client interface (`BaseLLMClient`) in `reliability_lab/llm/` supporting standardized message structures, generation parameters, latency tracking, and token usage accounting. Implement `MockLLMClient` for 100% offline, deterministic testing with configurable response queues, regex/keyword matchers, and programmable error triggers. Implement `GeminiLLMClient` supporting Gemini Flash models (`gemini-1.5-flash`, `gemini-2.0-flash`, `gemini-2.5-flash`) using resilient HTTP transport via `httpx`, bounded exponential backoff retries on transient 429/503 status codes, token-bucket rate limiting, structured JSON modes, and safe mock fallback when credentials are not configured. Provide a unified `get_llm_client()` factory function.
- **Consequences**:
  - Test suites and CI runners execute hermetically without external network calls, API keys, or cloud costs.
  - Live Gemini generation is supported out of the box when `GEMINI_API_KEY` is provided, with built-in retry and rate limiting protection.
  - Future providers (OpenAI, Anthropic, or local vLLM/Ollama engines) can be added seamlessly by extending `BaseLLMClient`.

## ADR-010: Schema-Validated LLM Judge with Strict Citation Provenance and NLI Directional Alignment
- **Status**: Accepted
- **Context**: LLM-as-a-judge approaches often suffer from uncalibrated verdicts, free-form non-deterministic text responses, susceptibility to prompt injection within retrieved chunks, and hallucinated citation IDs. Furthermore, evaluating whether an LLM judge correlates with deterministic local NLI models requires consistent directionality and robust handling of tied or zero-variance distributions.
- **Decision**: Implement `LLMJudge` (`reliability_lab/llm/judge.py`) utilizing schema-validated JSON completions and deterministic sentence claim segmentation (`extract_claim_spans`). Protect against prompt hijacking by escaping XML delimiters in evidence chunks and explicitly instructing the model to treat evidence as untrusted data. Enforce strict evidence verification by validating cited chunk IDs against provided retrieval results and demoting unsupported claims. Implement `calculate_llm_nli_correlation` aligning LLM unsupportedness risk ($1.0 - \text{faithfulness\_score}$) with NLI unsupportedness risk (`unsupported_risk`), computing Pearson $r$ and Spearman $\rho$ with fractional rank averaging for ties, and returning `None` with actionable warnings for zero variance or $N < 3$.
- **Consequences**:
  - LLM evaluations are hermetic, testable, schema-safe, and resilient against prompt injection.
  - Directional correlation between LLM judges and local cross-encoder NLI models is mathematically sound and transparent without fabricated coefficients.
  - Zero private chain-of-thought traces are stored or logged; only public justifications and structured diagnostics are retained.

## ADR-011: Unified End-to-End RAG Evaluation Pipeline Orchestration with Explicit Provenance and Decoupled Evaluation Stages
- **Status**: Accepted
- **Context**: The repository contains robust modular building blocks: document chunking, sparse/dense/hybrid retrieval, lexical groundedness, citation parsing and validation, local NLI entailment, provider-independent LLM generation, structured LLM judging, and SQLite experiment persistence. However, executing a complete RAG evaluation previously required manual, ad-hoc stitching of these subsystems. This manual invocation risked data contract divergence, accidental leakage of benchmark ground-truth answers into generation prompts, silent failure to track mock versus live LLM inference provenance, and fragmented reporting.
- **Decision**: Implement a unified backend orchestration service (`RAGEvaluationRunner` in `reliability_lab/pipeline/`) coordinating the full evaluation lifecycle:
  1. *Retrieval*: Retrieves top-$K$ chunks using any `BaseRetriever` and computes standard Cranfield ranking metrics (Recall, Precision, Hit Rate, MRR, NDCG).
  2. *Answer Generation*: Generates document-grounded responses using `BaseLLMClient` with prompt injection resilience (escaping evidence delimiters, treating chunks as untrusted data), strict exclusion of benchmark ground-truth answers from prompts, and explicit citation directives.
  3. *Multi-Faceted Evaluation*: Evaluates lexical groundedness (`GroundednessEvaluator`), citation validity and precision/recall (`CitationEvaluator`), local NLI entailment (`NLIEntailmentEvaluator`), and structured factual judging (`LLMJudge`). Preserves the distinct semantics and failure modes of each evaluation dimension without artificial score collapse.
  4. *Statistical Alignment*: Computes directional Pearson $r$ and Spearman $\rho$ correlation between LLM judge unsupportedness risk and local NLI unsupportedness risk when aligned claim pairs exist, preserving existing handling of small samples ($N < 3$) or zero variance.
  5. *Storage & Provenance Integration*: Emits canonical `EvaluationReport` structures persisted transactionally to SQLite via `ExperimentRepository`, retaining granular per-query diagnostics, token accounting, latency percentiles, and prominent mock/live provenance flags (`is_mock`).
  6. *CLI Entrypoint*: Exposes `reliability-lab run-rag-evaluation` with explicit provider configuration and clear warnings whenever mock generation or judging is employed.
- **Consequences**:
  - Provides a single, clean backend entrypoint for complete RAG evaluation without rewriting indexing algorithms or coupling to specific presentation frameworks.
  - Guarantees zero credential leakage and prevents synthetic mock evaluation from masquerading as live cloud inference.
  - Ensures full reproducibility, backward compatibility with existing retrieval-only benchmarks, and transaction-safe experiment persistence.

## ADR-012: Multi-Stage Containerization, Adapter-Node Deployment, and Reproducible Benchmark Topology
- **Status**: Accepted
- **Context**: AI Reliability Lab requires an industrial-grade, reproducible local execution environment (Milestone 11, PUB-001) that enables developers, researchers, and CI systems to run the FastAPI backend adapter, the SvelteKit diagnostics studio, automated test suites, and retrieval/RAG benchmarks without manually configuring Python or Node.js environments. The design must prevent data loss across container lifecycle events, enforce non-root security standards, keep runtime containers lean without heavy neural dependencies, and maintain browser-to-backend communication parity.
- **Decision**:
  1. *Backend Containerization*: Implement a multi-stage root `Dockerfile` using `python:3.11-slim` with distinct `test` (containing dev tooling, ruff, mypy, pytest) and `runtime` stages. The runtime stage runs as an unprivileged user (`appuser`), avoids wheel caching, excludes development dependencies, and launches Uvicorn binding to `0.0.0.0:8000`.
  2. *Frontend Containerization*: Transition frontend build configuration from `@sveltejs/adapter-auto` to `@sveltejs/adapter-node` to produce an optimized, standalone Node server (`build/index.js`). Package in a multi-stage `frontend/Dockerfile` using `node:22-alpine` with `npm ci` for reproducible dependency resolution, non-root execution (`USER node`), and published port `5173`. Configure browser API access via build-time `VITE_API_URL` pointing to the published backend address (`http://127.0.0.1:8000`).
  3. *Compose Topology*: Implement `docker-compose.yml` declaring `api`, `frontend`, `tests` (profile `test`), and `benchmark` (profile `benchmark`) services with healthchecks and dependencies. Persist SQLite experiment storage in a shared named volume (`reliability-lab-data`) mounted at `/data`, setting `RELIABILITY_LAB_DB=/data/experiments.sqlite3`.
  4. *Reproducible Benchmark Script*: Provide `scripts/run_benchmarks.sh` with strict bash error handling (`set -euo pipefail`), automated validation of corpus and dataset paths, mock provenance warnings, and non-destructive execution across retrieval benchmarks and end-to-end RAG evaluations.
- **Consequences**:
  - Entire platform builds and boots in a single command (`docker compose up --build -d`) with zero host dependency requirements.
  - Experiment history is safely preserved across container recreation and restarts via named Docker volumes.
  - Test suites and benchmarks can be executed reproducibly in isolated container environments.
  - Production images are secure, minimal, and non-root without embedding secrets or heavy unneeded dependencies.

## ADR-013: Local LLM Inference Provider Integration via Ollama HTTP API
- **Status**: Accepted
- **Context**: AI Reliability Lab v2 introduces offline Document Intelligence, enabling users to upload documents, ask questions, and verify citation provenance without sending private data to external cloud APIs. Relying solely on cloud LLM providers (e.g., Google Gemini) introduces vendor lock-in, data privacy concerns for sensitive documents, recurring API billing, rate limiting, and network availability constraints. The platform requires a local, privacy-preserving, zero-cost, offline LLM inference provider integrated seamlessly into the existing provider-agnostic `BaseLLMClient` architecture.
- **Alternatives Considered**:
  - *Direct `llama-cpp-python` Bindings*: Provides direct in-process C++ inference, but introduces heavy platform-specific compilation toolchains, wheel compatibility issues across OS environments (Windows/macOS/Linux), and fragile CUDA runtime dependencies.
  - *Local `vLLM` Serving*: Highly optimized for production GPU batch inference, but heavy memory footprint and lack of native first-class Windows support makes it unsuitable for standard local developer workstations.
  - *HuggingFace `transformers` In-Process*: Incurs massive Python memory overhead, requires manual model weight caching, and couples inference runtime to PyTorch environment dependencies.
  - *Ollama HTTP REST API*: Independent, platform-native local daemon with precompiled binaries for macOS, Linux, and Windows; manages model weight downloads and quantization formats (GGUF); provides hardware-accelerated local inference; exposes a clean, standardized HTTP REST API.
- **Decision**:
  1. *Native REST Transport via `httpx`*: Implement `OllamaLLMClient` in `reliability_lab/llm/client.py` communicating directly with Ollama's local HTTP API (`POST /api/chat` with `stream: false`) using the existing `httpx` library already integrated in the project. Do not introduce an external Ollama Python SDK.
  2. *Standardized Message and Parameter Translation*: Map `LLMMessage` role definitions (`system`, `user`, `assistant`), translate generation parameters (`temperature`, `max_output_tokens` -> `num_predict`, `top_p`, `top_k`, `stop`), and support schema-constrained JSON extraction via Ollama's native `"format": "json"` payload attribute.
  3. *Accurate Token and Latency Accounting*: Capture Ollama's reported evaluation statistics (`prompt_eval_count` -> `prompt_tokens`, `eval_count` -> `completion_tokens`) and track execution latency.
  4. *Explicit Provenance Distinction*: Designate `provider="ollama"` and `is_mock=False` in `GenerationProvenance`, SQLite database persistence, and CLI banners, ensuring local real AI inference is distinctly identifiable from deterministic synthetic mocks (`is_mock=True`) and cloud APIs (`provider="gemini"`).
  5. *Fail-Fast, Actionable Error Handling*: Raise dedicated `LLMProviderUnavailableError` (providing exact `ollama serve` commands upon connection failure) and `LLMInvalidRequestError` (providing `ollama pull <model>` instructions upon 404 model errors) without silent, misleading fallback to `MockLLMClient`.
  6. *Configurable Defaults*: Configure endpoint and model resolution through `OLLAMA_BASE_URL` (defaulting to `http://localhost:11434`) and `OLLAMA_MODEL` (defaulting to `llama3.2`), ensuring provider-specific model names are cleanly decoupled from Gemini defaults.
- **Consequences**:
  - Enables true 100% offline, privacy-preserving local RAG evaluation and document intelligence.
  - Test suites and CI runners execute hermetically without an active Ollama daemon using `httpx.MockTransport`.
  - Production and local developer environments remain lightweight with zero additional external package dependencies.





