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
