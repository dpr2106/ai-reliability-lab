# Project Status: AI Reliability Lab

## Verified Operational State
- **Base Branch**: `main`
- **Active Milestone**: Milestone 11 — Reproducibility, Packaging & Portfolio Presentation (COMPLETE)
- **Test Suite Baseline**: 648 passed in 85.10s, 93% coverage (historical baseline from Milestone 11 / PUB-002 release audit)
- **V2 Test Suite Total**: 1,028 passed unit tests, 0 failures, 6 warnings (verified on main after V2-005 PR #65 merge)

## PR Review & Gating Architecture State
- **Branch Protection**: `main` requires strict checks `test-and-lint (3.11)` and `test-and-lint (3.12)`; direct and force pushes are prohibited, and linear history is enforced (ADR-007).
- **Merge Strategy**: Native auto-merge is enabled; repository settings enforce rebase merges (ADR-007) and disallow merge commits and squash merges.
- **Code Review Status**: CI checks are the required merge gates. An empty webhooks API response does not establish that no GitHub Apps are installed. Copilot could not review PR #3 because its review quota was exhausted. An absent or unperformed review must not be reported as passed.

## Task Pointers
- **Next READY Task**: `NONE`

Task IDs and statuses are defined in `docs/roadmap/TASKS.md`. All 11 roadmap milestones and tasks are verified as DONE. This snapshot intentionally excludes a purported latest merge SHA, which would become stale and could refer to the commit containing this file.

## V2 Product Status

- **Product direction**: STING — evidence-first, local-first RAG reliability workbench built on AI Reliability Lab.
- **V2-001**: Complete — Ollama local inference provider (PR #57 merged).
- **V2-002**: Complete — Local document upload and indexing pipeline (PR #58, #59 merged).
- **V2-003**: Complete — Evidence-First Query and Trace Orchestrator, Persistent Sessions & Turn History, and True LLM Streaming via SSE (PR #61, #62, #63 merged).
- **V2-004**: Complete — Cross-Document Evidence Intelligence Workbench (PR #64 merged). Multi-document evidence extraction, thematic comparison matrix, source passage inspector, and markdown report export.
- **V2-005**: Complete — RAG Evaluation, Comparison and Regression Studio (PR #65 merged). Golden benchmark dataset manager, fail-closed corpus validation, background experiment runner with incremental progress, diagnostic query failure classification, baseline regression comparison, and multi-tier run reproducibility manifests.
- **V2-006**: READY — Reproducible Demo and Portfolio Release. Sample redistributable dataset, clean-checkout setup, reproducible execution walkthroughs, demo assets, and portfolio evidence.
- **Product differentiator**: Cross-document evidence investigation, retrieval traceability, citation provenance, claim-level evidence inspection, safe abstention, and reproducible RAG evaluation—not generic local chat.
- **Portfolio rule**: Describe only capabilities that are merged, tested, and demonstrated. Keep benchmark dataset sizes and metric methodology visible.
