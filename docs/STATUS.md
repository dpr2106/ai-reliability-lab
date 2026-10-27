# Project Status: AI Reliability Lab

## Verified Operational State
- **Base Branch**: `main`
- **Active Milestone**: Milestone 0 — Foundations, Governance & Test Infrastructure
- **Test Suite Baseline**: 84 passed in 0.89s, 93% coverage (verified baseline from ING-001)

## PR Review & Gating Architecture State
- **Branch Protection**: `main` requires strict checks `test-and-lint (3.11)` and `test-and-lint (3.12)`; direct and force pushes are prohibited, and linear history is enforced (ADR-006).
- **Merge Strategy**: Native auto-merge is enabled; repository settings allow squash merges and disallow merge commits and rebase merges.
- **Code Review Status**: CI checks are the required merge gates. An empty webhooks API response does not establish that no GitHub Apps are installed. Copilot could not review PR #3 because its review quota was exhausted. An absent or unperformed review must not be reported as passed.

## Task Pointers
- **Active Task**: `ING-001`
- **Next READY Task**: `DENSE-001`

Task IDs and statuses are defined in `docs/roadmap/TASKS.md`. This snapshot intentionally excludes a purported latest merge SHA, which would become stale and could refer to the commit containing this file.
