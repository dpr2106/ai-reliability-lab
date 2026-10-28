# Main Branch Commit Progress: AI Reliability Lab

## Purpose & Long-Term Requirement
AI Reliability Lab requires a rich, granular, reviewable git history of at least **300 meaningful engineering commits** reachable from the default branch (`main`).

Prior to GOV-006, all pull requests were squash-merged (ADR-006). While this kept `main` linear and simple, it collapsed each feature branch into a single commit upon merge, erasing distinct engineering steps and making the 300-commit milestone artificial or unachievable without hundreds of micro-PRs.

ADR-007 transitions the repository to a **rebase-only merge policy** (`allow_rebase_merge: true`, `allow_squash_merge: false`, `allow_merge_commit: false`) under strict linear history enforcement. Under this model, genuine, reviewable commits authored on feature branches are preserved individually on `main`.

---

## Canonical Measurement

### Measurement Command
```bash
git rev-list --count origin/main
```

### Commit Progress Ledger
| Stage / Milestone | Baseline Commit SHA | Main Branch Commit Count | Target | Delta Remaining | Notes |
|---|---|---|---|---|---|
| Initial Audit (pre-GOV-006) | `41119a073f4e857ddca76fa78e622b724490f055` | 19 | 300 | 281 | Verified starting count following PR #18 |
| Governance Scaffolding (PR #19) | `9de7e1951f28b499e74d1a4983fb08d98d633c77` | 20 | 300 | 280 | PR #19 merged under final squash policy |
| Rebase Verification (PR #20) | `4b192ad219d36bb66e01a884efb88950d859424d` | 22 | 300 | 278 | Rebase-merge verified: 2 commits preserved individually on main |

---

## Commit Distinction & Hygiene Policy

1. **Feature Branch Commits vs. Main Commits**:
   - Commits authored on feature branches only count toward the 300-commit requirement once they are successfully rebase-merged and reachable from `origin/main`.
   - Feature branches must not contain low-quality, temporary, or noise commits (e.g. "fix typo", "wip", "update").
   - Every commit on a feature branch must be meaningful, atomic, follow Conventional Commits, and pass linting and tests independently.

2. **No Commit Manipulation**:
   - Commits must represent genuine, reviewable engineering work: tests, domain models, algorithms, documentation, diagnostics, and refactors.
   - Artificial commits, whitespace churn, or fake feature commits solely to inflate the commit count are strictly prohibited.
