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
| EVAL-003 Baseline (PR #21) | `c9cb14aa8074a66aa39879a761cb87ce530b4afc` | 23 | 300 | 277 | Verified baseline for EVAL-003 after PR #21 rebase-merge |
| EVAL-003 Rebase Merge (PR #22) | `15c3cc53fc027aac5d2dc2babd6eb676500965ce` | 28 | 300 | 272 | Rebase-merge verified: 5 commits preserved individually on main |
| DENSE-001 Baseline (PR #23) | `59e0c3bd3453303ca144e7e422e04eeb869de8a0` | 29 | 300 | 271 | Verified baseline for DENSE-001 following PR #23 governance finalization |
| DENSE-001 Rebase Merge (PR #24) | `181cfb49afaa6c05bb10b96638233515c355bf1a` | 33 | 300 | 267 | Rebase-merge verified: 4 commits preserved individually on main |
| DENSE-002 Baseline (PR #25) | `83dc5a8e3d5d658c18495734ca0d06622306b63a` | 34 | 300 | 266 | Verified baseline for DENSE-002 following PR #25 governance finalization |
| DENSE-002 Rebase Merge (PR #26) | `602d3b76d3222f98a228ec0486098977c5676638` | 38 | 300 | 262 | Rebase-merge verified: 4 commits preserved individually on main |
| DENSE-003 Baseline (PR #27) | `7cbaa15b626c4be806b483f2dddbcaebd8a1056e` | 39 | 300 | 261 | Verified baseline for DENSE-003 following PR #27 governance finalization |
| DENSE-003 Rebase Merge (PR #28) | `5ef45895a8a460cccbd0c6653625b2b98124f779` | 42 | 300 | 258 | Rebase-merge verified: 3 commits preserved individually on main |
| GROUND-001 Baseline (PR #29) | `fb142aecdf49e38f7d47f74bb072d9b992f1d38d` | 43 | 300 | 257 | Verified baseline for GROUND-001 following PR #29 governance finalization |
| GROUND-001 Rebase Merge (PR #30) | `a5d35e8b7097c7bbaf07271e8ade36b073fa05ad` | 47 | 300 | 253 | Rebase-merge verified: 4 commits preserved individually on main |

---

## Commit Distinction & Hygiene Policy

1. **Feature Branch Commits vs. Main Commits**:
   - Commits authored on feature branches only count toward the 300-commit requirement once they are successfully rebase-merged and reachable from `origin/main`.
   - Feature branches must not contain low-quality, temporary, or noise commits (e.g. "fix typo", "wip", "update").
   - Every commit on a feature branch must be meaningful, atomic, follow Conventional Commits, and pass linting and tests independently.

2. **No Commit Manipulation**:
   - Commits must represent genuine, reviewable engineering work: tests, domain models, algorithms, documentation, diagnostics, and refactors.
   - Artificial commits, whitespace churn, or fake feature commits solely to inflate the commit count are strictly prohibited.
