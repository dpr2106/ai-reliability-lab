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
| GROUND-002 Baseline (PR #31) | `d7e5ac3140dd62e76f0f9b566993be122915813a` | 48 | 300 | 252 | Verified baseline for GROUND-002 following PR #31 governance finalization |
| GROUND-002 Rebase Merge (PR #32) | `edf5cc598f6e71ef9ab7f7939eb278e9f2054f58` | 51 | 300 | 249 | Rebase-merge verified: 3 commits preserved individually on main |
| GROUND-003 Baseline (PR #33) | `25d115288eaedc232ec345ffa66c5b8d1977421e` | 52 | 300 | 248 | Verified baseline for GROUND-003 following PR #33 governance finalization |
| GROUND-003 Rebase Merge (PR #34) | `8abb88963692e9e975c95aec8256117d5506ba23` | 55 | 300 | 245 | Rebase-merge verified: 3 commits preserved individually on main |
| EXP-001 Baseline (PR #35) | `803779ca20c7949daa0a2b25e2433c941d544325` | 56 | 300 | 244 | Verified baseline for EXP-001 following PR #35 governance finalization |
| EXP-001 Rebase Merge (PR #36) | `49aee42ad4f992f5d09d3da2eecc05173919acb2` | 59 | 300 | 241 | Rebase-merge verified: 3 commits preserved individually on main |
| EXP-002 Baseline (PR #37) | `e2f6216b9ce5fd64e3839c3ecd078b00ed3f9b8c` | 60 | 300 | 240 | Verified baseline for EXP-002 following PR #37 governance finalization |
| EXP-002 Rebase Merge (PR #38) | `536d7c8276c8a59a699ba92debdc97d00980fef8` | 63 | 300 | 237 | Rebase-merge verified: 3 commits preserved individually on main |
| UI-001 Baseline (PR #39) | `8f4921ba2a1472c31df4c1079c66234a7490cb9a` | 64 | 300 | 236 | Verified baseline for UI-001 following PR #39 governance finalization |
| UI-001 Rebase Merge (PR #40) | `d8c970a71fa59d3cf53cefcc1340064c28989770` | 67 | 300 | 233 | Rebase-merge verified: 3 commits preserved individually on main |
| UI-002 Baseline (PR #41) | `42f4921b853f7755b96e2cef52343488128ae619` | 68 | 300 | 232 | Verified baseline for UI-002 following PR #41 governance finalization |
| UI-002 Rebase Merge (PR #42) | `f770e2baffb6aa8cc61805cb6c9849b20f635a72` | 73 | 300 | 227 | Rebase-merge verified: 5 commits preserved individually on main |
| LLM-001 Baseline (PR #43) | `52fc753c3fd20d0c6ec00975ec94d908ae777e9b` | 74 | 300 | 226 | Verified baseline for LLM-001 following PR #43 governance finalization |
| LLM-001 Rebase Merge (PR #44) | `0d0f88a0104bf42130f040bef8071ff92bc693d9` | 79 | 300 | 221 | Rebase-merge verified: 5 commits preserved individually on main |
| LLM-002 Baseline (PR #45) | `b4f16ef96a3c30bd1ec0484de5ea790b937e67c9` | 80 | 300 | 220 | Verified baseline for LLM-002 following PR #45 governance finalization |
| LLM-002 Rebase Merge (PR #46) | `9449e84c31452ce4e02b0f0cabd800eb3e6f54ac` | 83 | 300 | 217 | Rebase-merge verified: 3 commits preserved individually on main |

---

## Commit Distinction & Hygiene Policy

1. **Feature Branch Commits vs. Main Commits**:
   - Commits authored on feature branches only count toward the 300-commit requirement once they are successfully rebase-merged and reachable from `origin/main`.
   - Feature branches must not contain low-quality, temporary, or noise commits (e.g. "fix typo", "wip", "update").
   - Every commit on a feature branch must be meaningful, atomic, follow Conventional Commits, and pass linting and tests independently.

2. **No Commit Manipulation**:
   - Commits must represent genuine, reviewable engineering work: tests, domain models, algorithms, documentation, diagnostics, and refactors.
   - Artificial commits, whitespace churn, or fake feature commits solely to inflate the commit count are strictly prohibited.
