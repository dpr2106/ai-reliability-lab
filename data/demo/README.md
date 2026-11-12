# STING Demo Corpus: Northstar Architecture, Benchmark & Incident Suite

## 1. Overview & Provenance

This directory contains the canonical, redistributable demonstration dataset for **STING** (Evidence-First RAG Reliability Workbench). 

All documents in this collection are **original, self-authored fictional technical documents** created specifically to demonstrate STING's evidence-first retrieval, cross-document investigation, citation verification, and evaluation studio workflows.

- **Fictional Entity**: "Northstar", a distributed document retrieval and evidence verification system.
- **Copyright & License**: Open source under Apache License 2.0 (see [LICENSE](LICENSE)). Completely free of third-party copyrighted material, proprietary corporate data, or private personal information.
- **Scientific Status**: Fictional engineering scenario designed for algorithmic demonstration and software evaluation; not real scientific literature.

## 2. Corpus Composition

The corpus consists of three coherent, complementary technical Markdown documents in `corpus/`:

| Document Filename | Document ID | Purpose & Role | Key Topics Covered |
|---|---|---|---|
| `architecture.md` | `NORTHSTAR-ARCH-2026-V2` | Baseline system architecture & design | 500-char chunking, Okapi BM25, FAISS 384-d `all-MiniLM-L6-v2`, Hybrid RRF (k=60), inline citation rules |
| `benchmark_report.md` | `NORTHSTAR-BENCH-2026-Q3` | Controlled IR benchmarking & metrics | 40-query benchmark, Recall@5 (0.9625), NDCG@5 (0.9420), P50 latency, limitation excluding alphanumeric SKUs |
| `incident_report.md` | `NORTHSTAR-INC-4091` | Post-mortem quality incident analysis | Incident INC-4091, RRF rank dilution on alphanumeric SKUs, contradiction with Q3 report, corrective actions |

## 3. Thematic Design & Cross-Document Dynamics

These documents were purposefully authored to provide realistic multi-document relationships:
1. **Complementary Agreement**: `architecture.md` specifies the RRF k=60 topology, and `benchmark_report.md` measures its quantitative advantage over isolated BM25 and FAISS baselines.
2. **Contextual Disagreement / Nuance**: While `benchmark_report.md` recommends Hybrid RRF as universally superior for general technical inquiries, `incident_report.md` reveals that Hybrid RRF suffered severe rank dilution on exact alphanumeric model identifiers (`NS-9000-XL`), pushing exact BM25 matches out of top-5.
3. **Traceable Methodology**: Both `benchmark_report.md` and `incident_report.md` document explicit experimental constraints and limitations that explain the divergence.
4. **Actionable Remediation**: `incident_report.md` establishes concrete corrective actions (lexical exact-match promotion, dense score thresholding, and safe abstention enforcement).
5. **Abstention Target**: Queries regarding unmentioned topics (e.g. quantum annealing, RLHF training, or graph neural networks) cleanly trigger STING's safe abstention.

