# Northstar Retrieval Evaluation & Benchmark Report

Document ID: NORTHSTAR-BENCH-2026-Q3
Version: 1.1.0
Publication Date: 2026-09-01
Author: Northstar Evaluation & Reliability Team

## 1. Evaluation Scope & Methodology

This report details the empirical retrieval performance of Northstar v2.2 across a canonical evaluation dataset of 40 technical queries against a 200-document knowledge repository. All experiments were conducted under uniform CPU conditions with top-K set to 5.

## 2. Quantitative Retrieval Metrics

Experiments compared standalone Okapi BM25, dense FAISS vectors, and Hybrid Reciprocal Rank Fusion (k=60):

| Metric | Okapi BM25 | Dense FAISS | Hybrid RRF (k=60) | Notes |
|---|---|---|---|---|
| **Recall@5** | 0.8250 | 0.9125 | **0.9625** | Hybrid achieves highest total coverage |
| **Precision@5** | 0.3600 | 0.4100 | **0.4350** | Multi-source agreement prioritizes top hits |
| **Hit Rate@5** | 0.9500 | 0.9750 | **1.0000** | At least one relevant chunk in top-5 for all queries |
| **MRR@5** | 0.8875 | 0.9250 | **0.9500** | First relevant hit consistently placed at rank 1 or 2 |
| **NDCG@5** | 0.8640 | 0.9180 | **0.9420** | Significant ranking quality improvement with RRF |
| **P50 Latency** | **0.12 ms** | 32.50 ms | 33.10 ms | BM25 is ~270x faster than neural vector inference |

## 3. Findings & Primary Recommendations

1. **Hybrid Retrieval Superiority**: Across general technical questions and paraphrased inquiries, Hybrid RRF achieved superior Recall@5 (0.9625) and NDCG@5 (0.9420), outperforming both isolated sparse and dense baselines.
2. **Dense Vector Strength in Conceptual Queries**: Dense vector retrieval consistently recovered conceptually related chunks where lexical overlap was zero.
3. **Latency Profile**: While BM25 demonstrated sub-millisecond execution (P50: 0.12 ms), the end-to-end latency of Hybrid RRF (33.10 ms) remained well within the interactive SLA budget of 50 ms.

## 4. Evaluation Limitations & Validity Constraints

All measurements in this report reflect natural language queries formulated by technical domain specialists. Alphanumeric SKU lookups and single-token error code searches were excluded from the 40-query evaluation set, representing a known coverage limitation.

