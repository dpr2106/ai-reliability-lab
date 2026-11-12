# Northstar Incident Post-Mortem: Retrieval Regression on Alphanumeric Codes

Document ID: NORTHSTAR-INC-4091
Version: 1.0.0
Date of Incident: 2026-09-20
Report Date: 2026-09-22
Author: Site Reliability & Search Infrastructure Team

## 1. Incident Overview

On September 20, 2026, production telemetry revealed a severe retrieval failure rate on technical support queries containing exact product SKUs and model numbers (e.g., `NS-9000-XL` and `ERR-404-BUS`). Users reported that the system hallucinated specifications or returned completely irrelevant hardware manuals.

## 2. Failure Analysis & Root Cause

Investigation revealed an architectural failure in the Reciprocal Rank Fusion (RRF) pipeline:
1. **Lexical Retrieval Performance**: Okapi BM25 successfully identified the exact target document at rank 1 for 100% of the affected queries.
2. **Dense Vector Failure**: The embedding model `all-MiniLM-L6-v2` split unhyphenated alphanumeric codes into fragmented subwords, yielding diffuse cosine similarity scores across hundreds of unrelated documents.
3. **RRF Rank Dilution**: Because dense retrieval populated its top ranks with irrelevant documents that clustered tightly, the combined RRF formula diluted the single high-confidence BM25 hit, pushing the correct chunk down to rank 7 or 8 (outside top-K=5).

## 3. Discrepancy with Q3 Benchmark Report

The Q3 Benchmark Report (NORTHSTAR-BENCH-2026-Q3) reported a near-perfect Hit Rate@5 of 1.0000 for Hybrid RRF. However, as noted in its limitations section, that benchmark excluded exact alphanumeric SKUs and single-token identifiers. The incident demonstrated that general benchmark dominance does not guarantee reliability across specialized query distributions.

## 4. Corrective Actions & Architectural Remediation

To remediate this failure mode in Northstar v2.3, the team approved two mandatory architectural changes:
1. **Lexical Exact-Match Promotion**: When a query contains an exact alphanumeric token matching indexed hardware codes, the BM25 top hit is automatically pinned into the top-3 candidate evidence set.
2. **Dense Confidence Thresholding**: Dense vector candidates with cosine similarity scores below 0.45 are disqualified from contributing to RRF aggregation.
3. **Safe Abstention Enforcement**: When neither lexical exact match nor dense similarity satisfies minimum confidence bounds, the system must trigger safe abstention rather than passing low-confidence candidates to the generator.

