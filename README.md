# AI Reliability Lab

Industrial-grade reliability, evaluation, and regression testing platform for document-based AI and RAG architectures.

## Overview
AI Reliability Lab evaluates retrieval quality, verifies groundedness against source documents, validates citations, and detects regressions across model and configuration changes.

## Engineering Standards
- **Baseline-First IR**: Exact lexical baselines (BM25) before vector and hybrid approaches.
- **Deterministic Metrics**: Hand-calculated Cranfield ranking metrics (Recall@K, MRR@K, NDCG@K) and claim-level verification.
- **Protected CI/CD & Auto-Merge**: Automated regression gating on every pull request.
- **Modular Architecture**: Decoupled domain models, parsers, retrievers, metrics, and persistence layers.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/dpr2106/ai-reliability-lab.git
cd ai-reliability-lab

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate

# Install development dependencies
pip install -e .[dev]

# Run static checks and tests
ruff check .
mypy reliability_lab
pytest
```
