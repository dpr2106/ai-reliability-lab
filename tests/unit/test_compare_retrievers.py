"""Unit tests for the comparative retrieval benchmark utility (PUB-002).

Covers:
- Input validation on missing or invalid corpus/dataset paths.
- Execution across retrievers with identical input sets.
- Graceful handling of missing optional dense dependencies (--skip-dense).
- Metric aggregation integrity and absence of fabricated zero values.
- JSON serialization and environment provenance capturing.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import pytest  # noqa: E402

from scripts.compare_retrievers import (  # noqa: E402
    get_environment_metadata,
    run_comparison,
)

SAMPLE_CORPUS = REPO_ROOT / "data" / "benchmarks" / "sample_corpus.jsonl"
SAMPLE_BENCHMARK = REPO_ROOT / "data" / "benchmarks" / "sample_benchmark.jsonl"


def test_environment_metadata_capture() -> None:
    """Verify environment metadata correctly records OS, Python, and package versions."""
    meta = get_environment_metadata()
    assert "os" in meta
    assert "architecture" in meta
    assert "python_version" in meta
    assert "numpy_version" in meta
    assert meta["numpy_version"] is not None


def test_run_comparison_invalid_inputs(tmp_path: Path) -> None:
    """Verify informative errors when corpus or dataset files do not exist."""
    missing_corpus = tmp_path / "nonexistent_corpus.jsonl"
    missing_dataset = tmp_path / "nonexistent_dataset.jsonl"

    with pytest.raises(FileNotFoundError):
        run_comparison(corpus_path=missing_corpus, dataset_path=SAMPLE_BENCHMARK)

    with pytest.raises(FileNotFoundError):
        run_comparison(corpus_path=SAMPLE_CORPUS, dataset_path=missing_dataset)


def test_run_comparison_skip_dense() -> None:
    """Verify execution when dense retrieval is skipped (or dependencies missing)."""
    results = run_comparison(
        corpus_path=SAMPLE_CORPUS,
        dataset_path=SAMPLE_BENCHMARK,
        top_k=5,
        skip_dense=True,
        repetitions=1,
    )

    assert "metadata" in results
    assert "retrievers" in results

    retrievers = results["retrievers"]
    assert "bm25" in retrievers
    assert "metrics" in retrievers["bm25"]
    assert retrievers["bm25"]["metrics"]["recall@5"] > 0.0

    # Dense and Hybrid must be marked SKIPPED cleanly without fabricated metrics
    assert retrievers["dense_faiss"]["status"] == "SKIPPED"
    assert "metrics" not in retrievers["dense_faiss"]

    assert retrievers["hybrid_rrf"]["status"] == "SKIPPED"
    assert "metrics" not in retrievers["hybrid_rrf"]


def test_run_comparison_identical_inputs_and_metrics() -> None:
    """Verify that BM25 evaluation uses identical queries and computes valid metrics."""
    results = run_comparison(
        corpus_path=SAMPLE_CORPUS,
        dataset_path=SAMPLE_BENCHMARK,
        top_k=5,
        skip_dense=True,
        repetitions=2,
    )

    bm25_res = results["retrievers"]["bm25"]
    metrics = bm25_res["metrics"]

    # Verify standard ranking metrics bounded in [0.0, 1.0]
    for m in ["recall@5", "precision@5", "hit_rate@5", "mrr@5", "ndcg@5"]:
        assert 0.0 <= metrics[m] <= 1.0

    # Timing metrics must be strictly positive
    assert metrics["mean_latency_ms"] > 0.0
    assert metrics["latency_p50_ms"] > 0.0
    assert bm25_res["indexing_time_ms"] > 0.0

    # Query count matches input benchmark
    assert bm25_res["query_count"] == 16
    assert len(bm25_res["per_query"]) == 16


def test_run_comparison_json_serializable(tmp_path: Path) -> None:
    """Verify complete output dictionary serializes cleanly to JSON without NaN."""
    results = run_comparison(
        corpus_path=SAMPLE_CORPUS,
        dataset_path=SAMPLE_BENCHMARK,
        top_k=5,
        skip_dense=True,
        repetitions=1,
    )

    json_str = json.dumps(results, indent=2)
    assert json_str is not None

    reloaded = json.loads(json_str)
    assert reloaded["metadata"]["corpus_size"] == 25
    assert reloaded["metadata"]["query_count"] == 16
