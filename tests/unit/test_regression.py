"""Unit tests for regression detection engine, metric diffing, and comparability checks."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest

from reliability_lab.core.models import EvaluationReport
from reliability_lab.storage import (
    DEFAULT_TOLERANCE,
    ExperimentRepository,
    IncomparableRunsError,
    InvalidToleranceError,
    RegressionDetector,
    RegressionResult,
    RegressionStatus,
    is_ranking_metric,
    normalize_metric_name,
)


def _make_report(
    report_id: str,
    experiment_name: str = "exp",
    metrics: dict[str, float] | None = None,
    per_query_results: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    timestamp: str = "2026-10-28T12:00:00+00:00",
) -> EvaluationReport:
    """Helper to construct an EvaluationReport for testing."""
    return EvaluationReport(
        experiment_name=experiment_name,
        metrics=metrics if metrics is not None else {"recall@5": 0.80, "mrr@5": 0.70},
        report_id=report_id,
        per_query_results=per_query_results if per_query_results is not None else [],
        timestamp=timestamp,
        config=config if config is not None else {"k1": 1.5},
    )


@pytest.fixture
def repo() -> Generator[ExperimentRepository, None, None]:
    """Yield a transient in-memory ExperimentRepository with guaranteed cleanup."""
    repository = ExperimentRepository(":memory:")
    try:
        yield repository
    finally:
        repository.close()


@pytest.fixture
def detector(repo: ExperimentRepository) -> RegressionDetector:
    """Yield a RegressionDetector instance."""
    return RegressionDetector(repo)


# ===========================================================================
# 1. Metric Normalization & Helper Tests
# ===========================================================================


def test_is_ranking_metric() -> None:
    """Verify recognition of canonical retrieval ranking metrics and aliases."""
    assert is_ranking_metric("recall@5") is True
    assert is_ranking_metric("MRR@10") is True
    assert is_ranking_metric("ndcg@5") is True
    assert is_ranking_metric("precision@1") is True
    assert is_ranking_metric("hit_rate@5") is True
    assert is_ranking_metric("map@10") is True
    assert is_ranking_metric("mean_recall@5") is True
    assert is_ranking_metric("mean_mrr@10") is True

    # Non-ranking metrics
    assert is_ranking_metric("latency_ms") is False
    assert is_ranking_metric("p95_latency_ms") is False
    assert is_ranking_metric("zero_hit_rate") is False
    assert is_ranking_metric("total_queries") is False
    assert is_ranking_metric("hallucination_score") is False


def test_normalize_metric_name() -> None:
    """Verify stripping of aggregate 'mean_' prefix while preserving cutoff."""
    assert normalize_metric_name("recall@5") == "recall@5"
    assert normalize_metric_name("mean_recall@5") == "recall@5"
    assert normalize_metric_name("  Mean_MRR@10 ") == "mrr@10"
    assert normalize_metric_name("p95_latency_ms") == "p95_latency_ms"


# ===========================================================================
# 2. Basic Comparison, Signed Deltas & Mathematical Direction
# ===========================================================================


def test_compare_runs_clean_pass(repo: ExperimentRepository, detector: RegressionDetector) -> None:
    """Verify clean PASS when candidate matches or improves baseline within tolerance.

    Baseline: recall@5 = 0.80, mrr@5 = 0.70
    Candidate: recall@5 = 0.81 (+0.01), mrr@5 = 0.70 (0.00)
    Both degradation magnitudes <= 0.02.
    """
    b_report = _make_report("base-1", metrics={"recall@5": 0.80, "mrr@5": 0.70})
    c_report = _make_report("cand-1", metrics={"recall@5": 0.81, "mrr@5": 0.70})
    repo.save_report(b_report)
    repo.save_report(c_report)

    result = detector.compare_runs("base-1", "cand-1")

    assert isinstance(result, RegressionResult)
    assert result.status == RegressionStatus.PASS
    assert result.has_regression is False
    assert result.is_complete is True
    assert len(result.regressed_aggregate_metrics) == 0

    # Check signed deltas and degradation
    r_delta = result.aggregate_deltas["recall@5"]
    assert r_delta.baseline_score == pytest.approx(0.80)
    assert r_delta.candidate_score == pytest.approx(0.81)
    assert r_delta.delta == pytest.approx(0.01)  # cand - base
    assert r_delta.degradation == pytest.approx(-0.01)  # base - cand
    assert r_delta.is_regressed is False

    m_delta = result.aggregate_deltas["mrr@5"]
    assert m_delta.delta == pytest.approx(0.0)
    assert m_delta.degradation == pytest.approx(0.0)
    assert m_delta.is_regressed is False


def test_compare_runs_aggregate_regression(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify REGRESSION when candidate aggregate metric drops by more than tolerance.

    Baseline: mrr@5 = 0.80
    Candidate: mrr@5 = 0.75
    Degradation = 0.80 - 0.75 = 0.05 > default tolerance 0.02.
    """
    b_report = _make_report("base-2", metrics={"mrr@5": 0.80})
    c_report = _make_report("cand-2", metrics={"mrr@5": 0.75})
    repo.save_report(b_report)
    repo.save_report(c_report)

    result = detector.compare_runs("base-2", "cand-2")

    assert result.status == RegressionStatus.REGRESSION
    assert result.has_regression is True
    assert result.regressed_aggregate_metrics == ["mrr@5"]

    m_delta = result.aggregate_deltas["mrr@5"]
    assert m_delta.delta == pytest.approx(-0.05)
    assert m_delta.degradation == pytest.approx(0.05)
    assert m_delta.tolerance == pytest.approx(DEFAULT_TOLERANCE)
    assert m_delta.is_regressed is True


def test_strictly_exceeding_tolerance_vs_equality(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify degradation exactly equal to tolerance is NOT reported as a regression.

    Tolerance = 0.02.
    Case A: degradation = 0.020000 -> is_regressed is False (PASS).
    Case B: degradation = 0.020001 -> is_regressed is True (REGRESSION).
    """
    # Case A: exactly equal
    r_base_a = _make_report("base-eq", metrics={"recall@5": 0.80})
    r_cand_a = _make_report("cand-eq", metrics={"recall@5": 0.78})
    repo.save_report(r_base_a)
    repo.save_report(r_cand_a)

    res_eq = detector.compare_runs("base-eq", "cand-eq", tolerance=0.02)
    assert res_eq.aggregate_deltas["recall@5"].degradation == pytest.approx(0.02)
    assert res_eq.aggregate_deltas["recall@5"].is_regressed is False
    assert res_eq.status == RegressionStatus.PASS

    # Case B: strictly exceeding by 1e-4
    r_base_b = _make_report("base-exceed", metrics={"recall@5": 0.80})
    r_cand_b = _make_report("cand-exceed", metrics={"recall@5": 0.7799})
    repo.save_report(r_base_b)
    repo.save_report(r_cand_b)

    res_exceed = detector.compare_runs("base-exceed", "cand-exceed", tolerance=0.02)
    assert res_exceed.aggregate_deltas["recall@5"].degradation > 0.02
    assert res_exceed.aggregate_deltas["recall@5"].is_regressed is True
    assert res_exceed.status == RegressionStatus.REGRESSION


def test_per_metric_tolerance_overrides(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify custom per-metric tolerance overrides take precedence over default tolerance.

    Default tolerance = 0.01.
    Override: ndcg@5 = 0.05.
    Degradation in ndcg@5 = 0.03.
    Under default (0.01), it would regress; under override (0.05), it passes.
    """
    b_report = _make_report("base-tol", metrics={"ndcg@5": 0.80, "recall@5": 0.80})
    c_report = _make_report("cand-tol", metrics={"ndcg@5": 0.77, "recall@5": 0.80})
    repo.save_report(b_report)
    repo.save_report(c_report)

    # Without override -> 0.03 > 0.01 -> REGRESSION
    res_default = detector.compare_runs("base-tol", "cand-tol", tolerance=0.01)
    assert res_default.status == RegressionStatus.REGRESSION

    # With override -> 0.03 <= 0.05 -> PASS
    res_override = detector.compare_runs(
        "base-tol",
        "cand-tol",
        tolerance=0.01,
        metric_tolerances={"ndcg@5": 0.05},
    )
    assert res_override.status == RegressionStatus.PASS
    assert res_override.aggregate_deltas["ndcg@5"].tolerance == pytest.approx(0.05)


# ===========================================================================
# 3. Per-Query Regression Analysis
# ===========================================================================


def test_per_query_regression_detection(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify per-query metric comparison and individual regression flags.

    Queries:
    Q-001: recall drops from 1.0 to 0.5 (deg = 0.5 > 0.02) -> REGRESSED
    Q-002: recall improves from 0.6 to 0.9 (deg = -0.3) -> IMPROVED
    Q-003: recall drops from 0.80 to 0.79 (deg = 0.01 <= 0.02) -> WITHIN TOLERANCE
    """
    b_queries = [
        {"query_id": "Q-001", "text": "BM25 query", "metrics": {"recall@5": 1.0}},
        {"query_id": "Q-002", "text": "HNSW query", "metrics": {"recall@5": 0.6}},
        {"query_id": "Q-003", "text": "Hybrid query", "metrics": {"recall@5": 0.80}},
    ]
    c_queries = [
        {"query_id": "Q-001", "text": "BM25 query", "metrics": {"recall@5": 0.5}},
        {"query_id": "Q-002", "text": "HNSW query", "metrics": {"recall@5": 0.9}},
        {"query_id": "Q-003", "text": "Hybrid query", "metrics": {"recall@5": 0.79}},
    ]

    b_report = _make_report("base-q", per_query_results=b_queries, metrics={"recall@5": 0.80})
    c_report = _make_report("cand-q", per_query_results=c_queries, metrics={"recall@5": 0.73})
    repo.save_report(b_report)
    repo.save_report(c_report)

    result = detector.compare_runs("base-q", "cand-q")

    assert result.status == RegressionStatus.REGRESSION
    assert result.total_queries_compared == 3
    assert result.regressed_query_ids == ["Q-001"]

    # Verify query ordering is deterministic by query_id ASC
    assert [q.query_id for q in result.query_regressions] == ["Q-001", "Q-002", "Q-003"]

    q1 = result.query_regressions[0]
    assert q1.has_regression is True
    assert q1.regressed_metrics == ["recall@5"]
    assert q1.metrics["recall@5"].degradation == pytest.approx(0.5)

    q2 = result.query_regressions[1]
    assert q2.has_regression is False
    assert q2.metrics["recall@5"].delta == pytest.approx(0.3)
    assert q2.metrics["recall@5"].degradation == pytest.approx(-0.3)

    q3 = result.query_regressions[2]
    assert q3.has_regression is False
    assert q3.metrics["recall@5"].degradation == pytest.approx(0.01)

    # Worst degradations list check
    assert len(result.worst_degradations) >= 1
    assert result.worst_degradations[0].query_id == "Q-001"
    assert result.worst_degradations[0].degradation == pytest.approx(0.5)


def test_per_query_mixed_metrics(repo: ExperimentRepository, detector: RegressionDetector) -> None:
    """Verify that a query regressing on one metric while improving on another preserves both."""
    b_queries = [
        {"query_id": "Q-MIX", "metrics": {"recall@5": 1.0, "mrr@5": 0.50}},
    ]
    c_queries = [
        {"query_id": "Q-MIX", "metrics": {"recall@5": 0.5, "mrr@5": 0.90}},
    ]
    b_report = _make_report(
        "b-mix",
        per_query_results=b_queries,
        metrics={"recall@5": 1.0, "mrr@5": 0.5},
    )
    c_report = _make_report(
        "c-mix",
        per_query_results=c_queries,
        metrics={"recall@5": 0.5, "mrr@5": 0.9},
    )
    repo.save_report(b_report)
    repo.save_report(c_report)

    result = detector.compare_runs("b-mix", "c-mix")

    q = result.query_regressions[0]
    assert q.has_regression is True
    assert q.regressed_metrics == ["recall@5"]  # recall regressed (1.0 -> 0.5)
    assert q.metrics["mrr@5"].is_regressed is False  # mrr improved (0.5 -> 0.9)
    assert q.metrics["mrr@5"].delta == pytest.approx(0.40)


# ===========================================================================
# 4. Comparability & Fail-Safe Incomplete Status
# ===========================================================================


def test_query_set_mismatch_yields_incomplete_when_no_regressions(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify that missing queries yield INCOMPLETE status (never false PASS)."""
    b_queries = [
        {"query_id": "Q-001", "metrics": {"recall@5": 0.8}},
        {"query_id": "Q-002", "metrics": {"recall@5": 0.8}},
    ]
    c_queries = [
        {"query_id": "Q-001", "metrics": {"recall@5": 0.8}},
        # Q-002 missing in candidate!
    ]
    b_report = _make_report("b-mismatch", per_query_results=b_queries)
    c_report = _make_report("c-mismatch", per_query_results=c_queries)
    repo.save_report(b_report)
    repo.save_report(c_report)

    result = detector.compare_runs("b-mismatch", "c-mismatch")

    assert result.status == RegressionStatus.INCOMPLETE
    assert result.is_complete is False
    assert result.has_regression is False
    assert result.baseline_only_query_ids == ["Q-002"]
    assert any("exist only in baseline" in w for w in result.warnings)


def test_query_set_mismatch_with_regressions_reports_regression(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify that if query sets mismatch BUT a regression is observed, status is REGRESSION."""
    b_queries = [
        {"query_id": "Q-001", "metrics": {"recall@5": 1.0}},
        {"query_id": "Q-002", "metrics": {"recall@5": 0.8}},
    ]
    c_queries = [
        {"query_id": "Q-001", "metrics": {"recall@5": 0.1}},  # severe drop
    ]
    b_report = _make_report("b-deg-mismatch", per_query_results=b_queries)
    c_report = _make_report("c-deg-mismatch", per_query_results=c_queries)
    repo.save_report(b_report)
    repo.save_report(c_report)

    result = detector.compare_runs("b-deg-mismatch", "c-deg-mismatch")

    assert result.status == RegressionStatus.REGRESSION
    assert result.is_complete is False
    assert result.has_regression is True
    assert "Q-001" in result.regressed_query_ids


def test_dataset_reference_mismatch_adds_warning(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify differing dataset_references record a comparability warning."""
    b_rep = _make_report("b-ds")
    c_rep = _make_report("c-ds")
    repo.save_report(b_rep, dataset_reference="ds-v1.jsonl")
    repo.save_report(c_rep, dataset_reference="ds-v2.jsonl")

    res = detector.compare_runs("b-ds", "c-ds")
    assert any("Dataset reference mismatch" in w for w in res.warnings)


def test_config_hash_divergence_adds_warning(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify different index configurations record a warning without aborting comparison."""
    b_rep = _make_report("b-cfg", config={"k1": 1.2})
    c_rep = _make_report("c-cfg", config={"k1": 1.8})
    repo.save_report(b_rep)
    repo.save_report(c_rep)

    res = detector.compare_runs("b-cfg", "c-cfg")
    assert any("Configuration hash divergence" in w for w in res.warnings)


# ===========================================================================
# 5. Aggregate Alias Handling & Validation
# ===========================================================================


def test_aggregate_mean_alias_deduplication(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify that canonical 'recall@5' and alias 'mean_recall@5' are not double-counted."""
    b_metrics = {"recall@5": 0.80, "mean_recall@5": 0.80}
    c_metrics = {"recall@5": 0.70, "mean_recall@5": 0.70}  # drop of 0.10

    b_rep = _make_report("b-alias", metrics=b_metrics)
    c_rep = _make_report("c-alias", metrics=c_metrics)
    repo.save_report(b_rep)
    repo.save_report(c_rep)

    res = detector.compare_runs("b-alias", "c-alias")
    assert res.status == RegressionStatus.REGRESSION
    # Only 1 aggregate regression counted, not 2
    assert res.regressed_aggregate_metrics == ["recall@5"]
    assert "mean_recall@5" not in res.aggregate_deltas


def test_conflicting_canonical_and_alias_values_raises_error(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify that contradictory values for recall@5 and mean_recall@5 raise an error."""
    conflicting = {"recall@5": 0.80, "mean_recall@5": 0.20}
    b_rep = _make_report("b-conflict", metrics=conflicting)
    c_rep = _make_report("c-conflict", metrics={"recall@5": 0.80})

    repo.save_report(b_rep)
    repo.save_report(c_rep)

    with pytest.raises(IncomparableRunsError, match="Conflicting alias values"):
        detector.compare_runs("b-conflict", "c-conflict")


# ===========================================================================
# 6. Error Handling & Defensive Boundaries
# ===========================================================================


def test_missing_run_ids_raise_error(detector: RegressionDetector) -> None:
    """Verify attempting to compare non-existent run IDs raises IncomparableRunsError."""
    with pytest.raises(IncomparableRunsError, match="Baseline run 'unknown-base' not found"):
        detector.compare_runs("unknown-base", "cand-x")


def test_empty_or_invalid_run_ids_raise_error(detector: RegressionDetector) -> None:
    """Verify empty or non-string run IDs are rejected."""
    with pytest.raises(IncomparableRunsError, match="baseline_run_id must be a non-empty string"):
        detector.compare_runs("", "cand-1")
    with pytest.raises(IncomparableRunsError, match="candidate_run_id must be a non-empty string"):
        detector.compare_runs("base-1", "   ")


def test_invalid_tolerance_values(detector: RegressionDetector) -> None:
    """Verify negative, boolean, NaN, and non-finite tolerances raise InvalidToleranceError."""
    b_rep = _make_report("b-t")
    c_rep = _make_report("c-t")

    with pytest.raises(InvalidToleranceError, match="non-negative"):
        detector.compare_reports(b_rep, c_rep, tolerance=-0.05)

    with pytest.raises(InvalidToleranceError, match="finite non-negative number"):
        detector.compare_reports(b_rep, c_rep, tolerance=True)  # type: ignore[arg-type]

    with pytest.raises(InvalidToleranceError, match="finite non-negative number"):
        detector.compare_reports(b_rep, c_rep, tolerance=float("nan"))

    with pytest.raises(InvalidToleranceError, match="finite non-negative number"):
        detector.compare_reports(b_rep, c_rep, tolerance=float("inf"))

    with pytest.raises(InvalidToleranceError, match="tolerance\\[ndcg@5\\]"):
        detector.compare_reports(b_rep, c_rep, metric_tolerances={"ndcg@5": -0.01})


def test_non_ranking_metrics_ignored_in_ranking_comparison(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify non-ranking metrics like latency_ms or zero_hit_rate do not trigger regressions."""
    b_rep = _make_report(
        "b-lat",
        metrics={"recall@5": 0.80, "mean_latency_ms": 10.0, "zero_hit_rate": 0.0},
    )
    c_rep = _make_report(
        "c-lat",
        metrics={"recall@5": 0.80, "mean_latency_ms": 500.0, "zero_hit_rate": 0.5},
    )
    repo.save_report(b_rep)
    repo.save_report(c_rep)

    res = detector.compare_runs("b-lat", "c-lat")
    # Only recall@5 was evaluated as ranking metric
    assert set(res.aggregate_deltas.keys()) == {"recall@5"}
    assert res.status == RegressionStatus.PASS


def test_duplicate_query_ids_in_results_raise_error(detector: RegressionDetector) -> None:
    """Verify malformed per-query lists with duplicate query_ids raise IncomparableRunsError."""
    dupe_queries = [
        {"query_id": "Q-001", "metrics": {"recall@5": 1.0}},
        {"query_id": "Q-001", "metrics": {"recall@5": 0.5}},
    ]
    b_rep = _make_report("b-dupe", per_query_results=dupe_queries)
    c_rep = _make_report(
        "c-valid",
        per_query_results=[{"query_id": "Q-001", "metrics": {"recall@5": 0.9}}],
    )

    with pytest.raises(IncomparableRunsError, match="Duplicate query_id 'Q-001'"):
        detector.compare_reports(b_rep, c_rep)


# ===========================================================================
# 7. Serialization Tests
# ===========================================================================


def test_regression_result_to_dict_and_to_json(
    repo: ExperimentRepository, detector: RegressionDetector
) -> None:
    """Verify that RegressionResult serializes cleanly to dict and JSON."""
    b_queries = [{"query_id": "Q-001", "text": "BM25 query", "metrics": {"recall@5": 0.8}}]
    c_queries = [{"query_id": "Q-001", "text": "BM25 query", "metrics": {"recall@5": 0.7}}]
    b_rep = _make_report("b-ser", per_query_results=b_queries, metrics={"recall@5": 0.8})
    c_rep = _make_report("c-ser", per_query_results=c_queries, metrics={"recall@5": 0.7})
    repo.save_report(b_rep)
    repo.save_report(c_rep)

    res = detector.compare_runs("b-ser", "c-ser")

    res_dict = res.to_dict()
    assert res_dict["status"] == "REGRESSION"
    assert res_dict["has_regression"] is True
    assert "recall@5" in res_dict["aggregate_deltas"]
    assert len(res_dict["query_regressions"]) == 1

    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["status"] == "REGRESSION"
    assert parsed["baseline_run_id"] == "b-ser"
    assert parsed["candidate_run_id"] == "c-ser"
