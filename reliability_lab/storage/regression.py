"""Regression detection engine and metric diffing for baseline vs candidate experiment runs."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reliability_lab.core.models import EvaluationReport
from reliability_lab.storage.repository import ExperimentRepository, StoredRun

DEFAULT_TOLERANCE: float = 0.02

# Pattern matching recognized retrieval ranking metrics with cutoffs (e.g. recall@5, mrr@10)
RANKING_METRIC_PATTERN = re.compile(
    r"^(recall|mrr|ndcg|precision|hit_rate|map)@(\d+)$",
    re.IGNORECASE,
)


class RegressionError(Exception):
    """Base exception for regression detection operations."""


class IncomparableRunsError(RegressionError):
    """Raised when runs cannot be compared due to missing records or incompatible schemas."""


class InvalidToleranceError(RegressionError):
    """Raised when a configured tolerance is negative, boolean, NaN, or non-finite."""


class RegressionStatus(str, Enum):
    """Overall classification status for an experiment run comparison."""

    PASS = "PASS"
    REGRESSION = "REGRESSION"
    INCOMPLETE = "INCOMPLETE"


class MetricDirection(str, Enum):
    """Optimization direction of an evaluation metric."""

    HIGHER_IS_BETTER = "HIGHER_IS_BETTER"
    LOWER_IS_BETTER = "LOWER_IS_BETTER"


def is_ranking_metric(metric_name: str) -> bool:
    """Return True if the metric is a recognized ranking metric with cutoff."""
    norm = metric_name.lower().removeprefix("mean_")
    return bool(RANKING_METRIC_PATTERN.match(norm))


def normalize_metric_name(metric_name: str) -> str:
    """Normalize a metric name by stripping aggregate alias prefixes like 'mean_'."""
    lowered = metric_name.strip().lower()
    if lowered.startswith("mean_") and is_ranking_metric(lowered):
        return lowered.removeprefix("mean_")
    return lowered


def is_degradation_regressed(degradation: float, tolerance: float) -> bool:
    """Return True if degradation strictly exceeds tolerance beyond floating point noise.

    Equality with the tolerance is never treated as a regression.
    """
    if math.isclose(degradation, tolerance, rel_tol=1e-9, abs_tol=1e-9):
        return False
    return degradation > tolerance


def validate_tolerance(val: Any, name: str = "tolerance") -> float:
    """Validate that a tolerance value is a finite non-negative float and not a boolean."""
    if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
        raise InvalidToleranceError(
            f"Tolerance '{name}' must be a finite non-negative number, got {val}."
        )
    float_val = float(val)
    if float_val < 0.0:
        raise InvalidToleranceError(f"Tolerance '{name}' must be non-negative, got {float_val}.")
    return float_val


@dataclass(frozen=True)
class MetricDelta:
    """Represents the comparison of one aggregate metric between baseline and candidate runs."""

    metric_name: str
    baseline_score: float
    candidate_score: float
    delta: float  # candidate_score - baseline_score
    degradation: float  # magnitude of deterioration relative to direction
    tolerance: float
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    is_regressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize metric delta to a dictionary."""
        return {
            "metric_name": self.metric_name,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "delta": self.delta,
            "degradation": self.degradation,
            "tolerance": self.tolerance,
            "direction": self.direction.value,
            "is_regressed": self.is_regressed,
        }


@dataclass(frozen=True)
class QueryMetricDelta:
    """Represents a metric comparison for an individual query."""

    query_id: str
    metric_name: str
    baseline_score: float
    candidate_score: float
    delta: float  # candidate - baseline
    degradation: float
    tolerance: float
    direction: MetricDirection = MetricDirection.HIGHER_IS_BETTER
    is_regressed: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialize query metric delta to a dictionary."""
        return {
            "query_id": self.query_id,
            "metric_name": self.metric_name,
            "baseline_score": self.baseline_score,
            "candidate_score": self.candidate_score,
            "delta": self.delta,
            "degradation": self.degradation,
            "tolerance": self.tolerance,
            "direction": self.direction.value,
            "is_regressed": self.is_regressed,
        }


@dataclass(frozen=True)
class QueryRegression:
    """Detailed per-query comparison result across all evaluated metrics."""

    query_id: str
    query_text: str | None
    metrics: dict[str, QueryMetricDelta]
    regressed_metrics: list[str]
    max_degradation: float
    has_regression: bool
    failure_trace: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize query regression diagnostics to a dictionary."""
        return {
            "query_id": self.query_id,
            "query_text": self.query_text,
            "metrics": {k: v.to_dict() for k, v in self.metrics.items()},
            "regressed_metrics": list(self.regressed_metrics),
            "max_degradation": self.max_degradation,
            "has_regression": self.has_regression,
            "failure_trace": dict(self.failure_trace) if self.failure_trace else None,
        }


@dataclass(frozen=True)
class RegressionResult:
    """Complete serialized outcome of an experiment run regression comparison."""

    status: RegressionStatus
    baseline_run_id: str
    candidate_run_id: str
    baseline_config_hash: str | None
    candidate_config_hash: str | None
    baseline_dataset: str | None
    candidate_dataset: str | None
    aggregate_deltas: dict[str, MetricDelta]
    query_regressions: list[QueryRegression]
    regressed_aggregate_metrics: list[str]
    regressed_query_ids: list[str]
    total_queries_compared: int
    baseline_only_query_ids: list[str]
    candidate_only_query_ids: list[str]
    worst_degradations: list[QueryMetricDelta]
    warnings: list[str]
    is_complete: bool
    has_regression: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize complete regression result to a dictionary."""
        return {
            "status": self.status.value,
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "baseline_config_hash": self.baseline_config_hash,
            "candidate_config_hash": self.candidate_config_hash,
            "baseline_dataset": self.baseline_dataset,
            "candidate_dataset": self.candidate_dataset,
            "aggregate_deltas": {k: v.to_dict() for k, v in self.aggregate_deltas.items()},
            "query_regressions": [q.to_dict() for q in self.query_regressions],
            "regressed_aggregate_metrics": list(self.regressed_aggregate_metrics),
            "regressed_query_ids": list(self.regressed_query_ids),
            "total_queries_compared": self.total_queries_compared,
            "baseline_only_query_ids": list(self.baseline_only_query_ids),
            "candidate_only_query_ids": list(self.candidate_only_query_ids),
            "worst_degradations": [w.to_dict() for w in self.worst_degradations],
            "warnings": list(self.warnings),
            "is_complete": self.is_complete,
            "has_regression": self.has_regression,
        }

    def to_json(self) -> str:
        """Serialize complete regression result to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=2)


class RegressionDetector:
    """Engine for comparing baseline and candidate runs to detect performance degradation."""

    def __init__(self, repository: ExperimentRepository) -> None:
        """Initialize RegressionDetector with an ExperimentRepository instance.

        Args:
            repository: ExperimentRepository backed by SQLite.
        """
        if not isinstance(repository, ExperimentRepository):
            raise TypeError(
                f"repository must be an ExperimentRepository, got {type(repository).__name__}."
            )
        self._repo = repository

    @property
    def repository(self) -> ExperimentRepository:
        """Return the underlying ExperimentRepository."""
        return self._repo

    def compare_runs(
        self,
        baseline_run_id: str,
        candidate_run_id: str,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        metric_tolerances: Mapping[str, float] | None = None,
    ) -> RegressionResult:
        """Compare two persisted runs retrieved by ID from the repository.

        Args:
            baseline_run_id: Stored run ID to serve as historical baseline.
            candidate_run_id: Stored run ID to evaluate for regressions.
            tolerance: Default absolute degradation tolerance (default 0.02).
            metric_tolerances: Optional per-metric override mappings.

        Returns:
            RegressionResult containing aggregate and per-query comparisons.

        Raises:
            IncomparableRunsError: If either run ID does not exist or runs are malformed.
            InvalidToleranceError: If any tolerance value is invalid.
        """
        if not isinstance(baseline_run_id, str) or not baseline_run_id.strip():
            raise IncomparableRunsError("baseline_run_id must be a non-empty string.")
        if not isinstance(candidate_run_id, str) or not candidate_run_id.strip():
            raise IncomparableRunsError("candidate_run_id must be a non-empty string.")

        baseline = self._repo.get_run(baseline_run_id)
        if baseline is None:
            raise IncomparableRunsError(
                f"Baseline run '{baseline_run_id}' not found in repository."
            )

        candidate = self._repo.get_run(candidate_run_id)
        if candidate is None:
            raise IncomparableRunsError(
                f"Candidate run '{candidate_run_id}' not found in repository."
            )

        return self.compare_reports(
            baseline,
            candidate,
            tolerance=tolerance,
            metric_tolerances=metric_tolerances,
        )

    def compare_reports(
        self,
        baseline: StoredRun | EvaluationReport,
        candidate: StoredRun | EvaluationReport,
        *,
        tolerance: float = DEFAULT_TOLERANCE,
        metric_tolerances: Mapping[str, float] | None = None,
    ) -> RegressionResult:
        """Compare two evaluation runs directly (either StoredRun or EvaluationReport).

        Args:
            baseline: Baseline run object.
            candidate: Candidate run object.
            tolerance: Default absolute degradation tolerance (default 0.02).
            metric_tolerances: Optional per-metric override mappings.

        Returns:
            RegressionResult with complete diff diagnostics and regression classifications.
        """
        default_tol = validate_tolerance(tolerance, "default_tolerance")
        overrides: dict[str, float] = {}
        if metric_tolerances is not None:
            if not isinstance(metric_tolerances, Mapping):
                raise TypeError("metric_tolerances must be a Mapping.")
            for k, v in metric_tolerances.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError("Metric tolerance override keys must be non-empty strings.")
                overrides[normalize_metric_name(k)] = validate_tolerance(v, f"tolerance[{k}]")

        def _get_tol(m: str) -> float:
            norm = normalize_metric_name(m)
            return overrides.get(norm, default_tol)

        warnings: list[str] = []

        # Extract identifiers and metadata
        b_id = baseline.run_id if isinstance(baseline, StoredRun) else baseline.report_id
        c_id = candidate.run_id if isinstance(candidate, StoredRun) else candidate.report_id

        b_cfg_hash = baseline.config_hash if isinstance(baseline, StoredRun) else None
        c_cfg_hash = candidate.config_hash if isinstance(candidate, StoredRun) else None

        b_dataset = baseline.dataset_reference if isinstance(baseline, StoredRun) else None
        c_dataset = candidate.dataset_reference if isinstance(candidate, StoredRun) else None

        if b_dataset != c_dataset:
            warnings.append(
                f"Dataset reference mismatch: baseline '{b_dataset}' vs candidate '{c_dataset}'."
            )

        if b_cfg_hash is not None and c_cfg_hash is not None and b_cfg_hash != c_cfg_hash:
            warnings.append(
                f"Configuration hash divergence: baseline '{b_cfg_hash[:8]}' "
                f"vs candidate '{c_cfg_hash[:8]}'."
            )

        # -------------------------------------------------------------------
        # 1. Normalize and compare aggregate metrics
        # -------------------------------------------------------------------
        base_agg = self._normalize_aggregate_metrics(baseline.metrics, "baseline")
        cand_agg = self._normalize_aggregate_metrics(candidate.metrics, "candidate")

        aggregate_deltas: dict[str, MetricDelta] = {}
        regressed_aggregate_metrics: list[str] = []

        # Compare metrics present in both
        common_metrics = sorted(set(base_agg.keys()) & set(cand_agg.keys()))
        for m_name in common_metrics:
            b_val = base_agg[m_name]
            c_val = cand_agg[m_name]
            tol = _get_tol(m_name)

            delta = c_val - b_val
            degradation = b_val - c_val  # Higher is better: degradation = base - cand
            is_reg = is_degradation_regressed(degradation, tol)

            m_delta = MetricDelta(
                metric_name=m_name,
                baseline_score=b_val,
                candidate_score=c_val,
                delta=delta,
                degradation=degradation,
                tolerance=tol,
                direction=MetricDirection.HIGHER_IS_BETTER,
                is_regressed=is_reg,
            )
            aggregate_deltas[m_name] = m_delta
            if is_reg:
                regressed_aggregate_metrics.append(m_name)

        # -------------------------------------------------------------------
        # 2. Compare per-query results
        # -------------------------------------------------------------------
        b_queries_map = self._map_query_results(baseline.per_query_results, "baseline")
        c_queries_map = self._map_query_results(candidate.per_query_results, "candidate")

        b_qids = set(b_queries_map.keys())
        c_qids = set(c_queries_map.keys())

        common_qids = sorted(b_qids & c_qids)
        b_only = sorted(b_qids - c_qids)
        c_only = sorted(c_qids - b_qids)

        is_complete = True
        if b_only:
            warnings.append(f"{len(b_only)} queries exist only in baseline (e.g. {b_only[:3]}).")
            is_complete = False
        if c_only:
            warnings.append(f"{len(c_only)} queries exist only in candidate (e.g. {c_only[:3]}).")
            is_complete = False

        query_regressions: list[QueryRegression] = []
        regressed_query_ids: list[str] = []
        all_query_metric_deltas: list[QueryMetricDelta] = []

        for qid in common_qids:
            b_q = b_queries_map[qid]
            c_q = c_queries_map[qid]

            q_metrics_delta: dict[str, QueryMetricDelta] = {}
            q_regressed_metrics: list[str] = []
            max_deg = 0.0

            b_m = b_q.get("metrics", {})
            c_m = c_q.get("metrics", {})

            # Exact canonical matching for ranking metrics
            common_qm = sorted(
                set(b_m.keys()) & set(c_m.keys()) & {m for m in b_m.keys() if is_ranking_metric(m)}
            )

            for m_name in common_qm:
                b_s = float(b_m[m_name])
                c_s = float(c_m[m_name])
                tol = _get_tol(m_name)

                delta = c_s - b_s
                deg = b_s - c_s
                is_reg = is_degradation_regressed(deg, tol)

                q_delta = QueryMetricDelta(
                    query_id=qid,
                    metric_name=m_name,
                    baseline_score=b_s,
                    candidate_score=c_s,
                    delta=delta,
                    degradation=deg,
                    tolerance=tol,
                    direction=MetricDirection.HIGHER_IS_BETTER,
                    is_regressed=is_reg,
                )
                q_metrics_delta[m_name] = q_delta
                all_query_metric_deltas.append(q_delta)

                if deg > max_deg:
                    max_deg = deg
                if is_reg:
                    q_regressed_metrics.append(m_name)

            has_q_reg = len(q_regressed_metrics) > 0
            if has_q_reg:
                regressed_query_ids.append(qid)

            query_regressions.append(
                QueryRegression(
                    query_id=qid,
                    query_text=c_q.get("text") or b_q.get("text"),
                    metrics=q_metrics_delta,
                    regressed_metrics=q_regressed_metrics,
                    max_degradation=max_deg,
                    has_regression=has_q_reg,
                    failure_trace=c_q.get("failure_trace") or b_q.get("failure_trace"),
                )
            )

        # Sort worst degradations descending by degradation magnitude
        worst_degradations = sorted(
            [d for d in all_query_metric_deltas if d.degradation > 0.0],
            key=lambda x: (x.degradation, x.query_id, x.metric_name),
            reverse=True,
        )[:15]

        # -------------------------------------------------------------------
        # 3. Aggregate Status Classification Contract
        # -------------------------------------------------------------------
        has_regression = bool(regressed_aggregate_metrics or regressed_query_ids)

        if has_regression:
            status = RegressionStatus.REGRESSION
        elif not is_complete or len(common_metrics) == 0:
            status = RegressionStatus.INCOMPLETE
        else:
            status = RegressionStatus.PASS

        return RegressionResult(
            status=status,
            baseline_run_id=b_id,
            candidate_run_id=c_id,
            baseline_config_hash=b_cfg_hash,
            candidate_config_hash=c_cfg_hash,
            baseline_dataset=b_dataset,
            candidate_dataset=c_dataset,
            aggregate_deltas=aggregate_deltas,
            query_regressions=query_regressions,
            regressed_aggregate_metrics=regressed_aggregate_metrics,
            regressed_query_ids=regressed_query_ids,
            total_queries_compared=len(common_qids),
            baseline_only_query_ids=b_only,
            candidate_only_query_ids=c_only,
            worst_degradations=worst_degradations,
            warnings=warnings,
            is_complete=is_complete,
            has_regression=has_regression,
        )

    def _normalize_aggregate_metrics(
        self, metrics: Mapping[str, Any], run_label: str
    ) -> dict[str, float]:
        """Normalize aggregate metrics, resolving aliases and validating values."""
        if not isinstance(metrics, Mapping):
            raise IncomparableRunsError(f"{run_label} metrics must be a Mapping.")

        canonical_dict: dict[str, float] = {}

        for raw_name, raw_val in metrics.items():
            if not isinstance(raw_name, str) or not raw_name.strip():
                raise IncomparableRunsError(f"Invalid metric name in {run_label}: {raw_name}")
            if (
                isinstance(raw_val, bool)
                or not isinstance(raw_val, (int, float))
                or not math.isfinite(raw_val)
            ):
                raise IncomparableRunsError(
                    f"Metric '{raw_name}' in {run_label} must be a finite numerical value, "
                    f"got {raw_val}."
                )

            float_val = float(raw_val)
            norm_name = normalize_metric_name(raw_name)

            if not is_ranking_metric(norm_name):
                continue  # Focus on retrieval ranking metrics

            if norm_name in canonical_dict:
                # Both canonical and alias exist (e.g. mrr@5 and mean_mrr@5)
                existing_val = canonical_dict[norm_name]
                if abs(existing_val - float_val) > 1e-5:
                    raise IncomparableRunsError(
                        f"Conflicting alias values in {run_label} for '{norm_name}': "
                        f"{existing_val} vs {float_val}."
                    )
                # If equal within tolerance, keep canonical
            else:
                canonical_dict[norm_name] = float_val

        return canonical_dict

    def _map_query_results(
        self, per_query_results: Sequence[Any], run_label: str
    ) -> dict[str, dict[str, Any]]:
        """Map and validate per-query records by query_id."""
        if not isinstance(per_query_results, Sequence):
            raise IncomparableRunsError(f"{run_label} per_query_results must be a Sequence.")

        mapped: dict[str, dict[str, Any]] = {}
        for idx, rec in enumerate(per_query_results):
            if not isinstance(rec, dict):
                raise IncomparableRunsError(
                    f"{run_label} query record at index {idx} must be a dict."
                )
            qid = rec.get("query_id")
            if not isinstance(qid, str) or not qid.strip():
                raise IncomparableRunsError(
                    f"{run_label} query record at index {idx} missing valid 'query_id'."
                )
            if qid in mapped:
                raise IncomparableRunsError(
                    f"Duplicate query_id '{qid}' in {run_label} per_query_results."
                )

            # Validate metric values if present
            q_metrics = rec.get("metrics")
            if q_metrics is not None:
                if not isinstance(q_metrics, dict):
                    raise IncomparableRunsError(f"metrics in query '{qid}' must be a dictionary.")
                for m_name, m_val in q_metrics.items():
                    if (
                        isinstance(m_val, bool)
                        or not isinstance(m_val, (int, float))
                        or not math.isfinite(m_val)
                    ):
                        raise IncomparableRunsError(
                            f"Metric '{m_name}' in query '{qid}' must be a finite float, "
                            f"got {m_val}."
                        )

            mapped[qid] = dict(rec)

        return mapped
