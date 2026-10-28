"""Repository abstraction for persisting, retrieving, and auditing experiment evaluation runs."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from reliability_lab.core.models import EvaluationReport
from reliability_lab.storage.database import Database


class StorageError(Exception):
    """Base exception for storage and repository operations."""


class DuplicateRunError(StorageError):
    """Raised when persisting a run ID that already exists with conflicting data."""


class ConfigurationConflictError(StorageError):
    """Raised when config hash matches an existing record but canonical JSON differs."""


class RunNotFoundError(StorageError):
    """Raised when a requested experiment run cannot be found."""


def canonicalize_config(config: Mapping[str, Any]) -> str:
    """Return a deterministic, canonical UTF-8 JSON representation of a configuration mapping.

    Sorts all keys recursively, applies compact standard separators (',', ':'),
    disallows non-finite floats (NaN, Inf, -Inf), and preserves unicode characters.

    Args:
        config: Configuration mapping to canonicalize.

    Returns:
        Deterministic JSON string.

    Raises:
        TypeError: If keys are not strings or values are un-serializable.
        ValueError: If configuration contains non-finite numerical values.
    """
    if not isinstance(config, Mapping):
        raise TypeError(f"Configuration must be a Mapping, got: {type(config).__name__}")

    def _sanitize(item: Any) -> Any:
        if isinstance(item, bool):
            return item
        if isinstance(item, (int, float)):
            if not math.isfinite(item):
                raise ValueError(f"Cannot canonicalize non-finite numerical value: {item}")
            return item
        if isinstance(item, str):
            return item
        if item is None:
            return None
        if isinstance(item, Mapping):
            sanitized_dict: dict[str, Any] = {}
            for k, v in item.items():
                if not isinstance(k, str):
                    raise TypeError(
                        f"Configuration dictionary keys must be strings, got: {type(k).__name__}"
                    )
                sanitized_dict[k] = _sanitize(v)
            return sanitized_dict
        if isinstance(item, (list, tuple, Sequence)):
            return [_sanitize(elem) for elem in item]
        raise TypeError(f"Un-serializable object of type {type(item).__name__} in configuration.")

    sanitized = _sanitize(config)
    try:
        return json.dumps(
            sanitized,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Failed to canonicalize configuration: {exc}") from exc


def compute_config_hash(config: Mapping[str, Any]) -> str:
    """Compute the SHA-256 hexadecimal digest of a configuration's canonical JSON representation.

    Args:
        config: Configuration mapping.

    Returns:
        64-character lowercase hexadecimal SHA-256 hash.
    """
    canonical_json = canonicalize_config(config)
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class StoredRun:
    """Represents a fully reconstructed experiment run from persistent storage."""

    run_id: str
    experiment_name: str
    run_timestamp: str
    config_hash: str
    config: dict[str, Any]
    dataset_reference: str | None
    metadata: dict[str, Any]
    created_at: str
    metrics: dict[str, float]
    per_query_results: list[dict[str, Any]]

    def to_report(self) -> EvaluationReport:
        """Reconstruct the canonical EvaluationReport entity."""
        return EvaluationReport(
            experiment_name=self.experiment_name,
            metrics=dict(self.metrics),
            report_id=self.run_id,
            per_query_results=[dict(r) for r in self.per_query_results],
            timestamp=self.run_timestamp,
            config=dict(self.config),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize stored run to a dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "run_timestamp": self.run_timestamp,
            "config_hash": self.config_hash,
            "config": dict(self.config),
            "dataset_reference": self.dataset_reference,
            "metadata": dict(self.metadata),
            "created_at": self.created_at,
            "metrics": dict(self.metrics),
            "per_query_results": [dict(r) for r in self.per_query_results],
        }


@dataclass(frozen=True)
class StoredRunSummary:
    """Summary record for listing experiment runs without query-level payload overhead."""

    run_id: str
    experiment_name: str
    run_timestamp: str
    config_hash: str
    dataset_reference: str | None
    created_at: str
    metric_count: int
    query_count: int
    metrics: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Serialize run summary to a dictionary."""
        return {
            "run_id": self.run_id,
            "experiment_name": self.experiment_name,
            "run_timestamp": self.run_timestamp,
            "config_hash": self.config_hash,
            "dataset_reference": self.dataset_reference,
            "created_at": self.created_at,
            "metric_count": self.metric_count,
            "query_count": self.query_count,
            "metrics": dict(self.metrics),
        }


class ExperimentRepository:
    """Relational SQLite repository for persisting and querying experiment runs and metrics."""

    def __init__(self, database: Database | str | Path = ":memory:") -> None:
        """Initialize ExperimentRepository with a Database instance or SQLite connection path."""
        if isinstance(database, Database):
            self._db = database
        else:
            self._db = Database(database)

    @property
    def database(self) -> Database:
        """Return the underlying Database instance."""
        return self._db

    def close(self) -> None:
        """Close the underlying Database instance and release connections."""
        self._db.close()

    def __enter__(self) -> ExperimentRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def save_report(
        self,
        report: EvaluationReport,
        *,
        index_config: Mapping[str, Any] | None = None,
        dataset_reference: str | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> str:
        """Persist an EvaluationReport transactionally into SQLite.

        Validates all fields, canonicalizes and deduplicates configurations, saves
        aggregate metrics and per-query records, and rejects conflicting duplicate writes.

        Args:
            report: EvaluationReport domain model instance.
            index_config: Optional configuration dictionary to merge with report.config.
            dataset_reference: Optional benchmark or corpus reference identifier.
            metadata: Optional additional JSON-serializable experiment metadata.

        Returns:
            The saved run_id.

        Raises:
            TypeError: If input types are invalid.
            ValueError: If metrics, queries, or configs contain non-finite numbers.
            DuplicateRunError: If run_id already exists with conflicting contents.
            ConfigurationConflictError: If config_hash matches a different config.
        """
        if not isinstance(report, EvaluationReport):
            raise TypeError(f"report must be an EvaluationReport, got: {type(report).__name__}")

        if not isinstance(report.report_id, str) or not report.report_id.strip():
            raise ValueError("report.report_id must be a non-empty string.")
        if not isinstance(report.experiment_name, str) or not report.experiment_name.strip():
            raise ValueError("report.experiment_name must be a non-empty string.")
        if not isinstance(report.timestamp, str) or not report.timestamp.strip():
            raise ValueError("report.timestamp must be a non-empty string.")

        # Validate aggregate metrics (reject booleans and non-finite numbers)
        if not isinstance(report.metrics, dict):
            raise TypeError("report.metrics must be a dictionary.")
        clean_metrics: dict[str, float] = {}
        for m_name, m_val in report.metrics.items():
            if not isinstance(m_name, str) or not m_name.strip():
                raise ValueError("Metric names must be non-empty strings.")
            if (
                isinstance(m_val, bool)
                or not isinstance(m_val, (int, float))
                or not math.isfinite(m_val)
            ):
                raise ValueError(
                    f"Metric score for '{m_name}' must be a finite numerical value, got {m_val}."
                )
            clean_metrics[m_name] = float(m_val)

        # Validate per-query results and metric scores
        if not isinstance(report.per_query_results, list):
            raise TypeError("report.per_query_results must be a list.")
        seen_query_ids: set[str] = set()
        clean_query_records: list[dict[str, Any]] = []

        for q_idx, q_rec in enumerate(report.per_query_results):
            if not isinstance(q_rec, dict):
                raise TypeError(f"Query record at index {q_idx} must be a dictionary.")
            qid = q_rec.get("query_id")
            if not isinstance(qid, str) or not qid.strip():
                raise ValueError(f"Query record at index {q_idx} missing valid 'query_id'.")
            if qid in seen_query_ids:
                raise ValueError(f"Duplicate query_id '{qid}' detected in per_query_results.")
            seen_query_ids.add(qid)

            latency = q_rec.get("latency_ms")
            if latency is not None:
                if (
                    isinstance(latency, bool)
                    or not isinstance(latency, (int, float))
                    or not math.isfinite(latency)
                ):
                    raise ValueError(
                        f"latency_ms for query '{qid}' must be a finite float, got {latency}."
                    )

            q_metrics = q_rec.get("metrics")
            if q_metrics is not None:
                if not isinstance(q_metrics, dict):
                    raise TypeError(f"metrics for query '{qid}' must be a dictionary.")
                for qm_name, qm_val in q_metrics.items():
                    if not isinstance(qm_name, str) or not qm_name.strip():
                        raise ValueError(
                            f"Metric name for query '{qid}' must be a non-empty string."
                        )
                    if (
                        isinstance(qm_val, bool)
                        or not isinstance(qm_val, (int, float))
                        or not math.isfinite(qm_val)
                    ):
                        raise ValueError(
                            f"Query metric '{qm_name}' for query '{qid}' "
                            f"must be a finite float, got {qm_val}."
                        )

            clean_query_records.append(dict(q_rec))

        # Merge and canonicalize configuration
        final_config: dict[str, Any] = dict(report.config)
        if index_config is not None:
            if not isinstance(index_config, Mapping):
                raise TypeError("index_config must be a Mapping.")
            final_config.update(dict(index_config))

        canonical_config_json = canonicalize_config(final_config)
        config_hash = compute_config_hash(final_config)

        # Sanitize metadata
        clean_metadata = dict(metadata) if metadata is not None else {}
        try:
            metadata_json = json.dumps(
                clean_metadata, sort_keys=True, ensure_ascii=False, allow_nan=False
            )
        except (ValueError, TypeError) as exc:
            raise ValueError(f"Metadata is not valid JSON: {exc}") from exc

        now_utc = datetime.now(timezone.utc).isoformat()

        # Transactional write
        with self._db.transaction() as conn:
            # 1. Check for existing run
            existing_run = conn.execute(
                "SELECT run_id, experiment_name, run_timestamp, config_hash, dataset_reference "
                "FROM experiment_runs WHERE run_id = ?;",
                (report.report_id,),
            ).fetchone()

            if existing_run is not None:
                # Check for idempotent replay
                same_name = existing_run["experiment_name"] == report.experiment_name
                same_ts = existing_run["run_timestamp"] == report.timestamp
                same_hash = existing_run["config_hash"] == config_hash
                same_dataset = existing_run["dataset_reference"] == dataset_reference

                # Fetch existing metrics
                stored_metrics_rows = conn.execute(
                    "SELECT metric_name, score FROM run_metrics WHERE run_id = ?;",
                    (report.report_id,),
                ).fetchall()
                stored_metrics = {r["metric_name"]: float(r["score"]) for r in stored_metrics_rows}

                # Fetch existing queries count
                stored_q_count = conn.execute(
                    "SELECT COUNT(*) as cnt FROM query_results WHERE run_id = ?;",
                    (report.report_id,),
                ).fetchone()["cnt"]

                same_metrics = stored_metrics == clean_metrics
                same_q_count = stored_q_count == len(clean_query_records)

                if (
                    same_name
                    and same_ts
                    and same_hash
                    and same_dataset
                    and same_metrics
                    and same_q_count
                ):
                    # Idempotent re-save: return existing ID cleanly
                    return report.report_id
                raise DuplicateRunError(
                    f"Run ID '{report.report_id}' already exists with conflicting experiment data."
                )

            # 2. Check or store index_configuration
            existing_cfg = conn.execute(
                "SELECT config_json FROM index_configurations WHERE config_hash = ?;",
                (config_hash,),
            ).fetchone()

            if existing_cfg is not None:
                if existing_cfg["config_json"] != canonical_config_json:
                    raise ConfigurationConflictError(
                        f"Configuration hash collision detected for hash '{config_hash}'."
                    )
            else:
                conn.execute(
                    "INSERT INTO index_configurations (config_hash, config_json, created_at) "
                    "VALUES (?, ?, ?);",
                    (config_hash, canonical_config_json, now_utc),
                )

            # 3. Insert experiment_run header
            conn.execute(
                "INSERT INTO experiment_runs ("
                "run_id, experiment_name, run_timestamp, config_hash, "
                "dataset_reference, metadata_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?);",
                (
                    report.report_id,
                    report.experiment_name,
                    report.timestamp,
                    config_hash,
                    dataset_reference,
                    metadata_json,
                    now_utc,
                ),
            )

            # 4. Insert aggregate run_metrics
            for m_name, m_score in clean_metrics.items():
                conn.execute(
                    "INSERT INTO run_metrics (run_id, metric_name, score) VALUES (?, ?, ?);",
                    (report.report_id, m_name, m_score),
                )

            # 5. Insert query_results and query_metric_scores
            for q_rec in clean_query_records:
                qid = q_rec["query_id"]
                latency = q_rec.get("latency_ms")
                q_json = json.dumps(q_rec, sort_keys=True, ensure_ascii=False, allow_nan=False)

                conn.execute(
                    "INSERT INTO query_results (run_id, query_id, latency_ms, result_json) "
                    "VALUES (?, ?, ?, ?);",
                    (
                        report.report_id,
                        qid,
                        float(latency) if latency is not None else None,
                        q_json,
                    ),
                )

                q_metrics = q_rec.get("metrics")
                if q_metrics and isinstance(q_metrics, dict):
                    for qm_name, qm_val in q_metrics.items():
                        conn.execute(
                            "INSERT INTO query_metric_scores "
                            "(run_id, query_id, metric_name, score) "
                            "VALUES (?, ?, ?, ?);",
                            (report.report_id, qid, qm_name, float(qm_val)),
                        )

        return report.report_id

    def get_run(self, run_id: str) -> StoredRun | None:
        """Retrieve a saved experiment run by run_id, or None if not found."""
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("run_id must be a non-empty string.")

        with self._db.get_connection() as conn:
            run_row = conn.execute(
                "SELECT r.run_id, r.experiment_name, r.run_timestamp, r.config_hash, "
                "r.dataset_reference, r.metadata_json, r.created_at, c.config_json "
                "FROM experiment_runs r "
                "JOIN index_configurations c ON r.config_hash = c.config_hash "
                "WHERE r.run_id = ?;",
                (run_id,),
            ).fetchone()

            if run_row is None:
                return None

            # Retrieve aggregate metrics
            metric_rows = conn.execute(
                "SELECT metric_name, score FROM run_metrics "
                "WHERE run_id = ? ORDER BY metric_name ASC;",
                (run_id,),
            ).fetchall()
            metrics = {row["metric_name"]: float(row["score"]) for row in metric_rows}

            # Retrieve per-query results
            query_rows = conn.execute(
                "SELECT result_json FROM query_results WHERE run_id = ? ORDER BY query_id ASC;",
                (run_id,),
            ).fetchall()
            per_query_results = [json.loads(row["result_json"]) for row in query_rows]

            config = json.loads(run_row["config_json"])
            metadata = json.loads(run_row["metadata_json"]) if run_row["metadata_json"] else {}

            return StoredRun(
                run_id=run_row["run_id"],
                experiment_name=run_row["experiment_name"],
                run_timestamp=run_row["run_timestamp"],
                config_hash=run_row["config_hash"],
                config=config,
                dataset_reference=run_row["dataset_reference"],
                metadata=metadata,
                created_at=run_row["created_at"],
                metrics=metrics,
                per_query_results=per_query_results,
            )

    def list_runs(
        self,
        *,
        experiment_name: str | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[StoredRunSummary]:
        """List stored experiment runs with deterministic ordering (timestamp DESC, run_id ASC).

        Args:
            experiment_name: Optional filter for a specific experiment name.
            limit: Optional limit on the number of returned runs.
            offset: Number of runs to skip (default 0).

        Returns:
            List of StoredRunSummary objects.
        """
        sql = (
            "SELECT r.run_id, r.experiment_name, r.run_timestamp, r.config_hash, "
            "r.dataset_reference, r.created_at, "
            "(SELECT COUNT(*) FROM query_results q WHERE q.run_id = r.run_id) as query_count, "
            "(SELECT COUNT(*) FROM run_metrics m WHERE m.run_id = r.run_id) as metric_count "
            "FROM experiment_runs r "
        )
        params: list[Any] = []
        if experiment_name is not None:
            sql += "WHERE r.experiment_name = ? "
            params.append(experiment_name)

        sql += "ORDER BY r.run_timestamp DESC, r.run_id ASC "

        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be non-negative.")
            sql += "LIMIT ? OFFSET ? "
            params.extend([limit, max(0, offset)])
        elif offset > 0:
            sql += "LIMIT -1 OFFSET ? "
            params.append(offset)

        with self._db.get_connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            summaries: list[StoredRunSummary] = []
            for row in rows:
                run_id = row["run_id"]
                metric_rows = conn.execute(
                    "SELECT metric_name, score FROM run_metrics "
                    "WHERE run_id = ? ORDER BY metric_name ASC;",
                    (run_id,),
                ).fetchall()
                metrics = {m["metric_name"]: float(m["score"]) for m in metric_rows}

                summaries.append(
                    StoredRunSummary(
                        run_id=row["run_id"],
                        experiment_name=row["experiment_name"],
                        run_timestamp=row["run_timestamp"],
                        config_hash=row["config_hash"],
                        dataset_reference=row["dataset_reference"],
                        created_at=row["created_at"],
                        metric_count=row["metric_count"],
                        query_count=row["query_count"],
                        metrics=metrics,
                    )
                )
            return summaries

    def get_query_results(
        self, run_id: str, *, query_id: str | None = None
    ) -> list[dict[str, Any]]:
        """Retrieve per-query results and diagnostic payloads for a run."""
        sql = "SELECT result_json FROM query_results WHERE run_id = ? "
        params: list[Any] = [run_id]
        if query_id is not None:
            sql += "AND query_id = ? "
            params.append(query_id)
        sql += "ORDER BY query_id ASC;"

        with self._db.get_connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            return [json.loads(r["result_json"]) for r in rows]

    def get_query_metric_scores(
        self, run_id: str, *, metric_name: str | None = None
    ) -> dict[str, dict[str, float]]:
        """Retrieve individual metric scores grouped by query_id for a given run."""
        sql = "SELECT query_id, metric_name, score FROM query_metric_scores WHERE run_id = ? "
        params: list[Any] = [run_id]
        if metric_name is not None:
            sql += "AND metric_name = ? "
            params.append(metric_name)
        sql += "ORDER BY query_id ASC, metric_name ASC;"

        with self._db.get_connection() as conn:
            rows = conn.execute(sql, tuple(params)).fetchall()
            scores_by_query: dict[str, dict[str, float]] = {}
            for r in rows:
                qid = r["query_id"]
                if qid not in scores_by_query:
                    scores_by_query[qid] = {}
                scores_by_query[qid][r["metric_name"]] = float(r["score"])
            return scores_by_query

    def delete_run(self, run_id: str) -> bool:
        """Delete an experiment run and cascade delete its metrics and query records."""
        with self._db.transaction() as conn:
            cursor = conn.execute("DELETE FROM experiment_runs WHERE run_id = ?;", (run_id,))
            return cursor.rowcount > 0
