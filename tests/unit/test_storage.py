"""Unit tests for SQLite database lifecycle, schema integrity, and experiment repository."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from reliability_lab.core.models import EvaluationReport
from reliability_lab.storage import (
    ConfigurationConflictError,
    Database,
    DuplicateRunError,
    ExperimentRepository,
    StoredRun,
    canonicalize_config,
    compute_config_hash,
)


def _make_sample_report(
    report_id: str = "run-001",
    experiment_name: str = "bm25_baseline",
    metrics: dict[str, float] | None = None,
    per_query_results: list[dict[str, Any]] | None = None,
    config: dict[str, Any] | None = None,
    timestamp: str = "2026-10-28T12:00:00+00:00",
) -> EvaluationReport:
    """Helper to construct a deterministic, valid EvaluationReport."""
    default_metrics = {
        "recall@5": 0.85,
        "mrr@5": 0.72,
        "ndcg@5": 0.78,
    }
    default_queries = [
        {
            "query_id": "Q-001",
            "text": "What is Okapi BM25 default k1?",
            "latency_ms": 1.25,
            "metrics": {"recall@5": 1.0, "mrr@5": 1.0},
            "retrieved": [{"chunk_id": "CHK-001", "rank": 1, "score": 4.5}],
            "failure_trace": {"has_failure": False},
        },
        {
            "query_id": "Q-002",
            "text": "How does sliding window chunking work?",
            "latency_ms": 2.10,
            "metrics": {"recall@5": 0.7, "mrr@5": 0.5},
            "retrieved": [{"chunk_id": "CHK-002", "rank": 1, "score": 3.8}],
            "failure_trace": {"has_failure": True},
        },
    ]
    default_config = {
        "retriever": "bm25",
        "top_k": 5,
        "k1": 1.5,
        "b": 0.75,
    }
    return EvaluationReport(
        experiment_name=experiment_name,
        metrics=metrics if metrics is not None else default_metrics,
        report_id=report_id,
        per_query_results=per_query_results if per_query_results is not None else default_queries,
        timestamp=timestamp,
        config=config if config is not None else default_config,
    )


@pytest.fixture
def repo() -> Generator[ExperimentRepository, None, None]:
    """Yield a transient in-memory ExperimentRepository with guaranteed resource cleanup."""
    repository = ExperimentRepository(":memory:")
    try:
        yield repository
    finally:
        repository.close()


# ===========================================================================
# 1. Database Lifecycle & Schema Tests
# ===========================================================================


def test_database_creation_in_memory() -> None:
    """Test in-memory database creation and table presence."""
    db = Database(":memory:")
    assert db.is_memory is True
    assert db.path == ":memory:"

    with db.get_connection() as conn:
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = {row[0] for row in cursor.fetchall()}
        expected = {
            "index_configurations",
            "experiment_runs",
            "run_metrics",
            "query_results",
            "query_metric_scores",
        }
        assert expected.issubset(tables)

    db.close()


def test_database_idempotent_schema_initialization() -> None:
    """Test calling initialize_schema multiple times is safe and preserves data."""
    db = Database(":memory:")
    db.initialize_schema()
    db.initialize_schema()

    with db.get_connection() as conn:
        ver = conn.execute("PRAGMA user_version;").fetchone()[0]
        assert ver == 1
    db.close()


def test_database_unsupported_future_schema_version() -> None:
    """Test database rejects newer unsupported schema versions."""
    db = Database(":memory:")
    with db.get_connection() as conn:
        conn.execute("PRAGMA user_version = 99;")

    with pytest.raises(RuntimeError, match="Unsupported newer SQLite schema version 99"):
        db.initialize_schema()
    db.close()


def test_database_context_manager_and_closure() -> None:
    """Test Database context manager clean closure."""
    with Database(":memory:") as db:
        assert db.is_memory
    with pytest.raises(RuntimeError, match="Database connection has been closed"):
        with db.get_connection():
            pass


def test_file_backed_database_persistence(tmp_path: Path) -> None:
    """Test that file-backed database persists tables and rows across separate connections."""
    db_file = tmp_path / "experiments.db"
    db1 = Database(db_file)
    repo1 = ExperimentRepository(db1)

    report = _make_sample_report(report_id="run-persist-1")
    repo1.save_report(report)
    db1.close()

    # Open with a brand new Database instance pointing to same file
    db2 = Database(db_file)
    repo2 = ExperimentRepository(db2)
    stored = repo2.get_run("run-persist-1")
    assert stored is not None
    assert stored.run_id == "run-persist-1"
    assert stored.experiment_name == "bm25_baseline"
    assert len(stored.metrics) == 3
    db2.close()


def test_foreign_key_enforcement() -> None:
    """Test that SQLite foreign key constraints are actively enforced."""
    db = Database(":memory:")
    with db.get_connection() as conn:
        # Attempting to insert an experiment_run with non-existent config_hash must fail
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO experiment_runs ("
                "run_id, experiment_name, run_timestamp, config_hash, created_at"
                ") VALUES ('run-x', 'exp-x', '2026-10-28', 'non_existent_hash', '2026-10-28');"
            )
    db.close()


def test_transaction_rollback_on_error() -> None:
    """Test that errors within a transaction trigger an atomic rollback."""
    db = Database(":memory:")

    with pytest.raises(ValueError, match="Simulated mid-write error"):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO index_configurations (config_hash, config_json, created_at) "
                "VALUES ('h1', '{}', '2026-10-28');"
            )
            # Simulate failure midway
            raise ValueError("Simulated mid-write error")

    # Verify 'h1' was rolled back completely
    with db.get_connection() as conn:
        row = conn.execute(
            "SELECT config_hash FROM index_configurations WHERE config_hash = 'h1';"
        ).fetchone()
        assert row is None
    db.close()


# ===========================================================================
# 2. Configuration Canonicalization & Hashing Tests
# ===========================================================================


def test_canonicalize_config_key_order_invariance() -> None:
    """Test that dicts with different key order yield identical canonical strings."""
    cfg1 = {"z": 1, "a": {"sub_b": 2, "sub_a": 1}, "m": [3, 2, 1]}
    cfg2 = {"a": {"sub_a": 1, "sub_b": 2}, "m": [3, 2, 1], "z": 1}

    canon1 = canonicalize_config(cfg1)
    canon2 = canonicalize_config(cfg2)
    assert canon1 == canon2
    assert compute_config_hash(cfg1) == compute_config_hash(cfg2)


def test_canonicalize_config_unicode_and_separators() -> None:
    """Test compact separators and unicode preservation in configuration."""
    cfg = {"name": "bm25_test", "desc": "Évaluation de fidélité"}
    canon = canonicalize_config(cfg)
    assert ',"desc":' in canon or '{"desc":' in canon
    assert "Évaluation" in canon


def test_canonicalize_config_rejection_of_non_finite_floats() -> None:
    """Test rejection of NaN and Inf floats in configurations."""
    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_config({"k1": float("nan")})

    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_config({"k1": float("inf")})

    with pytest.raises(ValueError, match="non-finite"):
        canonicalize_config({"k1": float("-inf")})


def test_canonicalize_config_type_errors() -> None:
    """Test rejection of non-mapping inputs or un-serializable objects."""
    with pytest.raises(TypeError, match="must be a Mapping"):
        canonicalize_config(["not", "a", "dict"])  # type: ignore[arg-type]

    class Unserializable:
        pass

    with pytest.raises(TypeError, match="Un-serializable object"):
        canonicalize_config({"obj": Unserializable()})

    with pytest.raises(TypeError, match="keys must be strings"):
        canonicalize_config({123: "val"})  # type: ignore[dict-item]


def test_compute_config_hash_format() -> None:
    """Test hash format is a valid 64-character lowercase hex string."""
    cfg = {"retriever": "bm25", "k1": 1.5}
    h = compute_config_hash(cfg)
    assert len(h) == 64
    assert h == h.lower()
    assert all(c in "0123456789abcdef" for c in h)


# ===========================================================================
# 3. Repository Save & Retrieve Tests
# ===========================================================================


def test_save_and_get_run_roundtrip(repo: ExperimentRepository) -> None:
    """Test complete persistence and reconstruction of EvaluationReport."""
    report = _make_sample_report(
        report_id="test-run-123",
        experiment_name="hybrid_rrf_exp",
    )

    saved_id = repo.save_report(report, dataset_reference="benchmark_v1.jsonl")
    assert saved_id == "test-run-123"

    stored = repo.get_run("test-run-123")
    assert stored is not None
    assert isinstance(stored, StoredRun)
    assert stored.run_id == "test-run-123"
    assert stored.experiment_name == "hybrid_rrf_exp"
    assert stored.dataset_reference == "benchmark_v1.jsonl"
    assert stored.run_timestamp == "2026-10-28T12:00:00+00:00"
    assert stored.metrics["recall@5"] == pytest.approx(0.85)
    assert stored.metrics["mrr@5"] == pytest.approx(0.72)
    assert stored.metrics["ndcg@5"] == pytest.approx(0.78)

    # Verify per-query results reconstruction
    assert len(stored.per_query_results) == 2
    q1 = stored.per_query_results[0]
    assert q1["query_id"] == "Q-001"
    assert q1["latency_ms"] == pytest.approx(1.25)
    assert q1["metrics"]["recall@5"] == pytest.approx(1.0)
    assert q1["failure_trace"]["has_failure"] is False

    # Verify reconstructed EvaluationReport matches
    reconstructed_report = stored.to_report()
    assert isinstance(reconstructed_report, EvaluationReport)
    assert reconstructed_report.report_id == "test-run-123"
    assert reconstructed_report.experiment_name == "hybrid_rrf_exp"
    assert reconstructed_report.metrics == stored.metrics
    assert reconstructed_report.config == stored.config


def test_get_non_existent_run(repo: ExperimentRepository) -> None:
    """Test get_run returns None for unknown run IDs."""
    assert repo.get_run("unknown-run") is None


def test_get_run_invalid_id(repo: ExperimentRepository) -> None:
    """Test get_run raises ValueError for empty or non-string IDs."""
    with pytest.raises(ValueError, match="must be a non-empty string"):
        repo.get_run("")
    with pytest.raises(ValueError, match="must be a non-empty string"):
        repo.get_run("   ")


def test_configuration_deduplication_across_runs(repo: ExperimentRepository) -> None:
    """Test that runs sharing identical configs reuse the same index_configuration row."""
    shared_config = {"retriever": "bm25", "k1": 1.5, "b": 0.75}

    report1 = _make_sample_report(report_id="run-1", config=shared_config)
    report2 = _make_sample_report(report_id="run-2", config=shared_config)

    repo.save_report(report1)
    repo.save_report(report2)

    with repo.database.get_connection() as conn:
        rows = conn.execute("SELECT COUNT(*) as cnt FROM index_configurations;").fetchone()
        assert rows["cnt"] == 1

        run_rows = conn.execute("SELECT run_id, config_hash FROM experiment_runs;").fetchall()
        assert len(run_rows) == 2
        assert run_rows[0]["config_hash"] == run_rows[1]["config_hash"]


def test_save_report_with_index_config_override(repo: ExperimentRepository) -> None:
    """Test supplying explicit index_config merges with report configuration."""
    report = _make_sample_report(report_id="run-override", config={"base_param": 1})

    repo.save_report(report, index_config={"index_type": "IndexFlatIP", "dim": 384})
    stored = repo.get_run("run-override")
    assert stored is not None
    assert stored.config["base_param"] == 1
    assert stored.config["index_type"] == "IndexFlatIP"
    assert stored.config["dim"] == 384


# ===========================================================================
# 4. Duplicate Run & Conflict Semantics
# ===========================================================================


def test_idempotent_duplicate_run_save(repo: ExperimentRepository) -> None:
    """Test saving identical run contents twice is idempotent and returns run_id without error."""
    report = _make_sample_report(report_id="run-idemp-1")

    id1 = repo.save_report(report, dataset_reference="ds-ref")
    id2 = repo.save_report(report, dataset_reference="ds-ref")
    assert id1 == id2 == "run-idemp-1"

    # Verify only 1 run row exists
    with repo.database.get_connection() as conn:
        count = conn.execute(
            "SELECT COUNT(*) as cnt FROM experiment_runs WHERE run_id = 'run-idemp-1';"
        ).fetchone()["cnt"]
        assert count == 1


def test_conflicting_duplicate_run_id_raises_error(repo: ExperimentRepository) -> None:
    """Test saving same run_id with different metrics raises DuplicateRunError."""
    report1 = _make_sample_report(report_id="run-conflict-1", metrics={"recall@5": 0.5})
    repo.save_report(report1)

    report2 = _make_sample_report(report_id="run-conflict-1", metrics={"recall@5": 0.99})
    with pytest.raises(DuplicateRunError, match="already exists with conflicting"):
        repo.save_report(report2)


def test_configuration_hash_collision_detection(repo: ExperimentRepository) -> None:
    """Test that a simulated configuration hash collision raises ConfigurationConflictError."""
    report1 = _make_sample_report(report_id="run-hash-1", config={"param": "A"})
    repo.save_report(report1)

    report2 = _make_sample_report(report_id="run-hash-2", config={"param": "B"})

    # Mock compute_config_hash to force same hash despite different json
    with patch(
        "reliability_lab.storage.repository.compute_config_hash",
        return_value=compute_config_hash({"param": "A"}),
    ):
        with pytest.raises(ConfigurationConflictError, match="hash collision detected"):
            repo.save_report(report2)


# ===========================================================================
# 5. Listing Runs & Filtering
# ===========================================================================


def test_list_runs_deterministic_ordering_and_filtering(repo: ExperimentRepository) -> None:
    """Test listing runs sorted deterministically by timestamp DESC, run_id ASC."""
    r1 = _make_sample_report(
        report_id="run-old",
        experiment_name="baseline_bm25",
        timestamp="2026-10-28T10:00:00+00:00",
    )
    r2 = _make_sample_report(
        report_id="run-new-a",
        experiment_name="hybrid_rrf",
        timestamp="2026-10-28T12:00:00+00:00",
    )
    r3 = _make_sample_report(
        report_id="run-new-b",
        experiment_name="baseline_bm25",
        timestamp="2026-10-28T12:00:00+00:00",
    )

    repo.save_report(r1)
    repo.save_report(r2)
    repo.save_report(r3)

    all_runs = repo.list_runs()
    assert len(all_runs) == 3
    # Timestamp 12:00:00 runs come first; tie broken by run_id ASC ("run-new-a" < "run-new-b")
    assert all_runs[0].run_id == "run-new-a"
    assert all_runs[1].run_id == "run-new-b"
    assert all_runs[2].run_id == "run-old"

    # Filter by experiment_name
    bm25_runs = repo.list_runs(experiment_name="baseline_bm25")
    assert len(bm25_runs) == 2
    assert [r.run_id for r in bm25_runs] == ["run-new-b", "run-old"]

    # Limit and offset
    limited = repo.list_runs(limit=1, offset=1)
    assert len(limited) == 1
    assert limited[0].run_id == "run-new-b"


def test_list_runs_invalid_limit(repo: ExperimentRepository) -> None:
    """Test negative limit raises ValueError."""
    with pytest.raises(ValueError, match="limit must be non-negative"):
        repo.list_runs(limit=-5)


# ===========================================================================
# 6. Granular Query Results & Metric Scores Retrieval
# ===========================================================================


def test_get_query_results_and_scores(repo: ExperimentRepository) -> None:
    """Test granular retrieval of per-query records and individual metric scores."""
    report = _make_sample_report(report_id="run-granular-1")
    repo.save_report(report)

    # Retrieve all queries
    all_q = repo.get_query_results("run-granular-1")
    assert len(all_q) == 2
    assert all_q[0]["query_id"] == "Q-001"
    assert all_q[1]["query_id"] == "Q-002"

    # Retrieve specific query
    q1 = repo.get_query_results("run-granular-1", query_id="Q-001")
    assert len(q1) == 1
    assert q1[0]["text"] == "What is Okapi BM25 default k1?"

    # Retrieve query metric scores
    scores_by_q = repo.get_query_metric_scores("run-granular-1")
    assert scores_by_q["Q-001"]["recall@5"] == pytest.approx(1.0)
    assert scores_by_q["Q-001"]["mrr@5"] == pytest.approx(1.0)
    assert scores_by_q["Q-002"]["recall@5"] == pytest.approx(0.7)
    assert scores_by_q["Q-002"]["mrr@5"] == pytest.approx(0.5)

    # Filter metric scores by metric_name
    mrr_only = repo.get_query_metric_scores("run-granular-1", metric_name="mrr@5")
    assert set(mrr_only["Q-001"].keys()) == {"mrr@5"}
    assert mrr_only["Q-001"]["mrr@5"] == pytest.approx(1.0)


def test_delete_run_cascades_foreign_keys(repo: ExperimentRepository) -> None:
    """Test deleting a run cascades deletion across run_metrics and query records."""
    report = _make_sample_report(report_id="run-del-1")
    repo.save_report(report)

    assert repo.get_run("run-del-1") is not None
    assert repo.delete_run("run-del-1") is True
    assert repo.get_run("run-del-1") is None

    with repo.database.get_connection() as conn:
        assert (
            conn.execute(
                "SELECT COUNT(*) as c FROM run_metrics WHERE run_id='run-del-1';"
            ).fetchone()["c"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) as c FROM query_results WHERE run_id='run-del-1';"
            ).fetchone()["c"]
            == 0
        )
        assert (
            conn.execute(
                "SELECT COUNT(*) as c FROM query_metric_scores WHERE run_id='run-del-1';"
            ).fetchone()["c"]
            == 0
        )

    assert repo.delete_run("run-del-1") is False


# ===========================================================================
# 7. Input Validation & Edge Cases
# ===========================================================================


def test_save_report_rejects_non_evaluation_report(repo: ExperimentRepository) -> None:
    """Test save_report rejects objects that are not EvaluationReport instances."""
    with pytest.raises(TypeError, match="must be an EvaluationReport"):
        repo.save_report({"not": "a report"})  # type: ignore[arg-type]


def test_save_report_rejects_boolean_scores(repo: ExperimentRepository) -> None:
    """Test save_report rejects booleans masquerading as numerical floats."""
    report = _make_sample_report(report_id="run-bool-1")
    # In Python, isinstance(True, int) is True, so explicit boolean check is required
    report.metrics["recall@5"] = True  # type: ignore[assignment]

    with pytest.raises(ValueError, match="must be a finite numerical value"):
        repo.save_report(report)


def test_save_report_rejects_non_finite_metric_scores(repo: ExperimentRepository) -> None:
    """Test save_report rejects NaN and Inf aggregate scores."""
    report = _make_sample_report(report_id="run-nan-1")
    report.metrics["recall@5"] = float("nan")

    with pytest.raises(ValueError, match="must be a finite numerical value"):
        repo.save_report(report)


def test_save_report_rejects_duplicate_query_ids_in_results(repo: ExperimentRepository) -> None:
    """Test save_report rejects reports with duplicate query_ids in per_query_results."""
    dupe_queries = [
        {"query_id": "Q-001", "metrics": {"m": 1.0}},
        {"query_id": "Q-001", "metrics": {"m": 0.5}},
    ]
    report = _make_sample_report(report_id="run-dupe-q", per_query_results=dupe_queries)

    with pytest.raises(ValueError, match="Duplicate query_id 'Q-001'"):
        repo.save_report(report)


def test_save_report_parameterized_sql_injection_safety(repo: ExperimentRepository) -> None:
    """Test SQL injection strings in experiment_name or IDs are safely handled."""
    nasty_name = "test'; DROP TABLE experiment_runs; --"
    report = _make_sample_report(report_id="run-sql-inj", experiment_name=nasty_name)

    repo.save_report(report)
    stored = repo.get_run("run-sql-inj")
    assert stored is not None
    assert stored.experiment_name == nasty_name

    # Confirm table was not dropped!
    with repo.database.get_connection() as conn:
        assert conn.execute("SELECT COUNT(*) as c FROM experiment_runs;").fetchone()["c"] == 1


def test_stored_run_to_dict_and_summary_to_dict(repo: ExperimentRepository) -> None:
    """Test to_dict serialization of StoredRun and StoredRunSummary."""
    report = _make_sample_report(report_id="run-dict-1")
    repo.save_report(report)

    stored = repo.get_run("run-dict-1")
    assert stored is not None
    d = stored.to_dict()
    assert d["run_id"] == "run-dict-1"
    assert "metrics" in d
    assert "config" in d

    summaries = repo.list_runs()
    assert len(summaries) == 1
    sd = summaries[0].to_dict()
    assert sd["run_id"] == "run-dict-1"
    assert sd["query_count"] == 2
    assert sd["metric_count"] == 3
