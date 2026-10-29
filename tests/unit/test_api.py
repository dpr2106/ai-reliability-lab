"""Unit tests for the AI Reliability Lab FastAPI adapter (UI-002).

Tests health checks, run listing and pagination, run details, regression comparison,
citation auditing, and error handling using hermetic SQLite test databases.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from reliability_lab.api.main import create_app
from reliability_lab.core.models import EvaluationReport
from reliability_lab.storage.repository import ExperimentRepository


@pytest.fixture
def test_db_path(tmp_path: Path) -> Path:
    """Provide a temporary database path."""
    return tmp_path / "test_api_experiments.sqlite3"


@pytest.fixture
def populated_db(test_db_path: Path) -> tuple[Path, str, str, str]:
    """Create a populated experiment database with baseline, pass, and regression runs."""
    base_report = EvaluationReport(
        report_id="run-baseline",
        experiment_name="bm25-baseline",
        metrics={"mrr@5": 0.80, "recall@5": 0.90},
        per_query_results=[
            {
                "query_id": "q1",
                "text": "what is information retrieval?",
                "metrics": {"mrr@5": 1.0, "recall@5": 1.0},
                "candidates": [{"chunk_id": "CHK-001", "score": 2.5, "rank": 1}],
                "answer": "Information retrieval finds relevant material [CIT-001].",
                "citation_mapping": {"CIT-001": "CHK-001"},
                "required_citations": ["CIT-001"],
            },
            {
                "query_id": "q2",
                "text": "vector search basics",
                "metrics": {"mrr@5": 0.60, "recall@5": 0.80},
                "candidates": [{"chunk_id": "CHK-002", "score": 1.8, "rank": 1}],
                "answer": "Vectors represent embeddings [CIT-002].",
                "citation_mapping": {"CIT-002": "CHK-002"},
                "required_citations": ["CIT-002"],
            },
        ],
        timestamp="2026-10-29T10:00:00Z",
        config={"retriever": "BM25", "top_k": 5},
    )

    pass_report = EvaluationReport(
        report_id="run-pass",
        experiment_name="bm25-candidate",
        metrics={"mrr@5": 0.82, "recall@5": 0.92},
        per_query_results=[
            {
                "query_id": "q1",
                "text": "what is information retrieval?",
                "metrics": {"mrr@5": 1.0, "recall@5": 1.0},
                "candidates": [{"chunk_id": "CHK-001", "score": 2.6, "rank": 1}],
            },
            {
                "query_id": "q2",
                "text": "vector search basics",
                "metrics": {"mrr@5": 0.64, "recall@5": 0.84},
                "candidates": [{"chunk_id": "CHK-002", "score": 1.9, "rank": 1}],
            },
        ],
        timestamp="2026-10-29T10:05:00Z",
        config={"retriever": "BM25", "top_k": 5},
    )

    reg_report = EvaluationReport(
        report_id="run-regressed",
        experiment_name="bm25-degraded",
        metrics={"mrr@5": 0.60, "recall@5": 0.70},
        per_query_results=[
            {
                "query_id": "q1",
                "text": "what is information retrieval?",
                "metrics": {"mrr@5": 0.50, "recall@5": 0.50},
                "candidates": [{"chunk_id": "CHK-001", "score": 1.0, "rank": 1}],
            },
            {
                "query_id": "q2",
                "text": "vector search basics",
                "metrics": {"mrr@5": 0.70, "recall@5": 0.90},
                "candidates": [{"chunk_id": "CHK-002", "score": 1.2, "rank": 1}],
            },
        ],
        timestamp="2026-10-29T10:10:00Z",
        config={"retriever": "BM25", "top_k": 5},
    )

    with ExperimentRepository(test_db_path) as repo:
        repo.save_report(base_report, dataset_reference="sample_benchmark.jsonl")
        repo.save_report(pass_report, dataset_reference="sample_benchmark.jsonl")
        repo.save_report(reg_report, dataset_reference="sample_benchmark.jsonl")

    return test_db_path, "run-baseline", "run-pass", "run-regressed"


@pytest.fixture
def client(populated_db: tuple[Path, str, str, str]) -> Generator[TestClient, None, None]:
    """TestClient configured to use populated test database via environment variable."""
    db_path = populated_db[0]
    old_env = os.environ.get("RELIABILITY_LAB_DB")
    os.environ["RELIABILITY_LAB_DB"] = str(db_path)
    try:
        app = create_app()
        with TestClient(app) as test_client:
            yield test_client
    finally:
        if old_env is not None:
            os.environ["RELIABILITY_LAB_DB"] = old_env
        else:
            os.environ.pop("RELIABILITY_LAB_DB", None)


class TestHealthEndpoint:
    """Tests for GET /api/health."""

    def test_health_with_populated_db(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        db_path = populated_db[0]
        response = client.get("/api/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["database"] == str(db_path.resolve())
        assert data["database_exists"] is True
        assert data["total_runs"] == 3

    def test_health_with_missing_db(self, tmp_path: Path) -> None:
        missing_db = tmp_path / "nonexistent.sqlite3"
        old_env = os.environ.get("RELIABILITY_LAB_DB")
        os.environ["RELIABILITY_LAB_DB"] = str(missing_db)
        try:
            app = create_app()
            with TestClient(app) as test_client:
                response = test_client.get("/api/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "ok"
                assert data["database_exists"] is False
                assert data["total_runs"] == 0
        finally:
            if old_env is not None:
                os.environ["RELIABILITY_LAB_DB"] = old_env
            else:
                os.environ.pop("RELIABILITY_LAB_DB", None)


class TestRunsEndpoint:
    """Tests for GET /api/runs."""

    def test_list_runs_success(self, client: TestClient) -> None:
        response = client.get("/api/runs")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["runs"]) == 3
        first = data["runs"][0]
        assert "run_id" in first
        assert "experiment_name" in first
        assert "metrics" in first

    def test_list_runs_filter_by_name(self, client: TestClient) -> None:
        response = client.get("/api/runs?experiment_name=bm25-baseline")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["runs"][0]["experiment_name"] == "bm25-baseline"

    def test_list_runs_pagination(self, client: TestClient) -> None:
        response = client.get("/api/runs?limit=2&offset=0")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 3
        assert len(data["runs"]) == 2


class TestRunDetailEndpoint:
    """Tests for GET /api/runs/{run_id}."""

    def test_get_run_success(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        baseline_id = populated_db[1]
        response = client.get(f"/api/runs/{baseline_id}")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == baseline_id
        assert data["experiment_name"] == "bm25-baseline"
        assert data["query_count"] == 2
        assert len(data["per_query_results"]) == 2
        assert data["config"]["retriever"] == "BM25"

    def test_get_run_not_found(self, client: TestClient) -> None:
        response = client.get("/api/runs/nonexistent-run-id")
        assert response.status_code == 404
        assert "not found" in response.json()["detail"]


class TestRunQueriesEndpoint:
    """Tests for GET /api/runs/{run_id}/queries."""

    def test_get_run_queries_success(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        baseline_id = populated_db[1]
        response = client.get(f"/api/runs/{baseline_id}/queries")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == baseline_id
        assert data["total"] == 2
        assert len(data["queries"]) == 2

    def test_get_run_queries_filter_query_id(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        baseline_id = populated_db[1]
        response = client.get(f"/api/runs/{baseline_id}/queries?query_id=q1")
        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 1
        assert data["queries"][0]["query_id"] == "q1"


class TestCompareEndpoint:
    """Tests for POST /api/runs/compare."""

    def test_compare_runs_pass(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        _, baseline, cand_pass, _ = populated_db
        response = client.post(
            "/api/runs/compare",
            json={
                "baseline_run_id": baseline,
                "candidate_run_id": cand_pass,
                "tolerance": 0.02,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "PASS"
        assert data["baseline_run_id"] == baseline
        assert data["candidate_run_id"] == cand_pass
        assert "aggregate_deltas" in data

    def test_compare_runs_regression(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        _, baseline, _, cand_reg = populated_db
        response = client.post(
            "/api/runs/compare",
            json={
                "baseline_run_id": baseline,
                "candidate_run_id": cand_reg,
                "tolerance": 0.02,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "REGRESSION"
        assert len(data["regressed_aggregate_metrics"]) > 0

    def test_compare_runs_nonexistent_run_400(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        _, baseline, _, _ = populated_db
        response = client.post(
            "/api/runs/compare",
            json={
                "baseline_run_id": baseline,
                "candidate_run_id": "ghost-run",
            },
        )
        assert response.status_code == 400
        assert "not found" in response.json()["detail"]

    def test_compare_runs_invalid_tolerance_422(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        _, baseline, cand_pass, _ = populated_db
        response = client.post(
            "/api/runs/compare",
            json={
                "baseline_run_id": baseline,
                "candidate_run_id": cand_pass,
                "tolerance": -0.05,
            },
        )
        assert response.status_code == 422


class TestCitationAuditEndpoint:
    """Tests for GET /api/runs/{run_id}/citation-audit."""

    def test_citation_audit_with_answers(
        self, client: TestClient, populated_db: tuple[Path, str, str, str]
    ) -> None:
        baseline_id = populated_db[1]
        response = client.get(f"/api/runs/{baseline_id}/citation-audit")
        assert response.status_code == 200
        data = response.json()
        assert data["run_id"] == baseline_id
        assert data["total_queries"] == 2
        assert data["audited_queries_count"] == 2
        assert data["mean_citation_precision"] == 1.0
        assert data["mean_citation_recall"] == 1.0
        assert len(data["queries"]) == 2
        first = data["queries"][0]
        assert first["query_id"] == "q1"
        assert first["valid_citations"] == ["CIT-001"]
        assert "Citation validity measures" in data["limitations_note"]

    def test_citation_audit_not_found(self, client: TestClient) -> None:
        response = client.get("/api/runs/nonexistent-run/citation-audit")
        assert response.status_code == 404
