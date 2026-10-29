"""Unit tests for the headless command-line interface (UI-001).

Tests argument parsing, help diagnostics, benchmark execution, SQLite persistence,
regression detection gating, machine-readable JSON output, and deterministic process exit codes.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from reliability_lab.cli.main import (
    EXIT_INCOMPLETE,
    EXIT_OPERATIONAL_ERROR,
    EXIT_PASS,
    EXIT_REGRESSION,
    _parse_metric_tolerances,
    main,
)
from reliability_lab.core.models import Chunk, EvaluationReport, Query
from reliability_lab.storage.repository import ExperimentRepository

REPO_ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CORPUS = REPO_ROOT / "data" / "benchmarks" / "sample_corpus.jsonl"
SAMPLE_BENCHMARK = REPO_ROOT / "data" / "benchmarks" / "sample_benchmark.jsonl"


@pytest.fixture
def mini_corpus_and_dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Create a minimal hermetic corpus and dataset for quick deterministic CLI testing."""
    corpus_file = tmp_path / "mini_corpus.jsonl"
    dataset_file = tmp_path / "mini_dataset.jsonl"

    chunks = [
        Chunk(
            chunk_id="CHK-001",
            doc_id="doc-1",
            content=(
                "Information retrieval metrics include precision, recall, and mean reciprocal rank."
            ),
            page_number=1,
            char_start=0,
            char_end=82,
        ),
        Chunk(
            chunk_id="CHK-002",
            doc_id="doc-2",
            content="Vector embeddings and FAISS index flat inner product provide dense retrieval.",
            page_number=1,
            char_start=0,
            char_end=77,
        ),
    ]
    with corpus_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.to_dict()) + "\n")

    queries = [
        Query(
            query_id="Q-001",
            text="information retrieval metrics",
            relevant_chunk_ids=["CHK-001"],
            relevance_grades={"CHK-001": 2},
        ),
        Query(
            query_id="Q-002",
            text="vector embeddings retrieval",
            relevant_chunk_ids=["CHK-002"],
            relevance_grades={"CHK-002": 2},
        ),
    ]
    with dataset_file.open("w", encoding="utf-8") as f:
        for q in queries:
            f.write(json.dumps(q.to_dict()) + "\n")

    return corpus_file, dataset_file


class TestArgumentParsingAndHelp:
    """Tests for argument parsing, subcommand help outputs, and validation."""

    def test_no_arguments_returns_operational_error(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        exit_code = main([])
        assert exit_code == EXIT_OPERATIONAL_ERROR
        captured = capsys.readouterr()
        assert "usage: reliability-lab" in captured.err

    def test_root_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "run-benchmark" in captured.out
        assert "assert-regression" in captured.out

    def test_run_benchmark_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["run-benchmark", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--corpus" in captured.out
        assert "--dataset" in captured.out
        assert "--database" in captured.out
        assert "--top-k" in captured.out

    def test_assert_regression_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["assert-regression", "--help"])
        assert exc_info.value.code == 0
        captured = capsys.readouterr()
        assert "--baseline" in captured.out
        assert "--candidate" in captured.out
        assert "--tolerance" in captured.out

    def test_assert_regression_missing_required_args(self) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main(["assert-regression"])
        # argparse standard usage error is code 2
        assert exc_info.value.code == 2

    def test_parse_metric_tolerances_valid(self) -> None:
        result = _parse_metric_tolerances(["mrr@5=0.01", "recall@5=0.05"])
        assert result == {"mrr@5": 0.01, "recall@5": 0.05}

    def test_parse_metric_tolerances_empty(self) -> None:
        assert _parse_metric_tolerances(None) == {}
        assert _parse_metric_tolerances([]) == {}

    def test_parse_metric_tolerances_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Expected format: metric=value"):
            _parse_metric_tolerances(["invalid_without_equals"])

    def test_parse_metric_tolerances_non_numeric(self) -> None:
        with pytest.raises(ValueError, match="must be a float"):
            _parse_metric_tolerances(["mrr@5=abc"])

    def test_parse_metric_tolerances_negative_or_nan(self) -> None:
        with pytest.raises(ValueError, match="finite non-negative"):
            _parse_metric_tolerances(["mrr@5=-0.05"])
        with pytest.raises(ValueError, match="finite non-negative"):
            _parse_metric_tolerances(["mrr@5=nan"])


class TestRunBenchmarkCommand:
    """Tests for the `run-benchmark` command."""

    def test_run_benchmark_success_text(
        self,
        mini_corpus_and_dataset: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus_path, dataset_path = mini_corpus_and_dataset
        db_path = tmp_path / "experiments.sqlite3"

        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(corpus_path),
                "--dataset",
                str(dataset_path),
                "--database",
                str(db_path),
                "--top-k",
                "5",
                "--experiment-name",
                "test-run",
                "--output",
                "text",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        assert "Benchmark Run Completed: test-run" in captured.out
        assert "Run ID:" in captured.out
        assert "recall@5" in captured.out

        # Verify persisted run
        with ExperimentRepository(db_path) as repo:
            runs = repo.list_runs()
            assert len(runs) == 1
            assert runs[0].experiment_name == "test-run"

    def test_run_benchmark_success_json(
        self,
        mini_corpus_and_dataset: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus_path, dataset_path = mini_corpus_and_dataset
        db_path = tmp_path / "experiments.sqlite3"

        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(corpus_path),
                "--dataset",
                str(dataset_path),
                "--database",
                str(db_path),
                "--top-k",
                "3",
                "--experiment-name",
                "json-test",
                "--output",
                "json",
            ]
        )

        assert exit_code == 0
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "SUCCESS"
        assert data["experiment_name"] == "json-test"
        assert data["query_count"] == 2
        assert "run_id" in data
        assert "metrics" in data
        assert "mrr@3" in data["metrics"]

    def test_run_benchmark_paths_with_spaces(
        self,
        mini_corpus_and_dataset: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus_path, dataset_path = mini_corpus_and_dataset
        spaced_dir = tmp_path / "folder with spaces"
        spaced_dir.mkdir()
        db_path = spaced_dir / "my experiments.sqlite3"

        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(corpus_path),
                "--dataset",
                str(dataset_path),
                "--database",
                str(db_path),
                "--experiment-name",
                "space-test",
            ]
        )

        assert exit_code == 0
        assert db_path.is_file()

    def test_run_benchmark_invalid_top_k(
        self,
        mini_corpus_and_dataset: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus_path, dataset_path = mini_corpus_and_dataset
        db_path = tmp_path / "test.db"

        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(corpus_path),
                "--dataset",
                str(dataset_path),
                "--database",
                str(db_path),
                "--top-k",
                "0",
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "positive integer >= 1" in captured.err

    def test_run_benchmark_empty_experiment_name(
        self,
        mini_corpus_and_dataset: tuple[Path, Path],
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        corpus_path, dataset_path = mini_corpus_and_dataset
        db_path = tmp_path / "test.db"

        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(corpus_path),
                "--dataset",
                str(dataset_path),
                "--database",
                str(db_path),
                "--experiment-name",
                "   ",
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "non-empty string" in captured.err

    def test_run_benchmark_missing_corpus(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(tmp_path / "nonexistent.jsonl"),
                "--dataset",
                str(SAMPLE_BENCHMARK),
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Corpus file not found" in captured.err

    def test_run_benchmark_missing_dataset(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(SAMPLE_CORPUS),
                "--dataset",
                str(tmp_path / "nonexistent.jsonl"),
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Benchmark dataset file not found" in captured.err

    def test_run_benchmark_malformed_jsonl(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        bad_corpus = tmp_path / "bad.jsonl"
        bad_corpus.write_text("{invalid json", encoding="utf-8")

        exit_code = main(
            [
                "run-benchmark",
                "--corpus",
                str(bad_corpus),
                "--dataset",
                str(SAMPLE_BENCHMARK),
            ]
        )
        assert exit_code == 1
        captured = capsys.readouterr()
        assert "Error loading corpus" in captured.err


class TestAssertRegressionCommand:
    """Tests for the `assert-regression` command and CI exit codes."""

    @pytest.fixture
    def populated_db(self, tmp_path: Path) -> tuple[Path, str, str, str]:
        """Populate database with baseline, identical (pass), regressed, and subset runs."""
        db_path = tmp_path / "experiments.sqlite3"

        base_report = EvaluationReport(
            report_id="run-baseline",
            experiment_name="baseline-experiment",
            metrics={"mrr@5": 0.80, "recall@5": 0.90},
            per_query_results=[
                {
                    "query_id": "q1",
                    "text": "test query 1",
                    "metrics": {"mrr@5": 1.0, "recall@5": 1.0},
                },
                {
                    "query_id": "q2",
                    "text": "test query 2",
                    "metrics": {"mrr@5": 0.60, "recall@5": 0.80},
                },
            ],
            timestamp="2026-10-29T10:00:00Z",
        )

        pass_report = EvaluationReport(
            report_id="run-pass",
            experiment_name="candidate-pass",
            metrics={"mrr@5": 0.82, "recall@5": 0.91},
            per_query_results=[
                {
                    "query_id": "q1",
                    "text": "test query 1",
                    "metrics": {"mrr@5": 1.0, "recall@5": 1.0},
                },
                {
                    "query_id": "q2",
                    "text": "test query 2",
                    "metrics": {"mrr@5": 0.64, "recall@5": 0.82},
                },
            ],
            timestamp="2026-10-29T10:05:00Z",
        )

        regressed_report = EvaluationReport(
            report_id="run-regressed",
            experiment_name="candidate-regressed",
            metrics={"mrr@5": 0.60, "recall@5": 0.70},  # drop of 0.20 > tolerance 0.02
            per_query_results=[
                {
                    "query_id": "q1",
                    "text": "test query 1",
                    "metrics": {"mrr@5": 0.5, "recall@5": 0.5},
                },
                {
                    "query_id": "q2",
                    "text": "test query 2",
                    "metrics": {"mrr@5": 0.7, "recall@5": 0.9},
                },
            ],
            timestamp="2026-10-29T10:10:00Z",
        )

        incomplete_report = EvaluationReport(
            report_id="run-incomplete",
            experiment_name="candidate-incomplete",
            metrics={"mrr@5": 1.0, "recall@5": 1.0},
            per_query_results=[
                {
                    "query_id": "q1",
                    "text": "test query 1",
                    "metrics": {"mrr@5": 1.0, "recall@5": 1.0},
                },
                # Missing q2
            ],
            timestamp="2026-10-29T10:15:00Z",
        )

        with ExperimentRepository(db_path) as repo:
            repo.save_report(base_report)
            repo.save_report(pass_report)
            repo.save_report(regressed_report)
            repo.save_report(incomplete_report)

        return db_path, "run-baseline", "run-pass", "run-regressed"

    def test_assert_regression_pass_exit_0(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, candidate_pass, _ = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                candidate_pass,
                "--database",
                str(db_path),
                "--tolerance",
                "0.02",
                "--output",
                "text",
            ]
        )

        assert exit_code == EXIT_PASS
        captured = capsys.readouterr()
        assert "Regression Analysis: PASS" in captured.out
        assert "[OK]" in captured.out

    def test_assert_regression_regression_exit_1(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, _, candidate_reg = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                candidate_reg,
                "--database",
                str(db_path),
                "--tolerance",
                "0.02",
                "--output",
                "text",
            ]
        )

        assert exit_code == EXIT_REGRESSION
        captured = capsys.readouterr()
        assert "Regression Analysis: REGRESSION" in captured.out
        assert "[REGRESSION]" in captured.out
        assert "mrr@5" in captured.out

    def test_assert_regression_incomplete_exit_2(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, _, _ = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                "run-incomplete",
                "--database",
                str(db_path),
                "--tolerance",
                "0.02",
            ]
        )

        assert exit_code == EXIT_INCOMPLETE
        captured = capsys.readouterr()
        assert "Regression Analysis: INCOMPLETE" in captured.out

    def test_assert_regression_json_output(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, candidate_pass, _ = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                candidate_pass,
                "--database",
                str(db_path),
                "--output",
                "json",
            ]
        )

        assert exit_code == EXIT_PASS
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["status"] == "PASS"
        assert data["baseline_run_id"] == baseline
        assert data["candidate_run_id"] == candidate_pass
        assert "aggregate_deltas" in data

    def test_assert_regression_metric_override(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, candidate_pass, _ = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                candidate_pass,
                "--database",
                str(db_path),
                "--metric-tolerance",
                "mrr@5=0.001",
                "--metric-tolerance",
                "recall@5=0.005",
            ]
        )
        # Even with tight tolerance, pass run actually improved, so status is PASS
        assert exit_code == EXIT_PASS

    def test_assert_regression_missing_run_id_operational_error(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, _, _ = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                "nonexistent-run",
                "--database",
                str(db_path),
            ]
        )

        assert exit_code == EXIT_OPERATIONAL_ERROR
        captured = capsys.readouterr()
        assert "not found in repository" in captured.err

    def test_assert_regression_nonexistent_database(
        self,
        tmp_path: Path,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                "run-1",
                "--candidate",
                "run-2",
                "--database",
                str(tmp_path / "nonexistent.db"),
            ]
        )

        assert exit_code == EXIT_OPERATIONAL_ERROR
        captured = capsys.readouterr()
        assert "database does not exist" in captured.err

    def test_assert_regression_invalid_tolerance(
        self,
        populated_db: tuple[Path, str, str, str],
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        db_path, baseline, candidate_pass, _ = populated_db

        exit_code = main(
            [
                "assert-regression",
                "--baseline",
                baseline,
                "--candidate",
                candidate_pass,
                "--database",
                str(db_path),
                "--tolerance",
                "-0.05",
            ]
        )

        assert exit_code == EXIT_OPERATIONAL_ERROR
        captured = capsys.readouterr()
        assert "finite non-negative number" in captured.err


class TestExecutableInvocation:
    """Tests invoking the module via python -m."""

    def test_python_module_invocation(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "reliability_lab.cli.main", "--help"],
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
            check=False,
        )
        assert result.returncode == 0
        assert "AI Reliability Lab — Headless CLI" in result.stdout
