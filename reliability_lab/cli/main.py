"""Headless command-line interface for benchmarking and regression gating.

Provides two primary commands:
- `run-benchmark`: executes BM25 retrieval over golden benchmark datasets and persists
  EvaluationReport records into a relational SQLite experiment database.
- `assert-regression`: evaluates performance changes between baseline and candidate
  experiment runs using the regression detection engine, outputting deterministic exit codes
  for continuous integration gating.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

from reliability_lab.core.models import Chunk, EvaluationReport, Query
from reliability_lab.evaluation.benchmark_runner import (
    RetrievalBenchmarkRunner,
    load_benchmark_dataset,
    load_corpus,
)
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.storage.regression import (
    DEFAULT_TOLERANCE,
    IncomparableRunsError,
    InvalidToleranceError,
    RegressionDetector,
    RegressionResult,
    RegressionStatus,
)
from reliability_lab.storage.repository import (
    ExperimentRepository,
    RunNotFoundError,
    StorageError,
)

# Canonical CLI Exit Codes
EXIT_PASS: int = 0
EXIT_REGRESSION: int = 1
EXIT_INCOMPLETE: int = 2
EXIT_OPERATIONAL_ERROR: int = 3

DEFAULT_CORPUS_REL: str = "data/benchmarks/sample_corpus.jsonl"
DEFAULT_DATASET_REL: str = "data/benchmarks/sample_benchmark.jsonl"
DEFAULT_DATABASE_PATH: str = ".reliability-lab/experiments.sqlite3"


def _resolve_file_path(path_str: str) -> Path:
    """Resolve a file path predictably, checking both CWD and repository root."""
    p = Path(path_str)
    if p.is_file():
        return p.resolve()

    # Fallback to repo root if path is relative
    proj_root = Path(__file__).resolve().parents[2]
    candidate = proj_root / path_str
    if candidate.is_file():
        return candidate.resolve()

    return p.resolve()


def _parse_metric_tolerances(
    raw_tolerances: Sequence[str] | None,
) -> dict[str, float]:
    """Parse repeatable 'metric=value' tolerance override strings.

    Args:
        raw_tolerances: Sequence of 'metric=val' strings.

    Returns:
        Dictionary mapping metric names to validated float tolerances.

    Raises:
        ValueError: If any override string is malformed or tolerance is not finite/non-negative.
    """
    if not raw_tolerances:
        return {}

    parsed: dict[str, float] = {}
    for item in raw_tolerances:
        if "=" not in item:
            raise ValueError(
                f"Invalid format '{item}'. Expected format: metric=value (e.g. mrr@5=0.01)"
            )
        name, val_str = item.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"Empty metric name in tolerance override: '{item}'")
        try:
            val = float(val_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"Tolerance value for '{name}' must be a float, got '{val_str}'"
            ) from exc

        if not math.isfinite(val) or val < 0.0:
            raise ValueError(
                f"Tolerance for '{name}' must be a finite non-negative number, got {val}"
            )
        parsed[name] = val

    return parsed


def run_benchmark_command(args: argparse.Namespace) -> int:
    """Execute the `run-benchmark` CLI subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code (0 on success, 1 on operational error).
    """
    # 1. Validate arguments
    if args.top_k <= 0:
        sys.stderr.write(f"Error: --top-k must be a positive integer >= 1, got {args.top_k}\n")
        return 1

    experiment_name = args.experiment_name.strip()
    if not experiment_name:
        sys.stderr.write("Error: --experiment-name must be a non-empty string.\n")
        return 1

    corpus_path = _resolve_file_path(args.corpus)
    if not corpus_path.is_file():
        sys.stderr.write(f"Error: Corpus file not found: {corpus_path}\n")
        return 1

    dataset_path = _resolve_file_path(args.dataset)
    if not dataset_path.is_file():
        sys.stderr.write(f"Error: Benchmark dataset file not found: {dataset_path}\n")
        return 1

    db_path = Path(args.database).resolve()

    # 2. Load corpus and benchmark dataset
    try:
        chunks: list[Chunk] = load_corpus(corpus_path)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Error loading corpus '{corpus_path}': {exc}\n")
        return 1

    try:
        queries: list[Query] = load_benchmark_dataset(dataset_path, corpus_chunks=chunks)
    except (FileNotFoundError, ValueError) as exc:
        sys.stderr.write(f"Error loading benchmark dataset '{dataset_path}': {exc}\n")
        return 1

    # 3. Initialize BM25 index & run retrieval benchmark
    try:
        retriever = BM25Retriever()
        retriever.index(chunks)
        runner = RetrievalBenchmarkRunner(
            retriever=retriever,
            top_k=args.top_k,
            experiment_name=experiment_name,
        )
        report: EvaluationReport = runner.run(
            corpus=chunks,
            benchmark=queries,
            auto_index=False,
        )
    except Exception as exc:
        sys.stderr.write(f"Error running benchmark: {exc}\n")
        return 1

    # 4. Persist run through ExperimentRepository
    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        with ExperimentRepository(db_path) as repo:
            run_id = repo.save_report(
                report,
                index_config={
                    "retriever": "BM25Retriever",
                    "top_k": args.top_k,
                },
                dataset_reference=str(dataset_path),
                metadata={
                    "corpus_path": str(corpus_path),
                    "chunk_count": len(chunks),
                    "query_count": len(queries),
                },
            )
    except (StorageError, ValueError, TypeError) as exc:
        sys.stderr.write(f"Error persisting experiment run to '{db_path}': {exc}\n")
        return 1

    # 5. Output results
    if args.output == "json":
        output_payload = {
            "status": "SUCCESS",
            "run_id": run_id,
            "experiment_name": report.experiment_name,
            "database": str(db_path),
            "dataset_reference": str(dataset_path),
            "query_count": len(queries),
            "top_k": args.top_k,
            "metrics": report.metrics,
        }
        sys.stdout.write(json.dumps(output_payload, indent=2) + "\n")
    else:
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write(f"Benchmark Run Completed: {report.experiment_name}\n")
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write(f"Run ID:            {run_id}\n")
        sys.stdout.write(f"Database:          {db_path}\n")
        sys.stdout.write(f"Corpus Chunks:     {len(chunks)}\n")
        sys.stdout.write(f"Evaluated Queries: {len(queries)}\n")
        sys.stdout.write(f"Top-K Cutoff:      {args.top_k}\n")
        sys.stdout.write("-" * 60 + "\n")
        sys.stdout.write("Aggregate Metrics:\n")
        for metric_name, score in sorted(report.metrics.items()):
            sys.stdout.write(f"  {metric_name:<20}: {score:.4f}\n")
        sys.stdout.write("=" * 60 + "\n")

    return 0


def assert_regression_command(args: argparse.Namespace) -> int:
    """Execute the `assert-regression` CLI subcommand.

    Args:
        args: Parsed command-line arguments.

    Returns:
        Process exit code:
        - 0 (EXIT_PASS): comparison complete with no regressions.
        - 1 (EXIT_REGRESSION): regression detected exceeding tolerance.
        - 2 (EXIT_INCOMPLETE): incomplete comparison.
        - 3 (EXIT_OPERATIONAL_ERROR): operational or input failure.
    """
    baseline_id = args.baseline.strip()
    candidate_id = args.candidate.strip()

    if not baseline_id:
        sys.stderr.write("Error: --baseline run ID must be a non-empty string.\n")
        return EXIT_OPERATIONAL_ERROR
    if not candidate_id:
        sys.stderr.write("Error: --candidate run ID must be a non-empty string.\n")
        return EXIT_OPERATIONAL_ERROR

    # Validate default tolerance
    if not math.isfinite(args.tolerance) or args.tolerance < 0.0:
        sys.stderr.write(
            f"Error: --tolerance must be a finite non-negative number, got {args.tolerance}\n"
        )
        return EXIT_OPERATIONAL_ERROR

    # Validate per-metric overrides
    try:
        metric_tolerances = _parse_metric_tolerances(args.metric_tolerance)
    except ValueError as exc:
        sys.stderr.write(f"Error in metric tolerance overrides: {exc}\n")
        return EXIT_OPERATIONAL_ERROR

    db_path = Path(args.database).resolve()
    if not db_path.is_file():
        sys.stderr.write(f"Error: Experiment database does not exist: {db_path}\n")
        return EXIT_OPERATIONAL_ERROR

    # Compare runs via RegressionDetector
    try:
        with ExperimentRepository(db_path) as repo:
            detector = RegressionDetector(repo)
            result: RegressionResult = detector.compare_runs(
                baseline_run_id=baseline_id,
                candidate_run_id=candidate_id,
                tolerance=args.tolerance,
                metric_tolerances=metric_tolerances or None,
            )
    except (IncomparableRunsError, RunNotFoundError, InvalidToleranceError, StorageError) as exc:
        sys.stderr.write(f"Regression comparison error: {exc}\n")
        return EXIT_OPERATIONAL_ERROR
    except Exception as exc:
        sys.stderr.write(f"Unexpected operational error during comparison: {exc}\n")
        return EXIT_OPERATIONAL_ERROR

    # Output results
    if args.output == "json":
        sys.stdout.write(result.to_json() + "\n")
    else:
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write(f"Regression Analysis: {result.status.value}\n")
        sys.stdout.write("=" * 60 + "\n")
        sys.stdout.write(f"Baseline Run ID:   {result.baseline_run_id}\n")
        sys.stdout.write(f"Candidate Run ID:  {result.candidate_run_id}\n")
        sys.stdout.write(f"Default Tolerance: {args.tolerance}\n")
        if metric_tolerances:
            sys.stdout.write(f"Metric Overrides:  {metric_tolerances}\n")
        sys.stdout.write(f"Total Compared:    {result.total_queries_compared}\n")
        sys.stdout.write(f"Regressed Queries: {len(result.regressed_query_ids)}\n")
        sys.stdout.write("-" * 60 + "\n")
        sys.stdout.write("Aggregate Metric Deltas:\n")
        for m_name, delta in sorted(result.aggregate_deltas.items()):
            flag = "[REGRESSION]" if delta.is_regressed else "[OK]"
            sys.stdout.write(
                f"  {m_name:<16}: base={delta.baseline_score:.4f} "
                f"cand={delta.candidate_score:.4f} "
                f"delta={delta.delta:+.4f} {flag}\n"
            )

        if result.regressed_aggregate_metrics:
            sys.stdout.write("-" * 60 + "\n")
            sys.stdout.write(
                f"Regressed Aggregate Metrics: {', '.join(result.regressed_aggregate_metrics)}\n"
            )

        if result.worst_degradations:
            sys.stdout.write("-" * 60 + "\n")
            sys.stdout.write("Top Regressed Query/Metric Pairs:\n")
            for q_delta in result.worst_degradations[:5]:
                sys.stdout.write(
                    f"  Query '{q_delta.query_id}' metric '{q_delta.metric_name}': "
                    f"degradation={q_delta.degradation:.4f} (tolerance={q_delta.tolerance:.4f})\n"
                )

        if result.warnings:
            sys.stdout.write("-" * 60 + "\n")
            sys.stdout.write("Warnings:\n")
            for w in result.warnings:
                sys.stdout.write(f"  ! {w}\n")
        sys.stdout.write("=" * 60 + "\n")

    # Map RegressionStatus to deterministic exit code
    if result.status == RegressionStatus.PASS:
        return EXIT_PASS
    if result.status == RegressionStatus.REGRESSION:
        return EXIT_REGRESSION
    if result.status == RegressionStatus.INCOMPLETE:
        return EXIT_INCOMPLETE

    return EXIT_OPERATIONAL_ERROR


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser and subcommands."""
    parser = argparse.ArgumentParser(
        prog="reliability-lab",
        description="AI Reliability Lab — Headless CLI for Benchmarking & Regression Gating",
    )
    subparsers = parser.add_subparsers(
        dest="subcommand",
        help="Available subcommands",
    )

    # 1. run-benchmark subcommand
    bench_parser = subparsers.add_parser(
        "run-benchmark",
        help="Run BM25 retrieval benchmark on a corpus and dataset, persisting the report.",
    )
    bench_parser.add_argument(
        "--corpus",
        type=str,
        default=DEFAULT_CORPUS_REL,
        help=f"Path to JSONL corpus file (default: '{DEFAULT_CORPUS_REL}').",
    )
    bench_parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_REL,
        help=f"Path to JSONL benchmark query file (default: '{DEFAULT_DATASET_REL}').",
    )
    bench_parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE_PATH,
        help=f"Path to SQLite experiment database (default: '{DEFAULT_DATABASE_PATH}').",
    )
    bench_parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K retrieval cutoff (default: 5).",
    )
    bench_parser.add_argument(
        "--experiment-name",
        type=str,
        default="bm25-baseline",
        help="Experiment name label (default: 'bm25-baseline').",
    )
    bench_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' for human summary or 'json' for machine-readable output.",
    )

    # 2. assert-regression subcommand
    reg_parser = subparsers.add_parser(
        "assert-regression",
        help="Compare two saved experiment runs and gate on performance regression.",
    )
    reg_parser.add_argument(
        "--baseline",
        type=str,
        required=True,
        help="Baseline experiment run ID.",
    )
    reg_parser.add_argument(
        "--candidate",
        type=str,
        required=True,
        help="Candidate experiment run ID.",
    )
    reg_parser.add_argument(
        "--database",
        type=str,
        default=DEFAULT_DATABASE_PATH,
        help=f"Path to SQLite experiment database (default: '{DEFAULT_DATABASE_PATH}').",
    )
    reg_parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Default absolute degradation tolerance (default: {DEFAULT_TOLERANCE}).",
    )
    reg_parser.add_argument(
        "--metric-tolerance",
        action="append",
        type=str,
        help="Per-metric tolerance override in 'metric=value' format.",
    )
    reg_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Output format: 'text' for human summary or 'json' for machine-readable output.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Main entrypoint for the reliability-lab CLI.

    Args:
        argv: Optional command-line arguments (defaults to sys.argv[1:]).

    Returns:
        Integer process exit code.
    """
    parser = build_parser()

    if argv is None:
        argv = sys.argv[1:]

    # If invoked with no arguments at all, display help and return operational error code
    if not argv:
        parser.print_help(sys.stderr)
        return EXIT_OPERATIONAL_ERROR

    args = parser.parse_args(argv)

    if args.subcommand == "run-benchmark":
        return run_benchmark_command(args)
    if args.subcommand == "assert-regression":
        return assert_regression_command(args)

    parser.print_help(sys.stderr)
    return EXIT_OPERATIONAL_ERROR


if __name__ == "__main__":
    sys.exit(main())
