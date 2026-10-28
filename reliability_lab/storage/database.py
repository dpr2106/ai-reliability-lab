"""SQLite database lifecycle, connection management, and transactional schema orchestration."""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

SCHEMA_VERSION: int = 1

CREATE_TABLES_SQL = """
-- Deduplicated index and experiment configurations
CREATE TABLE IF NOT EXISTS index_configurations (
    config_hash TEXT PRIMARY KEY,
    config_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Experiment run headers
CREATE TABLE IF NOT EXISTS experiment_runs (
    run_id TEXT PRIMARY KEY,
    experiment_name TEXT NOT NULL,
    run_timestamp TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    dataset_reference TEXT,
    metadata_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (config_hash) REFERENCES index_configurations(config_hash)
);

-- Normalized aggregate metrics per run
CREATE TABLE IF NOT EXISTS run_metrics (
    run_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (run_id, metric_name),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
);

-- Per-query evaluation records and diagnostics
CREATE TABLE IF NOT EXISTS query_results (
    run_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    latency_ms REAL,
    result_json TEXT NOT NULL,
    PRIMARY KEY (run_id, query_id),
    FOREIGN KEY (run_id) REFERENCES experiment_runs(run_id) ON DELETE CASCADE
);

-- Per-query individual metric scores
CREATE TABLE IF NOT EXISTS query_metric_scores (
    run_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    metric_name TEXT NOT NULL,
    score REAL NOT NULL,
    PRIMARY KEY (run_id, query_id, metric_name),
    FOREIGN KEY (run_id, query_id) REFERENCES query_results(run_id, query_id) ON DELETE CASCADE
);

-- Performance and lookup indexes
CREATE INDEX IF NOT EXISTS idx_experiment_runs_name_ts
    ON experiment_runs(experiment_name, run_timestamp);

CREATE INDEX IF NOT EXISTS idx_experiment_runs_config
    ON experiment_runs(config_hash);

CREATE INDEX IF NOT EXISTS idx_run_metrics_metric_score
    ON run_metrics(metric_name, score);

CREATE INDEX IF NOT EXISTS idx_query_metric_scores_metric
    ON query_metric_scores(metric_name, score);

CREATE INDEX IF NOT EXISTS idx_query_results_query_id
    ON query_results(query_id);
"""


class Database:
    """Manages SQLite connections, idempotent schema initialization, and transactional boundaries.

    Supports both file-backed SQLite databases and persistent in-memory databases (`:memory:`).
    Enforces foreign keys on every connection and tracks schema migrations via PRAGMA user_version.
    """

    def __init__(self, path: str | Path = ":memory:", timeout: float = 30.0) -> None:
        """Initialize Database manager with target SQLite path.

        Args:
            path: Path to SQLite file or ':memory:' for transient in-memory database.
            timeout: Timeout in seconds for SQLite lock resolution.
        """
        self._path_str = str(path) if isinstance(path, (str, Path)) else ":memory:"
        self._timeout = float(timeout)
        self._is_memory = self._path_str == ":memory:"

        # For in-memory databases, maintain an open connection so data persists
        self._memory_connection: sqlite3.Connection | None = None
        if self._is_memory:
            self._memory_connection = self._create_connection()

        self._closed = False
        self.initialize_schema()

    @property
    def path(self) -> str:
        """Return the database connection path."""
        return self._path_str

    @property
    def is_memory(self) -> bool:
        """Return True if using an in-memory database."""
        return self._is_memory

    def _create_connection(self) -> sqlite3.Connection:
        """Create and configure a fresh SQLite connection with strict pragmas."""
        if not self._is_memory:
            db_path = Path(self._path_str)
            if db_path.parent != Path("."):
                db_path.parent.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(
            self._path_str,
            timeout=self._timeout,
            isolation_level=None,  # Autocommit mode by default; explicit transactions used
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.execute(f"PRAGMA busy_timeout = {int(self._timeout * 1000)};")
        return conn

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Yield an active SQLite connection with foreign keys enabled."""
        if self._closed:
            raise RuntimeError("Database connection has been closed.")

        if self._is_memory:
            assert self._memory_connection is not None
            yield self._memory_connection
        else:
            conn = self._create_connection()
            try:
                yield conn
            finally:
                conn.close()

    @contextmanager
    def transaction(self) -> Generator[sqlite3.Connection, None, None]:
        """Coordinate an atomic database transaction.

        Commits on clean exit; rolls back on unhandled exception.
        """
        with self.get_connection() as conn:
            conn.execute("BEGIN IMMEDIATE;")
            try:
                yield conn
                conn.execute("COMMIT;")
            except Exception:
                conn.execute("ROLLBACK;")
                raise

    def initialize_schema(self) -> None:
        """Idempotently initialize relational tables and indexes.

        Verifies `PRAGMA user_version`. Raises RuntimeError if encountering a newer
        schema version than supported.
        """
        with self.get_connection() as conn:
            cursor = conn.execute("PRAGMA user_version;")
            row = cursor.fetchone()
            current_version = int(row[0]) if row else 0

            if current_version > SCHEMA_VERSION:
                raise RuntimeError(
                    f"Unsupported newer SQLite schema version {current_version}; "
                    f"this codebase supports up to version {SCHEMA_VERSION}."
                )

            if current_version == 0:
                conn.executescript(CREATE_TABLES_SQL)
                conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION};")

    def execute_query(
        self, sql: str, params: tuple[Any, ...] | dict[str, Any] = ()
    ) -> list[sqlite3.Row]:
        """Execute a parameterized read query and return rows."""
        with self.get_connection() as conn:
            cursor = conn.execute(sql, params)
            return cursor.fetchall()

    def close(self) -> None:
        """Close active connections and release resources."""
        if self._closed:
            return
        if self._memory_connection is not None:
            self._memory_connection.close()
            self._memory_connection = None
        self._closed = True

    def __enter__(self) -> Database:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
