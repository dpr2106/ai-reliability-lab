"""SQLite repository for persistent RAG Evaluation Datasets and Benchmarks."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from reliability_lab.benchmark.models import BenchmarkDataset, BenchmarkDatasetRecord
from reliability_lab.core.models import Query
from reliability_lab.evaluation.benchmark_runner import load_benchmark_dataset
from reliability_lab.storage.database import Database

logger = logging.getLogger(__name__)

CREATE_DATASET_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS benchmark_datasets (
    dataset_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    description TEXT,
    document_scope_json TEXT NOT NULL DEFAULT '[]',
    is_synthetic INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS benchmark_queries (
    dataset_id TEXT NOT NULL,
    query_id TEXT NOT NULL,
    query_text TEXT NOT NULL,
    relevant_chunk_ids_json TEXT NOT NULL DEFAULT '[]',
    relevance_grades_json TEXT NOT NULL DEFAULT '{}',
    ground_truth_answer TEXT,
    required_citations_json TEXT NOT NULL DEFAULT '[]',
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    PRIMARY KEY (dataset_id, query_id),
    FOREIGN KEY (dataset_id) REFERENCES benchmark_datasets(dataset_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_benchmark_queries_ds ON benchmark_queries(dataset_id);
"""


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class BenchmarkDatasetRepository:
    """Relational storage repository for benchmark datasets and golden queries."""

    def __init__(self, database_or_path: Database | Path | str) -> None:
        if isinstance(database_or_path, Database):
            self._db = database_or_path
        else:
            self._db = Database(database_or_path)
        self._init_schema()

    @property
    def database(self) -> Database:
        return self._db

    def _init_schema(self) -> None:
        with self._db.get_connection() as conn:
            conn.executescript(CREATE_DATASET_TABLES_SQL)

    def create_dataset(
        self,
        name: str,
        description: str | None = None,
        document_scope: list[str] | None = None,
        is_synthetic: bool = False,
        metadata: dict[str, Any] | None = None,
        queries: Sequence[Query] | None = None,
        dataset_id: str | None = None,
    ) -> BenchmarkDataset:
        """Create and persist a new evaluation dataset."""
        if not name or not name.strip():
            raise ValueError("Dataset name must be a non-empty string.")

        ds_id = (
            dataset_id.strip() if dataset_id and dataset_id.strip() else f"ds_{uuid4().hex[:12]}"
        )
        now = _utc_now_iso()
        scope = list(document_scope or [])
        meta = dict(metadata or {})
        q_list = list(queries or [])

        # Validate duplicate query IDs within input
        seen_qids: set[str] = set()
        for q in q_list:
            if q.query_id in seen_qids:
                raise ValueError(f"Duplicate query_id '{q.query_id}' in dataset creation.")
            seen_qids.add(q.query_id)

        with self._db.get_connection() as conn:
            # Check for existing dataset_id
            existing = conn.execute(
                "SELECT dataset_id FROM benchmark_datasets WHERE dataset_id = ?",
                (ds_id,),
            ).fetchone()
            if existing:
                raise ValueError(f"Dataset with ID '{ds_id}' already exists.")

            conn.execute(
                "INSERT INTO benchmark_datasets ("
                "dataset_id, name, description, document_scope_json, "
                "is_synthetic, created_at, updated_at, metadata_json"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    ds_id,
                    name.strip(),
                    description.strip() if description else None,
                    json.dumps(scope),
                    1 if is_synthetic else 0,
                    now,
                    now,
                    json.dumps(meta),
                ),
            )

            for q in q_list:
                conn.execute(
                    "INSERT INTO benchmark_queries ("
                    "dataset_id, query_id, query_text, relevant_chunk_ids_json, "
                    "relevance_grades_json, ground_truth_answer, required_citations_json, "
                    "metadata_json, created_at"
                    ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        ds_id,
                        q.query_id,
                        q.text,
                        json.dumps(q.relevant_chunk_ids),
                        json.dumps(q.relevance_grades),
                        q.ground_truth_answer,
                        json.dumps(q.required_citations),
                        json.dumps(q.metadata),
                        now,
                    ),
                )

        record = BenchmarkDatasetRecord(
            dataset_id=ds_id,
            name=name.strip(),
            description=description.strip() if description else None,
            document_scope=scope,
            query_count=len(q_list),
            is_synthetic=is_synthetic,
            created_at=now,
            updated_at=now,
            metadata=meta,
        )
        return BenchmarkDataset(record=record, queries=q_list)

    def get_dataset(self, dataset_id: str) -> BenchmarkDataset | None:
        """Fetch dataset and all its benchmark queries by ID."""
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT dataset_id, name, description, document_scope_json, "
                "is_synthetic, created_at, updated_at, metadata_json "
                "FROM benchmark_datasets WHERE dataset_id = ?",
                (dataset_id,),
            ).fetchone()
            if not row:
                return None

            q_rows = conn.execute(
                "SELECT query_id, query_text, relevant_chunk_ids_json, "
                "relevance_grades_json, ground_truth_answer, required_citations_json, "
                "metadata_json FROM benchmark_queries WHERE dataset_id = ? "
                "ORDER BY rowid ASC",
                (dataset_id,),
            ).fetchall()

            queries: list[Query] = []
            for qr in q_rows:
                queries.append(
                    Query(
                        query_id=qr["query_id"],
                        text=qr["query_text"],
                        relevant_chunk_ids=json.loads(qr["relevant_chunk_ids_json"] or "[]"),
                        relevance_grades={
                            str(k): int(v)
                            for k, v in json.loads(qr["relevance_grades_json"] or "{}").items()
                        },
                        ground_truth_answer=qr["ground_truth_answer"],
                        required_citations=json.loads(qr["required_citations_json"] or "[]"),
                        metadata=json.loads(qr["metadata_json"] or "{}"),
                    )
                )

            record = BenchmarkDatasetRecord(
                dataset_id=row["dataset_id"],
                name=row["name"],
                description=row["description"],
                document_scope=json.loads(row["document_scope_json"] or "[]"),
                query_count=len(queries),
                is_synthetic=bool(row["is_synthetic"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )
            return BenchmarkDataset(record=record, queries=queries)

    def get_dataset_record(self, dataset_id: str) -> BenchmarkDatasetRecord | None:
        """Fetch dataset summary without loading all constituent queries."""
        with self._db.get_connection() as conn:
            row = conn.execute(
                "SELECT d.dataset_id, d.name, d.description, d.document_scope_json, "
                "d.is_synthetic, d.created_at, d.updated_at, d.metadata_json, "
                "COUNT(q.query_id) as q_cnt "
                "FROM benchmark_datasets d "
                "LEFT JOIN benchmark_queries q ON d.dataset_id = q.dataset_id "
                "WHERE d.dataset_id = ? GROUP BY d.dataset_id",
                (dataset_id,),
            ).fetchone()
            if not row:
                return None

            return BenchmarkDatasetRecord(
                dataset_id=row["dataset_id"],
                name=row["name"],
                description=row["description"],
                document_scope=json.loads(row["document_scope_json"] or "[]"),
                query_count=int(row["q_cnt"]),
                is_synthetic=bool(row["is_synthetic"]),
                created_at=row["created_at"],
                updated_at=row["updated_at"],
                metadata=json.loads(row["metadata_json"] or "{}"),
            )

    def list_datasets(self, limit: int = 50, offset: int = 0) -> list[BenchmarkDatasetRecord]:
        """Return paginated list of dataset summaries ordered by creation time descending."""
        with self._db.get_connection() as conn:
            rows = conn.execute(
                "SELECT d.dataset_id, d.name, d.description, d.document_scope_json, "
                "d.is_synthetic, d.created_at, d.updated_at, d.metadata_json, "
                "COUNT(q.query_id) as q_cnt "
                "FROM benchmark_datasets d "
                "LEFT JOIN benchmark_queries q ON d.dataset_id = q.dataset_id "
                "GROUP BY d.dataset_id ORDER BY d.created_at DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()

            records: list[BenchmarkDatasetRecord] = []
            for row in rows:
                records.append(
                    BenchmarkDatasetRecord(
                        dataset_id=row["dataset_id"],
                        name=row["name"],
                        description=row["description"],
                        document_scope=json.loads(row["document_scope_json"] or "[]"),
                        query_count=int(row["q_cnt"]),
                        is_synthetic=bool(row["is_synthetic"]),
                        created_at=row["created_at"],
                        updated_at=row["updated_at"],
                        metadata=json.loads(row["metadata_json"] or "{}"),
                    )
                )
            return records

    def count_datasets(self) -> int:
        with self._db.get_connection() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM benchmark_datasets").fetchone()
            return int(row["cnt"]) if row else 0

    def update_dataset(
        self,
        dataset_id: str,
        name: str | None = None,
        description: str | None = None,
        document_scope: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BenchmarkDatasetRecord:
        """Update dataset header fields."""
        existing = self.get_dataset_record(dataset_id)
        if not existing:
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        new_name = name.strip() if name is not None else existing.name
        if not new_name:
            raise ValueError("Dataset name cannot be empty.")
        new_desc = description.strip() if description is not None else existing.description
        new_scope = list(document_scope) if document_scope is not None else existing.document_scope
        new_meta = dict(metadata) if metadata is not None else existing.metadata
        now = _utc_now_iso()

        with self._db.get_connection() as conn:
            conn.execute(
                "UPDATE benchmark_datasets SET name = ?, description = ?, "
                "document_scope_json = ?, metadata_json = ?, updated_at = ? "
                "WHERE dataset_id = ?",
                (
                    new_name,
                    new_desc,
                    json.dumps(new_scope),
                    json.dumps(new_meta),
                    now,
                    dataset_id,
                ),
            )

        return BenchmarkDatasetRecord(
            dataset_id=dataset_id,
            name=new_name,
            description=new_desc,
            document_scope=new_scope,
            query_count=existing.query_count,
            is_synthetic=existing.is_synthetic,
            created_at=existing.created_at,
            updated_at=now,
            metadata=new_meta,
        )

    def delete_dataset(self, dataset_id: str) -> bool:
        """Delete dataset and cascading queries."""
        with self._db.get_connection() as conn:
            res = conn.execute(
                "DELETE FROM benchmark_datasets WHERE dataset_id = ?",
                (dataset_id,),
            )
            return res.rowcount > 0

    def add_query(self, dataset_id: str, query: Query) -> Query:
        """Add a single query to an existing dataset."""
        if not self.get_dataset_record(dataset_id):
            raise ValueError(f"Dataset '{dataset_id}' not found.")

        now = _utc_now_iso()
        with self._db.get_connection() as conn:
            existing = conn.execute(
                "SELECT query_id FROM benchmark_queries WHERE dataset_id = ? AND query_id = ?",
                (dataset_id, query.query_id),
            ).fetchone()
            if existing:
                raise ValueError(
                    f"Query with ID '{query.query_id}' already exists in dataset '{dataset_id}'."
                )

            conn.execute(
                "INSERT INTO benchmark_queries ("
                "dataset_id, query_id, query_text, relevant_chunk_ids_json, "
                "relevance_grades_json, ground_truth_answer, required_citations_json, "
                "metadata_json, created_at"
                ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    dataset_id,
                    query.query_id,
                    query.text,
                    json.dumps(query.relevant_chunk_ids),
                    json.dumps(query.relevance_grades),
                    query.ground_truth_answer,
                    json.dumps(query.required_citations),
                    json.dumps(query.metadata),
                    now,
                ),
            )
            conn.execute(
                "UPDATE benchmark_datasets SET updated_at = ? WHERE dataset_id = ?",
                (now, dataset_id),
            )
        return query

    def update_query(self, dataset_id: str, query: Query) -> Query:
        """Update an existing query in a dataset."""
        now = _utc_now_iso()
        with self._db.get_connection() as conn:
            res = conn.execute(
                "UPDATE benchmark_queries SET query_text = ?, relevant_chunk_ids_json = ?, "
                "relevance_grades_json = ?, ground_truth_answer = ?, "
                "required_citations_json = ?, metadata_json = ? "
                "WHERE dataset_id = ? AND query_id = ?",
                (
                    query.text,
                    json.dumps(query.relevant_chunk_ids),
                    json.dumps(query.relevance_grades),
                    query.ground_truth_answer,
                    json.dumps(query.required_citations),
                    json.dumps(query.metadata),
                    dataset_id,
                    query.query_id,
                ),
            )
            if res.rowcount == 0:
                raise ValueError(f"Query '{query.query_id}' not found in dataset '{dataset_id}'.")
            conn.execute(
                "UPDATE benchmark_datasets SET updated_at = ? WHERE dataset_id = ?",
                (now, dataset_id),
            )
        return query

    def delete_query(self, dataset_id: str, query_id: str) -> bool:
        """Delete a query from a dataset."""
        now = _utc_now_iso()
        with self._db.get_connection() as conn:
            res = conn.execute(
                "DELETE FROM benchmark_queries WHERE dataset_id = ? AND query_id = ?",
                (dataset_id, query_id),
            )
            if res.rowcount > 0:
                conn.execute(
                    "UPDATE benchmark_datasets SET updated_at = ? WHERE dataset_id = ?",
                    (now, dataset_id),
                )
                return True
            return False

    def seed_sample_dataset_if_empty(self) -> BenchmarkDataset | None:
        """Seed bundled sample benchmark dataset if no datasets exist."""
        if self.count_datasets() > 0:
            return None

        # Look for sample_benchmark.jsonl
        candidate_paths = [
            Path("data/benchmarks/sample_benchmark.jsonl"),
            Path(__file__).parent.parent.parent / "data" / "benchmarks" / "sample_benchmark.jsonl",
        ]
        sample_path: Path | None = None
        for p in candidate_paths:
            if p.is_file():
                sample_path = p
                break

        if not sample_path:
            logger.info("Sample benchmark dataset file not found; skipping seed.")
            return None

        try:
            queries = load_benchmark_dataset(sample_path)
            created = self.create_dataset(
                name="Sample Retrieval & Ingestion Benchmark (Synthetic)",
                description=(
                    "Bundled synthetic benchmark dataset demonstrating multi-aspect RAG "
                    "evaluation over lexical and dense indexing capabilities. Clearly "
                    "labelled as synthetic demo data."
                ),
                document_scope=[],
                is_synthetic=True,
                metadata={"source_file": "sample_benchmark.jsonl", "demonstration": True},
                queries=queries,
                dataset_id="ds_sample_synthetic",
            )
            logger.info(
                "Successfully seeded bundled synthetic benchmark dataset (%d queries).",
                len(queries),
            )
            return created
        except Exception as exc:
            logger.warning("Failed to seed sample benchmark dataset: %s", exc)
            return None
