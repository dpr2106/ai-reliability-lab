"""SQLite repository for persistent cross-document investigations and findings."""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from reliability_lab.investigation.models import (
    EvidenceFinding,
    InvestigationRecord,
    InvestigationReport,
)
from reliability_lab.storage.database import Database

logger = logging.getLogger(__name__)

CREATE_INVESTIGATION_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS investigations (
    investigation_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    query TEXT NOT NULL,
    comparison_aspects_json TEXT NOT NULL DEFAULT '[]',
    document_ids_json TEXT NOT NULL DEFAULT '[]',
    status TEXT NOT NULL,
    created_at TEXT NOT NULL,
    completed_at TEXT,
    report_json TEXT,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS investigation_findings (
    finding_id TEXT PRIMARY KEY,
    investigation_id TEXT NOT NULL,
    topic_id TEXT NOT NULL,
    topic_name TEXT NOT NULL,
    doc_id TEXT NOT NULL,
    doc_title TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    page_number INTEGER,
    claim TEXT NOT NULL,
    supporting_passage TEXT NOT NULL,
    relationship TEXT NOT NULL,
    confidence REAL NOT NULL,
    methodology_notes TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (investigation_id) REFERENCES investigations(investigation_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_investigations_created
    ON investigations(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_findings_investigation
    ON investigation_findings(investigation_id, topic_id);
"""


class InvestigationStorageError(Exception):
    """Base exception for investigation persistence errors."""


class InvestigationNotFoundError(InvestigationStorageError):
    """Raised when an investigation record is not found."""


class InvestigationRepository:
    """SQLite repository for managing cross-document investigations and findings."""

    def __init__(self, database: Database | str | Path = ":memory:") -> None:
        if isinstance(database, Database):
            self._db = database
            self._owns_db = False
        else:
            self._db = Database(database)
            self._owns_db = True
        self._init_tables()

    @property
    def database(self) -> Database:
        """Return the underlying Database instance."""
        return self._db

    def _init_tables(self) -> None:
        """Idempotently create investigation tables and indexes."""
        with self._db.get_connection() as conn:
            conn.executescript(CREATE_INVESTIGATION_TABLES_SQL)

    def close(self) -> None:
        """Close database connection if owned."""
        if self._owns_db:
            self._db.close()

    def __enter__(self) -> InvestigationRepository:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def create_investigation(
        self,
        query: str,
        document_ids: Sequence[str],
        title: str | None = None,
        comparison_aspects: Sequence[str] | None = None,
        *,
        aspects: Sequence[str] | None = None,
    ) -> InvestigationRecord:
        """Create and persist a new investigation entity."""
        effective_aspects = comparison_aspects if comparison_aspects is not None else aspects
        inv_id = f"inv_{uuid4().hex[:16]}"
        effective_title = (title or query.strip()[:80]).strip()
        aspects_list = list(effective_aspects) if effective_aspects else []
        docs_list = list(document_ids)
        now = datetime.now(timezone.utc).isoformat()

        sql = """
        INSERT INTO investigations (
            investigation_id, title, query, comparison_aspects_json,
            document_ids_json, status, created_at, completed_at, report_json, error_message
        ) VALUES (?, ?, ?, ?, ?, 'created', ?, NULL, NULL, NULL);
        """
        with self._db.transaction() as conn:
            conn.execute(
                sql,
                (
                    inv_id,
                    effective_title,
                    query.strip(),
                    json.dumps(aspects_list, ensure_ascii=False),
                    json.dumps(docs_list, ensure_ascii=False),
                    now,
                ),
            )

        return InvestigationRecord(
            investigation_id=inv_id,
            title=effective_title,
            query=query.strip(),
            comparison_aspects=aspects_list,
            document_ids=docs_list,
            status="created",
            created_at=now,
        )

    def get_investigation(self, investigation_id: str) -> InvestigationRecord | None:
        """Retrieve an investigation by ID, reconstructing full report if available."""
        sql = "SELECT * FROM investigations WHERE investigation_id = ?;"
        rows = self._db.execute_query(sql, (investigation_id.strip(),))
        if not rows:
            return None
        row = dict(rows[0])

        report: InvestigationReport | None = None
        if row.get("report_json"):
            try:
                report_data = json.loads(str(row["report_json"]))
                report = InvestigationReport.from_dict(report_data)
            except Exception as exc:
                logger.warning(
                    "Failed to deserialize report_json for investigation '%s': %s",
                    investigation_id,
                    exc,
                )

        aspects = (
            json.loads(str(row["comparison_aspects_json"]))
            if row.get("comparison_aspects_json")
            else []
        )
        docs = json.loads(str(row["document_ids_json"])) if row.get("document_ids_json") else []

        return InvestigationRecord(
            investigation_id=str(row["investigation_id"]),
            title=str(row["title"]),
            query=str(row["query"]),
            comparison_aspects=aspects,
            document_ids=docs,
            status=str(row["status"]),
            created_at=str(row["created_at"]),
            completed_at=row.get("completed_at"),
            report=report,
            error_message=row.get("error_message"),
        )

    def list_investigations(
        self, limit: int = 20, offset: int = 0
    ) -> tuple[list[InvestigationRecord], int]:
        """List investigations ordered by creation timestamp DESC."""
        safe_limit = max(1, min(limit, 100))
        safe_offset = max(0, offset)

        count_sql = "SELECT COUNT(*) as total FROM investigations;"
        count_rows = self._db.execute_query(count_sql)
        total = count_rows[0]["total"] if count_rows else 0

        list_sql = """
        SELECT * FROM investigations
        ORDER BY created_at DESC
        LIMIT ? OFFSET ?;
        """
        rows = self._db.execute_query(list_sql, (safe_limit, safe_offset))
        records: list[InvestigationRecord] = []
        for r_raw in rows:
            r = dict(r_raw)
            aspects = (
                json.loads(str(r["comparison_aspects_json"]))
                if r.get("comparison_aspects_json")
                else []
            )
            docs = json.loads(str(r["document_ids_json"])) if r.get("document_ids_json") else []
            report: InvestigationReport | None = None
            if r.get("report_json"):
                try:
                    report = InvestigationReport.from_dict(json.loads(str(r["report_json"])))
                except Exception:
                    pass
            records.append(
                InvestigationRecord(
                    investigation_id=str(r["investigation_id"]),
                    title=str(r["title"]),
                    query=str(r["query"]),
                    comparison_aspects=aspects,
                    document_ids=docs,
                    status=str(r["status"]),
                    created_at=str(r["created_at"]),
                    completed_at=r.get("completed_at"),
                    report=report,
                    error_message=r.get("error_message"),
                )
            )

        return records, total

    def update_status(
        self,
        investigation_id: str,
        status: str,
        error_message: str | None = None,
    ) -> None:
        """Update execution status of an investigation."""
        completed_at = (
            datetime.now(timezone.utc).isoformat() if status in ("completed", "failed") else None
        )
        sql = """
        UPDATE investigations
        SET status = ?, error_message = ?, completed_at = COALESCE(?, completed_at)
        WHERE investigation_id = ?;
        """
        with self._db.transaction() as conn:
            cursor = conn.execute(
                sql,
                (status, error_message, completed_at, investigation_id.strip()),
            )
            if cursor.rowcount == 0:
                raise InvestigationNotFoundError(f"Investigation '{investigation_id}' not found.")

    def save_report(
        self,
        investigation_id: str,
        report: InvestigationReport,
    ) -> InvestigationRecord:
        """Atomically persist complete report and normalized findings."""
        completed_at = report.completed_at or datetime.now(timezone.utc).isoformat()
        report_json = json.dumps(report.to_dict(), ensure_ascii=False)

        sql_update_inv = """
        UPDATE investigations
        SET status = 'completed', completed_at = ?, report_json = ?, error_message = NULL
        WHERE investigation_id = ?;
        """

        sql_insert_finding = """
        INSERT INTO investigation_findings (
            finding_id, investigation_id, topic_id, topic_name, doc_id, doc_title,
            chunk_id, page_number, claim, supporting_passage, relationship,
            confidence, methodology_notes, created_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        """

        with self._db.transaction() as conn:
            # 1. Update master record
            cursor = conn.execute(
                sql_update_inv, (completed_at, report_json, investigation_id.strip())
            )
            if cursor.rowcount == 0:
                raise InvestigationNotFoundError(f"Investigation '{investigation_id}' not found.")

            # 2. Clear old findings if re-run
            conn.execute(
                "DELETE FROM investigation_findings WHERE investigation_id = ?;",
                (investigation_id.strip(),),
            )

            # 3. Insert all structured findings
            for topic in report.topics:
                for f in topic.findings:
                    conn.execute(
                        sql_insert_finding,
                        (
                            f.finding_id,
                            investigation_id.strip(),
                            topic.topic_id,
                            topic.topic_name,
                            f.doc_id,
                            f.doc_title,
                            f.chunk_id,
                            f.page_number,
                            f.claim,
                            f.supporting_passage,
                            f.relationship,
                            f.confidence,
                            f.methodology_notes,
                            f.created_at,
                        ),
                    )

        updated = self.get_investigation(investigation_id)
        if updated is None:
            raise InvestigationNotFoundError(f"Investigation '{investigation_id}' not found.")
        return updated

    def get_findings_for_investigation(self, investigation_id: str) -> list[EvidenceFinding]:
        """Fetch all individual findings for an investigation."""
        sql = """
        SELECT * FROM investigation_findings
        WHERE investigation_id = ?
        ORDER BY topic_name ASC, doc_title ASC;
        """
        rows = self._db.execute_query(sql, (investigation_id.strip(),))
        findings: list[EvidenceFinding] = []
        for r_raw in rows:
            r = dict(r_raw)
            findings.append(
                EvidenceFinding(
                    finding_id=str(r["finding_id"]),
                    topic_id=str(r["topic_id"]),
                    doc_id=str(r["doc_id"]),
                    doc_title=str(r["doc_title"]),
                    chunk_id=str(r["chunk_id"]),
                    page_number=r.get("page_number"),
                    claim=str(r["claim"]),
                    supporting_passage=str(r["supporting_passage"]),
                    relationship=r["relationship"],
                    confidence=float(r["confidence"]),
                    methodology_notes=r.get("methodology_notes"),
                    created_at=str(r["created_at"]),
                )
            )
        return findings

    def delete_investigation(self, investigation_id: str) -> bool:
        """Delete an investigation and its cascading findings."""
        with self._db.transaction() as conn:
            cursor = conn.execute(
                "DELETE FROM investigations WHERE investigation_id = ?;",
                (investigation_id.strip(),),
            )
            return cursor.rowcount > 0
