"""Unit tests for InvestigationRepository persistence in SQLite."""

from __future__ import annotations

from pathlib import Path

from reliability_lab.investigation.models import (
    ComparisonTopic,
    EvidenceFinding,
    InvestigationReport,
)
from reliability_lab.investigation.repository import InvestigationRepository
from reliability_lab.storage.database import Database


class TestInvestigationRepository:
    """Test suite for SQLite investigation and finding persistence."""

    def test_create_and_get_investigation(self, tmp_path: Path) -> None:
        db = Database(str(tmp_path / "test.db"))
        repo = InvestigationRepository(db)

        inv = repo.create_investigation(
            title="Transformer KV Cache Audit",
            query="Compare PagedAttention with Standard Attention",
            document_ids=["doc_1", "doc_2"],
            aspects=["Memory", "Throughput"],
        )

        assert inv.investigation_id.startswith("inv_")
        assert inv.title == "Transformer KV Cache Audit"
        assert inv.query == "Compare PagedAttention with Standard Attention"
        assert inv.document_ids == ["doc_1", "doc_2"]
        assert inv.aspects == ["Memory", "Throughput"]
        assert inv.status == "created"
        assert inv.report_json is None

        fetched = repo.get_investigation(inv.investigation_id)
        assert fetched is not None
        assert fetched.investigation_id == inv.investigation_id
        assert fetched.title == inv.title
        assert fetched.document_ids == ["doc_1", "doc_2"]
        assert fetched.aspects == ["Memory", "Throughput"]

    def test_get_nonexistent_returns_none(self, tmp_path: Path) -> None:
        repo = InvestigationRepository(Database(str(tmp_path / "test.db")))
        assert repo.get_investigation("invalid_id") is None
        assert repo.get_investigation("") is None

    def test_list_investigations_and_pagination(self, tmp_path: Path) -> None:
        repo = InvestigationRepository(Database(str(tmp_path / "test.db")))

        for i in range(5):
            repo.create_investigation(
                title=f"Investigation {i}",
                query=f"Query {i}",
                document_ids=[f"doc_{i}"],
                aspects=["Aspect A"],
            )

        items, total = repo.list_investigations(limit=3, offset=0)
        assert total == 5
        assert len(items) == 3

        items_p2, total_p2 = repo.list_investigations(limit=3, offset=3)
        assert total_p2 == 5
        assert len(items_p2) == 2

    def test_update_status_and_save_report(self, tmp_path: Path) -> None:
        repo = InvestigationRepository(Database(str(tmp_path / "test.db")))

        inv = repo.create_investigation(
            title="Attention Benchmark",
            query="Throughput comparison",
            document_ids=["doc_a", "doc_b"],
        )

        repo.update_status(inv.investigation_id, "running")
        fetched = repo.get_investigation(inv.investigation_id)
        assert fetched is not None
        assert fetched.status == "running"

        finding = EvidenceFinding(
            finding_id="f_test_1",
            investigation_id=inv.investigation_id,
            topic_name="Throughput",
            doc_id="doc_a",
            doc_title="Doc A",
            chunk_id="chk_a",
            page_number=1,
            claim="High throughput achieved.",
            supporting_passage="Throughput exceeds 500 tok/sec.",
            relationship="supports",
            confidence=0.92,
        )

        topic = ComparisonTopic(
            topic_name="Throughput",
            description="Speed test",
            findings=[finding],
            agreements_summary="Both tested at batch size 32.",
        )

        report = InvestigationReport(
            investigation_id=inv.investigation_id,
            title=inv.title,
            query=inv.query,
            executive_summary="Executive summary text.",
            documents=[{"doc_id": "doc_a", "title": "Doc A", "chunk_count": 5}],
            topics=[topic],
            relationship_counts={"supports": 1},
            total_findings=1,
            methodology_synthesis="Evaluated on GPUs.",
            generated_at="2026-11-08T02:00:00Z",
        )

        saved = repo.save_report(inv.investigation_id, report)
        assert saved.status == "completed"
        assert saved.report_json is not None

        findings = repo.get_findings_for_investigation(inv.investigation_id)
        assert len(findings) == 1
        assert findings[0].finding_id == "f_test_1"
        assert findings[0].claim == "High throughput achieved."
        assert findings[0].confidence == 0.92

    def test_delete_investigation_cascades(self, tmp_path: Path) -> None:
        repo = InvestigationRepository(Database(str(tmp_path / "test.db")))

        inv = repo.create_investigation(
            title="To Delete",
            query="Delete test",
            document_ids=["doc_x"],
        )

        finding = EvidenceFinding(
            finding_id="f_del_1",
            investigation_id=inv.investigation_id,
            topic_name="Topic X",
            doc_id="doc_x",
            doc_title="Doc X",
            chunk_id="chk_x",
            page_number=None,
            claim="Claim X",
            supporting_passage="Passage X",
            relationship="supports",
            confidence=0.88,
        )

        topic = ComparisonTopic(topic_name="Topic X", description="Desc", findings=[finding])
        report = InvestigationReport(
            investigation_id=inv.investigation_id,
            title="To Delete",
            query="Delete test",
            executive_summary="Summary",
            documents=[{"doc_id": "doc_x", "title": "Doc X", "chunk_count": 1}],
            topics=[topic],
            relationship_counts={"supports": 1},
            total_findings=1,
            methodology_synthesis=None,
            generated_at="2026-11-08T02:00:00Z",
        )
        repo.save_report(inv.investigation_id, report)

        assert len(repo.get_findings_for_investigation(inv.investigation_id)) == 1

        deleted = repo.delete_investigation(inv.investigation_id)
        assert deleted is True

        assert repo.get_investigation(inv.investigation_id) is None
        assert len(repo.get_findings_for_investigation(inv.investigation_id)) == 0

        # Deleting non-existent returns False
        assert repo.delete_investigation("non_existent") is False
