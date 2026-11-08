"""Unit tests for Dataset Studio: schema validation, reference integrity, and import/export."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
from reliability_lab.benchmark.service import (
    BenchmarkDatasetService,
    DatasetValidationError,
)
from reliability_lab.core.models import Query
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.storage.database import Database
from tests.unit.test_investigation_hardening import DeterministicEmbedder, _ingest


@pytest.fixture
def dataset_env(tmp_path: Path) -> dict[str, Any]:
    db_path = tmp_path / "dataset_test.db"
    db = Database(str(db_path))
    doc_repo = DocumentRepository(db_path)
    ds_repo = BenchmarkDatasetRepository(db)
    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=DeterministicEmbedder(),
    )
    service = BenchmarkDatasetService(
        dataset_repository=ds_repo,
        document_repository=doc_repo,
    )
    return {
        "db": db,
        "doc_repo": doc_repo,
        "ds_repo": ds_repo,
        "ingestion": ingestion,
        "service": service,
    }


class TestDatasetManagementAndValidation:
    """Tests for dataset creation, reference validation, query management, and import/export."""

    def test_create_dataset_with_valid_queries(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        q1 = Query(
            query_id="Q-001",
            text="What is BM25 k1 parameter?",
            relevant_chunk_ids=["chunk_1"],
            relevance_grades={"chunk_1": 2},
            ground_truth_answer="k1 controls term frequency saturation.",
            required_citations=["chunk_1"],
        )
        ds = service.create_dataset(
            name="Information Retrieval Benchmark",
            description="Testing BM25 parameters",
            queries=[q1],
            validate_chunk_existence=False,
        )

        assert ds.dataset_id.startswith("ds_")
        assert ds.name == "Information Retrieval Benchmark"
        assert ds.query_count == 1
        assert len(ds.queries) == 1
        assert ds.queries[0].query_id == "Q-001"

    def test_reject_duplicate_query_ids(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        q1 = Query(query_id="DUP-1", text="First query")
        q2 = Query(query_id="DUP-1", text="Duplicate query")

        with pytest.raises(DatasetValidationError, match="Duplicate query ID"):
            service.create_dataset(
                name="Duplicate Test",
                queries=[q1, q2],
                validate_chunk_existence=False,
            )

    def test_reject_invalid_document_reference(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        q = Query(query_id="Q-1", text="Any query")
        with pytest.raises(DatasetValidationError, match="does not exist in repository"):
            service.create_dataset(
                name="Invalid Doc Scope",
                document_scope=["non_existent_doc_12345"],
                queries=[q],
                validate_chunk_existence=True,
            )

    def test_reject_invalid_chunk_reference(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]
        ingestion: DocumentIngestionService = dataset_env["ingestion"]

        doc_id = _ingest(ingestion, "doc1.txt", "Some content about database indices.")

        q = Query(
            query_id="Q-1",
            text="Any query",
            relevant_chunk_ids=["non_existent_chunk_9999"],
        )
        with pytest.raises(DatasetValidationError, match="unknown chunk 'non_existent_chunk_9999'"):
            service.create_dataset(
                name="Invalid Chunk Ref",
                document_scope=[doc_id],
                queries=[q],
                validate_chunk_existence=True,
            )

    def test_reject_invalid_relevance_grade(self, dataset_env: dict[str, Any]) -> None:
        # Note: Query model itself enforces non-negative int grades
        with pytest.raises(ValueError, match=r"relevance grade.*must be a non-negative integer"):
            Query(
                query_id="Q-1",
                text="Negative grade test",
                relevance_grades={"chk_1": -2},
            )

    def test_query_lifecycle_add_update_delete(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        ds = service.create_dataset(
            name="CRUD Test Dataset",
            queries=[],
            validate_chunk_existence=False,
        )
        assert ds.query_count == 0

        # Add query
        q = Query(
            query_id="Q-CRUD-1",
            text="Initial question",
            ground_truth_answer="Initial answer",
        )
        service.add_query(ds.dataset_id, q, validate_chunk_existence=False)

        fetched = service.get_dataset(ds.dataset_id)
        assert fetched.query_count == 1
        assert fetched.queries[0].text == "Initial question"

        # Update query
        q_updated = Query(
            query_id="Q-CRUD-1",
            text="Updated question text",
            ground_truth_answer="Updated answer",
        )
        service.update_query(ds.dataset_id, q_updated, validate_chunk_existence=False)

        fetched2 = service.get_dataset(ds.dataset_id)
        assert fetched2.queries[0].text == "Updated question text"

        # Delete query
        deleted = service.delete_query(ds.dataset_id, "Q-CRUD-1")
        assert deleted is True

        fetched3 = service.get_dataset(ds.dataset_id)
        assert fetched3.query_count == 0

    def test_export_and_import_jsonl(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        q1 = Query(
            query_id="Q-EXP-1",
            text="What is NDCG?",
            relevant_chunk_ids=["chk_1"],
            relevance_grades={"chk_1": 2, "chk_2": 1},
            ground_truth_answer="NDCG is Normalized Discounted Cumulative Gain.",
            required_citations=["chk_1"],
            metadata={"difficulty": "medium"},
        )
        q2 = Query(
            query_id="Q-EXP-2",
            text="What is MRR?",
            relevant_chunk_ids=["chk_3"],
            relevance_grades={"chk_3": 1},
            ground_truth_answer="MRR is Mean Reciprocal Rank.",
        )

        ds = service.create_dataset(
            name="Export Benchmark",
            queries=[q1, q2],
            validate_chunk_existence=False,
        )

        jsonl_str = service.export_jsonl(ds.dataset_id)
        assert "Q-EXP-1" in jsonl_str
        assert "Q-EXP-2" in jsonl_str
        assert "What is NDCG?" in jsonl_str

        # Import into new dataset
        imported = service.import_jsonl(
            name="Re-imported Benchmark",
            jsonl_content=jsonl_str,
            validate_chunk_existence=False,
        )
        assert imported.name == "Re-imported Benchmark"
        assert imported.query_count == 2
        assert imported.queries[0].query_id == "Q-EXP-1"
        assert (
            imported.queries[0].ground_truth_answer
            == "NDCG is Normalized Discounted Cumulative Gain."
        )
        assert imported.queries[0].relevance_grades["chk_1"] == 2

    def test_seed_sample_dataset_if_empty(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        # Listing datasets triggers seed
        datasets = service.list_datasets()
        assert len(datasets) >= 1
        sample_ds = datasets[0]
        assert sample_ds.is_synthetic is True
        assert "Synthetic" in sample_ds.name

    def test_reject_unready_document_in_scope(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]
        doc_repo: DocumentRepository = dataset_env["doc_repo"]

        # Insert a document in FAILED state
        from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus

        doc_rec = DocumentRecord(
            doc_id="failed_doc_1",
            filename="broken.pdf",
            storage_path="mock/path",
            file_type="pdf",
            file_size=1024,
            content_hash="hash123",
            status=DocumentStatus.FAILED,
        )
        doc_repo.insert_document(doc_rec)

        q = Query(query_id="Q-FAIL-1", text="Question about failed doc")
        with pytest.raises(DatasetValidationError, match="not in ready state"):
            service.create_dataset(
                name="Failed Doc Test",
                document_scope=["failed_doc_1"],
                queries=[q],
                validate_chunk_existence=False,
            )

    def test_import_jsonl_duplicate_citations_rejected(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        jsonl_line = (
            '{"query_id": "Q-DUP-CIT", "text": "What is precision?", '
            '"required_citations": ["c1", "c1"]}\n'
        )
        with pytest.raises(DatasetValidationError, match="Duplicate citation identifier 'c1'"):
            service.import_jsonl(
                name="Duplicate Citations Dataset",
                jsonl_content=jsonl_line,
                validate_chunk_existence=False,
            )

    def test_import_jsonl_invalid_metadata_rejected(self, dataset_env: dict[str, Any]) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]

        jsonl_line = (
            '{"query_id": "Q-BAD-META", "text": "What is recall?", '
            '"metadata": "not_a_json_object"}\n'
        )
        with pytest.raises(DatasetValidationError, match="Invalid metadata type"):
            service.import_jsonl(
                name="Invalid Meta Dataset",
                jsonl_content=jsonl_line,
                validate_chunk_existence=False,
            )

    def test_import_jsonl_citation_to_chunk_mapping_roundtrip(
        self, dataset_env: dict[str, Any]
    ) -> None:
        service: BenchmarkDatasetService = dataset_env["service"]
        ingestion: DocumentIngestionService = dataset_env["ingestion"]

        doc_id = _ingest(ingestion, "doc_cit.txt", "Text for citation roundtrip testing.")
        chunks = dataset_env["doc_repo"].get_chunks(doc_id)
        assert len(chunks) > 0
        target_chunk_id = chunks[0].chunk_id

        jsonl_line = (
            f'{{"query_id": "Q-CIT-MAP", "text": "What is citation mapping?", '
            f'"relevant_chunk_ids": ["{target_chunk_id}"], '
            f'"relevance_grades": {{"{target_chunk_id}": 2}}, '
            f'"required_citations": ['
            f'{{"citation_id": "cit_alpha", "chunk_id": "{target_chunk_id}"}}]}}\n'
        )

        ds = service.import_jsonl(
            name="Citation Mapping Dataset",
            jsonl_content=jsonl_line,
            document_scope=[doc_id],
            validate_chunk_existence=True,
        )
        assert ds.query_count == 1
        q = ds.queries[0]
        assert "cit_alpha" in q.required_citations
        assert q.metadata.get("citation_to_chunk_id", {}).get("cit_alpha") == target_chunk_id

        # Roundtrip export and reimport
        exported = service.export_jsonl(ds.dataset_id)
        reimported = service.import_jsonl(
            name="Reimported Mapping Dataset",
            jsonl_content=exported,
            document_scope=[doc_id],
            validate_chunk_existence=True,
        )
        assert reimported.query_count == 1
        rq = reimported.queries[0]
        assert "cit_alpha" in rq.required_citations
        assert rq.metadata.get("citation_to_chunk_id", {}).get("cit_alpha") == target_chunk_id
