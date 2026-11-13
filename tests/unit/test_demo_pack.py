"""Unit tests for STING Demo Pack manifest, evidence resolution, and preparation workflow."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliability_lab.core.models import Chunk, Document
from reliability_lab.ingestion.chunker import RecursiveCharacterChunker
from scripts.prepare_sting_demo import (
    prepare_demo_local,
    resolve_anchors_to_chunks,
    validate_demo_pack,
)


@pytest.fixture
def demo_dir() -> Path:
    """Return path to data/demo directory."""
    root = Path(__file__).resolve().parents[2]
    return root / "data" / "demo"


class TestDemoPackManifestAndCorpus:
    """Validate demo pack integrity, file presence, and question structure."""

    def test_demo_manifest_structure_and_completeness(self, demo_dir: Path) -> None:
        """Ensure manifest defines at least 3 documents and all files exist and are non-empty."""
        manifest, questions = validate_demo_pack(demo_dir)

        assert manifest["corpus_id"] == "demo-northstar-v2"
        assert len(manifest["documents"]) >= 3
        assert len(questions) >= 8

        corpus_dir = demo_dir / "corpus"
        for doc in manifest["documents"]:
            assert doc["filename"] in (
                "architecture.md",
                "benchmark_report.md",
                "incident_report.md",
            )
            fpath = corpus_dir / doc["filename"]
            assert fpath.exists(), f"Corpus file '{doc['filename']}' is missing."
            content = fpath.read_text(encoding="utf-8")
            assert len(content) > 500, f"Corpus file '{doc['filename']}' is unexpectedly small."
            assert doc["doc_identifier"] in content, f"Identifier missing from '{doc['filename']}'."

    def test_golden_questions_structure_and_coverage(self, demo_dir: Path) -> None:
        """Validate question schemas, required fields, and category coverage."""
        bench_file = demo_dir / "golden_questions.json"
        assert bench_file.exists()

        spec = json.loads(bench_file.read_text(encoding="utf-8"))
        questions = spec.get("questions", [])
        assert 8 <= len(questions) <= 12

        categories = {q["category"] for q in questions}
        assert "keyword_retrieval" in categories
        assert "semantic_retrieval" in categories
        assert "methodology" in categories
        assert "safe_abstention" in categories
        assert "failure_analysis" in categories

        seen_qids: set[str] = set()
        for q in questions:
            qid = q["query_id"]
            assert qid not in seen_qids, f"Duplicate query ID: {qid}"
            seen_qids.add(qid)
            assert len(q["text"].strip()) > 10

            if q["category"] == "safe_abstention":
                assert q["source_filename"] is None
                assert q["evidence_anchor"] is None
                assert q["relevance_grade"] == 0
                assert "INSUFFICIENT_EVIDENCE" in q["reference_answer"]
            else:
                assert q["source_filename"] in (
                    "architecture.md",
                    "benchmark_report.md",
                    "incident_report.md",
                )
                assert isinstance(q["evidence_anchor"], str)
                assert len(q["evidence_anchor"].strip()) > 5
                assert q["relevance_grade"] > 0
                assert q["required_citation"] is not None


class TestEvidenceAnchorResolution:
    """Validate fail-closed evidence anchor resolution to generated chunks."""

    @pytest.fixture
    def chunked_corpus(self, demo_dir: Path) -> dict[str, list[Chunk]]:
        """Chunk sample documents using standard recursive chunker."""
        chunker = RecursiveCharacterChunker(chunk_size=500, chunk_overlap=100)
        chunks_by_file: dict[str, list[Chunk]] = {}
        corpus_dir = demo_dir / "corpus"

        for fname in ("architecture.md", "benchmark_report.md", "incident_report.md"):
            text = (corpus_dir / fname).read_text(encoding="utf-8")
            doc = Document(doc_id=f"doc_{fname[:4]}", content=text, metadata={"filename": fname})
            chunks_by_file[fname] = chunker.chunk(doc)

        return chunks_by_file

    def test_unique_anchor_to_chunk_resolution(
        self, demo_dir: Path, chunked_corpus: dict[str, list[Chunk]]
    ) -> None:
        """Every golden question's anchor must resolve to exactly one chunk."""
        _, questions = validate_demo_pack(demo_dir)
        resolved = resolve_anchors_to_chunks(questions, chunked_corpus)

        assert len(resolved) == len(questions)

        for item in resolved:
            if item["metadata"]["category"] == "safe_abstention":
                assert item["relevant_chunk_ids"] == []
                assert item["relevance_grades"] == {}
                assert item["required_citations"] == []
            else:
                assert len(item["relevant_chunk_ids"]) == 1
                chunk_id = item["relevant_chunk_ids"][0]
                assert item["relevance_grades"][chunk_id] > 0
                assert len(item["required_citations"]) == 1
                cit = item["required_citations"][0]
                assert item["metadata"]["citation_to_chunk_id"][cit] == chunk_id

    def test_anchor_resolution_fail_closed_on_missing_anchor(
        self, chunked_corpus: dict[str, list[Chunk]]
    ) -> None:
        """Non-existent textual anchor raises actionable ValueError."""
        bogus_questions = [
            {
                "query_id": "ERR-01",
                "text": "Where is the phantom text?",
                "source_filename": "architecture.md",
                "evidence_anchor": "THIS_STRING_DOES_NOT_EXIST_IN_CORPUS",
                "relevance_grade": 2,
            }
        ]

        with pytest.raises(ValueError, match="was not found in any ingested chunk"):
            resolve_anchors_to_chunks(bogus_questions, chunked_corpus)

    def test_anchor_resolution_fail_closed_on_ambiguous_anchor(
        self, chunked_corpus: dict[str, list[Chunk]]
    ) -> None:
        """Ambiguous anchor matching multiple chunks raises actionable ValueError."""
        ambiguous_questions = [
            {
                "query_id": "ERR-02",
                "text": "What does Northstar do?",
                "source_filename": "architecture.md",
                "evidence_anchor": "Northstar",  # Common term appearing across several chunks
                "relevance_grade": 2,
            }
        ]

        with pytest.raises(ValueError, match=r"ambiguous anchor .* matched [0-9]+ distinct chunks"):
            resolve_anchors_to_chunks(ambiguous_questions, chunked_corpus)

    def test_anchor_resolution_fail_closed_on_unknown_document(
        self, chunked_corpus: dict[str, list[Chunk]]
    ) -> None:
        """Targeting an unindexed document raises actionable ValueError."""
        invalid_doc_questions = [
            {
                "query_id": "ERR-03",
                "text": "What about the missing file?",
                "source_filename": "non_existent_doc.md",
                "evidence_anchor": "Some anchor",
                "relevance_grade": 2,
            }
        ]

        with pytest.raises(ValueError, match=r"references unknown document 'non_existent_doc.md'"):
            resolve_anchors_to_chunks(invalid_doc_questions, chunked_corpus)


class TestRepeatableDemoPreparation:
    """Validate repeatable execution and data preservation in local service mode."""

    def test_safe_repeat_preparation_preserves_records(
        self, tmp_path: Path, demo_dir: Path
    ) -> None:
        """Running preparation twice reuses existing records without duplication or data loss."""
        db_path = tmp_path / "test_demo.sqlite3"
        storage_dir = tmp_path / "documents"
        index_dir = tmp_path / "indexes"

        # First run: Ingests documents and creates dataset
        res1 = prepare_demo_local(
            db_path=db_path,
            storage_dir=storage_dir,
            index_dir=index_dir,
            demo_dir=demo_dir,
            reuse_existing=True,
        )

        assert len(res1.documents) == 3
        assert res1.query_count == 10
        assert res1.dataset_id != ""

        # Second run: Must reuse existing document records and benchmark dataset
        res2 = prepare_demo_local(
            db_path=db_path,
            storage_dir=storage_dir,
            index_dir=index_dir,
            demo_dir=demo_dir,
            reuse_existing=True,
        )

        # Assert accurate golden query and anchor breakdown
        assert res1.query_count == 10
        assert res1.evidence_anchored_queries == 9
        assert res1.resolved_anchors == 9
        assert res1.intentional_abstentions == 1
        assert res1.mode == "local"
        assert res1.db_path == str(db_path)

        # Assert identical IDs reused
        assert res2.documents == res1.documents
        assert res2.dataset_id == res1.dataset_id
        assert res2.query_count == res1.query_count
        assert res2.evidence_anchored_queries == res1.evidence_anchored_queries
        assert res2.resolved_anchors == res1.resolved_anchors
        assert res2.intentional_abstentions == res1.intentional_abstentions

        # Verify database record counts
        from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
        from reliability_lab.ingestion.repository import DocumentRepository
        from reliability_lab.storage.database import Database

        db = Database(db_path)
        doc_repo = DocumentRepository(db)
        ds_repo = BenchmarkDatasetRepository(db)

        assert doc_repo.count_documents() == 3
        # 1 bundled default dataset seeded automatically + 1 Northstar demo dataset
        assert ds_repo.count_datasets() >= 1
        northstar_datasets = [
            ds for ds in ds_repo.list_datasets() if ds.name == "Northstar Golden Benchmark"
        ]
        assert len(northstar_datasets) == 1

    def test_api_mode_fails_closed_when_server_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """API mode must fail with code 1 and not fall back to local mode."""
        from scripts.prepare_sting_demo import main

        dummy_args = [
            "prepare_sting_demo.py",
            "--api-url",
            "http://127.0.0.1:59999",
            "--demo-dir",
            str(tmp_path),
        ]
        monkeypatch.setattr("sys.argv", dummy_args)
        exit_code = main()
        assert exit_code == 1

    def test_receipt_json_generation_and_schema(
        self, tmp_path: Path, demo_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Preparation script emits machine-readable JSON receipt with dynamic identifiers."""
        from scripts.prepare_sting_demo import main

        receipt_file = tmp_path / "receipt.json"
        db_file = tmp_path / "receipt_test.sqlite3"
        storage_dir = tmp_path / "storage"
        index_dir = tmp_path / "indexes"

        cli_args = [
            "prepare_sting_demo.py",
            "--local",
            "--demo-dir",
            str(demo_dir),
            "--db-path",
            str(db_file),
            "--storage-dir",
            str(storage_dir),
            "--index-dir",
            str(index_dir),
            "--receipt-json",
            str(receipt_file),
        ]
        monkeypatch.setattr("sys.argv", cli_args)
        exit_code = main()
        assert exit_code == 0
        assert receipt_file.exists()

        data = json.loads(receipt_file.read_text(encoding="utf-8"))
        assert data["mode"] == "local"
        assert data["db_path"] == str(db_file)
        assert data["dataset_name"] == "Northstar Golden Benchmark"
        assert data["dataset_id"].startswith("ds_")
        assert len(data["document_ids"]) == 3
        assert data["total_queries"] == 10
        assert data["evidence_anchored_queries"] == 9
        assert data["resolved_anchors"] == 9
        assert data["intentional_abstentions"] == 1
