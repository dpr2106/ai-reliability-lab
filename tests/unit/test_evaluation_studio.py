"""Unit tests for Evaluation Studio: execution, provenance, and failure diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from reliability_lab.api.main import create_app
from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
from reliability_lab.benchmark.service import BenchmarkDatasetService
from reliability_lab.core.models import Query
from reliability_lab.evaluation.execution_service import (
    EvaluationExecutionService,
    ExperimentExecutionConfig,
)
from reliability_lab.evaluation.metrics_meta import METRIC_DEFINITIONS, classify_query_failure
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import MockLLMClient
from reliability_lab.storage.database import Database
from reliability_lab.storage.repository import ExperimentRepository
from tests.unit.test_investigation_hardening import DeterministicEmbedder, _ingest


@pytest.fixture
def eval_env(tmp_path: Path) -> dict[str, Any]:
    db_path = tmp_path / "eval_test.db"
    db = Database(str(db_path))
    exp_repo = ExperimentRepository(db)
    doc_repo = DocumentRepository(db_path)
    ds_repo = BenchmarkDatasetRepository(db)

    embedder = DeterministicEmbedder()
    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )

    eval_service = EvaluationExecutionService(
        experiment_repository=exp_repo,
        document_repository=doc_repo,
        dataset_repository=ds_repo,
        ingestion_service=ingestion,
        embedder=embedder,
    )

    return {
        "db": db,
        "exp_repo": exp_repo,
        "doc_repo": doc_repo,
        "ds_repo": ds_repo,
        "ingestion": ingestion,
        "eval_service": eval_service,
        "embedder": embedder,
    }


class TestEvaluationExecutionAndProvenance:
    """Tests for pipeline execution, provenance tracking, and leak prevention."""

    def test_execute_evaluation_with_mock_provenance(self, eval_env: dict[str, Any]) -> None:
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_id = _ingest(
            ingestion,
            "sample_doc.txt",
            "This document explains that BM25 uses term frequency saturation parameter k1=1.5.",
        )
        chunks = eval_env["doc_repo"].get_chunks(doc_id)
        assert len(chunks) > 0

        q = Query(
            query_id="Q-BM25",
            text="What is BM25 parameter k1?",
            relevant_chunk_ids=[chunks[0].chunk_id],
            relevance_grades={chunks[0].chunk_id: 2},
            ground_truth_answer="k1 is 1.5",
        )
        ds = ds_repo.create_dataset(
            name="Sample Eval Benchmark",
            document_scope=[doc_id],
            queries=[q],
        )

        cfg = ExperimentExecutionConfig(
            experiment_name="Mock BM25 Test Run",
            dataset_id=ds.dataset_id,
            document_ids=[doc_id],
            retriever_type="bm25",
            provider="mock",
            model_name="mock-model",
            enable_judge=False,
            enable_nli=False,
        )

        report = eval_service.execute_evaluation(cfg)

        assert report.report_id is not None
        assert report.experiment_name == "Mock BM25 Test Run"
        assert any("recall" in k.lower() for k in report.metrics)
        assert report.metrics.get("recall@5", report.metrics.get("mean_recall@5")) == 1.0

        # Verify provenance was saved in repository metadata
        stored = eval_env["exp_repo"].get_run(report.report_id)
        assert stored is not None
        meta = stored.metadata or {}
        assert meta.get("provenance_type") == "mock"
        assert meta.get("is_mock") is True
        assert meta.get("mock_warning") is True

    def test_ground_truth_never_leaks_into_generation_prompt(
        self, eval_env: dict[str, Any]
    ) -> None:
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_id = _ingest(
            ingestion,
            "secret_doc.txt",
            "Public passage discussing quantum mechanics and wavefunctions.",
        )
        c = eval_env["doc_repo"].get_chunks(doc_id)[0]

        secret_answer = "SECRET_GROUND_TRUTH_TOKEN_99887766"
        secret_citation = "SECRET_CITATION_MAP_112233"

        q = Query(
            query_id="Q-LEAK",
            text="Explain wavefunctions.",
            relevant_chunk_ids=[c.chunk_id],
            relevance_grades={c.chunk_id: 2},
            ground_truth_answer=secret_answer,
            required_citations=[secret_citation],
        )
        ds = ds_repo.create_dataset(
            name="Leak Test Dataset",
            document_scope=[doc_id],
            queries=[q],
        )

        mock_generator = MockLLMClient()
        recorded_prompts: list[Any] = []

        orig_generate = mock_generator.generate

        def capture_generate(prompt: Any, config: Any = None) -> Any:
            recorded_prompts.append(prompt)
            return orig_generate(prompt, config)

        mock_generator.generate = capture_generate  # type: ignore[method-assign]

        with patch.object(eval_service, "_build_generator", return_value=mock_generator):
            cfg = ExperimentExecutionConfig(
                experiment_name="Leak Prevention Check",
                dataset_id=ds.dataset_id,
                document_ids=[doc_id],
                retriever_type="bm25",
                provider="mock",
                enable_judge=False,
                enable_nli=False,
            )
            eval_service.execute_evaluation(cfg)

        assert len(recorded_prompts) == 1
        all_prompt_text = " ".join(m.content for m in recorded_prompts[0] if hasattr(m, "content"))
        # CRITICAL: Ground-truth answer and citation must NEVER appear in generation prompt
        assert secret_answer not in all_prompt_text
        assert secret_citation not in all_prompt_text

    def test_unsupported_retriever_raises_error(self, eval_env: dict[str, Any]) -> None:
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_id = _ingest(ingestion, "doc_retriever_test.txt", "Some chunk content.")
        ds = ds_repo.create_dataset(
            name="Dummy DS",
            document_scope=[doc_id],
            queries=[Query(query_id="Q1", text="Sample")],
        )
        cfg = ExperimentExecutionConfig(
            experiment_name="Invalid Retriever Test",
            dataset_id=ds.dataset_id,
            document_ids=[doc_id],
            retriever_type="quantum_teleportation_retriever",
        )
        with pytest.raises(ValueError, match="Unsupported retriever_type"):
            eval_service.execute_evaluation(cfg)

    def test_background_job_execution(self, eval_env: dict[str, Any]) -> None:
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_id = _ingest(ingestion, "doc_bg.txt", "Content for async background execution.")
        q = Query(query_id="Q-BG", text="Question for background test")
        ds = ds_repo.create_dataset(name="Async Test", document_scope=[doc_id], queries=[q])

        cfg = ExperimentExecutionConfig(
            experiment_name="Async Job Test",
            dataset_id=ds.dataset_id,
            document_ids=[doc_id],
            retriever_type="bm25",
            provider="mock",
            enable_judge=False,
            enable_nli=False,
        )

        job = eval_service.start_evaluation_job(cfg)
        assert job.job_id.startswith("job_")
        assert job.status in ("pending", "running", "completed")

        # Wait briefly for synchronous thread execution in unit test
        import time

        for _ in range(50):
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert job.status == "completed"
        assert job.run_id is not None
        assert job.progress_percent == 100.0


class TestFailureExplorerAndMetricsMeta:
    """Tests for query failure classification and metric metadata catalog."""

    def test_failure_classification_categories(self) -> None:
        # Complete retrieval miss
        miss_q = {"metrics": {"HitRate@K": 0.0, "Recall@K": 0.0}}
        res_miss = classify_query_failure(miss_q)
        assert res_miss["classification"] == "retrieval_miss"
        assert res_miss["severity"] == "critical"
        assert res_miss["has_failure"] is True

        # Partial retrieval
        partial_q = {"metrics": {"HitRate@K": 1.0, "Recall@K": 0.5}}
        res_partial = classify_query_failure(partial_q)
        assert res_partial["classification"] == "partial_retrieval"
        assert res_partial["severity"] == "moderate"

        # Citation hallucination
        halluc_q = {
            "metrics": {
                "HitRate@K": 1.0,
                "Recall@K": 1.0,
                "citation_precision": 0.5,
                "citation_recall": 1.0,
            }
        }
        res_halluc = classify_query_failure(halluc_q)
        assert res_halluc["classification"] == "citation_hallucination"

        # Clean pass
        clean_q = {
            "metrics": {
                "HitRate@K": 1.0,
                "Recall@K": 1.0,
                "citation_precision": 1.0,
                "citation_recall": 1.0,
                "lexical_groundedness": 0.9,
            }
        }
        res_clean = classify_query_failure(clean_q)
        assert res_clean["classification"] == "passed"
        assert res_clean["has_failure"] is False

    def test_metrics_definitions_coverage(self) -> None:
        assert "Recall@K" in METRIC_DEFINITIONS
        assert "Precision@K" in METRIC_DEFINITIONS
        assert "HitRate@K" in METRIC_DEFINITIONS
        assert "MRR@K" in METRIC_DEFINITIONS
        assert "NDCG@K" in METRIC_DEFINITIONS
        assert "citation_precision" in METRIC_DEFINITIONS
        assert "citation_recall" in METRIC_DEFINITIONS
        assert "lexical_groundedness" in METRIC_DEFINITIONS
        assert "total_latency_ms" in METRIC_DEFINITIONS

        for _m_name, meta in METRIC_DEFINITIONS.items():
            assert "name" in meta
            assert "direction" in meta
            assert "description" in meta
            assert "limitations" in meta


class TestApiIntegrationEvaluationStudio:
    """End-to-end FastAPI endpoint integration tests."""

    @pytest.fixture
    def client(self, tmp_path: Path) -> TestClient:
        db_path = tmp_path / "api_test.db"
        db = Database(str(db_path))
        doc_repo = DocumentRepository(db_path)
        ds_repo = BenchmarkDatasetRepository(db)
        ds_service = BenchmarkDatasetService(ds_repo, doc_repo)

        app = create_app()
        from reliability_lab.api.main import (
            get_dataset_repository,
            get_dataset_service,
            get_document_repository,
        )

        app.dependency_overrides[get_dataset_repository] = lambda: ds_repo
        app.dependency_overrides[get_dataset_service] = lambda: ds_service
        app.dependency_overrides[get_document_repository] = lambda: doc_repo
        return TestClient(app)

    def test_api_dataset_crud_and_export(self, client: TestClient) -> None:
        # 1. Create dataset
        payload = {
            "name": "Integration API Dataset",
            "description": "API test dataset",
            "queries": [
                {
                    "query_id": "API-Q1",
                    "text": "What is hybrid retrieval?",
                    "relevant_chunk_ids": ["chk_1"],
                    "relevance_grades": {"chk_1": 2},
                    "ground_truth_answer": "Hybrid combines sparse and dense retrieval.",
                }
            ],
        }
        res = client.post("/api/datasets", json=payload)
        assert res.status_code == 201
        data = res.json()
        ds_id = data["dataset_id"]
        assert data["name"] == "Integration API Dataset"
        assert data["query_count"] == 1

        # 2. List datasets
        res_list = client.get("/api/datasets")
        assert res_list.status_code == 200
        assert any(d["dataset_id"] == ds_id for d in res_list.json()["datasets"])

        # 3. Add query
        res_q = client.post(
            f"/api/datasets/{ds_id}/queries",
            json={"query_id": "API-Q2", "text": "Second query text"},
        )
        assert res_q.status_code == 200
        assert res_q.json()["query_id"] == "API-Q2"

        # 4. Export JSONL
        res_exp = client.get(f"/api/datasets/{ds_id}/export")
        assert res_exp.status_code == 200
        assert "application/x-jsonlines" in res_exp.headers["content-type"]
        assert "API-Q1" in res_exp.text
        assert "API-Q2" in res_exp.text

        # 5. Delete dataset
        res_del = client.delete(f"/api/datasets/{ds_id}")
        assert res_del.status_code == 200
        assert res_del.json()["deleted"] is True

    def test_api_metrics_meta_endpoint(self, client: TestClient) -> None:
        res = client.get("/api/evaluation/metrics-meta")
        assert res.status_code == 200
        data = res.json()
        assert "metrics" in data
        assert "Recall@K" in data["metrics"]
        assert data["metrics"]["Recall@K"]["direction"] == "higher"


class TestHardenedEvaluationStudioBehaviors:
    """Targeted regression tests for Priorities 1, 2, 3, and 4."""

    def test_resolve_embedder_actionable_error_when_st_fails(
        self, eval_env: dict[str, Any]
    ) -> None:
        """Priority 1: Ensure missing sentence-transformers fails with actionable error."""
        service = EvaluationExecutionService(
            experiment_repository=eval_env["exp_repo"],
            document_repository=eval_env["doc_repo"],
            dataset_repository=eval_env["ds_repo"],
            embedder=None,
        )

        with patch(
            "reliability_lab.evaluation.execution_service.SentenceTransformerEmbedder",
            side_effect=ImportError("sentence-transformers not installed"),
        ):
            with pytest.raises(RuntimeError, match="Dense vector embedding engine is unavailable"):
                service._resolve_embedder()

    def test_multi_query_incremental_progress(self, eval_env: dict[str, Any]) -> None:
        """Priority 2: Progress updates incrementally after each evaluated query."""
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_id = _ingest(
            ingestion, "doc_multi_prog.txt", "Content for multi-query progress tracking."
        )
        q1 = Query(query_id="Q-P1", text="First progress query")
        q2 = Query(query_id="Q-P2", text="Second progress query")
        q3 = Query(query_id="Q-P3", text="Third progress query")

        ds = ds_repo.create_dataset(
            name="Multi Prog Dataset", document_scope=[doc_id], queries=[q1, q2, q3]
        )

        progress_observations: list[tuple[int, int]] = []

        cfg = ExperimentExecutionConfig(
            experiment_name="Multi Progress Run",
            dataset_id=ds.dataset_id,
            document_ids=[doc_id],
            retriever_type="bm25",
            provider="mock",
            enable_judge=False,
            enable_nli=False,
        )

        def _cb(completed: int, total: int) -> None:
            progress_observations.append((completed, total))

        eval_service.execute_evaluation(cfg, progress_callback=_cb)

        assert len(progress_observations) == 3
        assert progress_observations[0] == (1, 3)
        assert progress_observations[1] == (2, 3)
        assert progress_observations[2] == (3, 3)

    def test_job_early_failure_preserves_last_completed_count(
        self, eval_env: dict[str, Any]
    ) -> None:
        """Priority 2: Early job failure preserves queries_completed and records failure message."""
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        q = Query(query_id="Q-FAIL-EARLY", text="Failing query")
        ds = ds_repo.create_dataset(name="Fail Early DS", document_scope=[], queries=[q])

        # An experiment with missing document scope will raise ValueError during execute_evaluation
        cfg = ExperimentExecutionConfig(
            experiment_name="Fail Early Run",
            dataset_id=ds.dataset_id,
            document_ids=[],
            retriever_type="bm25",
            provider="mock",
        )

        job = eval_service.start_evaluation_job(cfg)
        import time

        for _ in range(50):
            if job.status in ("completed", "failed"):
                break
            time.sleep(0.05)

        assert job.status == "failed"
        assert job.error_message is not None
        assert "explicit document scope" in job.error_message
        assert job.queries_completed == 0
        assert job.progress_percent == 0.0

    def test_scope_divergence_rejected(self, eval_env: dict[str, Any]) -> None:
        """Priority 3: Experiment document selection cannot silently diverge from dataset scope."""
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc1 = _ingest(ingestion, "doc1_scoped.txt", "Content of doc1.")
        doc2 = _ingest(ingestion, "doc2_out.txt", "Content of doc2.")

        q = Query(query_id="Q-DIV", text="Question about doc1")
        ds = ds_repo.create_dataset(name="Scoped DS", document_scope=[doc1], queries=[q])

        cfg = ExperimentExecutionConfig(
            experiment_name="Divergence Test",
            dataset_id=ds.dataset_id,
            document_ids=[doc1, doc2],  # doc2 is outside dataset scope
            retriever_type="bm25",
            provider="mock",
        )

        with pytest.raises(ValueError, match="outside dataset scope"):
            eval_service.execute_evaluation(cfg)

    def test_empty_experiment_scope_defaults_to_dataset_scope(
        self, eval_env: dict[str, Any]
    ) -> None:
        """Priority 3: Empty experiment scope defaults to dataset scope without leakage."""
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_scoped = _ingest(ingestion, "scoped_only.txt", "Content for scoped query.")
        _ingest(ingestion, "leak_check.txt", "Unscoped content that must not be queried.")

        q = Query(query_id="Q-DFLT", text="Default scope question")
        ds = ds_repo.create_dataset(
            name="Default Scope DS", document_scope=[doc_scoped], queries=[q]
        )

        cfg = ExperimentExecutionConfig(
            experiment_name="Default Scope Run",
            dataset_id=ds.dataset_id,
            document_ids=[],  # empty should default to ds.record.document_scope
            retriever_type="bm25",
            provider="mock",
            enable_judge=False,
            enable_nli=False,
        )

        report = eval_service.execute_evaluation(cfg)
        saved_run = eval_env["exp_repo"].get_run(report.report_id)
        assert saved_run is not None
        assert saved_run.metadata.get("document_ids") == [doc_scoped]

    def test_missing_or_failed_corpus_document_fails_closed(self, eval_env: dict[str, Any]) -> None:
        """Priority 3: Missing or non-ready documents fail closed."""
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]
        doc_repo: DocumentRepository = eval_env["doc_repo"]

        q = Query(query_id="Q-MISS-DOC", text="Query with missing doc")
        ds = ds_repo.create_dataset(
            name="Missing Doc DS", document_scope=["non_existent_doc_id"], queries=[q]
        )

        cfg = ExperimentExecutionConfig(
            experiment_name="Missing Doc Run",
            dataset_id=ds.dataset_id,
            document_ids=["non_existent_doc_id"],
            retriever_type="bm25",
            provider="mock",
        )

        with pytest.raises(ValueError, match="not found in repository"):
            eval_service.execute_evaluation(cfg)

        # Non-ready status failure
        from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus

        doc_rec = DocumentRecord(
            doc_id="failing_doc_test",
            filename="failed.pdf",
            storage_path="mock/path",
            file_type="pdf",
            file_size=1024,
            content_hash="h1",
            status=DocumentStatus.FAILED,
        )
        doc_repo.insert_document(doc_rec)

        ds_failed = ds_repo.create_dataset(
            name="Failed Doc DS", document_scope=["failing_doc_test"], queries=[q]
        )

        cfg_failed = ExperimentExecutionConfig(
            experiment_name="Failed Doc Run",
            dataset_id=ds_failed.dataset_id,
            document_ids=["failing_doc_test"],
            retriever_type="bm25",
            provider="mock",
        )

        with pytest.raises(ValueError, match="not in ready state"):
            eval_service.execute_evaluation(cfg_failed)

    def test_reproducibility_distinguishes_tiers(self, eval_env: dict[str, Any]) -> None:
        """Priority 4: Reproducibility accurately distinguishes exact vs partial vs impossible."""
        eval_service: EvaluationExecutionService = eval_env["eval_service"]
        ingestion: DocumentIngestionService = eval_env["ingestion"]
        ds_repo: BenchmarkDatasetRepository = eval_env["ds_repo"]

        doc_id = _ingest(ingestion, "repro_doc.txt", "Content for reproducibility evaluation.")
        q = Query(query_id="Q-REPRO", text="Reproducibility question")
        ds = ds_repo.create_dataset(name="Repro DS", document_scope=[doc_id], queries=[q])

        cfg = ExperimentExecutionConfig(
            experiment_name="Repro Baseline Run",
            dataset_id=ds.dataset_id,
            document_ids=[doc_id],
            retriever_type="bm25",
            provider="mock",
            enable_judge=False,
            enable_nli=False,
        )

        report = eval_service.execute_evaluation(cfg)
        run_id = report.report_id

        app = create_app()
        from reliability_lab.api.main import (
            get_dataset_repository,
            get_document_repository,
            get_repository,
        )

        db_file = str(eval_env["db"].path)
        app.dependency_overrides[get_repository] = lambda: ExperimentRepository(Database(db_file))
        app.dependency_overrides[get_dataset_repository] = lambda: BenchmarkDatasetRepository(
            Database(db_file)
        )
        app.dependency_overrides[get_document_repository] = lambda: DocumentRepository(
            Path(db_file)
        )

        client = TestClient(app)

        # 1. Exact reproduction possible immediately after run
        res_exact = client.get(f"/api/runs/{run_id}/reproducibility")
        assert res_exact.status_code == 200
        data_exact = res_exact.json()
        assert data_exact["reproducibility_tier"] == "exact"
        assert data_exact["can_reproduce"] is True
        assert data_exact["dataset_unchanged"] is True
        assert data_exact["corpus_unchanged"] is True

        # 2. Modify dataset -> tier becomes 'partial', can_reproduce is False
        q_mod = Query(query_id="Q-REPRO", text="Modified query text that changes hash")
        ds_repo.update_query(ds.dataset_id, q_mod)

        res_partial = client.get(f"/api/runs/{run_id}/reproducibility")
        assert res_partial.status_code == 200
        data_partial = res_partial.json()
        assert data_partial["reproducibility_tier"] == "partial"
        assert data_partial["can_reproduce"] is False
        assert data_partial["dataset_unchanged"] is False

        # 3. Delete dataset -> tier becomes 'impossible'
        ds_repo.delete_dataset(ds.dataset_id)
        res_imp = client.get(f"/api/runs/{run_id}/reproducibility")
        assert res_imp.status_code == 200
        data_imp = res_imp.json()
        assert data_imp["reproducibility_tier"] == "impossible"
        assert data_imp["can_reproduce"] is False
