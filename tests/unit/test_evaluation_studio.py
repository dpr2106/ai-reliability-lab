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
