"""Unit and integration tests for POST /api/query endpoint (STING V2-003)."""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reliability_lab.api.main import (
    create_app,
    get_document_repository,
    get_ingestion_service,
    get_query_service,
)
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.models import DocumentRecord, DocumentStatus, IndexingStatus
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.llm.client import BaseLLMClient
from reliability_lab.llm.models import LLMMessage, LLMResponse, TokenUsage
from reliability_lab.query.orchestrator import DocumentQueryService


class ApiTestEmbedder(BaseEmbedder):
    """Deterministic 4D embedder for API tests."""

    @property
    def model_name(self) -> str:
        return "api-test-model"

    @property
    def embedding_dim(self) -> int:
        return 4

    def embed_texts(self, texts: Any, batch_size: int = 32) -> np.ndarray:
        vecs = []
        for t in texts:
            if "alpha" in str(t).lower():
                vecs.append([1.0, 0.0, 0.0, 0.0])
            elif "beta" in str(t).lower():
                vecs.append([0.0, 1.0, 0.0, 0.0])
            else:
                vecs.append([0.5, 0.5, 0.0, 0.0])
        return np.array(vecs, dtype=np.float32)


class ApiTestLLM(BaseLLMClient):
    """Deterministic mock LLM for API tests."""

    def __init__(self, response_text: str = "Grounded response citing [CIT-001].") -> None:
        self.response_text = response_text
        self.generated_prompts: list[list[LLMMessage]] = []

    @property
    def model_name(self) -> str:
        return "api-mock-llm"

    @property
    def provider_name(self) -> str:
        return "mock-provider"

    def generate(self, messages: list[LLMMessage], **kwargs: Any) -> LLMResponse:
        self.generated_prompts.append(messages)
        return LLMResponse(
            text=self.response_text,
            model=self.model_name,
            usage=TokenUsage(prompt_tokens=100, completion_tokens=25, total_tokens=125),
        )


@pytest.fixture
def app_client(tmp_path: Path):
    db_path = tmp_path / "api_test.db"
    storage_dir = tmp_path / "storage"
    index_dir = tmp_path / "indexes"
    storage_dir.mkdir(parents=True, exist_ok=True)
    index_dir.mkdir(parents=True, exist_ok=True)

    repo = DocumentRepository(db_path)
    embedder = ApiTestEmbedder()
    ingestion = DocumentIngestionService(
        repository=repo,
        storage_dir=storage_dir,
        index_dir=index_dir,
        embedder=embedder,
    )
    mock_llm = ApiTestLLM()
    query_service = DocumentQueryService(
        repository=repo,
        ingestion_service=ingestion,
        embedder=embedder,
        llm_client=mock_llm,
    )

    app = create_app()
    app.dependency_overrides[get_document_repository] = lambda: repo
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_query_service] = lambda: query_service

    with patch("reliability_lab.api.main.get_db_path", return_value=db_path):
        client = TestClient(app)
        yield client, repo, ingestion, mock_llm


def _upload_and_process(
    ingestion: DocumentIngestionService,
    filename: str,
    content: bytes,
) -> DocumentRecord:
    rec = ingestion.save_upload_stream(filename, io.BytesIO(content))
    return ingestion.process_document(rec.doc_id)


class TestQueryApiEndpoints:
    """Tests for POST /api/query."""

    def test_query_success_returns_200_and_trace(self, app_client) -> None:
        client, _repo, ingestion, _llm = app_client
        doc1 = _upload_and_process(ingestion, "doc1.txt", b"Alpha document information.")
        doc2 = _upload_and_process(ingestion, "doc2.txt", b"Beta document details.")

        resp = client.post(
            "/api/query",
            json={
                "query": "Alpha information",
                "document_ids": [doc1.doc_id, doc2.doc_id],
                "strategy": "hybrid",
                "top_k": 3,
            },
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "Grounded response" in data["answer"]
        assert data["query_text"] == "Alpha information"
        assert len(data["selected_doc_ids"]) == 2
        assert data["strategy"] == "hybrid"
        assert data["effective_strategy"] == "hybrid"
        assert data["abstained"] is False
        assert len(data["resolved_citations"]) == 1
        assert data["resolved_citations"][0]["alias"] == "CIT-001"
        assert "trace" in data
        assert data["trace"]["trace_id"].startswith("trc_")
        assert data["trace"]["prompt_tokens"] == 100
        assert data["trace"]["completion_tokens"] == 25

    def test_query_missing_document_returns_404(self, app_client) -> None:
        client, _, _, _ = app_client
        resp = client.post(
            "/api/query",
            json={
                "query": "test query",
                "document_ids": ["non_existent_doc_id"],
                "strategy": "hybrid",
            },
        )
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_query_skipped_empty_doc_returns_422(self, app_client) -> None:
        client, _, ingestion, _ = app_client
        doc = _upload_and_process(ingestion, "empty.txt", b"   ")
        assert doc.indexing_status == IndexingStatus.SKIPPED

        resp = client.post(
            "/api/query",
            json={
                "query": "test query",
                "document_ids": [doc.doc_id],
                "strategy": "hybrid",
            },
        )
        assert resp.status_code == 422
        assert "skipped during extraction" in resp.json()["detail"].lower()

    def test_query_dense_on_partial_doc_returns_422(self, app_client) -> None:
        client, repo, ingestion, _ = app_client
        doc = _upload_and_process(ingestion, "partial.txt", b"Partial document.")
        repo.update_document_status(
            doc.doc_id, status=DocumentStatus.READY, indexing_status=IndexingStatus.PARTIAL
        )

        resp = client.post(
            "/api/query",
            json={
                "query": "test query",
                "document_ids": [doc.doc_id],
                "strategy": "dense",
            },
        )
        assert resp.status_code == 422
        assert "dense index missing" in resp.json()["detail"].lower()

    def test_query_processing_doc_returns_409(self, app_client) -> None:
        client, _, ingestion, _ = app_client
        rec = ingestion.save_upload_stream("uploading.txt", io.BytesIO(b"Data"))

        resp = client.post(
            "/api/query",
            json={
                "query": "test query",
                "document_ids": [rec.doc_id],
                "strategy": "hybrid",
            },
        )
        assert resp.status_code == 409
        assert "processing" in resp.json()["detail"].lower()

    def test_query_abstention_returns_200_with_abstained_flag(self, app_client) -> None:
        client, _, ingestion, llm = app_client
        doc = _upload_and_process(ingestion, "fruit.txt", b"Apples and bananas.")

        # Out-of-vocabulary BM25 query
        resp = client.post(
            "/api/query",
            json={
                "query": "xylophone zebra quantum",
                "document_ids": [doc.doc_id],
                "strategy": "bm25",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["abstained"] is True
        assert data["retrieval_abstained"] is True
        assert len(llm.generated_prompts) == 0

    def test_query_privacy_zero_host_paths(self, app_client) -> None:
        client, _repo, ingestion, _ = app_client
        doc = _upload_and_process(ingestion, "privacy.txt", b"Privacy sensitive document.")

        resp = client.post(
            "/api/query",
            json={
                "query": "Privacy",
                "document_ids": [doc.doc_id],
                "strategy": "hybrid",
            },
        )
        assert resp.status_code == 200
        resp_text = resp.text
        # Must not contain host directory markers
        assert str(ingestion.index_dir) not in resp_text
        assert str(ingestion.storage_dir) not in resp_text
