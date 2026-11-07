"""Unit tests for Investigation REST API endpoints."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from reliability_lab.api.main import (
    create_app,
    get_document_repository,
    get_ingestion_service,
    get_investigation_repository,
    get_investigation_service,
)
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.investigation.repository import InvestigationRepository
from reliability_lab.investigation.service import InvestigationService
from reliability_lab.llm.client import MockLLMClient
from reliability_lab.storage.database import Database


class ApiTestEmbedder(BaseEmbedder):
    @property
    def model_name(self) -> str:
        return "api-test-model"

    @property
    def artifact_fingerprint(self) -> str:
        return "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    @property
    def embedding_dim(self) -> int:
        return 4

    def embed_texts(self, texts: Any, batch_size: int = 32) -> np.ndarray:
        return np.full((len(texts), 4), 0.5, dtype=np.float32)


@pytest.fixture
def api_env(tmp_path: Path):
    db_path = tmp_path / "api_test.db"
    db = Database(str(db_path))
    doc_repo = DocumentRepository(db_path)
    inv_repo = InvestigationRepository(db)

    embedder = ApiTestEmbedder()
    ingestion = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=tmp_path / "storage",
        index_dir=tmp_path / "indexes",
        embedder=embedder,
    )

    # Ingest two documents
    rec1 = ingestion.save_upload_stream("doc1.txt", io.BytesIO(b"Alpha paper content about speed."))
    ingestion.process_document(rec1.doc_id)

    rec2 = ingestion.save_upload_stream("doc2.txt", io.BytesIO(b"Beta paper content about memory."))
    ingestion.process_document(rec2.doc_id)

    chunks1 = doc_repo.get_chunks(rec1.doc_id)

    mock_llm = MockLLMClient(
        default_response=json.dumps(
            {
                "findings": [
                    {
                        "topic_name": "Performance",
                        "claim": "Speed is optimized.",
                        "chunk_id": chunks1[0].chunk_id,
                        "page_number": 1,
                        "supporting_passage": "Alpha paper content about speed.",
                        "relationship": "supports",
                        "confidence": 0.95,
                    }
                ]
            }
        )
    )

    inv_service = InvestigationService(
        investigation_repository=inv_repo,
        document_repository=doc_repo,
        llm_client=mock_llm,
    )

    app = create_app()
    app.dependency_overrides[get_document_repository] = lambda: doc_repo
    app.dependency_overrides[get_ingestion_service] = lambda: ingestion
    app.dependency_overrides[get_investigation_repository] = lambda: inv_repo
    app.dependency_overrides[get_investigation_service] = lambda: inv_service

    client = TestClient(app)

    return {
        "client": client,
        "doc1_id": rec1.doc_id,
        "doc2_id": rec2.doc_id,
        "inv_repo": inv_repo,
        "inv_service": inv_service,
    }


def test_create_and_run_investigation_api(api_env: dict[str, Any]) -> None:
    client = api_env["client"]
    doc1_id = api_env["doc1_id"]
    doc2_id = api_env["doc2_id"]

    # 1. Create with run_immediately=True
    create_payload = {
        "title": "Cross-Doc Speed Audit",
        "query": "Compare speed and latency",
        "document_ids": [doc1_id, doc2_id],
        "aspects": ["Performance"],
        "run_immediately": True,
    }
    resp = client.post("/api/investigations", json=create_payload)
    assert resp.status_code == 201
    data = resp.json()
    assert data["investigation_id"].startswith("inv_")
    assert data["title"] == "Cross-Doc Speed Audit"
    assert data["status"] == "completed"
    assert data["report"] is not None
    inv_id = data["investigation_id"]

    # 2. Get investigation detail
    get_resp = client.get(f"/api/investigations/{inv_id}")
    assert get_resp.status_code == 200
    detail = get_resp.json()
    assert detail["investigation_id"] == inv_id
    assert detail["status"] == "completed"
    assert len(detail["report"]["topics"]) == 1

    # 3. List investigations
    list_resp = client.get("/api/investigations")
    assert list_resp.status_code == 200
    list_data = list_resp.json()
    assert list_data["total"] >= 1
    assert any(i["investigation_id"] == inv_id for i in list_data["investigations"])

    # 4. Export Markdown
    export_resp = client.get(f"/api/investigations/{inv_id}/export")
    assert export_resp.status_code == 200
    assert "text/markdown" in export_resp.headers["content-type"]
    assert "# STING Investigation Report: Cross-Doc Speed Audit" in export_resp.text

    # 5. Delete investigation
    del_resp = client.delete(f"/api/investigations/{inv_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["deleted"] is True

    # 6. Verify 404 after deletion
    assert client.get(f"/api/investigations/{inv_id}").status_code == 404


def test_create_investigation_validation_error(api_env: dict[str, Any]) -> None:
    client = api_env["client"]

    # Empty query
    resp = client.post(
        "/api/investigations",
        json={"title": "T", "query": "", "document_ids": ["any_doc"]},
    )
    assert resp.status_code == 422

    # Invalid document ID
    resp = client.post(
        "/api/investigations",
        json={"title": "T", "query": "Valid query", "document_ids": ["invalid_doc_id"]},
    )
    assert resp.status_code == 404
