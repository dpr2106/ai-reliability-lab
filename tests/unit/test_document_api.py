"""Unit tests for FastAPI document upload, listing, retrieval, and deletion endpoints."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pymupdf
import pytest
from fastapi.testclient import TestClient

from reliability_lab.api.main import (
    create_app,
    get_document_repository,
    get_ingestion_service,
)
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.storage.database import Database


class MockAPIEmbedder(BaseEmbedder):
    """Hermetic embedder for API testing."""

    @property
    def model_name(self) -> str:
        return "mock-api-embedder"

    @property
    def embedding_dim(self) -> int:
        return 16

    def embed_texts(self, texts: list[str] | tuple[str, ...], batch_size: int = 32) -> np.ndarray:
        vectors = []
        for text in texts:
            val = float(sum(ord(c) for c in text) % 50) / 50.0
            vec = np.full(16, val, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            vectors.append(vec)
        return np.vstack(vectors).astype(np.float32)


def make_valid_pdf_bytes(text: str = "Test PDF document for API upload.") -> bytes:
    doc = pymupdf.open()
    page = doc.new_page()
    page.insert_text((50, 50), text)
    data = doc.tobytes()
    doc.close()
    return data


@pytest.fixture
def api_test_env(tmp_path: Path):
    """Set up isolated database and document ingestion service for FastAPI TestClient."""
    db_file = tmp_path / "api_test.sqlite3"
    db = Database(db_file)
    repo = DocumentRepository(db)
    service = DocumentIngestionService(
        repository=repo,
        storage_dir=tmp_path / "documents",
        index_dir=tmp_path / "indexes",
        max_upload_size=500_000,
        embedder=MockAPIEmbedder(),
    )

    app = create_app()

    # Override dependencies
    app.dependency_overrides[get_document_repository] = lambda: repo
    app.dependency_overrides[get_ingestion_service] = lambda: service

    with TestClient(app) as client:
        yield client, repo, service


class TestDocumentUploadAPI:
    """Tests verifying POST /api/documents/upload endpoint."""

    def test_upload_valid_text_document(self, api_test_env) -> None:
        client, _, _ = api_test_env

        files = {
            "file": ("report.txt", io.BytesIO(b"Hello world from uploaded file."), "text/plain")
        }
        response = client.post("/api/documents/upload", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["doc_id"].startswith("doc_")
        assert data["filename"] == "report.txt"
        assert data["file_type"] == "txt"
        assert data["status"] == "UPLOADED"
        assert "content_hash" in data

    def test_upload_valid_pdf_document(self, api_test_env) -> None:
        client, _, _ = api_test_env

        pdf_bytes = make_valid_pdf_bytes("Neural retrieval evaluation benchmark.")
        files = {"file": ("manual.pdf", io.BytesIO(pdf_bytes), "application/pdf")}
        response = client.post("/api/documents/upload", files=files)

        assert response.status_code == 201
        data = response.json()
        assert data["file_type"] == "pdf"
        assert data["filename"] == "manual.pdf"

    def test_upload_empty_file_returns_400(self, api_test_env) -> None:
        client, _, _ = api_test_env

        files = {"file": ("empty.txt", io.BytesIO(b""), "text/plain")}
        response = client.post("/api/documents/upload", files=files)

        assert response.status_code == 400
        assert "empty" in response.json()["detail"].lower()

    def test_upload_invalid_extension_returns_400(self, api_test_env) -> None:
        client, _, _ = api_test_env

        files = {"file": ("script.sh", io.BytesIO(b"echo 'evil'"), "application/x-sh")}
        response = client.post("/api/documents/upload", files=files)

        assert response.status_code == 400
        assert "unsupported file extension" in response.json()["detail"].lower()

    def test_upload_pdf_header_mismatch_returns_400(self, api_test_env) -> None:
        client, _, _ = api_test_env

        files = {"file": ("fake.pdf", io.BytesIO(b"not a valid pdf header"), "application/pdf")}
        response = client.post("/api/documents/upload", files=files)

        assert response.status_code == 400
        assert "magic header" in response.json()["detail"].lower()

    def test_upload_oversized_file_returns_413(self, api_test_env) -> None:
        client, _, service = api_test_env

        oversized_data = b"A" * (service.max_upload_size + 1000)
        files = {"file": ("large.txt", io.BytesIO(oversized_data), "text/plain")}
        response = client.post("/api/documents/upload", files=files)

        assert response.status_code == 413
        assert "limit exceeded" in response.json()["detail"].lower()

    def test_upload_content_length_header_rejects_early(self, api_test_env) -> None:
        client, _, service = api_test_env

        files = {"file": ("test.txt", io.BytesIO(b"Small content"), "text/plain")}
        headers = {"content-length": str(service.max_upload_size + 10000)}
        response = client.post("/api/documents/upload", files=files, headers=headers)

        assert response.status_code == 413
        assert "upload size limit exceeded" in response.json()["detail"].lower()


class TestDocumentQueryAndManagementAPI:
    """Tests verifying GET and DELETE /api/documents endpoints."""

    def test_list_and_get_documents(self, api_test_env) -> None:
        client, _, _ = api_test_env

        # Upload 3 documents
        for i in range(3):
            files = {
                "file": (f"doc_{i}.txt", io.BytesIO(f"Text content {i}".encode()), "text/plain")
            }
            client.post("/api/documents/upload", files=files)

        # List documents
        list_resp = client.get("/api/documents")
        assert list_resp.status_code == 200
        list_data = list_resp.json()
        assert list_data["total"] == 3
        assert len(list_data["documents"]) == 3

        # Get specific document detail
        first_id = list_data["documents"][0]["doc_id"]
        detail_resp = client.get(f"/api/documents/{first_id}")
        assert detail_resp.status_code == 200
        detail_data = detail_resp.json()
        assert detail_data["doc_id"] == first_id
        assert detail_data["file_type"] == "txt"
        # Confirm internal filesystem storage path is not exposed
        assert "storage_path" not in detail_data

    def test_get_document_chunks_paginated(self, api_test_env) -> None:
        client, _, service = api_test_env

        # Upload and process a document with multiple sentences
        text = "Sentence one about AI evaluation. " * 30
        files = {"file": ("chunks_doc.txt", io.BytesIO(text.encode()), "text/plain")}
        up_resp = client.post("/api/documents/upload", files=files)
        doc_id = up_resp.json()["doc_id"]

        # Run processing synchronously for test
        service.process_document(doc_id)

        # Query all chunks
        chunk_resp = client.get(f"/api/documents/{doc_id}/chunks?limit=50&offset=0")
        assert chunk_resp.status_code == 200
        data = chunk_resp.json()
        assert data["total"] > 1
        assert data["limit"] == 50
        assert data["offset"] == 0
        assert len(data["chunks"]) == data["total"]

        first_chunk = data["chunks"][0]
        assert first_chunk["doc_id"] == doc_id
        assert first_chunk["chunk_index"] == 0
        assert first_chunk["page_number"] == 1
        assert first_chunk["char_start"] == 0
        assert first_chunk["char_end"] > 0
        assert len(first_chunk["content"]) > 0

        # Query with pagination limit=1, offset=1
        paged_resp = client.get(f"/api/documents/{doc_id}/chunks?limit=1&offset=1")
        assert paged_resp.status_code == 200
        paged_data = paged_resp.json()
        assert paged_data["total"] == data["total"]
        assert len(paged_data["chunks"]) == 1
        assert paged_data["chunks"][0]["chunk_index"] == 1

    def test_get_document_chunks_nonexistent_returns_404(self, api_test_env) -> None:
        client, _, _ = api_test_env
        resp = client.get("/api/documents/doc_nonexistent_000/chunks")
        assert resp.status_code == 404

    def test_get_nonexistent_document_returns_404(self, api_test_env) -> None:
        client, _, _ = api_test_env
        resp = client.get("/api/documents/doc_nonexistent_999")
        assert resp.status_code == 404

    def test_delete_document_success(self, api_test_env) -> None:
        client, _, _ = api_test_env

        files = {"file": ("del.txt", io.BytesIO(b"To delete"), "text/plain")}
        up_resp = client.post("/api/documents/upload", files=files)
        doc_id = up_resp.json()["doc_id"]

        del_resp = client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 200
        assert del_resp.json()["status"] == "deleted"

        # Subsequent GET should return 404
        assert client.get(f"/api/documents/{doc_id}").status_code == 404

    def test_delete_document_filesystem_failure_returns_500(
        self, api_test_env, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        client, _, service = api_test_env

        files = {"file": ("protected.txt", io.BytesIO(b"Data"), "text/plain")}
        up_resp = client.post("/api/documents/upload", files=files)
        doc_id = up_resp.json()["doc_id"]

        from reliability_lab.ingestion.service import DocumentDeletionError

        def mock_delete_document(target_id: str) -> bool:
            raise DocumentDeletionError("I/O error: disk block locked by kernel")

        monkeypatch.setattr(service, "delete_document", mock_delete_document)

        del_resp = client.delete(f"/api/documents/{doc_id}")
        assert del_resp.status_code == 500
        assert "disk block locked" in del_resp.json()["detail"].lower()

    def test_delete_nonexistent_document_returns_404(self, api_test_env) -> None:
        client, _, _ = api_test_env
        del_resp = client.delete("/api/documents/doc_nonexistent_888")
        assert del_resp.status_code == 404
