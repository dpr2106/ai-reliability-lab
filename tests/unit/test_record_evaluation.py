"""Focused unit tests for the demo evaluation recorder and reproducibility provenance (V2-006.2)."""

import re

import httpx
import pytest

from scripts.record_demo_evaluation import (
    compare_run_metrics,
    extract_dataset_content_hash,
    poll_evaluation_job,
    request_json,
    select_golden_dataset,
    validate_document_scope,
)

SAMPLE_HASH = "88ee2c58555b5d86f789dd660f30365659acc59375809c6b5df528780280425e"


class TestDatasetHashExtraction:
    """Test truthful dataset_content_hash extraction and fail-closed validation."""

    def test_extract_valid_dataset_content_hash_from_repro_data(self) -> None:
        repro_data = {"metadata": {"dataset_content_hash": SAMPLE_HASH}}
        run_data = {"metadata": {}}
        hash_val = extract_dataset_content_hash(repro_data, run_data)
        assert hash_val == SAMPLE_HASH

    def test_extract_valid_dataset_content_hash_from_run_data_fallback(self) -> None:
        repro_data = {"metadata": {}}
        run_data = {"metadata": {"dataset_content_hash": SAMPLE_HASH}}
        hash_val = extract_dataset_content_hash(repro_data, run_data)
        assert hash_val == SAMPLE_HASH

    def test_extract_dataset_content_hash_fails_closed_when_missing(self) -> None:
        repro_data = {"metadata": {}}
        run_data = {"run_id": "run_test_001", "metadata": {}}
        with pytest.raises(ValueError, match="Verified dataset_content_hash is unavailable"):
            extract_dataset_content_hash(repro_data, run_data)

    def test_extract_dataset_content_hash_fails_closed_when_invalid_short_string(self) -> None:
        repro_data = {"metadata": {"dataset_content_hash": "short_hash"}}
        run_data = {"run_id": "run_test_002", "metadata": {}}
        with pytest.raises(ValueError, match="Verified dataset_content_hash is unavailable"):
            extract_dataset_content_hash(repro_data, run_data)


class TestMetricComparisonHonesty:
    """Test metric comparison without zero-filling and explicit incomparability marking."""

    def test_complete_metrics_match_identically(self) -> None:
        m1 = {"hit_rate@5": 0.8, "mrr@5": 0.7, "ndcg@5": 0.7262}
        m2 = {"hit_rate@5": 0.8, "mrr@5": 0.7, "ndcg@5": 0.7262}
        res = compare_run_metrics(m1, m2)

        assert res["is_comparable"] is True
        assert res["missing_keys"] == []
        assert res["metrics"]["hit_rate@5"]["delta"] == 0.0
        assert res["metrics"]["hit_rate@5"]["status"] == "exact"
        assert res["metrics"]["mrr@5"]["delta"] == 0.0
        assert res["metrics"]["ndcg@5"]["delta"] == 0.0

    def test_missing_metrics_are_represented_as_none_not_zero(self) -> None:
        m1 = {"hit_rate@5": 0.8, "mrr@5": 0.7}
        m2 = {"hit_rate@5": 0.8, "ndcg@5": 0.7262}
        res = compare_run_metrics(m1, m2)

        assert res["is_comparable"] is False
        assert set(res["missing_keys"]) == {"mrr@5", "ndcg@5"}

        # mrr@5 is present in run 1 but missing in run 2
        assert res["metrics"]["mrr@5"]["run_1"] == 0.7
        assert res["metrics"]["mrr@5"]["run_2"] is None
        assert res["metrics"]["mrr@5"]["delta"] is None
        assert res["metrics"]["mrr@5"]["status"] == "missing_key"

        # ndcg@5 is missing in run 1 but present in run 2
        assert res["metrics"]["ndcg@5"]["run_1"] is None
        assert res["metrics"]["ndcg@5"]["run_2"] == 0.7262
        assert res["metrics"]["ndcg@5"]["delta"] is None
        assert res["metrics"]["ndcg@5"]["status"] == "missing_key"

    def test_diverged_metrics_are_accurately_recorded(self) -> None:
        m1 = {"mean_latency_ms": 1.25}
        m2 = {"mean_latency_ms": 1.30}
        res = compare_run_metrics(m1, m2)

        assert res["is_comparable"] is True
        assert res["metrics"]["mean_latency_ms"]["delta"] == 0.05
        assert res["metrics"]["mean_latency_ms"]["status"] == "diverged"


class TestDatasetSelectionAndAmbiguity:
    """Test finding the golden dataset and rejecting missing or ambiguous matches."""

    def test_select_golden_dataset_success(self) -> None:
        datasets = [
            {"dataset_id": "ds_other", "name": "Other Benchmark"},
            {"dataset_id": "ds_northstar", "name": "Northstar Golden Benchmark"},
        ]
        chosen = select_golden_dataset(datasets)
        assert chosen["dataset_id"] == "ds_northstar"

    def test_select_golden_dataset_fails_when_missing(self) -> None:
        datasets = [{"dataset_id": "ds_other", "name": "Other Benchmark"}]
        with pytest.raises(ValueError, match="not found"):
            select_golden_dataset(datasets)

    def test_select_golden_dataset_fails_when_ambiguous(self) -> None:
        datasets = [
            {"dataset_id": "ds_northstar_1", "name": "Northstar Golden Benchmark"},
            {"dataset_id": "ds_northstar_2", "name": "Northstar Golden Benchmark"},
        ]
        with pytest.raises(ValueError, match="Ambiguous datasets: found 2 datasets"):
            select_golden_dataset(datasets)


class TestDocumentScopeValidation:
    """Test validation of expected Northstar corpus documents and statuses."""

    def test_validate_document_scope_success(self) -> None:
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
            {"doc_id": "doc_unrelated", "filename": "unrelated.md", "status": "READY"},
        ]
        doc_ids = validate_document_scope(target_ds, all_docs)
        assert doc_ids == ["doc_arch", "doc_bench", "doc_inc"]

    def test_validate_document_scope_fails_on_missing_doc(self) -> None:
        target_ds = {}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            # missing incident_report.md
        ]
        with pytest.raises(ValueError, match="missing from API"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_unready_doc(self) -> None:
        target_ds = {}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "INDEXING"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="is not READY"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_mismatched_dataset_ids(self) -> None:
        target_ds = {"document_ids": ["doc_foreign_123"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="unexpected document IDs"):
            validate_document_scope(target_ds, all_docs)


class TestHttpAndPollingBehavior:
    """Test HTTP status verification, job failure detection, and polling timeouts."""

    def test_request_json_raises_on_http_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal Server Error")

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="http://test")

        with pytest.raises(httpx.HTTPStatusError):
            request_json(client, "GET", "/api/error")

    def test_poll_evaluation_job_success(self) -> None:
        call_count = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return httpx.Response(200, json={"status": "running", "queries_completed": 5})
            return httpx.Response(
                200,
                json={"status": "completed", "queries_completed": 10, "run_id": "run_999"},
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="http://test")

        detail = poll_evaluation_job(client, "job_123", timeout_seconds=5.0, poll_interval=0.01)
        assert detail["status"] == "completed"
        assert detail["run_id"] == "run_999"

    def test_poll_evaluation_job_fails_on_job_failure(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"status": "failed", "error": "Retriever out of memory"},
            )

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="http://test")

        with pytest.raises(RuntimeError, match="failed: Retriever out of memory"):
            poll_evaluation_job(client, "job_failed", timeout_seconds=5.0, poll_interval=0.01)

    def test_poll_evaluation_job_raises_on_timeout(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"status": "running", "queries_completed": 0})

        transport = httpx.MockTransport(handler)
        client = httpx.Client(transport=transport, base_url="http://test")

        with pytest.raises(TimeoutError, match=re.escape("timed out after 0.05 seconds")):
            poll_evaluation_job(client, "job_slow", timeout_seconds=0.05, poll_interval=0.01)
