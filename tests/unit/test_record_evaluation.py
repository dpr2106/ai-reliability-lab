"""Focused unit tests for evaluation recorder and provenance (V2-006.2.1)."""

import re
from typing import Any

import httpx
import pytest

from scripts.record_demo_evaluation import (
    compare_run_metrics,
    extract_config_hash,
    extract_dataset_content_hash,
    extract_run_provenance_record,
    poll_evaluation_job,
    request_json,
    select_golden_dataset,
    validate_and_compare_runs,
    validate_document_scope,
    validate_sha256_digest,
)

SAMPLE_HASH = "88ee2c58555b5d86f789dd660f30365659acc59375809c6b5df528780280425e"
SAMPLE_CONFIG_HASH = "f6fa8444f4c72e2b8e97cb309644d0ee8b802f8ed7425d30687af02c7b0350b5"


class TestHashValidationStrictness:
    """Test strict 64-character hexadecimal SHA-256 digest validation."""

    def test_valid_lowercase_hash(self) -> None:
        val = validate_sha256_digest(SAMPLE_HASH, "test_hash")
        assert val == SAMPLE_HASH

    def test_valid_uppercase_hash_normalizes_to_lowercase(self) -> None:
        upper_hash = SAMPLE_HASH.upper()
        val = validate_sha256_digest(upper_hash, "test_hash")
        assert val == SAMPLE_HASH

    def test_short_hash_raises_value_error(self) -> None:
        short_hash = SAMPLE_HASH[:60]
        with pytest.raises(ValueError, match=re.escape("invalid length 60 (expected exactly 64")):
            validate_sha256_digest(short_hash, "test_hash")

    def test_long_hash_raises_value_error(self) -> None:
        long_hash = SAMPLE_HASH + "abcd"
        with pytest.raises(ValueError, match=re.escape("invalid length 68 (expected exactly 64")):
            validate_sha256_digest(long_hash, "test_hash")

    def test_non_hexadecimal_characters_raise_value_error(self) -> None:
        # Replace first characters with non-hex 'zzzz'
        non_hex = "zzzz" + SAMPLE_HASH[4:]
        with pytest.raises(ValueError, match="contains non-hexadecimal characters"):
            validate_sha256_digest(non_hex, "test_hash")

    def test_missing_or_empty_hash_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="missing or empty"):
            validate_sha256_digest(None, "test_hash")
        with pytest.raises(ValueError, match="missing or empty"):
            validate_sha256_digest("", "test_hash")

    def test_whitespace_padded_hashes_fail_closed(self) -> None:
        # Leading whitespace
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest(" " + SAMPLE_HASH, "test_hash")
        # Trailing whitespace
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest(SAMPLE_HASH + " ", "test_hash")
        # Leading tab
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest("\t" + SAMPLE_HASH, "test_hash")
        # Trailing tab
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest(SAMPLE_HASH + "\t", "test_hash")
        # Newline
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest(SAMPLE_HASH + "\n", "test_hash")
        # Carriage return / newline
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest(SAMPLE_HASH + "\r\n", "test_hash")
        # Internal space
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest(SAMPLE_HASH[:32] + " " + SAMPLE_HASH[33:], "test_hash")
        # All whitespace
        with pytest.raises(ValueError, match="contains whitespace characters"):
            validate_sha256_digest("   ", "test_hash")

    def test_extract_dataset_content_hash_success(self) -> None:
        repro_data = {"metadata": {"dataset_content_hash": SAMPLE_HASH}}
        run_data = {"metadata": {}}
        hash_val = extract_dataset_content_hash(repro_data, run_data)
        assert hash_val == SAMPLE_HASH

    def test_extract_config_hash_success(self) -> None:
        run_data = {"config_hash": SAMPLE_CONFIG_HASH}
        hash_val = extract_config_hash(run_data)
        assert hash_val == SAMPLE_CONFIG_HASH

    def test_extract_config_hash_fails_when_missing(self) -> None:
        run_data = {}
        with pytest.raises(ValueError, match="missing or empty"):
            extract_config_hash(run_data)


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


class TestExactDocumentScopeValidation:
    """Test strict validation of expected Northstar corpus documents and dataset scopes."""

    def test_validate_exact_document_scope_success(self) -> None:
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
            {"doc_id": "doc_unrelated", "filename": "unrelated.md", "status": "READY"},
        ]
        doc_ids = validate_document_scope(target_ds, all_docs)
        assert doc_ids == ["doc_arch", "doc_bench", "doc_inc"]

    def test_validate_document_scope_fails_on_missing_dataset_document_ids(self) -> None:
        # Dataset declares only 2 of the 3 required Northstar documents
        target_ds = {"document_ids": ["doc_arch", "doc_bench"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="missing expected Northstar document IDs"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_unexpected_dataset_document_ids(self) -> None:
        # Dataset contains extra/unexpected document ID
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc", "doc_foreign_999"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="unexpected document IDs"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_empty_dataset_scope(self) -> None:
        target_ds = {"document_ids": []}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="empty, missing, or not a list"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_duplicate_dataset_document_ids(self) -> None:
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc", "doc_arch"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="duplicate document IDs"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_duplicate_filenames_in_repository(self) -> None:
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_arch_dup", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="Duplicate document filename detected"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_duplicate_doc_ids_in_repository(self) -> None:
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_arch", "filename": "other.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "READY"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="Duplicate document ID detected in repository"):
            validate_document_scope(target_ds, all_docs)

    def test_validate_document_scope_fails_on_unready_document(self) -> None:
        target_ds = {"document_ids": ["doc_arch", "doc_bench", "doc_inc"]}
        all_docs = [
            {"doc_id": "doc_arch", "filename": "architecture.md", "status": "READY"},
            {"doc_id": "doc_bench", "filename": "benchmark_report.md", "status": "INDEXING"},
            {"doc_id": "doc_inc", "filename": "incident_report.md", "status": "READY"},
        ]
        with pytest.raises(ValueError, match="is not READY"):
            validate_document_scope(target_ds, all_docs)


class TestIndependentCrossRunValidation:
    """Test independent extraction and cross-run validation across both evaluation runs."""

    @pytest.fixture
    def valid_run_entries(self) -> tuple[dict[str, Any], dict[str, Any]]:
        metrics = {"hit_rate@5": 0.8, "mrr@5": 0.7, "ndcg@5": 0.7262}
        r1 = {
            "run_data": {
                "run_id": "run_001",
                "experiment_name": "northstar_baseline_run_1",
                "config_hash": SAMPLE_CONFIG_HASH,
                "config": {
                    "retriever": "bm25",
                    "top_k": 5,
                    "generator_provider": "mock",
                    "generator_model": "mock-pipeline",
                },
                "metadata": {
                    "dataset_content_hash": SAMPLE_HASH,
                    "document_ids": ["doc_1", "doc_2", "doc_3"],
                },
                "metrics": metrics,
                "query_count": 10,
            },
            "repro_data": {
                "dataset_id": "ds_8138dee2cb55",
                "reproducibility_tier": "exact",
                "can_reproduce": True,
                "corpus_document_ids": ["doc_1", "doc_2", "doc_3"],
                "metadata": {"dataset_content_hash": SAMPLE_HASH},
            },
        }
        r2 = {
            "run_data": {
                "run_id": "run_002",
                "experiment_name": "northstar_baseline_run_2",
                "config_hash": SAMPLE_CONFIG_HASH,
                "config": {
                    "retriever": "bm25",
                    "top_k": 5,
                    "generator_provider": "mock",
                    "generator_model": "mock-pipeline",
                },
                "metadata": {
                    "dataset_content_hash": SAMPLE_HASH,
                    "document_ids": ["doc_1", "doc_2", "doc_3"],
                },
                "metrics": metrics,
                "query_count": 10,
            },
            "repro_data": {
                "dataset_id": "ds_8138dee2cb55",
                "reproducibility_tier": "exact",
                "can_reproduce": True,
                "corpus_document_ids": ["doc_1", "doc_2", "doc_3"],
                "metadata": {"dataset_content_hash": SAMPLE_HASH},
            },
        }
        return r1, r2

    def test_identical_runs_pass_validation_with_exact_tier(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        val = validate_and_compare_runs(r1, r2)

        assert val["is_comparable"] is True
        assert val["provenance_compatible"] is True
        assert val["incompatibilities"] == []
        assert val["overall_reproducibility_tier"] == "exact"

    def test_conflicting_dataset_hashes_fails_cross_run_compatibility(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        # Modify Run 2 dataset hash
        alt_hash = "1111111111111111111111111111111111111111111111111111111111111111"
        r2["run_data"]["metadata"]["dataset_content_hash"] = alt_hash
        r2["repro_data"]["metadata"]["dataset_content_hash"] = alt_hash

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert any("Dataset content hash mismatch" in s for s in val["incompatibilities"])

    def test_conflicting_config_hashes_fails_cross_run_compatibility(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        alt_cfg = "2222222222222222222222222222222222222222222222222222222222222222"
        r2["run_data"]["config_hash"] = alt_cfg

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert any("Config hash mismatch" in s for s in val["incompatibilities"])

    def test_conflicting_document_scopes_fails_cross_run_compatibility(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r2["repro_data"]["corpus_document_ids"] = ["doc_1", "doc_2", "doc_diff"]
        r2["run_data"]["metadata"]["document_ids"] = ["doc_1", "doc_2", "doc_diff"]

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert any("Document scope mismatch" in s for s in val["incompatibilities"])

    def test_non_exact_tier_downgrades_overall_reproducibility(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r2["repro_data"]["reproducibility_tier"] = "partial"

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["overall_reproducibility_tier"] == "partial"
        assert any("tier is not exact" in s for s in val["incompatibilities"])

    def test_both_runs_missing_dataset_ids_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["repro_data"].pop("dataset_id")
        r2["repro_data"].pop("dataset_id")

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("dataset_id" in s for s in val["incompatibilities"])

    def test_both_runs_missing_document_scopes_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["repro_data"].pop("corpus_document_ids")
        r1["run_data"]["metadata"].pop("document_ids")
        r2["repro_data"].pop("corpus_document_ids")
        r2["run_data"]["metadata"].pop("document_ids")

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("document_scope" in s for s in val["incompatibilities"])

    def test_both_runs_missing_query_counts_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["run_data"].pop("query_count")
        r2["run_data"].pop("query_count")

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("query_count" in s for s in val["incompatibilities"])

    def test_both_runs_missing_retriever_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["run_data"]["config"].pop("retriever")
        r2["run_data"]["config"].pop("retriever")

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("retriever identifier" in s for s in val["incompatibilities"])

    def test_both_runs_missing_provider_or_model_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["run_data"]["config"].pop("generator_provider")
        r2["run_data"]["config"].pop("generator_model")

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any(
            "provider identifier" in s or "model identifier" in s for s in val["incompatibilities"]
        )

    def test_both_runs_missing_top_k_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["run_data"]["config"].pop("top_k")
        r2["run_data"]["config"].pop("top_k")

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("top_k" in s for s in val["incompatibilities"])

    def test_both_runs_containing_empty_metrics_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["run_data"]["metrics"] = {}
        r2["run_data"]["metrics"] = {}

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("metrics mapping" in s for s in val["incompatibilities"])

    def test_string_false_supplied_as_can_reproduce_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["repro_data"]["can_reproduce"] = "false"
        r2["repro_data"]["can_reproduce"] = "false"

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("can_reproduce" in s for s in val["incompatibilities"])

    def test_invalid_or_non_positive_numeric_config_fails_cross_run_validation(
        self, valid_run_entries: tuple[dict[str, Any], dict[str, Any]]
    ) -> None:
        r1, r2 = valid_run_entries
        r1["run_data"]["config"]["top_k"] = -5
        r2["run_data"]["config"]["top_k"] = 0

        val = validate_and_compare_runs(r1, r2)
        assert val["provenance_compatible"] is False
        assert val["is_comparable"] is False
        assert val["overall_reproducibility_tier"] != "exact"
        assert any("top_k" in s for s in val["incompatibilities"])


class TestRunProvenanceExtractionFailClosed:
    """Test individual run extraction fails closed on missing or invalid metadata."""

    @pytest.fixture
    def single_valid_run(self) -> dict[str, Any]:
        return {
            "run_data": {
                "run_id": "run_test_001",
                "experiment_name": "test_experiment",
                "config_hash": SAMPLE_CONFIG_HASH,
                "config": {
                    "retriever": "bm25",
                    "top_k": 5,
                    "generator_provider": "mock",
                    "generator_model": "mock-pipeline",
                },
                "metadata": {
                    "dataset_content_hash": SAMPLE_HASH,
                    "document_ids": ["doc_1", "doc_2"],
                },
                "metrics": {"hit_rate@5": 0.8},
                "query_count": 10,
            },
            "repro_data": {
                "dataset_id": "ds_test",
                "reproducibility_tier": "exact",
                "can_reproduce": True,
                "corpus_document_ids": ["doc_1", "doc_2"],
                "metadata": {"dataset_content_hash": SAMPLE_HASH},
            },
        }

    def test_missing_run_id_raises_value_error(self, single_valid_run: dict[str, Any]) -> None:
        single_valid_run["run_data"].pop("run_id")
        with pytest.raises(ValueError, match="run_id is missing or empty"):
            extract_run_provenance_record(single_valid_run)

    def test_missing_dataset_id_raises_value_error(self, single_valid_run: dict[str, Any]) -> None:
        single_valid_run["repro_data"].pop("dataset_id")
        with pytest.raises(ValueError, match="dataset_id is missing or empty"):
            extract_run_provenance_record(single_valid_run)

    def test_missing_document_scope_raises_value_error(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["repro_data"].pop("corpus_document_ids")
        single_valid_run["run_data"]["metadata"].pop("document_ids")
        with pytest.raises(ValueError, match=r"document_scope .* is missing, empty, or not a list"):
            extract_run_provenance_record(single_valid_run)

    def test_duplicate_document_scope_ids_raises_value_error(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["repro_data"]["corpus_document_ids"] = ["doc_1", "doc_1"]
        with pytest.raises(ValueError, match="contains duplicate IDs"):
            extract_run_provenance_record(single_valid_run)

    def test_query_count_zero_or_boolean_raises_value_error(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["run_data"]["query_count"] = 0
        with pytest.raises(ValueError, match=r"query_count .* must be a positive integer"):
            extract_run_provenance_record(single_valid_run)
        single_valid_run["run_data"]["query_count"] = True
        with pytest.raises(ValueError, match=r"query_count .* must be a positive integer"):
            extract_run_provenance_record(single_valid_run)

    def test_missing_retriever_raises_value_error(self, single_valid_run: dict[str, Any]) -> None:
        single_valid_run["run_data"]["config"].pop("retriever")
        with pytest.raises(ValueError, match="retriever identifier is missing or empty"):
            extract_run_provenance_record(single_valid_run)

    def test_missing_top_k_does_not_silently_default_to_5(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["run_data"]["config"].pop("top_k")
        with pytest.raises(
            ValueError, match=r"top_k .* must be an explicitly supplied positive integer"
        ):
            extract_run_provenance_record(single_valid_run)

    def test_invalid_top_k_raises_value_error(self, single_valid_run: dict[str, Any]) -> None:
        single_valid_run["run_data"]["config"]["top_k"] = -1
        with pytest.raises(
            ValueError, match=r"top_k .* must be an explicitly supplied positive integer"
        ):
            extract_run_provenance_record(single_valid_run)
        single_valid_run["run_data"]["config"]["top_k"] = True
        with pytest.raises(
            ValueError, match=r"top_k .* must be an explicitly supplied positive integer"
        ):
            extract_run_provenance_record(single_valid_run)

    def test_missing_provider_or_model_raises_value_error(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["run_data"]["config"].pop("generator_provider")
        with pytest.raises(ValueError, match="provider identifier is missing or empty"):
            extract_run_provenance_record(single_valid_run)

    def test_string_can_reproduce_does_not_coerce_to_true(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["repro_data"]["can_reproduce"] = "false"
        with pytest.raises(ValueError, match=r"can_reproduce .* must be an actual boolean"):
            extract_run_provenance_record(single_valid_run)

    def test_empty_metrics_mapping_raises_value_error(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["run_data"]["metrics"] = {}
        with pytest.raises(ValueError, match=r"metrics mapping .* is missing or empty"):
            extract_run_provenance_record(single_valid_run)

    def test_non_numeric_or_non_finite_metric_raises_value_error(
        self, single_valid_run: dict[str, Any]
    ) -> None:
        single_valid_run["run_data"]["metrics"] = {"hit_rate@5": "high"}
        with pytest.raises(ValueError, match="must be a numeric value"):
            extract_run_provenance_record(single_valid_run)
        single_valid_run["run_data"]["metrics"] = {"hit_rate@5": float("nan")}
        with pytest.raises(ValueError, match="is not a finite number"):
            extract_run_provenance_record(single_valid_run)
        single_valid_run["run_data"]["metrics"] = {"hit_rate@5": True}
        with pytest.raises(ValueError, match="must be a numeric value"):
            extract_run_provenance_record(single_valid_run)


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
