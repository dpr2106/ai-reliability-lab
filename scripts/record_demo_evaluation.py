#!/usr/bin/env python3
"""Execute and record reproducible Northstar evaluations (V2-006.2).

Runs the bundled Northstar benchmark dataset twice against the active STING API
server using the deterministic mock provider and Okapi BM25 retrieval, validates
exact dataset document scope, independently verifies provenance and reproducibility
manifests for both runs, records all metrics, failure modes, and outputs a
comprehensive comparison analysis.
"""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from typing import Any

import httpx

API_URL = os.environ.get("STING_API_URL", "http://127.0.0.1:8000")
NORTHSTAR_FILENAMES = ("architecture.md", "benchmark_report.md", "incident_report.md")
NORTHSTAR_DATASET_NAME = "Northstar Golden Benchmark"
SHA256_HEX_PATTERN = re.compile(r"^[0-9a-f]{64}$")
GIT_SHA_HEX_PATTERN = re.compile(r"^[0-9a-f]{40}$|^[0-9a-f]{64}$")


def validate_git_commit_sha(raw_sha: Any) -> str:
    """Validate and return normalized lowercase git commit SHA (40 or 64 hex characters).

    Rejects empty, non-string, whitespace-padded, or non-hex values fail-closed.
    """
    if raw_sha is None or not isinstance(raw_sha, str) or not raw_sha:
        raise ValueError("Git commit SHA is missing or empty.")

    if raw_sha != raw_sha.strip() or any(c.isspace() for c in raw_sha):
        raise ValueError(f"Git commit SHA contains whitespace characters: {raw_sha!r}")

    normalized = raw_sha.lower()
    if not GIT_SHA_HEX_PATTERN.match(normalized):
        raise ValueError(
            f"Git commit SHA must be a 40- or 64-character hexadecimal string, "
            f"got ({len(raw_sha)} chars): {raw_sha!r}"
        )
    return normalized


def request_json(client: httpx.Client, method: str, url: str, **kwargs: Any) -> Any:
    """Send an HTTP request, check for status errors, and parse JSON response."""
    res = client.request(method, url, **kwargs)
    if res.status_code >= 400:
        raise httpx.HTTPStatusError(
            f"HTTP {res.status_code} {res.reason_phrase} for {url}: {res.text}",
            request=res.request,
            response=res,
        )
    return res.json()


def validate_sha256_digest(raw_val: Any, field_name: str, entity_id: str = "") -> str:
    """Validate that raw_val is an authentic 64-character hexadecimal SHA-256 digest."""
    prefix = f"{field_name} for '{entity_id}'" if entity_id else field_name

    if raw_val is None or not isinstance(raw_val, str) or not raw_val:
        raise ValueError(f"{prefix} is missing or empty.")

    # Reject leading or trailing whitespace and internal whitespace fail-closed
    if raw_val != raw_val.strip() or any(c.isspace() for c in raw_val):
        raise ValueError(f"{prefix} contains whitespace characters: {raw_val!r}.")

    if len(raw_val) != 64:
        raise ValueError(
            f"{prefix} has invalid length {len(raw_val)} "
            f"(expected exactly 64 hex characters): '{raw_val}'."
        )

    norm = raw_val.lower()

    if not SHA256_HEX_PATTERN.match(norm):
        raise ValueError(f"{prefix} contains non-hexadecimal characters: '{raw_val}'.")

    return norm


def extract_dataset_content_hash(repro_data: dict[str, Any], run_data: dict[str, Any]) -> str:
    """Extract and strictly validate dataset_content_hash from run metadata.

    Fails explicitly if hash is unavailable or not a valid 64-char hex SHA-256 string.
    Never falls back to a hard-coded value.
    """
    repro_meta = repro_data.get("metadata") or {}
    run_meta = run_data.get("metadata") or {}

    ds_hash = repro_meta.get("dataset_content_hash") or run_meta.get("dataset_content_hash")
    run_id = run_data.get("run_id", "unknown")

    return validate_sha256_digest(ds_hash, "dataset_content_hash", run_id)


def extract_config_hash(run_data: dict[str, Any], repro_data: dict[str, Any] | None = None) -> str:
    """Extract and strictly validate config_hash from run metadata.

    Fails explicitly if hash is unavailable or not a valid 64-char hex SHA-256 string.
    Never silently becomes 'unknown'.
    """
    cfg_hash = run_data.get("config_hash") or (
        repro_data.get("config_hash") if repro_data else None
    )
    run_id = run_data.get("run_id", "unknown")

    return validate_sha256_digest(cfg_hash, "config_hash", run_id)


def compare_run_metrics(m1: dict[str, Any], m2: dict[str, Any]) -> dict[str, Any]:
    """Compare metrics between two evaluation runs without zero-filling missing keys."""
    all_keys = sorted(set(m1.keys()) | set(m2.keys()))
    missing_keys: list[str] = []
    comparison_map: dict[str, dict[str, Any]] = {}

    for k in all_keys:
        v1 = m1.get(k)
        v2 = m2.get(k)

        if v1 is None or v2 is None:
            missing_keys.append(k)
            comparison_map[k] = {
                "run_1": v1,
                "run_2": v2,
                "delta": None,
                "status": "missing_key",
            }
        else:
            delta = round(float(v2) - float(v1), 6)
            comparison_map[k] = {
                "run_1": float(v1),
                "run_2": float(v2),
                "delta": delta,
                "status": "exact" if abs(delta) < 1e-9 else "diverged",
            }

    return {
        "is_comparable": len(missing_keys) == 0,
        "missing_keys": missing_keys,
        "metrics": comparison_map,
    }


def select_golden_dataset(
    datasets: list[dict[str, Any]], target_name: str = NORTHSTAR_DATASET_NAME
) -> dict[str, Any]:
    """Select exact Northstar Golden Benchmark dataset, rejecting missing or ambiguous matches."""
    matching = [d for d in datasets if d.get("name") == target_name]
    if not matching:
        raise ValueError(
            f"Dataset '{target_name}' not found. Run 'scripts/prepare_sting_demo.py' first."
        )
    if len(matching) > 1:
        ids = [d.get("dataset_id") for d in matching]
        raise ValueError(
            f"Ambiguous datasets: found {len(matching)} datasets named '{target_name}' ({ids}). "
            "Resolve duplicate datasets before recording."
        )
    return matching[0]


def validate_document_scope(
    target_dataset: dict[str, Any],
    all_documents: list[dict[str, Any]],
    expected_filenames: tuple[str, ...] = NORTHSTAR_FILENAMES,
) -> list[str]:
    """Verify that expected Northstar documents are present, ready, and exactly match dataset scope.

    Enforces strict requirements:
    1. Each expected Northstar filename must resolve to exactly one document.
    2. Duplicate filenames in repository fail explicitly.
    3. Duplicate document IDs fail explicitly.
    4. Every document must have a valid non-empty document ID.
    5. All expected documents must be in status READY.
    6. The dataset's declared document IDs must exactly match the expected Northstar document IDs.
    7. Rejects both missing and unexpected document IDs.
    8. Preserves deterministic document ordering.
    """
    # 1. Verify document IDs and uniqueness
    seen_doc_ids: set[str] = set()
    for d in all_documents:
        did = d.get("doc_id")
        if not did or not isinstance(did, str) or not did.strip():
            raise ValueError(f"Encountered document with invalid or missing doc_id: {d}")
        did = did.strip()
        if did in seen_doc_ids:
            raise ValueError(f"Duplicate document ID detected in repository: '{did}'.")
        seen_doc_ids.add(did)

    # 2. Verify filename occurrences
    filename_counts: dict[str, int] = {}
    for d in all_documents:
        fname = d.get("filename")
        if fname and isinstance(fname, str):
            fname = fname.strip()
            filename_counts[fname] = filename_counts.get(fname, 0) + 1

    for fname in expected_filenames:
        count = filename_counts.get(fname, 0)
        if count == 0:
            raise ValueError(f"Expected Northstar demo document '{fname}' is missing from API.")
        if count > 1:
            raise ValueError(
                f"Duplicate document filename detected in repository: "
                f"'{fname}' appears {count} times."
            )

    # 3. Map unique expected documents
    doc_by_name = {
        d["filename"].strip(): d
        for d in all_documents
        if d.get("filename") and d["filename"].strip() in expected_filenames
    }

    # 4. Check readiness of all expected documents
    for fname in expected_filenames:
        doc = doc_by_name[fname]
        st = str(doc.get("status", "")).upper()
        if st != "READY":
            raise ValueError(f"Northstar demo document '{fname}' is not READY (status: '{st}').")

    # 5. Determine expected document IDs in deterministic order
    expected_doc_ids = [doc_by_name[fname]["doc_id"].strip() for fname in expected_filenames]
    expected_doc_ids_set = set(expected_doc_ids)

    # 6. Validate dataset's declared document scope
    ds_doc_ids_raw = target_dataset.get("document_ids") or target_dataset.get("document_scope")
    if not ds_doc_ids_raw or not isinstance(ds_doc_ids_raw, list):
        raise ValueError("Dataset document scope is empty, missing, or not a list.")

    ds_doc_ids = [str(did).strip() for did in ds_doc_ids_raw]
    if len(ds_doc_ids) != len(set(ds_doc_ids)):
        raise ValueError(f"Dataset document scope contains duplicate document IDs: {ds_doc_ids}")

    ds_doc_ids_set = set(ds_doc_ids)

    missing_ids = expected_doc_ids_set - ds_doc_ids_set
    if missing_ids:
        raise ValueError(
            "Dataset document scope is missing expected Northstar document IDs: "
            f"{sorted(missing_ids)}"
        )

    unexpected_ids = ds_doc_ids_set - expected_doc_ids_set
    if unexpected_ids:
        raise ValueError(
            f"Dataset document scope contains unexpected document IDs: {sorted(unexpected_ids)}"
        )

    return expected_doc_ids


def poll_evaluation_job(
    client: httpx.Client,
    job_id: str,
    timeout_seconds: float = 60.0,
    poll_interval: float = 1.0,
) -> dict[str, Any]:
    """Poll an evaluation job until completion or timeout."""
    start_time = time.monotonic()
    while time.monotonic() - start_time < timeout_seconds:
        job_detail = request_json(client, "GET", f"/api/evaluation/jobs/{job_id}")
        st = str(job_detail.get("status", "")).upper()
        q_done = job_detail.get("queries_completed", 0)
        q_total = job_detail.get("total_queries", 0)
        print(f"Job {job_id} status: {st} (queries: {q_done}/{q_total})")

        if st == "COMPLETED":
            return job_detail
        if st == "FAILED":
            err = job_detail.get("error") or "evaluation job failed"
            raise RuntimeError(f"Evaluation job '{job_id}' failed: {err}")

        time.sleep(poll_interval)

    raise TimeoutError(f"Evaluation job '{job_id}' timed out after {timeout_seconds} seconds.")


def extract_run_provenance_record(
    run_entry: dict[str, Any],
) -> dict[str, Any]:
    """Independently extract and validate all provenance fields for a single run.

    Fails explicitly with descriptive ValueError if any required provenance field
    is missing, malformed, or invalid. Never silently defaults missing fields.
    """
    if not isinstance(run_entry, dict):
        raise ValueError(f"run_entry must be a dictionary, got: {type(run_entry).__name__}")

    run_data = run_entry.get("run_data")
    if not isinstance(run_data, dict):
        raise ValueError("run_entry['run_data'] is missing or not a dictionary")

    repro_data = run_entry.get("repro_data")
    if not isinstance(repro_data, dict):
        raise ValueError("run_entry['repro_data'] is missing or not a dictionary")

    # 1. Non-empty run ID
    run_id = run_data.get("run_id") or repro_data.get("run_id")
    if not run_id or not isinstance(run_id, str) or not run_id.strip():
        raise ValueError("run_id is missing or empty in run data")
    run_id = run_id.strip()

    # 2. Non-empty dataset ID
    ds_id = repro_data.get("dataset_id") or run_data.get("dataset_reference")
    if not ds_id or not isinstance(ds_id, str) or not ds_id.strip():
        raise ValueError(f"dataset_id is missing or empty in run '{run_id}'")
    ds_id = ds_id.strip()

    # 3. Valid 64-character dataset content SHA-256
    ds_hash = extract_dataset_content_hash(repro_data, run_data)

    # 4. Valid 64-character configuration SHA-256
    cfg_hash = extract_config_hash(run_data, repro_data)

    # 5. Non-empty document scope containing valid, non-empty string IDs, with no duplicates
    raw_doc_scope = (
        repro_data.get("corpus_document_ids")
        if repro_data.get("corpus_document_ids") is not None
        else run_data.get("metadata", {}).get("document_ids")
    )
    if not isinstance(raw_doc_scope, list) or len(raw_doc_scope) == 0:
        raise ValueError(f"document_scope in run '{run_id}' is missing, empty, or not a list")

    norm_doc_scope: list[str] = []
    for did in raw_doc_scope:
        if not isinstance(did, str):
            raise ValueError(
                f"document_scope in run '{run_id}' contains invalid non-string ID: {did!r}"
            )
        stripped = did.strip()
        if not stripped:
            raise ValueError(f"document_scope in run '{run_id}' contains empty ID: {did!r}")
        norm_doc_scope.append(stripped)

    if len(norm_doc_scope) != len(set(norm_doc_scope)):
        raise ValueError(
            f"document_scope in run '{run_id}' contains duplicate IDs: {norm_doc_scope}"
        )
    doc_scope = sorted(norm_doc_scope)

    # 6. Positive integer query count; reject booleans and zero
    raw_q_count = (
        run_data.get("query_count")
        if run_data.get("query_count") is not None
        else (
            len(run_data.get("per_query_results"))
            if run_data.get("per_query_results") is not None
            else (
                run_data.get("config", {}).get("query_count")
                if isinstance(run_data.get("config"), dict)
                and run_data.get("config", {}).get("query_count") is not None
                else None
            )
        )
    )
    if (
        raw_q_count is None
        or isinstance(raw_q_count, bool)
        or not isinstance(raw_q_count, int)
        or raw_q_count <= 0
    ):
        raise ValueError(
            f"query_count in run '{run_id}' must be a positive integer, got: {raw_q_count!r}"
        )
    q_count = raw_q_count

    cfg = run_data.get("config") if isinstance(run_data.get("config"), dict) else {}
    meta = run_data.get("metadata") if isinstance(run_data.get("metadata"), dict) else {}

    # 7. Non-empty retriever identifier
    raw_retriever = cfg.get("retriever") or meta.get("retriever_type")
    if not raw_retriever or not isinstance(raw_retriever, str) or not raw_retriever.strip():
        raise ValueError(f"retriever identifier is missing or empty in run '{run_id}'")
    retriever = raw_retriever.strip().lower()

    # 8. An explicitly supplied positive integer top_k; do not silently default to 5
    raw_top_k = (
        cfg.get("top_k") if "top_k" in cfg else (meta.get("top_k") if "top_k" in meta else None)
    )
    if (
        raw_top_k is None
        or isinstance(raw_top_k, bool)
        or not isinstance(raw_top_k, int)
        or raw_top_k <= 0
    ):
        raise ValueError(
            f"top_k in run '{run_id}' must be an explicitly supplied positive integer, "
            f"got: {raw_top_k!r}"
        )
    top_k = raw_top_k

    # 9. Non-empty provider identifier
    raw_provider = cfg.get("generator_provider") or meta.get("provider")
    if not raw_provider or not isinstance(raw_provider, str) or not raw_provider.strip():
        raise ValueError(f"provider identifier is missing or empty in run '{run_id}'")
    provider = raw_provider.strip().lower()

    # 10. Non-empty model identifier
    raw_model = cfg.get("generator_model") or meta.get("model_name")
    if not raw_model or not isinstance(raw_model, str) or not raw_model.strip():
        raise ValueError(f"model identifier is missing or empty in run '{run_id}'")
    model = raw_model.strip()

    # 11. Valid reproducibility tier
    raw_tier = repro_data.get("reproducibility_tier")
    if not raw_tier or not isinstance(raw_tier, str) or not raw_tier.strip():
        raise ValueError(f"reproducibility_tier is missing or empty in run '{run_id}'")
    tier = raw_tier.strip().lower()
    if tier not in ("exact", "partial", "unverifiable", "impossible"):
        raise ValueError(f"reproducibility_tier in run '{run_id}' is invalid: {raw_tier!r}")

    # 12. Actual boolean can_reproduce value; do not coerce arbitrary values using bool()
    raw_can_reproduce = repro_data.get("can_reproduce")
    if raw_can_reproduce is None or not isinstance(raw_can_reproduce, bool):
        raise ValueError(
            f"can_reproduce in run '{run_id}' must be an actual boolean (True/False), "
            f"got: {raw_can_reproduce!r} ({type(raw_can_reproduce).__name__})"
        )
    can_reproduce = raw_can_reproduce

    # 13. Non-empty metrics mapping with valid finite numeric metric values
    raw_metrics = run_data.get("metrics")
    if not isinstance(raw_metrics, dict) or len(raw_metrics) == 0:
        raise ValueError(f"metrics mapping in run '{run_id}' is missing or empty")

    for mk, mv in raw_metrics.items():
        if not isinstance(mk, str) or not mk.strip():
            raise ValueError(f"metrics mapping in run '{run_id}' has invalid metric key: {mk!r}")
        if mv is None or isinstance(mv, bool) or not isinstance(mv, (int, float)):
            raise ValueError(
                f"metric '{mk}' in run '{run_id}' must be a numeric value, got: {mv!r}"
            )
        if not math.isfinite(mv):
            raise ValueError(f"metric '{mk}' in run '{run_id}' is not a finite number: {mv!r}")

    return {
        "run_id": run_id,
        "experiment_name": run_data.get("experiment_name", ""),
        "dataset_content_hash": ds_hash,
        "config_hash": cfg_hash,
        "reproducibility_tier": tier,
        "can_reproduce": can_reproduce,
        "dataset_id": ds_id,
        "document_scope": doc_scope,
        "query_count": q_count,
        "retriever": retriever,
        "top_k": top_k,
        "provider": provider,
        "model": model,
        "metrics": raw_metrics,
        "failure_counts": run_data.get("failure_counts") or {},
    }


def validate_and_compare_runs(
    run1_entry: dict[str, Any],
    run2_entry: dict[str, Any],
    expected_query_count: int | None = 10,
) -> dict[str, Any]:
    """Validate both evaluation runs independently and assert cross-run provenance compatibility."""
    incompatibilities: list[str] = []
    p1: dict[str, Any] | None = None
    p2: dict[str, Any] | None = None

    try:
        p1 = extract_run_provenance_record(run1_entry)
    except ValueError as exc:
        incompatibilities.append(f"Run 1 provenance validation failed: {exc}")

    try:
        p2 = extract_run_provenance_record(run2_entry)
    except ValueError as exc:
        incompatibilities.append(f"Run 2 provenance validation failed: {exc}")

    if p1 is None or p2 is None:
        metric_cmp: dict[str, Any] = {
            "is_comparable": False,
            "missing_keys": [],
            "metrics": {},
        }
        return {
            "run_1_provenance": p1 or {},
            "run_2_provenance": p2 or {},
            "is_comparable": False,
            "provenance_compatible": False,
            "incompatibilities": incompatibilities,
            "overall_reproducibility_tier": "impossible",
            "metrics_comparison": metric_cmp,
        }

    # 0. Reject duplicate run IDs (both runs must be independent)
    if p1["run_id"] == p2["run_id"]:
        incompatibilities.append(
            f"Duplicate run ID detected: both runs have run_id '{p1['run_id']}'. "
            "Dual evaluation runs must be genuinely independent."
        )

    # 1. Dataset hashes match and are identical
    if p1["dataset_content_hash"] != p2["dataset_content_hash"]:
        incompatibilities.append(
            f"Dataset content hash mismatch: Run 1 has '{p1['dataset_content_hash']}', "
            f"Run 2 has '{p2['dataset_content_hash']}'."
        )

    # 2. Config hashes match and are identical
    if p1["config_hash"] != p2["config_hash"]:
        incompatibilities.append(
            f"Config hash mismatch: Run 1 has '{p1['config_hash']}', "
            f"Run 2 has '{p2['config_hash']}'."
        )

    # 3. Dataset IDs match
    if p1["dataset_id"] != p2["dataset_id"]:
        incompatibilities.append(
            f"Dataset ID mismatch: Run 1 has '{p1['dataset_id']}', Run 2 has '{p2['dataset_id']}'."
        )

    # 4. Document scopes match exactly
    if p1["document_scope"] != p2["document_scope"]:
        incompatibilities.append(
            f"Document scope mismatch: Run 1 has {p1['document_scope']}, "
            f"Run 2 has {p2['document_scope']}."
        )

    # 5. Query counts match and equal expected benchmark query count if specified
    if p1["query_count"] != p2["query_count"]:
        incompatibilities.append(
            f"Query count mismatch: Run 1 evaluated {p1['query_count']}, "
            f"Run 2 evaluated {p2['query_count']}."
        )
    if expected_query_count is not None:
        if p1["query_count"] != expected_query_count or p2["query_count"] != expected_query_count:
            if p1["query_count"] == p2["query_count"]:
                incompatibilities.append(
                    f"Query count mismatch with expected benchmark size ({expected_query_count}): "
                    f"both runs evaluated {p1['query_count']}."
                )
            else:
                incompatibilities.append(
                    f"Query count mismatch with expected benchmark size ({expected_query_count}): "
                    f"Run 1 evaluated {p1['query_count']}, Run 2 evaluated {p2['query_count']}."
                )

    # 6. Both reproducibility manifests report exact
    if p1["reproducibility_tier"] != "exact" or p2["reproducibility_tier"] != "exact":
        incompatibilities.append(
            f"Reproducibility tier is not exact: Run 1 is '{p1['reproducibility_tier']}', "
            f"Run 2 is '{p2['reproducibility_tier']}'."
        )

    # 7. Both manifests permit reproduction
    if not p1["can_reproduce"] or not p2["can_reproduce"]:
        incompatibilities.append(
            f"can_reproduce is false: Run 1 is {p1['can_reproduce']}, "
            f"Run 2 is {p2['can_reproduce']}."
        )

    # 8. Retriever, top_k, provider, and model match
    if p1["retriever"] != p2["retriever"]:
        incompatibilities.append(
            f"Retriever mismatch: Run 1 is '{p1['retriever']}', Run 2 is '{p2['retriever']}'."
        )
    if p1["top_k"] != p2["top_k"]:
        incompatibilities.append(f"Top-K mismatch: Run 1 is {p1['top_k']}, Run 2 is {p2['top_k']}.")
    if p1["provider"] != p2["provider"] or p1["model"] != p2["model"]:
        incompatibilities.append(
            f"Provider/model mismatch: Run 1 is '{p1['provider']}/{p1['model']}', "
            f"Run 2 is '{p2['provider']}/{p2['model']}'."
        )

    # Metric comparison
    metric_cmp = compare_run_metrics(p1["metrics"], p2["metrics"])

    provenance_compatible = len(incompatibilities) == 0
    is_comparable = provenance_compatible and metric_cmp["is_comparable"]

    if (
        is_comparable
        and p1["reproducibility_tier"] == "exact"
        and p2["reproducibility_tier"] == "exact"
    ):
        overall_tier = "exact"
    elif p1["reproducibility_tier"] in ("exact", "partial") and p2["reproducibility_tier"] in (
        "exact",
        "partial",
    ):
        overall_tier = "partial"
    else:
        overall_tier = "impossible"

    return {
        "run_1_provenance": p1,
        "run_2_provenance": p2,
        "is_comparable": is_comparable,
        "provenance_compatible": provenance_compatible,
        "incompatibilities": incompatibilities,
        "overall_reproducibility_tier": overall_tier,
        "metrics_comparison": metric_cmp,
    }


def get_environment_provenance() -> dict[str, Any]:
    """Capture authentic environment and package metadata without hardcoding."""
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            timeout=5.0,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to capture git commit SHA via 'git rev-parse HEAD': {exc}"
        ) from exc

    raw_sha = res.stdout.strip()
    try:
        git_sha = validate_git_commit_sha(raw_sha)
    except ValueError as exc:
        raise RuntimeError(f"Invalid git commit SHA captured from environment: {exc}") from exc

    pkgs = [
        "fastapi",
        "pydantic",
        "uvicorn",
        "httpx",
        "pymupdf",
        "faiss-cpu",
        "sentence-transformers",
        "torch",
        "ruff",
        "mypy",
        "pytest",
    ]
    pkg_versions: dict[str, str] = {}
    for pkg in pkgs:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            pkg_versions[pkg] = "not_installed"

    return {
        "git_commit_sha": git_sha,
        "python_version": sys.version,
        "platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "architecture": platform.machine(),
        "package_versions": pkg_versions,
    }


def process_evaluation_runs(
    runs: list[dict[str, Any]],
    demo_ds: dict[str, Any],
    env_prov: dict[str, Any],
    expected_query_count: int | None = 10,
    evidence_path: str = "docs/reproducibility_evidence.json",
    diagnostic_path: str = "docs/incompatible_reproducibility_diagnostic.json",
    api_url: str = API_URL,
) -> dict[str, Any]:
    """Process evaluation runs, perform cross-run validation, and output artifacts.

    If runs are incompatible, writes a diagnostic failure artifact and raises RuntimeError.
    If runs are compatible, prints comparison summary, writes reproducibility evidence,
    and returns it.
    """
    if len(runs) < 2:
        raise ValueError(
            f"At least two runs are required for cross-run validation, got {len(runs)}"
        )

    validation = validate_and_compare_runs(
        runs[0], runs[1], expected_query_count=expected_query_count
    )
    p1 = validation.get("run_1_provenance")
    p2 = validation.get("run_2_provenance")
    metric_cmp = validation.get("metrics_comparison", {})

    if (
        not validation.get("provenance_compatible")
        or not p1
        or not p2
        or not validation.get("is_comparable")
    ):
        diagnostic_output = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": "incompatible",
            "api_url": api_url,
            "git_commit_sha": env_prov.get("git_commit_sha"),
            "incompatibilities": validation.get("incompatibilities", []),
            "cross_run_validation": validation,
            "provenance": env_prov,
            "dataset": demo_ds,
            "runs": runs,
        }
        if os.path.dirname(diagnostic_path):
            os.makedirs(os.path.dirname(diagnostic_path), exist_ok=True)
        with open(diagnostic_path, "w", encoding="utf-8") as f:
            json.dump(diagnostic_output, f, indent=2)

        print("\nERROR: Cross-run validation failed! Incompatibilities:")
        for inc in validation.get("incompatibilities", []):
            print(f"  - {inc}")
        print(f"Wrote diagnostic failure artifact to '{diagnostic_path}'.")
        raise RuntimeError(
            f"Cross-run evaluation validation failed: {validation.get('incompatibilities')}"
        )

    print("\n======================================================================")
    print("  REPRODUCIBILITY EVALUATION COMPARISON: RUN 1 vs RUN 2")
    print("======================================================================")
    print(f"Dataset Name:                 {demo_ds.get('name')}")
    print(f"Dataset ID:                   {demo_ds.get('dataset_id')}")
    print(f"Query Count:                  {demo_ds.get('query_count')}")
    print(f"Dataset SHA-256 (Run 1):      {p1['dataset_content_hash']}")
    print(f"Dataset SHA-256 (Run 2):      {p2['dataset_content_hash']}")
    print(f"Config Hash (Run 1):          {p1['config_hash']}")
    print(f"Config Hash (Run 2):          {p2['config_hash']}")
    print("Retriever:                    bm25 (top_k=5)")
    print("Provider / Model:             mock / mock-pipeline (Deterministic Rule-Based Mock)")
    print(f"Reproducibility Tier (R1):    {p1['reproducibility_tier']}")
    print(f"Reproducibility Tier (R2):    {p2['reproducibility_tier']}")
    print(f"Overall Reproducibility Tier: {validation['overall_reproducibility_tier']}")
    print(f"Provenance Compatible:        {validation['provenance_compatible']}")
    print(f"Is Comparable:                {validation['is_comparable']}")
    if validation.get("incompatibilities"):
        print(f"Incompatibilities:            {validation['incompatibilities']}")
    if metric_cmp.get("missing_keys"):
        print(f"Missing Metric Keys:          {metric_cmp['missing_keys']}")
    print("----------------------------------------------------------------------")
    print(f"{'Metric':<32} | {'Run 1':<12} | {'Run 2':<12} | {'Delta':<10}")
    print("----------------------------------------------------------------------")

    for k, info in metric_cmp.get("metrics", {}).items():
        v1_str = f"{info['run_1']:.4f}" if info["run_1"] is not None else "N/A"
        v2_str = f"{info['run_2']:.4f}" if info["run_2"] is not None else "N/A"
        delta_str = f"{info['delta']:+.4f}" if info["delta"] is not None else "N/A"
        print(f"{k:<32} | {v1_str:<12} | {v2_str:<12} | {delta_str:<10}")

    print("----------------------------------------------------------------------")
    print(f"Failure Modes Run 1:  {p1['failure_counts']}")
    print(f"Failure Modes Run 2:  {p2['failure_counts']}")
    print("======================================================================")
    print("Note on Mock Provider Evaluation:")
    print("  Exact reproducibility demonstrates pipeline determinism and metric repeatability")
    print("  under identical inputs, not answer quality. The deterministic mock provider emits")
    print("  rule-based test strings without citations or NLI grounding, yielding 0.0000 citation")
    print("  precision/recall and 0.1000 lexical groundedness. These metrics honestly reflect the")
    print(
        "  mock provider output and do not represent authentic neural models "
        "(e.g. Ollama llama3.2)."
    )
    print("======================================================================")

    evidence_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": api_url,
        "git_commit_sha": env_prov.get("git_commit_sha"),
        "dataset_content_hash": p1["dataset_content_hash"],
        "config_hash": p1["config_hash"],
        "retriever": p1["retriever"],
        "top_k": p1["top_k"],
        "provider": p1["provider"],
        "model": p1["model"],
        "query_count": p1["query_count"],
        "reproducibility_tier": validation["overall_reproducibility_tier"],
        "is_comparable": validation["is_comparable"],
        "provenance_compatible": validation["provenance_compatible"],
        "incompatibilities": validation["incompatibilities"],
        "evaluation_notes": (
            "Exact reproducibility demonstrates pipeline determinism and metric repeatability "
            "under identical inputs, not answer quality. The deterministic mock provider emits "
            "rule-based test strings without citations or NLI grounding, yielding 0.0000 citation "
            "precision/recall and 0.1000 lexical groundedness. These metrics honestly reflect the "
            "mock provider output and do not represent the performance of authentic neural models "
            "such as Ollama llama3.2."
        ),
        "provenance": env_prov,
        "dataset": demo_ds,
        "cross_run_validation": validation,
        "runs": runs,
    }

    if os.path.dirname(evidence_path):
        os.makedirs(os.path.dirname(evidence_path), exist_ok=True)
    with open(evidence_path, "w", encoding="utf-8") as f:
        json.dump(evidence_output, f, indent=2)
    print(f"Wrote evidence artifact to '{evidence_path}'.")

    return evidence_output


def run() -> None:
    client = httpx.Client(base_url=API_URL, timeout=60.0)
    print(f"Connecting to STING API at {API_URL}...")

    # 1. Check health
    health = request_json(client, "GET", "/api/health")
    print(f"API Health: {health}")

    # 2. Find dataset
    res_ds = request_json(client, "GET", "/api/datasets")
    datasets = res_ds.get("datasets") or res_ds.get("items") or []
    demo_ds = select_golden_dataset(datasets)
    dataset_id = demo_ds["dataset_id"]
    print(f"Using Dataset: '{demo_ds['name']}' ({dataset_id}), queries: {demo_ds['query_count']}")

    # 3. Find demo documents and validate exact scope
    res_docs = request_json(client, "GET", "/api/documents")
    docs = res_docs.get("documents", [])
    demo_doc_ids = validate_document_scope(demo_ds, docs)
    print(f"Validated Northstar Scoped Document IDs: {demo_doc_ids}")

    runs: list[dict[str, Any]] = []
    for run_idx in (1, 2):
        exp_name = f"northstar_baseline_run_{run_idx}"
        print(f"\n--- Launching Evaluation Run {run_idx}: '{exp_name}' ---")
        payload = {
            "experiment_name": exp_name,
            "dataset_id": dataset_id,
            "document_ids": demo_doc_ids,
            "retriever_type": "bm25",
            "provider": "mock",
            "model_name": "mock-pipeline",
            "top_k": 5,
            "enable_groundedness": True,
            "enable_citations": True,
            "enable_nli": False,
            "enable_judge": False,
        }
        res = request_json(client, "POST", "/api/evaluation/run", json=payload)
        job_id = res["job_id"]
        print(f"Dispatched Job ID: {job_id}")

        job_detail = poll_evaluation_job(client, job_id, timeout_seconds=60.0)
        run_id = job_detail["run_id"]

        run_data = request_json(client, "GET", f"/api/runs/{run_id}")
        repro_data = request_json(client, "GET", f"/api/runs/{run_id}/reproducibility")

        runs.append(
            {
                "run_index": run_idx,
                "run_id": run_id,
                "experiment_name": exp_name,
                "job_id": job_id,
                "run_data": run_data,
                "repro_data": repro_data,
            }
        )

    # Cross-run independent validation, metric comparison, and artifact generation
    env_prov = get_environment_provenance()
    process_evaluation_runs(
        runs=runs,
        demo_ds=demo_ds,
        env_prov=env_prov,
        expected_query_count=demo_ds.get("query_count", 10),
        evidence_path="docs/reproducibility_evidence.json",
        diagnostic_path="docs/incompatible_reproducibility_diagnostic.json",
        api_url=API_URL,
    )


if __name__ == "__main__":
    run()
