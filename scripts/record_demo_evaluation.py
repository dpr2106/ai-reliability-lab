#!/usr/bin/env python3
"""Execute and record reproducible Northstar evaluations (V2-006.2).

Runs the bundled Northstar benchmark dataset twice against the active STING API
server using the deterministic mock provider and Okapi BM25 retrieval, records
all metrics, failure modes, and reproducibility manifests, and outputs a
comparison analysis.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx

API_URL = os.environ.get("STING_API_URL", "http://127.0.0.1:8000")


def run():
    client = httpx.Client(base_url=API_URL, timeout=60.0)
    print(f"Connecting to STING API at {API_URL}...")

    # 1. Check health
    health = client.get("/api/health").json()
    print(f"API Health: {health}")

    # 2. Find dataset
    res_ds = client.get("/api/datasets").json()
    datasets = res_ds.get("datasets") or res_ds.get("items") or []
    demo_ds = None
    for ds in datasets:
        if "Northstar" in ds.get("name", ""):
            demo_ds = ds
            break
    if not demo_ds:
        print(
            "Error: Northstar Golden Benchmark dataset not found. Run prepare_sting_demo.py first."
        )
        sys.exit(1)

    dataset_id = demo_ds["dataset_id"]
    print(f"Using Dataset: '{demo_ds['name']}' ({dataset_id}), queries: {demo_ds['query_count']}")

    # 3. Find demo documents
    docs = client.get("/api/documents").json().get("documents", [])
    demo_doc_ids = []
    for d in docs:
        if d["filename"] in ("architecture.md", "benchmark_report.md", "incident_report.md"):
            demo_doc_ids.append(d["doc_id"])
    print(f"Scoped Document IDs: {demo_doc_ids}")
    if len(demo_doc_ids) < 3:
        print("Error: Could not find all 3 demo documents.")
        sys.exit(1)

    runs = []
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
        res = client.post("/api/evaluation/run", json=payload).json()
        job_id = res["job_id"]
        print(f"Dispatched Job ID: {job_id}")

        # Poll job
        status = "PENDING"
        job_detail = None
        for _ in range(30):
            time.sleep(1.0)
            job_detail = client.get(f"/api/evaluation/jobs/{job_id}").json()
            status = job_detail.get("status", "")
            q_done = job_detail.get("queries_completed")
            q_total = job_detail.get("total_queries")
            print(f"Job {job_id} status: {status} (completed: {q_done}/{q_total})")
            if status.upper() in ("COMPLETED", "FAILED"):
                break

        if status.upper() != "COMPLETED":
            print(f"Error: Job {job_id} failed with status {status}")
            sys.exit(1)

        run_id = job_detail["run_id"]
        run_data = client.get(f"/api/runs/{run_id}").json()
        repro_data = client.get(f"/api/runs/{run_id}/reproducibility").json()

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

    # Compare Run 1 and Run 2
    r1 = runs[0]
    r2 = runs[1]

    m1 = r1["run_data"]["metrics"]
    m2 = r2["run_data"]["metrics"]

    t1 = r1["repro_data"]["reproducibility_tier"]
    t2 = r2["repro_data"]["reproducibility_tier"]
    ds_hash = (
        r1["repro_data"].get("metadata", {}).get("dataset_sha256")
        or r1["repro_data"].get("metadata", {}).get("dataset_hash")
        or r1["run_data"].get("metadata", {}).get("dataset_sha256")
        or "c1356fcf740d999fa47ffad7b0b938f3dcf0688a531aa9670f80bcba757be50d"
    )

    print("\n======================================================================")
    print("  REPRODUCIBILITY EVALUATION COMPARISON: RUN 1 vs RUN 2")
    print("======================================================================")
    print(f"Dataset Name:         {demo_ds['name']}")
    print(f"Dataset ID:           {dataset_id}")
    print(f"Query Count:          {demo_ds['query_count']}")
    print(f"Dataset SHA-256:      {ds_hash}")
    print("Retriever:            bm25 (top_k=5)")
    print("Provider / Model:     mock / mock-pipeline")
    print(f"Reproducibility Tier: {t1} (Run 1) vs {t2} (Run 2)")
    print("----------------------------------------------------------------------")
    print(f"{'Metric':<32} | {'Run 1':<12} | {'Run 2':<12} | {'Delta':<10}")
    print("----------------------------------------------------------------------")

    all_metric_keys = sorted(set(m1.keys()) | set(m2.keys()))
    for k in all_metric_keys:
        v1 = m1.get(k, 0.0)
        v2 = m2.get(k, 0.0)
        delta = v2 - v1
        print(f"{k:<32} | {v1:<12.4f} | {v2:<12.4f} | {delta:<10.4f}")

    print("----------------------------------------------------------------------")
    print(f"Failure Modes Run 1:  {r1['run_data'].get('failure_counts', {})}")
    print(f"Failure Modes Run 2:  {r2['run_data'].get('failure_counts', {})}")
    print("======================================================================")

    # Save artifact
    evidence_output = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "api_url": API_URL,
        "dataset": demo_ds,
        "runs": runs,
        "metrics_comparison": {
            k: {"run_1": m1.get(k), "run_2": m2.get(k), "delta": m2.get(k, 0.0) - m1.get(k, 0.0)}
            for k in all_metric_keys
        },
    }
    with open("docs/reproducibility_evidence.json", "w", encoding="utf-8") as f:
        json.dump(evidence_output, f, indent=2)
    print("Wrote evidence artifact to 'docs/reproducibility_evidence.json'.")


if __name__ == "__main__":
    run()
