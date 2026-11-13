#!/usr/bin/env python3
"""Hermetic End-to-End Demo Smoke Test for STING (V2-006.2).

Executes the complete STING workflow against an isolated, temporary SQLite
database and temporary storage directories without requiring live Ollama or
downloading external model weights:
1. Validates the Northstar demo corpus manifest.
2. Ingests all 3 documents via POST /api/documents/upload.
3. Verifies all documents reach READY status.
4. Retrieves runtime chunks via GET /api/documents/{id}/chunks.
5. Dynamically resolves all 9 evidence anchors to chunk IDs fail-closed.
6. Creates the golden benchmark dataset via POST /api/datasets.
7. Validates dataset query count (10) and document scope.
8. Executes a document-grounded query via POST /api/query.
9. Verifies citation provenance and retrieval trace.
10. Submits an ungrounded query and verifies safe abstention.
11. Exercises cross-document investigation via POST /api/investigations and exports report.
12. Launches a deterministic evaluation run via POST /api/evaluation/run.
13. Polls the evaluation job until completion with bounded timeout.
14. Verifies the stored evaluation run exists and contains query metrics.
15. Retrieves the run reproducibility manifest via GET /api/runs/{run_id}/reproducibility.
16. Validates dataset and source document provenance integrity in the manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import re
import sys
import tempfile
import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from fastapi.testclient import TestClient

from reliability_lab.api.main import (
    create_app,
    get_dataset_repository,
    get_dataset_service,
    get_document_repository,
    get_evaluation_service,
    get_ingestion_service,
    get_investigation_repository,
    get_investigation_service,
    get_query_service,
    get_repository,
)
from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
from reliability_lab.benchmark.service import BenchmarkDatasetService
from reliability_lab.evaluation.execution_service import EvaluationExecutionService
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.ingestion.repository import DocumentRepository
from reliability_lab.ingestion.service import DocumentIngestionService
from reliability_lab.investigation.repository import InvestigationRepository
from reliability_lab.investigation.service import InvestigationService
from reliability_lab.llm.client import BaseLLMClient
from reliability_lab.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamChunk,
    TokenUsage,
)
from reliability_lab.query.orchestrator import DocumentQueryService
from reliability_lab.storage.repository import ExperimentRepository

# Ensure repository root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts.prepare_sting_demo import resolve_anchors_to_chunks, validate_demo_pack  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("verify_demo")


class HermeticDeterministicEmbedder(BaseEmbedder):
    """Deterministic 384-dimensional embedder requiring zero external downloads."""

    def __init__(self, dim: int = 384) -> None:
        self._dim = dim

    @property
    def model_name(self) -> str:
        return "hermetic-deterministic-embedder-v1"

    @property
    def artifact_fingerprint(self) -> str:
        return "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: Any, batch_size: int = 32) -> np.ndarray:
        vecs = []
        for t in texts:
            h = hashlib.sha256(str(t).encode("utf-8")).digest()
            seed = int.from_bytes(h[:4], "little")
            rng = np.random.RandomState(seed)
            v = rng.randn(self._dim).astype(np.float32)
            norm = float(np.linalg.norm(v))
            if norm > 0:
                v = v / norm
            vecs.append(v)
        return np.array(vecs, dtype=np.float32)


class ControlledSmokeLLMClient(BaseLLMClient):
    """Controlled test double for grounded answers, safe abstention, and investigation JSON."""

    def __init__(self, model_name: str = "controlled-smoke-llm") -> None:
        self._model_name = model_name

    @property
    def model_name(self) -> str:
        return self._model_name

    def generate(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        full_text = ""
        if isinstance(prompt, str):
            full_text = prompt
        else:
            for m in prompt:
                if isinstance(m, dict):
                    full_text += " " + m.get("content", "")
                elif isinstance(m, LLMMessage):
                    full_text += " " + m.content

        # Handle abstention requests (RLHF, PPO, unmentioned hyperparameters)
        if any(term in full_text for term in ("RLHF", "PPO", "reinforcement learning")):
            return LLMResponse(
                text=(
                    "[ABSTAIN: INSUFFICIENT_EVIDENCE] The retrieved Northstar documents "
                    "do not mention RLHF reward models or PPO fine-tuning hyperparameters."
                ),
                model=self._model_name,
                usage=TokenUsage(prompt_tokens=25, completion_tokens=30, total_tokens=55),
                finish_reason="stop",
            )

        # Standard grounded question response with CIT-1
        answer = (
            "According to the Northstar architectural specification [CIT-1], "
            "document ingestion uses a target chunk size of 500 characters and "
            "100 character overlap."
        )
        return LLMResponse(
            text=answer,
            model=self._model_name,
            usage=TokenUsage(prompt_tokens=50, completion_tokens=40, total_tokens=90),
            finish_reason="stop",
        )

    def generate_json(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Produce structured findings JSON for cross-document investigation."""
        full_text = ""
        if isinstance(prompt, str):
            full_text = prompt
        else:
            for m in prompt:
                if isinstance(m, dict):
                    full_text += " " + m.get("content", "")
                elif isinstance(m, LLMMessage):
                    full_text += " " + m.content

        # Extract available chunk IDs and passages from prompt
        chunk_matches = re.findall(
            r"--- \[CHUNK_ID:\s*([^\]]+)\]\s*\([^\)]*\)\s*---\n(.*?)(?=(?:--- \[CHUNK_ID:|$))",
            full_text,
            flags=re.DOTALL,
        )

        findings = []
        if chunk_matches:
            cid, passage = chunk_matches[0]
            clean_cid = cid.strip()
            # Pick a verbatim snippet from the passage
            clean_snippet = passage.strip().split("\n")[0][:80]
            rel = "supports"
            if "incident" in full_text.lower():
                rel = "contradicts"

            findings.append(
                {
                    "topic_name": "Retrieval Performance",
                    "claim": f"Performance evaluation finding for chunk {clean_cid}",
                    "chunk_id": clean_cid,
                    "page_number": 1,
                    "supporting_passage": clean_snippet,
                    "relationship": rel,
                    "confidence": 0.95,
                    "methodology_notes": "Hermetic smoke test verified finding",
                }
            )

        return {"findings": findings}

    def generate_stream(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> Iterator[LLMStreamChunk]:
        resp = self.generate(prompt=prompt, config=config)
        yield LLMStreamChunk(
            delta=resp.text,
            model=resp.model,
            done=True,
            usage=resp.usage,
            finish_reason="stop",
        )


def run_hermetic_demo_smoke_test(
    demo_dir: Path | None = None,
    temp_dir_path: Path | None = None,
) -> dict[str, Any]:
    """Execute complete end-to-end hermetic demonstration smoke test."""
    if demo_dir is None:
        demo_dir = Path("data/demo")

    logger.info("Step 1: Validating demo manifest and corpus...")
    manifest, raw_questions = validate_demo_pack(demo_dir)
    assert len(manifest["documents"]) >= 3
    assert len(raw_questions) >= 8

    # Create isolated temp workspace
    with tempfile.TemporaryDirectory() as tmp_str:
        work_dir = temp_dir_path or Path(tmp_str)
        db_path = work_dir / "smoke.sqlite3"
        storage_dir = work_dir / "documents"
        index_dir = work_dir / "indexes"

        os.environ["RELIABILITY_LAB_DB"] = str(db_path)

        embedder = HermeticDeterministicEmbedder(dim=384)
        smoke_llm = ControlledSmokeLLMClient()

        def make_doc_repo() -> DocumentRepository:
            return DocumentRepository(db_path)

        def make_exp_repo() -> ExperimentRepository:
            return ExperimentRepository(db_path)

        def make_ds_repo() -> BenchmarkDatasetRepository:
            return BenchmarkDatasetRepository(db_path)

        def make_inv_repo() -> InvestigationRepository:
            return InvestigationRepository(db_path)

        def make_ingestion() -> DocumentIngestionService:
            return DocumentIngestionService(
                repository=make_doc_repo(),
                storage_dir=storage_dir,
                index_dir=index_dir,
                embedder=embedder,
            )

        def make_query_service() -> DocumentQueryService:
            return DocumentQueryService(
                repository=make_doc_repo(),
                ingestion_service=make_ingestion(),
                embedder=embedder,
                llm_client=smoke_llm,
            )

        def make_inv_service() -> InvestigationService:
            return InvestigationService(
                investigation_repository=make_inv_repo(),
                document_repository=make_doc_repo(),
                query_service=make_query_service(),
                llm_client=smoke_llm,
            )

        def make_ds_service() -> BenchmarkDatasetService:
            return BenchmarkDatasetService(
                dataset_repository=make_ds_repo(),
                document_repository=make_doc_repo(),
            )

        eval_service = EvaluationExecutionService(
            experiment_repository=make_exp_repo(),
            document_repository=make_doc_repo(),
            dataset_repository=make_ds_repo(),
            ingestion_service=make_ingestion(),
            embedder=embedder,
        )

        app = create_app()
        app.dependency_overrides[get_repository] = make_exp_repo
        app.dependency_overrides[get_document_repository] = make_doc_repo
        app.dependency_overrides[get_dataset_repository] = make_ds_repo
        app.dependency_overrides[get_investigation_repository] = make_inv_repo
        app.dependency_overrides[get_ingestion_service] = make_ingestion
        app.dependency_overrides[get_query_service] = make_query_service
        app.dependency_overrides[get_investigation_service] = make_inv_service
        app.dependency_overrides[get_dataset_service] = make_ds_service
        app.dependency_overrides[get_evaluation_service] = lambda: eval_service

        client = TestClient(app)

        # Health check
        r_health = client.get("/api/health")
        assert r_health.status_code == 200, f"Health check failed: {r_health.text}"

        logger.info("Step 2 & 3: Ingesting 3 documents and verifying READY status...")
        doc_ids: dict[str, str] = {}
        corpus_dir = demo_dir / "corpus"

        for doc_entry in manifest["documents"]:
            fname = doc_entry["filename"]
            fpath = corpus_dir / fname
            with open(fpath, "rb") as fh:
                r_up = client.post(
                    "/api/documents/upload",
                    files={"file": (fname, fh, "text/markdown")},
                )
            assert r_up.status_code in (200, 201), f"Upload failed for {fname}: {r_up.text}"
            doc_id = r_up.json()["doc_id"]
            doc_ids[fname] = doc_id

            # Poll document status
            r_doc = client.get(f"/api/documents/{doc_id}")
            assert r_doc.status_code == 200
            assert r_doc.json()["status"].upper() == "READY"
            logger.info("  Document '%s' successfully reached READY (%s)", fname, doc_id)

        logger.info("Step 4 & 5: Fetching chunks and resolving dynamic evidence anchors...")
        doc_chunks_by_fname: dict[str, list[Any]] = {}
        for fname, did in doc_ids.items():
            r_chunks = client.get(f"/api/documents/{did}/chunks?limit=100")
            assert r_chunks.status_code == 200
            chunk_items = r_chunks.json().get("chunks", [])
            assert len(chunk_items) > 0, f"Expected chunks for {fname}"
            doc_chunks_by_fname[fname] = chunk_items

        resolved_queries = resolve_anchors_to_chunks(raw_questions, doc_chunks_by_fname)
        assert len(resolved_queries) == len(raw_questions)

        anchored_cnt = sum(1 for q in resolved_queries if len(q.get("relevant_chunk_ids", [])) == 1)
        abstention_cnt = sum(
            1 for q in resolved_queries if len(q.get("relevant_chunk_ids", [])) == 0
        )
        assert anchored_cnt == 9
        assert abstention_cnt == 1
        logger.info("  Resolved 9 anchored queries and 1 intentional safe-abstention query.")

        logger.info("Step 6 & 7: Creating benchmark dataset and verifying scope...")
        target_scope = sorted(doc_ids.values())
        ds_payload = {
            "name": "Northstar Golden Benchmark",
            "description": "Hermetic verified demo benchmark dataset",
            "document_scope": target_scope,
            "is_synthetic": True,
            "queries": resolved_queries,
        }
        r_ds = client.post("/api/datasets", json=ds_payload)
        assert r_ds.status_code in (200, 201), f"Dataset creation failed: {r_ds.text}"
        dataset_id = r_ds.json()["dataset_id"]

        r_get_ds = client.get(f"/api/datasets/{dataset_id}")
        assert r_get_ds.status_code == 200
        ds_data = r_get_ds.json()
        assert ds_data["query_count"] == 10
        assert sorted(ds_data["document_scope"]) == target_scope
        logger.info("  Dataset '%s' created with 10 queries and verified scope.", dataset_id)

        logger.info("Step 8 & 9: Running document-scoped evidence query with citation trace...")
        arch_id = doc_ids["architecture.md"]
        q_payload = {
            "query": "What target chunk size and overlap are specified for Northstar?",
            "document_ids": [arch_id],
            "strategy": "bm25",
            "top_k": 5,
        }
        r_query = client.post("/api/query", json=q_payload)
        assert r_query.status_code == 200, f"Evidence query failed: {r_query.text}"
        q_resp = r_query.json()
        assert q_resp["abstained"] is False
        assert len(q_resp["resolved_citations"]) > 0
        assert q_resp["resolved_citations"][0]["is_valid"] is True
        assert q_resp["trace"] is not None
        assert len(q_resp["trace"]["evidence_chunks"]) > 0
        logger.info("  Evidence query verified: citation resolved to candidate chunk.")

        logger.info("Step 10: Verifying safe abstention on ungrounded question...")
        abstain_payload = {
            "query": (
                "What reinforcement learning from human feedback (RLHF) reward model "
                "and PPO hyperparameters are used in Northstar?"
            ),
            "document_ids": list(doc_ids.values()),
            "strategy": "bm25",
            "top_k": 5,
        }
        r_abstain = client.post("/api/query", json=abstain_payload)
        assert r_abstain.status_code == 200
        abstain_resp = r_abstain.json()
        assert abstain_resp["abstained"] is True or "[ABSTAIN" in abstain_resp["answer"]
        logger.info("  Safe abstention verified for ungrounded topic.")

        logger.info("Step 11: Exercising cross-document evidence investigation...")
        inv_payload = {
            "title": "Smoke Investigation: Retrieval Disagreements",
            "query": "Compare reported benchmark retrieval performance with incident findings.",
            "document_ids": list(doc_ids.values()),
            "run_immediately": True,
        }
        r_inv = client.post("/api/investigations", json=inv_payload)
        assert r_inv.status_code in (200, 201), f"Investigation creation failed: {r_inv.text}"
        inv_id = r_inv.json()["investigation_id"]

        r_inv_get = client.get(f"/api/investigations/{inv_id}")
        assert r_inv_get.status_code == 200
        inv_data = r_inv_get.json()
        assert inv_data["status"] == "completed"
        assert inv_data["report"] is not None

        r_export = client.get(f"/api/investigations/{inv_id}/export")
        assert r_export.status_code == 200
        assert "# STING Investigation Report:" in r_export.text
        logger.info("  Cross-document investigation and markdown export verified.")

        logger.info("Step 12 & 13: Launching evaluation run via POST /api/evaluation/run...")
        eval_payload = {
            "experiment_name": "smoke_hermetic_eval",
            "dataset_id": dataset_id,
            "document_ids": list(doc_ids.values()),
            "retriever_type": "bm25",
            "provider": "mock",
            "model_name": "mock-pipeline",
            "top_k": 5,
            "enable_groundedness": True,
            "enable_citations": True,
            "enable_nli": False,
            "enable_judge": False,
        }
        r_eval = client.post("/api/evaluation/run", json=eval_payload)
        assert r_eval.status_code == 200, f"Evaluation launch failed: {r_eval.text}"
        job_id = r_eval.json()["job_id"]

        # Poll job until completed with bounded timeout
        terminal_status = ""
        run_id = ""
        for _ in range(60):
            r_job = client.get(f"/api/evaluation/jobs/{job_id}")
            assert r_job.status_code == 200
            jdata = r_job.json()
            st = jdata.get("status", "")
            if st in ("completed", "failed"):
                terminal_status = st
                run_id = jdata.get("run_id") or ""
                break
            time.sleep(0.5)

        assert terminal_status == "completed", f"Job ended with status '{terminal_status}'"
        assert run_id != "", "Job completed without run_id"
        logger.info("  Evaluation completed successfully with run ID '%s'.", run_id)

        logger.info("Step 14: Verifying run query metrics...")
        r_run = client.get(f"/api/runs/{run_id}")
        assert r_run.status_code == 200
        run_data = r_run.json()
        assert run_data["query_count"] == 10
        assert "metrics" in run_data
        logger.info("  Run contains 10 evaluated queries and aggregate metrics.")

        logger.info("Step 15 & 16: Fetching and validating reproducibility manifest...")
        r_repro = client.get(f"/api/runs/{run_id}/reproducibility")
        assert r_repro.status_code == 200
        repro_data = r_repro.json()
        assert repro_data["run_id"] == run_id
        assert repro_data["dataset_id"] == dataset_id
        assert repro_data["dataset_exists"] is True
        assert repro_data["dataset_unchanged"] is True
        assert set(repro_data["corpus_document_ids"]) == set(doc_ids.values())
        assert repro_data["corpus_documents_exist"] is True
        logger.info("  Reproducibility manifest successfully validated provenance.")

        return {
            "status": "passed",
            "documents": doc_ids,
            "dataset_id": dataset_id,
            "query_count": 10,
            "evidence_anchors": 9,
            "abstentions": 1,
            "run_id": run_id,
            "job_id": job_id,
            "reproducibility_tier": repro_data.get("reproducibility_tier", "exact"),
        }


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Run hermetic end-to-end smoke verification for STING demonstration."
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/demo"),
        help="Path to demo corpus directory (default: data/demo).",
    )
    args = parser.parse_args()

    try:
        result = run_hermetic_demo_smoke_test(demo_dir=args.demo_dir)
        print("\n" + "=" * 70)
        print("  HERMETIC DEMO SMOKE TEST RESULT: PASSED")
        print("=" * 70)
        print(f"Documents Ingested:    {len(result['documents'])}")
        print(f"Dataset Created:       {result['dataset_id']} (10 queries)")
        print(f"Evaluation Run:        {result['run_id']}")
        print(f"Reproducibility Tier:  {result['reproducibility_tier']}")
        print("All 16 validation checkpoints verified cleanly.")
        print("=" * 70 + "\n")
        return 0
    except Exception as exc:
        logger.error("Hermetic demo smoke test failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
