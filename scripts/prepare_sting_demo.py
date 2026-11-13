#!/usr/bin/env python3
"""STING Reproducible Demo Preparation Workflow (V2-006.1).

Coordinates document ingestion, chunk retrieval, fail-closed evidence anchor
resolution, and benchmark dataset creation against a running STING instance or
via local storage services.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("prepare_demo")


@dataclass
class DemoPreparationResult:
    """Outcome of a demo preparation execution."""

    documents: dict[str, str]  # filename -> doc_id
    chunk_counts: dict[str, int]  # filename -> chunk_count
    dataset_id: str
    dataset_name: str
    query_count: int
    resolved_queries: list[dict[str, Any]] = field(default_factory=list)
    mode: str = "api"
    db_path: str | None = None
    evidence_anchored_queries: int = 0
    resolved_anchors: int = 0
    intentional_abstentions: int = 0


def compute_file_hash(path: Path) -> str:
    """Compute SHA-256 digest of a local file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def validate_demo_pack(demo_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Validate demo manifest, corpus files, and golden benchmark specifications."""
    manifest_path = demo_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Demo manifest not found at: {manifest_path}")

    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    corpus_dir = demo_dir / "corpus"
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"Demo corpus directory not found at: {corpus_dir}")

    doc_specs = manifest.get("documents", [])
    if len(doc_specs) < 3:
        raise ValueError(f"Demo manifest requires at least 3 documents, found {len(doc_specs)}.")

    for doc in doc_specs:
        fname = doc.get("filename")
        if not fname:
            raise ValueError(f"Document entry missing filename: {doc}")
        fpath = corpus_dir / fname
        if not fpath.exists():
            raise FileNotFoundError(f"Corpus file '{fname}' not found at: {fpath}")
        if fpath.stat().st_size == 0:
            raise ValueError(f"Corpus file '{fname}' is empty.")

    bench_rel = manifest.get("benchmark_file", "golden_questions.json")
    bench_path = demo_dir / bench_rel
    if not bench_path.exists():
        raise FileNotFoundError(f"Golden benchmark specification not found at: {bench_path}")

    with open(bench_path, encoding="utf-8") as f:
        bench_spec = json.load(f)

    questions = bench_spec.get("questions", [])
    if len(questions) < 8:
        raise ValueError(
            f"Benchmark specification requires 8-12 questions, found {len(questions)}."
        )

    return manifest, questions


def resolve_anchors_to_chunks(
    questions: list[dict[str, Any]],
    doc_chunks_by_filename: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Resolve textual evidence anchors to runtime chunk records fail-closed.

    Raises ValueError if an anchor is not found, or matches more than one chunk.
    """
    resolved: list[dict[str, Any]] = []

    for q in questions:
        qid = q.get("query_id")
        text = q.get("text")
        if not qid or not text:
            raise ValueError(f"Malformed question entry: {q}")

        fname = q.get("source_filename")
        anchor = q.get("evidence_anchor")
        grade = int(q.get("relevance_grade", 2))
        cit_label = q.get("required_citation") or f"CIT-{qid}"

        # 1. Abstention / Out-of-scope query
        if not fname:
            resolved.append(
                {
                    "query_id": qid,
                    "text": text,
                    "relevant_chunk_ids": [],
                    "relevance_grades": {},
                    "ground_truth_answer": q.get("reference_answer"),
                    "required_citations": [],
                    "metadata": {
                        "category": q.get("category", "safe_abstention"),
                        "reference_answer": q.get("reference_answer"),
                        "source_filename": None,
                        "citation_to_chunk_id": {},
                    },
                }
            )
            continue

        # 2. Evidence-grounded query
        if fname not in doc_chunks_by_filename:
            raise ValueError(
                f"Query '{qid}' references unknown document '{fname}'. "
                f"Available documents: {list(doc_chunks_by_filename.keys())}."
            )

        chunks = doc_chunks_by_filename[fname]
        if not anchor or not str(anchor).strip():
            raise ValueError(
                f"Query '{qid}' targets document '{fname}' but provides no evidence anchor."
            )

        def _get_chunk_text(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("content") or c.get("text") or "")
            return str(getattr(c, "content", None) or getattr(c, "text", "") or "")

        def _get_chunk_id(c: Any) -> str:
            if isinstance(c, dict):
                return str(c.get("chunk_id", ""))
            return str(getattr(c, "chunk_id", ""))

        anchor_str = str(anchor).strip()
        matching_chunks = [c for c in chunks if anchor_str in _get_chunk_text(c)]

        if not matching_chunks:
            raise ValueError(
                f"Provenance resolution failure for query '{qid}': "
                f"anchor '{anchor_str[:40]}...' was not found in any ingested chunk of '{fname}'."
            )
        if len(matching_chunks) > 1:
            raise ValueError(
                f"Provenance resolution failure for query '{qid}': "
                f"ambiguous anchor '{anchor_str[:40]}...' matched {len(matching_chunks)} distinct "
                f"chunks in '{fname}'. Anchors must be unique."
            )

        target_chunk = matching_chunks[0]
        chunk_id = _get_chunk_id(target_chunk)

        resolved.append(
            {
                "query_id": qid,
                "text": text,
                "relevant_chunk_ids": [chunk_id],
                "relevance_grades": {chunk_id: grade},
                "ground_truth_answer": q.get("reference_answer"),
                "required_citations": [cit_label],
                "metadata": {
                    "category": q.get("category", "evidence_grounded"),
                    "reference_answer": q.get("reference_answer"),
                    "source_filename": fname,
                    "evidence_anchor": anchor_str,
                    "citation_to_chunk_id": {cit_label: chunk_id},
                },
            }
        )

    return resolved


def prepare_demo_api(
    api_url: str,
    demo_dir: Path,
    timeout: float = 60.0,
    reuse_existing: bool = True,
) -> DemoPreparationResult:
    """Execute demo preparation against a running STING HTTP server."""
    import httpx

    manifest, questions = validate_demo_pack(demo_dir)
    corpus_dir = demo_dir / "corpus"
    base_url = api_url.rstrip("/")

    # Check health
    try:
        r = httpx.get(f"{base_url}/api/health", timeout=5.0)
        r.raise_for_status()
    except Exception as exc:
        raise ConnectionError(
            f"Unable to reach STING backend at '{base_url}/api/health'. "
            "Ensure the server is running (e.g. uvicorn reliability_lab.api.main:app) "
            "or run with --local to use direct database services."
        ) from exc

    # Fetch existing documents
    existing_docs_by_name: dict[str, dict[str, Any]] = {}
    if reuse_existing:
        try:
            r = httpx.get(f"{base_url}/api/documents?limit=200", timeout=10.0)
            if r.status_code == 200:
                for d in r.json().get("documents", []):
                    existing_docs_by_name[d["filename"]] = d
        except Exception as exc:
            logger.warning("Could not query existing documents: %s", exc)

    doc_ids: dict[str, str] = {}
    chunk_counts: dict[str, int] = {}
    doc_chunks_by_filename: dict[str, list[Any]] = {}

    for doc_spec in manifest["documents"]:
        fname = doc_spec["filename"]
        fpath = corpus_dir / fname
        fhash = compute_file_hash(fpath)

        # Check if we can safely reuse existing READY document
        reused = False
        if reuse_existing and fname in existing_docs_by_name:
            cand = existing_docs_by_name[fname]
            if cand.get("content_hash") == fhash and str(cand.get("status")).upper() == "READY":
                doc_id = cand["doc_id"]
                logger.info("Reusing existing ingested document '%s' (%s)", fname, doc_id)
                doc_ids[fname] = doc_id
                reused = True

        if not reused:
            logger.info("Uploading '%s' to %s/api/documents/upload ...", fname, base_url)
            with open(fpath, "rb") as f:
                content_type = (
                    "text/markdown" if fname.endswith(".md") else "application/octet-stream"
                )
                files = {"file": (fname, f, content_type)}
                r = httpx.post(f"{base_url}/api/documents/upload", files=files, timeout=30.0)
                if r.status_code not in (200, 201):
                    raise RuntimeError(f"Upload failed for '{fname}': {r.status_code} {r.text}")
                doc_id = r.json()["doc_id"]
                doc_ids[fname] = doc_id

            # Poll for readiness
            start_t = time.time()
            ready = False
            while time.time() - start_t < timeout:
                r = httpx.get(f"{base_url}/api/documents/{doc_id}", timeout=5.0)
                if r.status_code == 200:
                    data = r.json()
                    st = str(data.get("status", "")).upper()
                    if st == "READY":
                        ready = True
                        break
                    if st == "FAILED":
                        err_msg = data.get("error_message")
                        raise RuntimeError(
                            f"Ingestion failed for document '{fname}' ({doc_id}): {err_msg}"
                        )
                time.sleep(0.5)

            if not ready:
                raise TimeoutError(
                    f"Timed out after {timeout}s waiting for document '{fname}' to index."
                )

        # Fetch chunks for anchor resolution
        r = httpx.get(f"{base_url}/api/documents/{doc_ids[fname]}/chunks?limit=500", timeout=10.0)
        if r.status_code != 200:
            raise RuntimeError(
                f"Failed fetching chunks for document '{fname}' ({doc_ids[fname]}): {r.text}"
            )
        chunks_data = r.json().get("chunks", [])

        # Convert to lightweight objects
        @dataclass
        class SimpleChunk:
            chunk_id: str
            content: str

        chunk_objs = [
            SimpleChunk(chunk_id=c["chunk_id"], content=c["content"]) for c in chunks_data
        ]
        doc_chunks_by_filename[fname] = chunk_objs
        chunk_counts[fname] = len(chunk_objs)
        logger.info("Document '%s' ready with %d chunks.", fname, len(chunk_objs))

    # Resolve anchors to runtime chunk IDs
    logger.info("Resolving %d benchmark query anchors against runtime chunks...", len(questions))
    resolved_queries = resolve_anchors_to_chunks(questions, doc_chunks_by_filename)

    # Check existing datasets
    dataset_name = "Northstar Golden Benchmark"
    dataset_id = ""
    target_scope = sorted(doc_ids.values())

    if reuse_existing:
        try:
            r = httpx.get(f"{base_url}/api/datasets?limit=100", timeout=10.0)
            if r.status_code == 200:
                for ds in r.json().get("datasets", []):
                    if (
                        ds.get("name") == dataset_name
                        and sorted(ds.get("document_scope", [])) == target_scope
                    ):
                        dataset_id = ds["dataset_id"]
                        logger.info(
                            "Reusing existing benchmark dataset '%s' (%s)", dataset_name, dataset_id
                        )
                        break
        except Exception as exc:
            logger.warning("Could not query existing datasets: %s", exc)

    if not dataset_id:
        logger.info(
            "Creating benchmark dataset '%s' with %d queries...",
            dataset_name,
            len(resolved_queries),
        )
        payload = {
            "name": dataset_name,
            "description": (
                "10 manually verified benchmark queries for the Northstar demonstration corpus."
            ),
            "document_scope": target_scope,
            "is_synthetic": True,
            "metadata": {"corpus_id": manifest.get("corpus_id", "demo-northstar-v2")},
            "queries": resolved_queries,
        }
        r = httpx.post(f"{base_url}/api/datasets", json=payload, timeout=20.0)
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Failed creating benchmark dataset: {r.status_code} {r.text}")
        dataset_id = r.json()["dataset_id"]
        logger.info("Successfully created dataset '%s' (%s).", dataset_name, dataset_id)

    abstentions = sum(
        1
        for q in resolved_queries
        if (
            q.get("metadata", {}).get("category") == "safe_abstention"
            or not q.get("relevant_chunk_ids")
        )
    )
    anchored = len(resolved_queries) - abstentions
    resolved_cnt = sum(1 for q in resolved_queries if len(q.get("relevant_chunk_ids", [])) == 1)

    return DemoPreparationResult(
        documents=doc_ids,
        chunk_counts=chunk_counts,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        query_count=len(resolved_queries),
        resolved_queries=resolved_queries,
        mode="api",
        db_path=None,
        evidence_anchored_queries=anchored,
        resolved_anchors=resolved_cnt,
        intentional_abstentions=abstentions,
    )


def prepare_demo_local(
    db_path: Path,
    storage_dir: Path,
    index_dir: Path,
    demo_dir: Path,
    reuse_existing: bool = True,
) -> DemoPreparationResult:
    """Execute demo preparation using direct Python service dependencies."""
    from reliability_lab.benchmark.repository import BenchmarkDatasetRepository
    from reliability_lab.benchmark.service import BenchmarkDatasetService
    from reliability_lab.core.models import Query
    from reliability_lab.ingestion.repository import DocumentRepository
    from reliability_lab.ingestion.service import DocumentIngestionService
    from reliability_lab.storage.database import Database

    manifest, questions = validate_demo_pack(demo_dir)
    corpus_dir = demo_dir / "corpus"

    db = Database(db_path)
    doc_repo = DocumentRepository(db)
    dataset_repo = BenchmarkDatasetRepository(db)
    ingestion_service = DocumentIngestionService(
        repository=doc_repo,
        storage_dir=storage_dir,
        index_dir=index_dir,
    )
    dataset_service = BenchmarkDatasetService(
        dataset_repository=dataset_repo,
        document_repository=doc_repo,
    )

    doc_ids: dict[str, str] = {}
    chunk_counts: dict[str, int] = {}
    doc_chunks_by_filename: dict[str, list[Any]] = {}

    for doc_spec in manifest["documents"]:
        fname = doc_spec["filename"]
        fpath = corpus_dir / fname
        fhash = compute_file_hash(fpath)

        reused = False
        if reuse_existing:
            for d in doc_repo.list_documents(limit=100):
                if d.filename == fname and d.content_hash == fhash:
                    st = getattr(d, "status", None)
                    st_val = getattr(st, "value", None)
                    st_str = str(st_val if st_val is not None else (st or ""))
                    if st_str.upper() == "READY":
                        doc_ids[fname] = d.doc_id
                        reused = True
                        logger.info("Reusing existing local document '%s' (%s)", fname, d.doc_id)
                        break

        if not reused:
            logger.info("Ingesting local document '%s' ...", fname)
            with open(fpath, "rb") as f:
                rec = ingestion_service.save_upload_stream(filename=fname, file_stream=f)
            rec = ingestion_service.process_document(rec.doc_id)
            doc_ids[fname] = rec.doc_id

        chunks = doc_repo.get_chunks(doc_ids[fname])
        doc_chunks_by_filename[fname] = chunks
        chunk_counts[fname] = len(chunks)
        logger.info("Local document '%s' indexed with %d chunks.", fname, len(chunks))

    # Resolve anchors
    resolved_queries = resolve_anchors_to_chunks(questions, doc_chunks_by_filename)

    dataset_name = "Northstar Golden Benchmark"
    dataset_id = ""
    target_scope = sorted(doc_ids.values())

    if reuse_existing:
        for ds in dataset_service.list_datasets(limit=100):
            if ds.name == dataset_name and sorted(ds.document_scope) == target_scope:
                dataset_id = ds.dataset_id
                logger.info(
                    "Reusing existing local benchmark dataset '%s' (%s)", dataset_name, dataset_id
                )
                break

    if not dataset_id:
        query_objs = [
            Query(
                query_id=q["query_id"],
                text=q["text"],
                relevant_chunk_ids=q["relevant_chunk_ids"],
                relevance_grades=q["relevance_grades"],
                ground_truth_answer=q.get("ground_truth_answer"),
                required_citations=q.get("required_citations", []),
                metadata=q.get("metadata", {}),
            )
            for q in resolved_queries
        ]
        created = dataset_service.create_dataset(
            name=dataset_name,
            description=(
                "10 manually verified benchmark queries for the Northstar demonstration corpus."
            ),
            document_scope=target_scope,
            is_synthetic=True,
            metadata={"corpus_id": manifest.get("corpus_id", "demo-northstar-v2")},
            queries=query_objs,
        )
        dataset_id = created.dataset_id
        logger.info("Created local benchmark dataset '%s' (%s).", dataset_name, dataset_id)

    abstentions = sum(
        1
        for q in resolved_queries
        if (
            q.get("metadata", {}).get("category") == "safe_abstention"
            or not q.get("relevant_chunk_ids")
        )
    )
    anchored = len(resolved_queries) - abstentions
    resolved_cnt = sum(1 for q in resolved_queries if len(q.get("relevant_chunk_ids", [])) == 1)

    return DemoPreparationResult(
        documents=doc_ids,
        chunk_counts=chunk_counts,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        query_count=len(resolved_queries),
        resolved_queries=resolved_queries,
        mode="local",
        db_path=str(db_path),
        evidence_anchored_queries=anchored,
        resolved_anchors=resolved_cnt,
        intentional_abstentions=abstentions,
    )


def print_receipt(result: DemoPreparationResult) -> None:
    """Print formatted execution receipt for the user."""
    print("\n" + "=" * 70)
    print("  STING DEMO PACK PREPARATION RECEIPT")
    print("=" * 70)
    print(f"Execution Mode:             {result.mode.upper()}")
    if result.db_path:
        print(f"Database Path:              {result.db_path}")
    print(f"Benchmark Dataset:          {result.dataset_name}")
    print(f"Dataset Identifier:         {result.dataset_id}")
    print(f"Total Golden Queries:       {result.query_count}")
    print(
        f"  * Evidence-Anchored:      {result.evidence_anchored_queries} "
        f"(resolved to unique chunks: {result.resolved_anchors})"
    )
    print(
        f"  * Safe Abstention:        {result.intentional_abstentions} "
        f"(intentional zero-evidence ground truth)"
    )
    print("-" * 70)
    print("Ingested Documents:")
    total_chunks = sum(result.chunk_counts.values())
    for fname, did in result.documents.items():
        cnt = result.chunk_counts.get(fname, 0)
        print(f"  * {fname:25s} -> Doc ID: {did} ({cnt} chunks)")
    print(f"Total Ingested Chunks:      {total_chunks}")
    print("-" * 70)
    print("Next Steps for STING Demonstration:")
    print("  1. Evidence Queries:      POST /api/query or /api/query/stream")
    print(f"     Scoped Doc IDs:        {', '.join(result.documents.values())}")
    print("  2. Cross-Document Invest: POST /api/investigations")
    print(
        f"  3. Evaluation Studio:     POST /api/evaluation/run with dataset '{result.dataset_id}'"
    )
    print("-" * 70)
    arch_id = result.documents.get("architecture.md", "")
    bench_id = result.documents.get("benchmark_report.md", "")
    inc_id = result.documents.get("incident_report.md", "")
    print("Quick Shell Variables (Bash / Zsh):")
    print(f'  export DOC_ARCH="{arch_id}"')
    print(f'  export DOC_BENCH="{bench_id}"')
    print(f'  export DOC_INCIDENT="{inc_id}"')
    print(f'  export DEMO_DATASET_ID="{result.dataset_id}"')
    print("\nQuick PowerShell Variables:")
    print(f'  $env:DOC_ARCH = "{arch_id}"')
    print(f'  $env:DOC_BENCH = "{bench_id}"')
    print(f'  $env:DOC_INCIDENT = "{inc_id}"')
    print(f'  $env:DEMO_DATASET_ID = "{result.dataset_id}"')
    print("=" * 70 + "\n")


def main() -> int:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        description="Prepare the STING Northstar demo documents and golden benchmark dataset."
    )
    parser.add_argument(
        "--api-url",
        default=os.getenv("STING_API_URL", "http://127.0.0.1:8000"),
        help="FastAPI server URL (default: http://127.0.0.1:8000).",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help=(
            "Force direct execution using local SQLite and storage directories without HTTP server."
        ),
    )
    parser.add_argument(
        "--demo-dir",
        type=Path,
        default=Path("data/demo"),
        help="Path to demo directory containing manifest.json (default: data/demo).",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path(os.getenv("RELIABILITY_LAB_DB", ".reliability-lab/experiments.sqlite3")),
        help="Database file path for local mode (default: .reliability-lab/experiments.sqlite3).",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=Path(".reliability-lab/documents"),
        help="Document storage directory for local mode.",
    )
    parser.add_argument(
        "--index-dir",
        type=Path,
        default=Path(".reliability-lab/indexes"),
        help="Index directory for local mode.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Timeout in seconds for document indexing in API mode (default: 60s).",
    )
    parser.add_argument(
        "--no-reuse",
        dest="reuse_existing",
        action="store_false",
        help="Force new uploads and dataset creation instead of reusing existing records.",
    )
    parser.add_argument(
        "--receipt-json",
        type=Path,
        default=None,
        help=(
            "Optional path to write a machine-readable JSON receipt with document and dataset IDs."
        ),
    )

    args = parser.parse_args()

    try:
        if args.local:
            logger.info("Executing in explicit LOCAL mode using database '%s'.", args.db_path)
            result = prepare_demo_local(
                db_path=args.db_path,
                storage_dir=args.storage_dir,
                index_dir=args.index_dir,
                demo_dir=args.demo_dir,
                reuse_existing=args.reuse_existing,
            )
        else:
            # API mode: require running server, fail-closed without silent local fallback
            import httpx

            target_url = args.api_url.rstrip("/")
            try:
                r = httpx.get(f"{target_url}/api/health", timeout=3.0)
                if r.status_code != 200:
                    raise RuntimeError(
                        f"STING API server at {target_url} returned unhealthy status "
                        f"{r.status_code}."
                    )
            except Exception as exc:
                raise ConnectionError(
                    f"Cannot connect to STING API server at '{target_url}': {exc}\n"
                    "To run in API mode, first start the STING server:\n"
                    "  uvicorn reliability_lab.api.main:app --port 8000\n"
                    "Or pass '--local' to execute directly against local SQLite and storage "
                    "without a server."
                ) from exc

            logger.info("Executing in API mode against server at '%s'.", target_url)
            result = prepare_demo_api(
                api_url=target_url,
                demo_dir=args.demo_dir,
                timeout=args.timeout,
                reuse_existing=args.reuse_existing,
            )

        print_receipt(result)

        if args.receipt_json:
            receipt_payload = {
                "mode": result.mode,
                "db_path": result.db_path,
                "dataset_id": result.dataset_id,
                "dataset_name": result.dataset_name,
                "document_ids": result.documents,
                "chunk_counts": result.chunk_counts,
                "total_chunks": sum(result.chunk_counts.values()),
                "total_queries": result.query_count,
                "evidence_anchored_queries": result.evidence_anchored_queries,
                "resolved_anchors": result.resolved_anchors,
                "intentional_abstentions": result.intentional_abstentions,
            }
            args.receipt_json.parent.mkdir(parents=True, exist_ok=True)
            args.receipt_json.write_text(json.dumps(receipt_payload, indent=2), encoding="utf-8")
            logger.info("Machine-readable receipt written to '%s'.", args.receipt_json)

        return 0
    except Exception as exc:
        logger.error("Demo preparation failed: %s", exc, exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
