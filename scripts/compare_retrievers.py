#!/usr/bin/env python3
"""Empirical retrieval benchmark comparison utility.

Evaluates BM25, FAISS dense vector retrieval, and Hybrid Reciprocal Rank Fusion (RRF)
across the canonical benchmark corpus and queries under identical evaluation protocols.
Emits human-readable summaries and machine-readable JSON reports.
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from reliability_lab.core.models import Chunk, Query
from reliability_lab.evaluation.benchmark_runner import (
    RetrievalBenchmarkRunner,
    load_benchmark_dataset,
    load_corpus,
)
from reliability_lab.indexing.bm25_index import BM25Retriever

logger = logging.getLogger("compare_retrievers")


def get_environment_metadata() -> dict[str, Any]:
    """Capture runtime environment and installed package versions."""
    metadata: dict[str, Any] = {
        "os": f"{platform.system()} {platform.release()}",
        "architecture": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python_version": platform.python_version(),
    }

    # Record package versions safely
    for pkg in ["numpy", "faiss", "torch", "sentence_transformers", "transformers"]:
        try:
            mod = __import__(pkg)
            metadata[f"{pkg}_version"] = getattr(mod, "__version__", "installed")
        except ImportError:
            metadata[f"{pkg}_version"] = None

    return metadata


def evaluate_retriever_runner(
    retriever: Any,
    corpus: list[Chunk],
    queries: list[Query],
    top_k: int = 5,
    experiment_name: str = "eval",
    repetitions: int = 5,
) -> dict[str, Any]:
    """Evaluate retrieval metrics using RetrievalBenchmarkRunner and warm latency repeats."""
    runner = RetrievalBenchmarkRunner(
        retriever=retriever,
        top_k=top_k,
        experiment_name=experiment_name,
    )
    report = runner.run(corpus=corpus, benchmark=queries, auto_index=False)

    # Timing benchmark with warm-up pass
    for query in queries[:3]:
        _ = retriever.retrieve(query.text, top_k=top_k)

    latencies_ms: list[float] = []
    for _ in range(max(1, repetitions)):
        for query in queries:
            t0 = time.perf_counter()
            _ = retriever.retrieve(query.text, top_k=top_k)
            t1 = time.perf_counter()
            latencies_ms.append((t1 - t0) * 1000.0)

    lat_arr = np.array(latencies_ms, dtype=np.float64)

    # Format metrics and merge timing distributions
    metrics = dict(report.metrics)
    metrics["mean_latency_ms"] = float(np.mean(lat_arr))
    metrics["latency_p50_ms"] = float(np.percentile(lat_arr, 50))
    metrics["latency_p90_ms"] = float(np.percentile(lat_arr, 90))
    metrics["latency_p95_ms"] = float(np.percentile(lat_arr, 95))
    metrics["latency_p99_ms"] = float(np.percentile(lat_arr, 99))
    metrics["min_latency_ms"] = float(np.min(lat_arr))
    metrics["max_latency_ms"] = float(np.max(lat_arr))

    # Query details
    per_query_rankings: list[dict[str, Any]] = []
    for query in queries:
        results = retriever.retrieve(query.text, top_k=top_k)
        retrieved_ids = [r.chunk.chunk_id for r in results]
        per_query_rankings.append(
            {
                "query_id": query.query_id,
                "query_text": query.text,
                "relevant_chunk_ids": query.relevant_chunk_ids,
                "retrieved_chunk_ids": retrieved_ids,
            }
        )

    return {
        "metrics": metrics,
        "query_count": len(queries),
        "repetitions": repetitions,
        "per_query": per_query_rankings,
    }


def run_comparison(
    corpus_path: str | Path,
    dataset_path: str | Path,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
    skip_dense: bool = False,
    repetitions: int = 5,
) -> dict[str, Any]:
    """Execute end-to-end comparative benchmark across available retrievers."""
    corpus: list[Chunk] = load_corpus(corpus_path)
    queries: list[Query] = load_benchmark_dataset(dataset_path, corpus_chunks=corpus)

    env_meta = get_environment_metadata()
    results: dict[str, Any] = {
        "metadata": {
            "environment": env_meta,
            "corpus_path": str(corpus_path),
            "corpus_size": len(corpus),
            "dataset_path": str(dataset_path),
            "query_count": len(queries),
            "top_k": top_k,
            "repetitions": repetitions,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "retrievers": {},
    }

    # -------------------------------------------------------------------------
    # 1. Okapi BM25
    # -------------------------------------------------------------------------
    bm25 = BM25Retriever(k1=1.5, b=0.75)
    t0 = time.perf_counter()
    bm25.index(corpus)
    bm25_index_time_ms = (time.perf_counter() - t0) * 1000.0

    bm25_eval = evaluate_retriever_runner(
        retriever=bm25,
        corpus=corpus,
        queries=queries,
        top_k=top_k,
        experiment_name="comparative-bm25",
        repetitions=repetitions,
    )
    bm25_eval["indexing_time_ms"] = bm25_index_time_ms
    bm25_eval["config"] = {"algorithm": "bm25", "k1": 1.5, "b": 0.75}
    results["retrievers"]["bm25"] = bm25_eval

    # -------------------------------------------------------------------------
    # 2. Dense FAISS Vector Retriever
    # -------------------------------------------------------------------------
    faiss_retriever = None
    has_dense_deps = (
        env_meta.get("sentence_transformers_version") is not None
        and env_meta.get("faiss_version") is not None
    )

    if not skip_dense and has_dense_deps:
        try:
            from reliability_lab.indexing.embeddings import SentenceTransformerEmbedder
            from reliability_lab.indexing.faiss_index import FaissVectorRetriever

            embedder = SentenceTransformerEmbedder(
                model_name_or_path=model_name,
                expected_dim=384,
                normalize_embeddings=True,
            )
            faiss_retriever = FaissVectorRetriever(
                embedder=embedder,
                index_type="flat",
                metric="ip",
                normalize=True,
            )

            t0 = time.perf_counter()
            faiss_retriever.index(corpus)
            faiss_index_time_ms = (time.perf_counter() - t0) * 1000.0

            faiss_eval = evaluate_retriever_runner(
                retriever=faiss_retriever,
                corpus=corpus,
                queries=queries,
                top_k=top_k,
                experiment_name="comparative-dense-faiss",
                repetitions=repetitions,
            )
            faiss_eval["indexing_time_ms"] = faiss_index_time_ms
            faiss_eval["config"] = {
                "algorithm": "dense_faiss",
                "model_name": model_name,
                "dimension": 384,
                "index_type": "flat",
                "metric": "inner_product",
                "normalized": True,
            }
            results["retrievers"]["dense_faiss"] = faiss_eval
        except Exception as e:
            logger.warning("Dense FAISS retrieval failed: %s", e)
            results["retrievers"]["dense_faiss"] = {
                "status": "SKIPPED",
                "reason": str(e),
            }
    else:
        results["retrievers"]["dense_faiss"] = {
            "status": "SKIPPED",
            "reason": "Dense retrieval skipped (dependencies missing or --skip-dense requested)",
        }

    # -------------------------------------------------------------------------
    # 3. Hybrid Reciprocal Rank Fusion (RRF)
    # -------------------------------------------------------------------------
    if faiss_retriever is not None:
        try:
            from reliability_lab.indexing.hybrid import HybridRetriever

            hybrid = HybridRetriever(
                sparse_retriever=bm25,
                dense_retriever=faiss_retriever,
                rrf_k=60,
                name="hybrid_rrf",
            )
            t0 = time.perf_counter()
            hybrid.index(corpus)
            hybrid_index_time_ms = (time.perf_counter() - t0) * 1000.0

            hybrid_eval = evaluate_retriever_runner(
                retriever=hybrid,
                corpus=corpus,
                queries=queries,
                top_k=top_k,
                experiment_name="comparative-hybrid-rrf",
                repetitions=repetitions,
            )
            hybrid_eval["indexing_time_ms"] = hybrid_index_time_ms
            hybrid_eval["config"] = {
                "algorithm": "hybrid_rrf",
                "sparse_component": "bm25",
                "dense_component": "faiss_flat_ip",
                "rrf_k": 60,
            }
            results["retrievers"]["hybrid_rrf"] = hybrid_eval
        except Exception as e:
            logger.warning("Hybrid RRF retrieval failed: %s", e)
            results["retrievers"]["hybrid_rrf"] = {
                "status": "SKIPPED",
                "reason": str(e),
            }
    else:
        results["retrievers"]["hybrid_rrf"] = {
            "status": "SKIPPED",
            "reason": "Hybrid retrieval skipped because dense retriever is unavailable",
        }

    return results


def print_summary_table(results: dict[str, Any]) -> None:
    """Print an aligned ASCII summary table comparing retrieval metrics and latency."""
    meta = results["metadata"]
    top_k = meta["top_k"]

    print("=" * 86)
    print("AI Reliability Lab — Comparative Retrieval Benchmark")
    print("=" * 86)
    print(f"Corpus Chunks:     {meta['corpus_size']}")
    print(f"Benchmark Queries: {meta['query_count']}")
    print(f"Top-K Cutoff:      {top_k}")
    print(f"Latency Repeats:   {meta['repetitions']} (per query, with warm-up)")
    print("-" * 86)

    headers = ["Metric / Dimension", "Okapi BM25", "Dense FAISS", "Hybrid RRF"]
    print(f"{headers[0]:<26} | {headers[1]:<16} | {headers[2]:<16} | {headers[3]:<16}")
    print("-" * 86)

    retrievers = results["retrievers"]
    bm25 = retrievers.get("bm25", {}).get("metrics", {})
    dense = retrievers.get("dense_faiss", {}).get("metrics", {})
    hybrid = retrievers.get("hybrid_rrf", {}).get("metrics", {})

    def _fmt(val_dict: dict[str, Any], key: str, is_ms: bool = False) -> str:
        if not val_dict or key not in val_dict:
            return "N/A"
        val = val_dict[key]
        if is_ms:
            return f"{val:.4f} ms"
        return f"{val:.4f}"

    metrics_to_show = [
        (f"Recall@{top_k}", f"recall@{top_k}", False),
        (f"Precision@{top_k}", f"precision@{top_k}", False),
        (f"Hit Rate@{top_k}", f"hit_rate@{top_k}", False),
        (f"MRR@{top_k}", f"mrr@{top_k}", False),
        (f"NDCG@{top_k}", f"ndcg@{top_k}", False),
        ("Mean Latency", "mean_latency_ms", True),
        ("P50 Latency", "latency_p50_ms", True),
        ("P95 Latency", "latency_p95_ms", True),
        ("P99 Latency", "latency_p99_ms", True),
    ]

    for label, key, is_ms in metrics_to_show:
        b_str = _fmt(bm25, key, is_ms)
        d_str = _fmt(dense, key, is_ms)
        h_str = _fmt(hybrid, key, is_ms)
        print(f"{label:<26} | {b_str:<16} | {d_str:<16} | {h_str:<16}")

    print("-" * 86)
    # Index build times
    b_idx = f"{retrievers.get('bm25', {}).get('indexing_time_ms', 0):.2f} ms"
    d_idx = (
        f"{retrievers.get('dense_faiss', {}).get('indexing_time_ms', 0):.2f} ms"
        if "indexing_time_ms" in retrievers.get("dense_faiss", {})
        else "N/A"
    )
    h_idx = (
        f"{retrievers.get('hybrid_rrf', {}).get('indexing_time_ms', 0):.2f} ms"
        if "indexing_time_ms" in retrievers.get("hybrid_rrf", {})
        else "N/A"
    )
    print(f"{'Index Build Time':<26} | {b_idx:<16} | {d_idx:<16} | {h_idx:<16}")
    print("=" * 86)


def main() -> int:
    """CLI entrypoint for compare_retrievers script."""
    parser = argparse.ArgumentParser(
        description="Benchmark and compare BM25, FAISS, and Hybrid RRF retrievers."
    )
    parser.add_argument(
        "--corpus",
        type=str,
        default="data/benchmarks/sample_corpus.jsonl",
        help="Path to JSONL corpus file.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/benchmarks/sample_benchmark.jsonl",
        help="Path to JSONL queries file.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top-K ranking cutoff (default: 5).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Sentence-Transformers model name for dense vector retrieval.",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Repetitions per query for timing distribution (default: 5).",
    )
    parser.add_argument(
        "--skip-dense",
        action="store_true",
        help="Skip dense FAISS and Hybrid retrievers (lexical BM25 only).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to output JSON file (or '-' for stdout).",
    )

    args = parser.parse_args()

    results = run_comparison(
        corpus_path=args.corpus,
        dataset_path=args.dataset,
        top_k=args.top_k,
        model_name=args.model,
        skip_dense=args.skip_dense,
        repetitions=args.repetitions,
    )

    print_summary_table(results)

    if args.output:
        json_str = json.dumps(results, indent=2)
        if args.output == "-":
            print(json_str)
        else:
            out_path = Path(args.output)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json_str, encoding="utf-8")
            print(f"Comparative report saved to: {out_path.resolve()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
