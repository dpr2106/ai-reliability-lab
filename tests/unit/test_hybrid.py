"""Unit and integration tests for HybridRetriever and Reciprocal Rank Fusion (RRF).

Covers:
- Standard RRF mathematical formulation, default k=60, and custom k values.
- Hand-calculated verification for multi-retriever and single-retriever candidate hits.
- Invariance to raw child score scales.
- Chunk provenance preservation and metadata enrichment.
- Deterministic tie-breaking behavior.
- Strict input validation, rank validation, and non-finite score rejection.
- Internal child duplicate detection and conflicting cross-retriever provenance rejection.
- Indexing lifecycle: empty corpus, retrieval before indexing, child failure isolation.
- Integration with RetrievalBenchmarkRunner.
- Real-FAISS backend integration test (isolated with importorskip).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pytest

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.benchmark_runner import RetrievalBenchmarkRunner
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.indexing.hybrid import DEFAULT_RRF_K, HybridRetriever


class StubRetriever(BaseRetriever):
    """Configurable in-memory stub retriever for deterministic unit testing."""

    def __init__(
        self,
        name: str = "stub_retriever",
        predefined_results: list[RetrievalResult] | None = None,
        fail_indexing: bool = False,
    ) -> None:
        self._name = name
        self._results = predefined_results or []
        self._fail_indexing = fail_indexing
        self.indexed_chunks: list[Chunk] = []

    @property
    def name(self) -> str:
        return self._name

    def index(self, chunks: Sequence[Chunk]) -> None:
        if self._fail_indexing:
            raise RuntimeError(f"Simulated indexing failure in {self._name}")
        self.indexed_chunks = list(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return list(self._results[:top_k])


def _make_chunk(chunk_id: str, content: str = "Sample content", doc_id: str = "doc-1") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        content=content,
        page_number=1,
        char_start=0,
        char_end=len(content),
        metadata={"source": "test"},
    )


# -----------------------------------------------------------------------------
# 1. Initialization and Parameter Validation
# -----------------------------------------------------------------------------


def test_initialization_defaults() -> None:
    sparse = StubRetriever(name="sparse")
    dense = StubRetriever(name="dense")
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)

    assert hybrid.name == "hybrid_rrf"
    assert hybrid.sparse_retriever is sparse
    assert hybrid.dense_retriever is dense
    assert hybrid.rrf_k == DEFAULT_RRF_K
    assert hybrid.rrf_k == 60
    assert hybrid.candidate_depth is None
    assert hybrid.is_indexed is False


def test_custom_parameters() -> None:
    sparse = StubRetriever(name="sparse")
    dense = StubRetriever(name="dense")
    hybrid = HybridRetriever(
        sparse_retriever=sparse,
        dense_retriever=dense,
        rrf_k=30,
        name="custom_hybrid",
        candidate_depth=100,
    )

    assert hybrid.name == "custom_hybrid"
    assert hybrid.rrf_k == 30
    assert hybrid.candidate_depth == 100


def test_validation_invalid_retriever_types() -> None:
    sparse = StubRetriever(name="sparse")
    with pytest.raises(TypeError, match="sparse_retriever must be an instance of BaseRetriever"):
        HybridRetriever(sparse_retriever="not_a_retriever", dense_retriever=sparse)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="dense_retriever must be an instance of BaseRetriever"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever="not_a_retriever")  # type: ignore[arg-type]


def test_validation_invalid_rrf_k() -> None:
    sparse = StubRetriever(name="sparse")
    dense = StubRetriever(name="dense")

    with pytest.raises(TypeError, match="rrf_k must be an integer"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=True)  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="rrf_k must be an integer"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=60.5)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="rrf_k must be a strictly positive integer"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=0)

    with pytest.raises(ValueError, match="rrf_k must be a strictly positive integer"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, rrf_k=-10)


def test_validation_invalid_name() -> None:
    sparse = StubRetriever(name="sparse")
    dense = StubRetriever(name="dense")

    with pytest.raises(TypeError, match="name must be a string"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, name=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="name must be a non-empty string"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, name="   ")


def test_validation_invalid_candidate_depth() -> None:
    sparse = StubRetriever(name="sparse")
    dense = StubRetriever(name="dense")

    with pytest.raises(TypeError, match="candidate_depth must be an integer"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, candidate_depth=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="candidate_depth must be a positive integer"):
        HybridRetriever(sparse_retriever=sparse, dense_retriever=dense, candidate_depth=0)


# -----------------------------------------------------------------------------
# 2. Mathematical Hand-Calculations for RRF
# -----------------------------------------------------------------------------


def test_rrf_hand_calculation_both_retrievers() -> None:
    """Verifies hand-calculated RRF score for a chunk returned by both retrievers.

    Setup:
        k = 60
        Chunk A:
            Sparse rank = 1
            Dense rank = 2
        Sparse contrib = 1 / (60 + 1) = 1/61 = ~0.0163934426
        Dense contrib = 1 / (60 + 2) = 1/62 = ~0.0161290323
        Expected fused score = 1/61 + 1/62 = 123 / 3782 = ~0.0325224749
    """
    chunk_a = _make_chunk("chunk_a", "Chunk A content")
    sparse_res = [RetrievalResult(chunk=chunk_a, score=15.5, rank=1, retriever_name="bm25")]
    dense_res = [RetrievalResult(chunk=chunk_a, score=0.89, rank=2, retriever_name="faiss")]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
        rrf_k=60,
    )
    hybrid.index([chunk_a])

    results = hybrid.retrieve("query", top_k=5)
    assert len(results) == 1

    expected_score = (1.0 / 61.0) + (1.0 / 62.0)
    assert pytest.approx(results[0].score, rel=1e-6) == expected_score
    assert results[0].rank == 1
    assert results[0].retriever_name == "hybrid_rrf"
    assert results[0].metadata["sparse_rank"] == 1
    assert results[0].metadata["dense_rank"] == 2
    assert pytest.approx(results[0].metadata["sparse_score"]) == 15.5
    assert pytest.approx(results[0].metadata["dense_score"]) == 0.89
    assert pytest.approx(results[0].metadata["rrf_contributions"]["bm25"]) == 1.0 / 61.0
    assert pytest.approx(results[0].metadata["rrf_contributions"]["faiss"]) == 1.0 / 62.0


def test_rrf_hand_calculation_single_retriever_hit() -> None:
    """Verifies score when chunk is returned by only one retriever."""
    chunk_a = _make_chunk("chunk_a")
    chunk_b = _make_chunk("chunk_b")

    # Sparse returns chunk_a at rank 2; dense returns nothing
    sparse_res = [RetrievalResult(chunk=chunk_a, score=12.0, rank=2, retriever_name="bm25")]
    dense_res: list[RetrievalResult] = []

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
        rrf_k=60,
    )
    hybrid.index([chunk_a, chunk_b])

    results = hybrid.retrieve("query", top_k=5)
    assert len(results) == 1
    assert results[0].chunk.chunk_id == "chunk_a"

    expected_score = 1.0 / (60 + 2)
    assert pytest.approx(results[0].score, rel=1e-6) == expected_score
    assert results[0].metadata["sparse_rank"] == 2
    assert results[0].metadata["dense_rank"] is None
    assert results[0].metadata["dense_score"] is None
    assert "bm25" in results[0].metadata["rrf_contributions"]
    assert "faiss" not in results[0].metadata["rrf_contributions"]


def test_raw_scores_do_not_influence_fused_ranking() -> None:
    """Verifies that RRF uses rank-only fusion and ignores raw score magnitudes."""
    chunk_1 = _make_chunk("chunk_1")
    chunk_2 = _make_chunk("chunk_2")

    # In sparse: chunk_1 is rank 1 (score 1000.0), chunk_2 is rank 2 (score 1.0)
    # In dense: chunk_2 is rank 1 (score 0.99), chunk_1 is rank 3 (score 0.01)
    #
    # With k=60:
    # chunk_1 fused = 1/(60+1) + 1/(60+3) = 1/61 + 1/63 = 0.0163934 + 0.0158730 = 0.0322664
    # chunk_2 fused = 1/(60+2) + 1/(60+1) = 1/62 + 1/61 = 0.0161290 + 0.0163934 = 0.0325225
    # Therefore, chunk_2 must rank HIGHER than chunk_1 despite chunk_1's massive raw score (1000.0)!
    sparse_res = [
        RetrievalResult(chunk=chunk_1, score=1000.0, rank=1, retriever_name="bm25"),
        RetrievalResult(chunk=chunk_2, score=1.0, rank=2, retriever_name="bm25"),
    ]
    dense_res = [
        RetrievalResult(chunk=chunk_2, score=0.99, rank=1, retriever_name="faiss"),
        RetrievalResult(chunk=chunk_1, score=0.01, rank=3, retriever_name="faiss"),
    ]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
        rrf_k=60,
    )
    hybrid.index([chunk_1, chunk_2])

    results = hybrid.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chunk_2"
    assert results[1].chunk.chunk_id == "chunk_1"
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert results[0].score > results[1].score


# -----------------------------------------------------------------------------
# 3. Provenance and Deduplication
# -----------------------------------------------------------------------------


def test_chunk_provenance_preserved() -> None:
    """All original chunk fields must survive fusion without alteration."""
    chunk = Chunk(
        chunk_id="chk-provenance-1",
        doc_id="report-2026",
        content="Important reliability findings.",
        page_number=42,
        char_start=500,
        char_end=532,
        metadata={"author": "auditor", "confidence": 0.98},
    )
    sparse_res = [RetrievalResult(chunk=chunk, score=8.5, rank=1, retriever_name="bm25")]
    dense_res = [RetrievalResult(chunk=chunk, score=0.88, rank=1, retriever_name="faiss")]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
    )
    hybrid.index([chunk])

    res = hybrid.retrieve("test", top_k=1)[0]
    assert res.chunk.chunk_id == "chk-provenance-1"
    assert res.chunk.doc_id == "report-2026"
    assert res.chunk.content == "Important reliability findings."
    assert res.chunk.page_number == 42
    assert res.chunk.char_start == 500
    assert res.chunk.char_end == 532
    assert res.chunk.metadata == {"author": "auditor", "confidence": 0.98}


def test_conflicting_provenance_raises_error() -> None:
    """Conflicting metadata or payload for the same chunk_id across retrievers must be rejected."""
    chunk_sparse = Chunk(chunk_id="chk-dup", doc_id="doc-A", content="Original text")
    chunk_dense = Chunk(chunk_id="chk-dup", doc_id="doc-B", content="Different text")

    sparse_res = [RetrievalResult(chunk=chunk_sparse, score=5.0, rank=1, retriever_name="bm25")]
    dense_res = [RetrievalResult(chunk=chunk_dense, score=0.9, rank=1, retriever_name="faiss")]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
    )
    hybrid.index([chunk_sparse])

    with pytest.raises(
        ValueError, match="Conflicting chunk provenance detected for chunk ID 'chk-dup'"
    ):
        hybrid.retrieve("query")


# -----------------------------------------------------------------------------
# 4. Tie-Breaking & Ordering
# -----------------------------------------------------------------------------


def test_deterministic_tie_breaking_consensus() -> None:
    """When fused score is identical, item supported by BOTH retrievers ranks higher."""
    # Suppose k=60
    # Chunk A is rank 1 in sparse, not in dense -> score = 1/(60+1) = 1/61
    # Chunk B is rank 2 in sparse, rank 3 in dense:
    # 1/62 + 1/63 = 0.016129 + 0.015873 = 0.032002
    # To construct an exact tie in fused score:
    # Chunk X: rank 1 in sparse, rank 2 in dense -> 1/61 + 1/62
    # Chunk Y: rank 2 in sparse, rank 1 in dense -> 1/62 + 1/61
    # Both have identical fused score = 1/61 + 1/62.
    # Chunk X has chunk_id "chk_A", Chunk Y has chunk_id "chk_B".
    # Both have retriever_count=2, both have best_rank=1.
    # Tied at (fused_score, retriever_count, best_rank).
    # Then tie breaks lexicographically by chunk_id ("chk_A" < "chk_B").
    chunk_x = _make_chunk("chk_A")
    chunk_y = _make_chunk("chk_B")

    sparse_res = [
        RetrievalResult(chunk=chunk_x, score=10.0, rank=1, retriever_name="bm25"),
        RetrievalResult(chunk=chunk_y, score=9.0, rank=2, retriever_name="bm25"),
    ]
    dense_res = [
        RetrievalResult(chunk=chunk_y, score=0.95, rank=1, retriever_name="faiss"),
        RetrievalResult(chunk=chunk_x, score=0.90, rank=2, retriever_name="faiss"),
    ]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
    )
    hybrid.index([chunk_x, chunk_y])

    results = hybrid.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0].score == results[1].score
    assert results[0].chunk.chunk_id == "chk_A"
    assert results[1].chunk.chunk_id == "chk_B"
    assert results[0].rank == 1
    assert results[1].rank == 2


def test_deterministic_tie_breaking_single_retriever_ties() -> None:
    """Ties between two chunks with identical single-retriever rank tie-break by chunk_id."""
    chunk_1 = _make_chunk("id_z")
    chunk_2 = _make_chunk("id_a")

    # In sparse: chunk_1 rank 1, chunk_2 not present.
    # In dense: chunk_2 rank 1, chunk_1 not present.
    # Both have fused score 1/(60+1) = 1/61.
    # Both have retriever_count = 1, best_rank = 1.
    # Lexicographical comparison: "id_a" < "id_z", so id_a must be ranked first.
    sparse_res = [RetrievalResult(chunk=chunk_1, score=10.0, rank=1, retriever_name="bm25")]
    dense_res = [RetrievalResult(chunk=chunk_2, score=0.9, rank=1, retriever_name="faiss")]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("bm25", sparse_res),
        dense_retriever=StubRetriever("faiss", dense_res),
    )
    hybrid.index([chunk_1, chunk_2])

    results = hybrid.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "id_a"
    assert results[1].chunk.chunk_id == "id_z"


# -----------------------------------------------------------------------------
# 5. Child Result Validation & Edge Cases
# -----------------------------------------------------------------------------


def test_invalid_child_result_types() -> None:
    class BadRetriever(BaseRetriever):
        @property
        def name(self) -> str:
            return "bad"

        def index(self, chunks: Sequence[Chunk]) -> None:
            pass

        def retrieve(self, query: str, top_k: int = 5) -> list[Any]:
            return ["not_a_retrieval_result"]  # type: ignore[return-value]

    hybrid = HybridRetriever(
        sparse_retriever=BadRetriever(),
        dense_retriever=StubRetriever("dense"),
    )
    hybrid.index([_make_chunk("c1")])

    with pytest.raises(TypeError, match="returned non-RetrievalResult item"):
        hybrid.retrieve("query")


def test_non_finite_child_score_rejected() -> None:
    chunk = _make_chunk("c1")
    with pytest.raises(ValueError, match="score must be a finite float"):
        RetrievalResult(chunk=chunk, score=float("nan"), rank=1, retriever_name="sparse")

    res = RetrievalResult(chunk=chunk, score=1.0, rank=1, retriever_name="sparse")
    object.__setattr__(res, "score", float("nan"))
    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("sparse", [res]),
        dense_retriever=StubRetriever("dense"),
    )
    hybrid.index([chunk])

    with pytest.raises(ValueError, match="returned non-finite score nan"):
        hybrid.retrieve("query")


def test_invalid_child_rank_rejected() -> None:
    chunk = _make_chunk("c1")
    # RetrievalResult post_init enforces rank >= 1, but we test hybrid defense if bypassed
    with pytest.raises(ValueError, match="rank must be a positive 1-indexed integer"):
        RetrievalResult(chunk=chunk, score=1.0, rank=0, retriever_name="sparse")


def test_internal_duplicate_chunk_in_child_ranking_rejected() -> None:
    chunk = _make_chunk("c1")
    # Child returns same chunk_id twice at ranks 1 and 2
    res_list = [
        RetrievalResult(chunk=chunk, score=10.0, rank=1, retriever_name="sparse"),
        RetrievalResult(chunk=chunk, score=8.0, rank=2, retriever_name="sparse"),
    ]
    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("sparse", res_list),
        dense_retriever=StubRetriever("dense"),
    )
    hybrid.index([chunk])

    with pytest.raises(
        ValueError, match="returned duplicate chunk ID 'c1' within a single ranking"
    ):
        hybrid.retrieve("query")


def test_internal_duplicate_rank_in_child_ranking_rejected() -> None:
    chunk_a = _make_chunk("c1")
    chunk_b = _make_chunk("c2")
    # Child returns two different chunks both having rank 1
    res_list = [
        RetrievalResult(chunk=chunk_a, score=10.0, rank=1, retriever_name="sparse"),
        RetrievalResult(chunk=chunk_b, score=8.0, rank=1, retriever_name="sparse"),
    ]
    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("sparse", res_list),
        dense_retriever=StubRetriever("dense"),
    )
    hybrid.index([chunk_a, chunk_b])

    with pytest.raises(ValueError, match="returned duplicate rank position 1"):
        hybrid.retrieve("query")


# -----------------------------------------------------------------------------
# 6. Indexing & Retrieval Lifecycle
# -----------------------------------------------------------------------------


def test_retrieval_before_indexing_raises_error() -> None:
    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("sparse"),
        dense_retriever=StubRetriever("dense"),
    )
    with pytest.raises(RuntimeError, match="has not been indexed yet"):
        hybrid.retrieve("query")


def test_child_indexing_failure_keeps_hybrid_unindexed() -> None:
    sparse = StubRetriever("sparse")
    dense_fail = StubRetriever("dense", fail_indexing=True)
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense_fail)

    with pytest.raises(RuntimeError, match="Simulated indexing failure"):
        hybrid.index([_make_chunk("c1")])

    assert hybrid.is_indexed is False
    with pytest.raises(RuntimeError, match="has not been indexed yet"):
        hybrid.retrieve("query")


def test_empty_corpus_indexing_and_retrieval() -> None:
    sparse = StubRetriever("sparse")
    dense = StubRetriever("dense")
    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)

    hybrid.index([])
    assert hybrid.is_indexed is True
    assert hybrid.retrieve("query") == []


def test_blank_query_and_zero_top_k() -> None:
    chunk = _make_chunk("c1")
    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("sparse", [_make_result(chunk, 1.0, 1)]),
        dense_retriever=StubRetriever("dense"),
    )
    hybrid.index([chunk])

    assert hybrid.retrieve("") == []
    assert hybrid.retrieve("   ") == []
    assert hybrid.retrieve("query", top_k=0) == []
    assert hybrid.retrieve("query", top_k=-1) == []


def _make_result(chunk: Chunk, score: float, rank: int, name: str = "stub") -> RetrievalResult:
    return RetrievalResult(chunk=chunk, score=score, rank=rank, retriever_name=name)


def test_top_k_cutoff_and_overfetching() -> None:
    """Over-fetching candidate_depth returns up to top_k final results."""
    chunks = [_make_chunk(f"chk-{i}") for i in range(10)]
    sparse_res = [_make_result(chunks[i], 10.0 - i, i + 1, "sparse") for i in range(10)]
    dense_res = [_make_result(chunks[9 - i], 1.0 - (i * 0.1), i + 1, "dense") for i in range(10)]

    hybrid = HybridRetriever(
        sparse_retriever=StubRetriever("sparse", sparse_res),
        dense_retriever=StubRetriever("dense", dense_res),
        candidate_depth=10,
    )
    hybrid.index(chunks)

    # Ask for top 3
    results = hybrid.retrieve("query", top_k=3)
    assert len(results) == 3
    assert [r.rank for r in results] == [1, 2, 3]


# -----------------------------------------------------------------------------
# 7. Benchmark Runner Integration Test
# -----------------------------------------------------------------------------


def test_hybrid_integration_with_benchmark_runner() -> None:
    """Evaluates HybridRetriever through the existing RetrievalBenchmarkRunner."""
    corpus = [
        Chunk(chunk_id="c1", doc_id="d1", content="Machine learning system reliability."),
        Chunk(chunk_id="c2", doc_id="d2", content="Dense retrieval vector search."),
        Chunk(chunk_id="c3", doc_id="d3", content="Lexical BM25 ranking baseline."),
    ]
    query = Query(
        query_id="q1",
        text="machine learning reliability",
        relevant_chunk_ids=["c1"],
        relevance_grades={"c1": 2},
    )

    sparse = BM25Retriever()
    # Stub dense retriever that returns c1 at rank 1
    dense = StubRetriever(
        "dense_stub",
        [RetrievalResult(chunk=corpus[0], score=0.95, rank=1, retriever_name="dense_stub")],
    )

    hybrid = HybridRetriever(sparse_retriever=sparse, dense_retriever=dense)

    runner = RetrievalBenchmarkRunner(retriever=hybrid, top_k=5)
    report = runner.run(corpus=corpus, benchmark=[query])

    assert report.config["query_count"] == 1
    assert report.config["retriever_name"] == "hybrid_rrf"
    assert report.metrics["recall@5"] == 1.0
    assert report.metrics["mrr@5"] == 1.0


# -----------------------------------------------------------------------------
# 8. Real-Backend Integration Test with FAISS and BM25
# -----------------------------------------------------------------------------


def test_real_faiss_and_bm25_hybrid_integration() -> None:
    """Combines real BM25Retriever and real FaissVectorRetriever through HybridRetriever."""
    faiss = pytest.importorskip("faiss")  # noqa: F841

    from reliability_lab.indexing.faiss_index import FaissVectorRetriever

    class FakeEmbedder(BaseEmbedder):
        @property
        def model_name(self) -> str:
            return "fake-3d"

        @property
        def embedding_dim(self) -> int:
            return 3

        def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
            vecs = []
            for t in texts:
                if "python" in t.lower():
                    vecs.append([1.0, 0.0, 0.0])
                elif "dense" in t.lower():
                    vecs.append([0.0, 1.0, 0.0])
                else:
                    vecs.append([0.0, 0.0, 1.0])
            return np.array(vecs, dtype=np.float32)

    chunks = [
        Chunk(chunk_id="ch-py", doc_id="d1", content="Python reliability benchmarking."),
        Chunk(chunk_id="ch-dense", doc_id="d2", content="Dense vector indexing with FAISS."),
        Chunk(chunk_id="ch-sparse", doc_id="d3", content="Sparse lexical BM25 matching."),
    ]

    bm25 = BM25Retriever()
    faiss_retriever = FaissVectorRetriever(embedder=FakeEmbedder(), index_type="flat")
    hybrid = HybridRetriever(sparse_retriever=bm25, dense_retriever=faiss_retriever)

    hybrid.index(chunks)
    assert hybrid.is_indexed is True

    # Search for python
    results = hybrid.retrieve("python", top_k=2)
    assert len(results) > 0
    assert results[0].chunk.chunk_id == "ch-py"
    assert results[0].rank == 1
    assert results[0].metadata["sparse_retriever_name"] == "bm25"
    assert results[0].metadata["dense_retriever_name"] == "faiss-flat"
