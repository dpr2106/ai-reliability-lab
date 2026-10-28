"""Unit and integration tests for FaissVectorRetriever.

Covers:
- Real FAISS backend execution for IndexFlatIP (exact) and IndexHNSWFlat (approximate).
- Integer-ID mapping with IndexIDMap2 and Chunk provenance preservation.
- Incremental additions, corpus replacement, and duplicate prevention.
- Safe atomic persistence (save/load) and post-reload incremental expansion.
- Lazy import behavior and BM25 isolation.
- Error handling, shape validation, and non-finite value rejection.
- Latency vs recall comparison between Flat and HNSW indices.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from reliability_lab.core.models import Chunk
from reliability_lab.indexing.bm25_index import BM25Retriever
from reliability_lab.indexing.embeddings import BaseEmbedder
from reliability_lab.indexing.faiss_index import (
    FaissVectorRetriever,
    _require_faiss,
)


class DeterministicFakeEmbedder(BaseEmbedder):
    """Deterministic in-memory embedder for reproducible testing."""

    def __init__(self, dim: int = 4, mapping: dict[str, list[float]] | None = None) -> None:
        self._dim = dim
        self._mapping = mapping or {}

    @property
    def model_name(self) -> str:
        return "deterministic-fake-embedder"

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        vectors: list[list[float]] = []
        for text in texts:
            if text in self._mapping:
                vectors.append(self._mapping[text])
            else:
                h = sum(ord(c) for c in text)
                vec = [float((h * (i + 1)) % 100) / 100.0 for i in range(self._dim)]
                vectors.append(vec)
        return np.array(vectors, dtype=np.float32)


def _make_sample_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="chk-001",
            doc_id="doc-A",
            content="Alpha chunk text about retrieval reliability and validation.",
            page_number=1,
            char_start=0,
            char_end=60,
            metadata={"source": "test"},
        ),
        Chunk(
            chunk_id="chk-002",
            doc_id="doc-A",
            content="Beta chunk text describing dense vector similarity search.",
            page_number=1,
            char_start=61,
            char_end=120,
            metadata={"source": "test"},
        ),
        Chunk(
            chunk_id="chk-003",
            doc_id="doc-B",
            content="Gamma chunk text discussing BM25 lexical ranking baseline.",
            page_number=2,
            char_start=0,
            char_end=58,
            metadata={"source": "test"},
        ),
    ]


# -----------------------------------------------------------------------------
# 1. Real FAISS Backend: Initialization & Properties
# -----------------------------------------------------------------------------


def test_real_faiss_flat_initialization_and_properties() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    assert retriever.name == "faiss-flat"
    assert retriever.index_type == "flat"
    assert retriever.dimension == 4
    assert retriever.metric == "ip"
    assert retriever.normalize is True
    assert retriever.size == 0
    assert len(retriever) == 0
    assert retriever.chunk_ids == []
    assert retriever.embedder is embedder


def test_real_faiss_hnsw_initialization_and_properties() -> None:
    embedder = DeterministicFakeEmbedder(dim=8)
    retriever = FaissVectorRetriever(
        embedder=embedder,
        index_type="hnsw",
        m=16,
        ef_construction=32,
        ef_search=16,
    )

    assert retriever.name == "faiss-hnsw"
    assert retriever.index_type == "hnsw"
    assert retriever.dimension == 8
    assert retriever.m == 16
    assert retriever.ef_construction == 32
    assert retriever.ef_search == 16
    assert retriever.size == 0


# -----------------------------------------------------------------------------
# 2. Real FAISS Backend: Integer-ID Mapping & Exact Flat Ranking
# -----------------------------------------------------------------------------


def test_real_faiss_exact_flat_ranking_and_scores() -> None:
    # 3 orthogonal vectors in 3D
    mapping = {
        "doc1": [1.0, 0.0, 0.0],
        "doc2": [0.0, 1.0, 0.0],
        "doc3": [0.0, 0.0, 1.0],
        "query_doc1": [1.0, 0.0, 0.0],
    }
    embedder = DeterministicFakeEmbedder(dim=3, mapping=mapping)
    chunks = [
        Chunk(chunk_id="c1", doc_id="d1", content="doc1"),
        Chunk(chunk_id="c2", doc_id="d2", content="doc2"),
        Chunk(chunk_id="c3", doc_id="d3", content="doc3"),
    ]

    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat", normalize=True)
    retriever.index(chunks)

    assert retriever.size == 3
    assert "c1" in retriever
    assert "c2" in retriever
    assert "c3" in retriever
    assert retriever.get_chunk("c2") == chunks[1]
    assert retriever.get_chunk("c_missing") is None

    results = retriever.retrieve("query_doc1", top_k=3)
    assert len(results) == 3

    # Top result should be c1 with cosine similarity 1.0
    assert results[0].chunk.chunk_id == "c1"
    assert pytest.approx(results[0].score, abs=1e-5) == 1.0
    assert results[0].rank == 1
    assert results[0].retriever_name == "faiss-flat"
    assert results[0].metadata["faiss_id"] == 0
    assert results[0].metadata["index_type"] == "flat"

    # Other orthogonal results have score ~0.0
    assert pytest.approx(results[1].score, abs=1e-5) == 0.0
    assert pytest.approx(results[2].score, abs=1e-5) == 0.0


def test_real_faiss_hnsw_integer_id_mapping_and_retrieval() -> None:
    """Verifies HNSW sequential integer ID allocation via IndexIDMap2 wrapper."""
    mapping = {
        "text_1": [1.0, 0.1, 0.0, 0.0],
        "text_2": [0.0, 1.0, 0.0, 0.0],
        "text_3": [0.0, 0.0, 1.0, 0.0],
        "query": [0.99, 0.05, 0.0, 0.0],
    }
    embedder = DeterministicFakeEmbedder(dim=4, mapping=mapping)
    chunks = [
        Chunk(chunk_id="chk-10", doc_id="d1", content="text_1"),
        Chunk(chunk_id="chk-20", doc_id="d2", content="text_2"),
        Chunk(chunk_id="chk-30", doc_id="d3", content="text_3"),
    ]

    retriever = FaissVectorRetriever(
        embedder=embedder,
        index_type="hnsw",
        m=16,
        ef_construction=64,
        ef_search=32,
    )
    retriever.index(chunks)

    assert retriever.size == 3
    results = retriever.retrieve("query", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "chk-10"
    assert results[0].rank == 1
    assert results[0].metadata["index_type"] == "hnsw"
    assert results[0].metadata["ef_search"] == 32
    assert results[0].metadata["m"] == 16


# -----------------------------------------------------------------------------
# 3. Incremental Additions & Corpus Replacement
# -----------------------------------------------------------------------------


def test_real_faiss_incremental_add_behavior() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    chunks = _make_sample_chunks()

    # Initial add of 2 chunks
    retriever.add(chunks[:2])
    assert retriever.size == 2
    assert retriever.chunk_ids == ["chk-001", "chk-002"]

    # Incrementally add third chunk
    retriever.add([chunks[2]])
    assert retriever.size == 3
    assert retriever.chunk_ids == ["chk-001", "chk-002", "chk-003"]

    # Retrieve top 3
    results = retriever.retrieve("dense vector", top_k=3)
    assert len(results) == 3
    found_ids = [r.chunk.chunk_id for r in results]
    assert set(found_ids) == {"chk-001", "chk-002", "chk-003"}


def test_real_faiss_corpus_replacement_index_method() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    initial_chunks = _make_sample_chunks()
    retriever.index(initial_chunks)
    assert retriever.size == 3

    replacement_chunks = [
        Chunk(chunk_id="rep-01", doc_id="new-doc", content="Replacement content one"),
        Chunk(chunk_id="rep-02", doc_id="new-doc", content="Replacement content two"),
    ]
    retriever.index(replacement_chunks)
    assert retriever.size == 2
    assert "chk-001" not in retriever
    assert "rep-01" in retriever
    assert "rep-02" in retriever


# -----------------------------------------------------------------------------
# 4. Save and Load Roundtrip
# -----------------------------------------------------------------------------


@pytest.mark.parametrize("index_type", ["flat", "hnsw"])
def test_real_faiss_save_and_load_roundtrip(tmp_path: Path, index_type: str) -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(
        embedder=embedder,
        index_type=index_type,
        m=16,
        ef_construction=32,
        ef_search=16,
    )
    chunks = _make_sample_chunks()
    retriever.index(chunks)

    # Initial retrieval before save
    results_before = retriever.retrieve("retrieval validation", top_k=3)

    # Save to disk
    save_dir = tmp_path / f"saved_{index_type}"
    retriever.save(save_dir)

    assert (save_dir / "index.faiss").is_file()
    assert (save_dir / "metadata.json").is_file()

    # Load back
    loaded_retriever = FaissVectorRetriever.load(save_dir, embedder=embedder)

    assert loaded_retriever.name == f"faiss-{index_type}"
    assert loaded_retriever.size == 3
    assert loaded_retriever.dimension == 4
    if index_type == "hnsw":
        assert loaded_retriever.ef_search == 16

    # Verify identical retrieval scores and ranks
    results_after = loaded_retriever.retrieve("retrieval validation", top_k=3)
    assert len(results_after) == len(results_before)
    for r_b, r_a in zip(results_before, results_after, strict=True):
        assert r_b.chunk.chunk_id == r_a.chunk.chunk_id
        assert pytest.approx(r_b.score, abs=1e-5) == r_a.score
        assert r_b.rank == r_a.rank

    # Test subsequent incremental addition on loaded index!
    new_chunk = Chunk(chunk_id="post-load-01", doc_id="doc-X", content="Post load added chunk")
    loaded_retriever.add([new_chunk])
    assert loaded_retriever.size == 4
    assert "post-load-01" in loaded_retriever


def test_real_faiss_save_and_load_empty_index(tmp_path: Path) -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    save_dir = tmp_path / "empty_index"
    retriever.save(save_dir)

    loaded = FaissVectorRetriever.load(save_dir, embedder=embedder)
    assert loaded.size == 0
    assert loaded.retrieve("query") == []


# -----------------------------------------------------------------------------
# 5. Edge Cases & Normalization
# -----------------------------------------------------------------------------


def test_real_faiss_empty_corpus_and_edge_cases() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    # Empty index
    assert retriever.retrieve("query") == []

    # Populate index
    retriever.index(_make_sample_chunks())

    # Invalid query strings / empty
    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []

    # top_k <= 0
    assert retriever.retrieve("query", top_k=0) == []
    assert retriever.retrieve("query", top_k=-5) == []


def test_real_faiss_zero_norm_vector_handling() -> None:
    """Zero vectors must be handled without producing NaNs."""
    mapping = {
        "zero": [0.0, 0.0, 0.0, 0.0],
        "nonzero": [1.0, 0.0, 0.0, 0.0],
    }
    embedder = DeterministicFakeEmbedder(dim=4, mapping=mapping)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat", normalize=True)

    chunks = [
        Chunk(chunk_id="c_zero", doc_id="d1", content="zero"),
        Chunk(chunk_id="c_norm", doc_id="d2", content="nonzero"),
    ]
    retriever.index(chunks)
    assert retriever.size == 2

    results = retriever.retrieve("nonzero", top_k=2)
    assert len(results) == 2
    assert results[0].chunk.chunk_id == "c_norm"


# -----------------------------------------------------------------------------
# 6. Controlled Latency vs Recall Benchmark (Flat vs HNSW)
# -----------------------------------------------------------------------------


def test_real_faiss_flat_vs_hnsw_benchmark() -> None:
    """Compares exact FlatIP against approximate HNSWFlat on synthetic vectors."""
    dim = 16
    n_chunks = 100
    rng = np.random.default_rng(42)

    # Generate synthetic embeddings
    synthetic_vecs = rng.standard_normal((n_chunks, dim)).astype(np.float32)
    mapping = {f"item_{i}": synthetic_vecs[i].tolist() for i in range(n_chunks)}
    embedder = DeterministicFakeEmbedder(dim=dim, mapping=mapping)

    chunks = [
        Chunk(chunk_id=f"chk-{i}", doc_id="synth", content=f"item_{i}") for i in range(n_chunks)
    ]

    flat_retriever = FaissVectorRetriever(embedder=embedder, index_type="flat", normalize=True)
    flat_retriever.index(chunks)

    hnsw_retriever = FaissVectorRetriever(
        embedder=embedder,
        index_type="hnsw",
        m=16,
        ef_construction=64,
        ef_search=32,
        normalize=True,
    )
    hnsw_retriever.index(chunks)

    # Perform 10 test queries
    top_k = 5
    recalls: list[float] = []
    for q_idx in range(10):
        q_vec = rng.standard_normal((1, dim)).astype(np.float32).tolist()[0]
        q_embedder = DeterministicFakeEmbedder(dim=dim, mapping={f"query_{q_idx}": q_vec})

        # Inject query vector into embedder
        flat_retriever._embedder = q_embedder
        hnsw_retriever._embedder = q_embedder

        flat_res = flat_retriever.retrieve(f"query_{q_idx}", top_k=top_k)
        hnsw_res = hnsw_retriever.retrieve(f"query_{q_idx}", top_k=top_k)

        flat_ids = {r.chunk.chunk_id for r in flat_res}
        hnsw_ids = {r.chunk.chunk_id for r in hnsw_res}

        overlap = len(flat_ids & hnsw_ids)
        recalls.append(overlap / top_k)

    mean_recall = sum(recalls) / len(recalls)
    # HNSW with efSearch=32 on 100 items should achieve >= 90% recall vs exact Flat
    assert mean_recall >= 0.90


# -----------------------------------------------------------------------------
# 7. Validation & Error Handling
# -----------------------------------------------------------------------------


def test_validation_invalid_embedder_type() -> None:
    with pytest.raises(TypeError, match="embedder must be an instance of BaseEmbedder"):
        FaissVectorRetriever(embedder="not-an-embedder")  # type: ignore[arg-type]


def test_validation_invalid_index_type() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    with pytest.raises(ValueError, match="Invalid index_type 'unsupported'"):
        FaissVectorRetriever(embedder=embedder, index_type="unsupported")


def test_validation_invalid_metric() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    with pytest.raises(ValueError, match="Unsupported metric 'l2'"):
        FaissVectorRetriever(embedder=embedder, metric="l2")


def test_validation_invalid_hnsw_parameters() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    with pytest.raises(ValueError, match="m must be an integer >= 4"):
        FaissVectorRetriever(embedder=embedder, index_type="hnsw", m=2)

    with pytest.raises(ValueError, match="ef_construction must be a positive integer"):
        FaissVectorRetriever(embedder=embedder, index_type="hnsw", ef_construction=0)

    with pytest.raises(ValueError, match="ef_search must be a positive integer"):
        FaissVectorRetriever(embedder=embedder, index_type="hnsw", ef_search=-1)


def test_validation_duplicate_chunk_in_batch() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    batch = [
        Chunk(chunk_id="dup-01", doc_id="d1", content="Text A"),
        Chunk(chunk_id="dup-01", doc_id="d1", content="Text B"),
    ]
    with pytest.raises(ValueError, match="Duplicate chunk ID 'dup-01' found in the input batch"):
        retriever.add(batch)

    # State must not be corrupted
    assert retriever.size == 0


def test_validation_duplicate_chunk_with_existing() -> None:
    embedder = DeterministicFakeEmbedder(dim=4)
    retriever = FaissVectorRetriever(embedder=embedder, index_type="flat")

    chunk1 = Chunk(chunk_id="c1", doc_id="d1", content="Text 1")
    retriever.add([chunk1])
    assert retriever.size == 1

    chunk_dup = Chunk(chunk_id="c1", doc_id="d1", content="Text 1 duplicate")
    with pytest.raises(ValueError, match="Chunk ID 'c1' already exists in index"):
        retriever.add([chunk_dup])

    # State must remain clean with original 1 chunk
    assert retriever.size == 1


def test_validation_embedder_dimension_mismatch() -> None:
    class BadEmbedder(BaseEmbedder):
        @property
        def model_name(self) -> str:
            return "bad"

        @property
        def embedding_dim(self) -> int:
            return 4

        def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
            # Returns 3D vector instead of 4D
            return np.ones((len(texts), 3), dtype=np.float32)

    retriever = FaissVectorRetriever(embedder=BadEmbedder(), index_type="flat")
    with pytest.raises(ValueError, match=r"Embedding shape .* does not match expected dimension"):
        retriever.add([Chunk(chunk_id="c1", doc_id="d1", content="content")])


def test_validation_non_finite_embeddings_rejected() -> None:
    class NanEmbedder(BaseEmbedder):
        @property
        def model_name(self) -> str:
            return "nan"

        @property
        def embedding_dim(self) -> int:
            return 4

        def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
            arr = np.ones((len(texts), 4), dtype=np.float32)
            arr[0, 1] = float("nan")
            return arr

    retriever = FaissVectorRetriever(embedder=NanEmbedder(), index_type="flat")
    with pytest.raises(ValueError, match="Embeddings contain non-finite values"):
        retriever.add([Chunk(chunk_id="c1", doc_id="d1", content="content")])


def test_validation_load_errors(tmp_path: Path) -> None:
    embedder = DeterministicFakeEmbedder(dim=4)

    # Missing directory
    with pytest.raises(FileNotFoundError, match="FAISS index directory not found"):
        FaissVectorRetriever.load(tmp_path / "non_existent", embedder=embedder)

    # Missing index binary
    empty_dir = tmp_path / "missing_files"
    empty_dir.mkdir()
    with pytest.raises(FileNotFoundError, match="Index binary not found"):
        FaissVectorRetriever.load(empty_dir, embedder=embedder)

    # Corrupted metadata
    (empty_dir / "index.faiss").write_bytes(b"dummy")
    (empty_dir / "metadata.json").write_text("not json", encoding="utf-8")
    with pytest.raises(ValueError, match="Failed to parse metadata file"):
        FaissVectorRetriever.load(empty_dir, embedder=embedder)

    # Dimension mismatch
    meta_mismatch: dict[str, Any] = {
        "dimension": 8,
        "index_type": "flat",
        "chunks": {},
    }
    (empty_dir / "metadata.json").write_text(json.dumps(meta_mismatch), encoding="utf-8")
    expected_msg = r"Saved index dimension \(8\) does not match embedder dimension \(4\)"
    with pytest.raises(ValueError, match=expected_msg):
        FaissVectorRetriever.load(empty_dir, embedder=embedder)


# -----------------------------------------------------------------------------
# 8. Lazy Import & Isolation Verification
# -----------------------------------------------------------------------------


def test_lazy_import_missing_faiss_raises_actionable_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulates absence of faiss module to verify actionable error message."""
    import builtins

    real_import = builtins.__import__

    def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "faiss":
            raise ImportError("No module named 'faiss'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", mock_import)

    with pytest.raises(ImportError, match="faiss is required for FaissVectorRetriever"):
        _require_faiss()


def test_bm25_workflow_completely_independent_of_faiss() -> None:
    """Verifies that BM25 retriever functions without requiring or touching FAISS."""
    chunks = _make_sample_chunks()
    bm25 = BM25Retriever()
    bm25.index(chunks)
    assert len(bm25._chunks) == 3

    results = bm25.retrieve("retrieval reliability", top_k=2)
    assert len(results) > 0
    assert results[0].retriever_name == "bm25"
