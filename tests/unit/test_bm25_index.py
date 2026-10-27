"""Unit tests for in-memory Okapi BM25 lexical retriever (RET-001).

Tests mathematical correctness, hand-calculated score verification,
ranking invariants, tokenization, lifecycle, and edge cases.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, RetrievalResult
from reliability_lab.indexing.bm25_index import BM25Retriever


def _make_chunk(chunk_id: str, content: str, doc_id: str = "doc-1") -> Chunk:
    """Helper to create minimal valid Chunk entities for index testing."""
    return Chunk(chunk_id=chunk_id, doc_id=doc_id, content=content)


class TestBM25Configuration:
    """Verify parameter validation and default settings."""

    def test_default_parameters(self) -> None:
        retriever = BM25Retriever()
        assert retriever.k1 == 1.5
        assert retriever.b == 0.75
        assert retriever.name == "bm25"
        assert isinstance(retriever, BaseRetriever)

    def test_custom_valid_parameters(self) -> None:
        retriever = BM25Retriever(k1=1.2, b=0.8)
        assert retriever.k1 == 1.2
        assert retriever.b == 0.8

    @pytest.mark.parametrize("invalid_k1", [0, -1.0, -0.01, float("inf"), float("nan")])
    def test_reject_invalid_k1_values(self, invalid_k1: float) -> None:
        with pytest.raises(ValueError, match="strictly positive finite"):
            BM25Retriever(k1=invalid_k1)

    @pytest.mark.parametrize("invalid_k1_type", ["1.5", None, True, False, [1.5]])
    def test_reject_invalid_k1_types(self, invalid_k1_type: Any) -> None:
        with pytest.raises(TypeError, match="finite float or int"):
            BM25Retriever(k1=invalid_k1_type)

    @pytest.mark.parametrize("invalid_b", [-0.1, 1.01, 2.0, float("inf"), float("nan")])
    def test_reject_invalid_b_values(self, invalid_b: float) -> None:
        with pytest.raises(ValueError, match=r"range \[0.0, 1.0\]"):
            BM25Retriever(b=invalid_b)

    @pytest.mark.parametrize("invalid_b_type", ["0.75", None, True, False, (0.75,)])
    def test_reject_invalid_b_types(self, invalid_b_type: Any) -> None:
        with pytest.raises(TypeError, match="finite float or int"):
            BM25Retriever(b=invalid_b_type)

    def test_boundary_b_values(self) -> None:
        retriever_b0 = BM25Retriever(b=0.0)
        assert retriever_b0.b == 0.0

        retriever_b1 = BM25Retriever(b=1.0)
        assert retriever_b1.b == 1.0


class TestBM25MathematicalCorrectness:
    """Hand-calculated mathematical verification of BM25 and IDF formulations."""

    @pytest.fixture
    def small_corpus(self) -> list[Chunk]:
        """Corpus:
        Chunk 0: "apple banana" (length 2)
        Chunk 1: "apple apple cherry" (length 3)
        Total tokens = 5, N = 2, avgdl = 2.5
        """
        return [
            _make_chunk("c0", "apple banana"),
            _make_chunk("c1", "apple apple cherry"),
        ]

    def test_hand_calculated_single_term_query(self, small_corpus: list[Chunk]) -> None:
        """Hand calculations:
        k1 = 1.5, b = 0.75
        N = 2, avgdl = 2.5

        term: "apple", df = 2
        IDF(apple) = ln(1 + (2 - 2 + 0.5) / (2 + 0.5)) = ln(1 + 0.2) = ln(1.2) ~= 0.1823215568

        Chunk 0: |D0| = 2, f(apple, D0) = 1
        len_ratio = 2 / 2.5 = 0.8
        denom = 1 + 1.5 * (1 - 0.75 + 0.75 * 0.8) = 1 + 1.5 * (0.25 + 0.6) = 1 + 1.275 = 2.275
        num = 1 * (1.5 + 1) = 2.5
        score_c0 = ln(1.2) * (2.5 / 2.275) ~= 0.200353359

        Chunk 1: |D1| = 3, f(apple, D1) = 2
        len_ratio = 3 / 2.5 = 1.2
        denom = 2 + 1.5 * (1 - 0.75 + 0.75 * 1.2) = 2 + 1.5 * (0.25 + 0.9) = 2 + 1.725 = 3.725
        num = 2 * (1.5 + 1) = 5.0
        score_c1 = ln(1.2) * (5.0 / 3.725) ~= 0.244726922
        """
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(small_corpus)

        results = retriever.retrieve("apple", top_k=5)
        assert len(results) == 2

        expected_idf_apple = math.log(1.2)
        expected_score_c0 = expected_idf_apple * (2.5 / 2.275)
        expected_score_c1 = expected_idf_apple * (5.0 / 3.725)

        # Chunk 1 ranks #1
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].rank == 1
        assert math.isclose(results[0].score, expected_score_c1, rel_tol=1e-6)

        # Chunk 0 ranks #2
        assert results[1].chunk.chunk_id == "c0"
        assert results[1].rank == 2
        assert math.isclose(results[1].score, expected_score_c0, rel_tol=1e-6)

    def test_hand_calculated_selective_term_query(self, small_corpus: list[Chunk]) -> None:
        """Hand calculations for "banana":
        term: "banana", df = 1
        IDF(banana) = ln(1 + (2 - 1 + 0.5) / (1 + 0.5)) = ln(1 + 1.0) = ln(2) ~= 0.69314718056

        Chunk 0: f(banana, D0) = 1, denom = 2.275, num = 2.5
        score_c0 = ln(2) * (2.5 / 2.275) ~= 0.761700198
        Chunk 1: does not contain "banana" -> not returned.
        """
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(small_corpus)

        results = retriever.retrieve("banana", top_k=5)
        assert len(results) == 1

        expected_idf_banana = math.log(2.0)
        expected_score_c0 = expected_idf_banana * (2.5 / 2.275)

        assert results[0].chunk.chunk_id == "c0"
        assert results[0].rank == 1
        assert math.isclose(results[0].score, expected_score_c0, rel_tol=1e-6)

    def test_hand_calculated_multi_term_additive_score(self, small_corpus: list[Chunk]) -> None:
        """Query "apple banana":
        Chunk 0 scores for both "apple" and "banana".
        score_c0 = score(apple, D0) + score(banana, D0)
        """
        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index(small_corpus)

        results = retriever.retrieve("apple banana", top_k=5)
        assert len(results) == 2

        score_apple_c0 = math.log(1.2) * (2.5 / 2.275)
        score_banana_c0 = math.log(2.0) * (2.5 / 2.275)
        score_apple_c1 = math.log(1.2) * (5.0 / 3.725)

        expected_c0 = score_apple_c0 + score_banana_c0
        expected_c1 = score_apple_c1

        assert results[0].chunk.chunk_id == "c0"
        assert math.isclose(results[0].score, expected_c0, rel_tol=1e-6)

        assert results[1].chunk.chunk_id == "c1"
        assert math.isclose(results[1].score, expected_c1, rel_tol=1e-6)

    def test_term_frequency_saturation(self) -> None:
        """Verify that score increases with tf but saturates towards (k1 + 1) * IDF."""
        c1 = _make_chunk("c1", "keyword")
        c2 = _make_chunk("c2", "keyword " * 10)
        c3 = _make_chunk("c3", "keyword " * 100)

        # Use b=0 to isolate term frequency from length normalization
        retriever = BM25Retriever(k1=1.5, b=0.0)
        retriever.index([c1, c2, c3])

        results = retriever.retrieve("keyword", top_k=3)
        assert len(results) == 3

        # Scores must be strictly increasing: score(100) > score(10) > score(1)
        score_1 = next(r.score for r in results if r.chunk.chunk_id == "c1")
        score_10 = next(r.score for r in results if r.chunk.chunk_id == "c2")
        score_100 = next(r.score for r in results if r.chunk.chunk_id == "c3")

        assert score_100 > score_10 > score_1

        # Saturation theoretical ceiling = (k1 + 1) * IDF = 2.5 * ln(1 + (3 - 3 + 0.5) / 3.5)
        idf = math.log(1.0 + 0.5 / 3.5)
        ceiling = 2.5 * idf
        assert score_100 < ceiling
        assert math.isclose(score_100, ceiling, rel_tol=0.05)

    def test_document_length_normalization(self) -> None:
        """With b > 0, an identical term match in a shorter document scores higher."""
        short_chunk = _make_chunk("short", "target")
        long_chunk = _make_chunk("long", "target filler filler filler filler filler filler filler")

        retriever = BM25Retriever(k1=1.5, b=0.75)
        retriever.index([short_chunk, long_chunk])

        results = retriever.retrieve("target", top_k=2)
        assert results[0].chunk.chunk_id == "short"
        assert results[0].score > results[1].score

    def test_zero_length_normalization_when_b_is_zero(self) -> None:
        """When b=0, document length penalty is completely disabled."""
        short_chunk = _make_chunk("short", "target")
        long_chunk = _make_chunk("long", "target filler filler filler filler filler filler filler")

        retriever = BM25Retriever(k1=1.5, b=0.0)
        retriever.index([short_chunk, long_chunk])

        results = retriever.retrieve("target", top_k=2)
        # Equal term frequency (1) without length normalization yields identical scores
        assert math.isclose(results[0].score, results[1].score, rel_tol=1e-9)


class TestRankingAndTieBreaking:
    """Test deterministic ranking, stable tie-breaking, and top_k bounds."""

    def test_stable_tie_breaking_preserves_input_order(self) -> None:
        """When multiple chunks have identical scores, original input order breaks ties."""
        chunks = [
            _make_chunk("chunk_first", "sample test content"),
            _make_chunk("chunk_second", "sample test content"),
            _make_chunk("chunk_third", "sample test content"),
        ]
        retriever = BM25Retriever()
        retriever.index(chunks)

        results = retriever.retrieve("sample", top_k=3)
        assert len(results) == 3

        # All scores are identical
        assert math.isclose(results[0].score, results[1].score, rel_tol=1e-9)
        assert math.isclose(results[1].score, results[2].score, rel_tol=1e-9)

        # Ranks are strictly 1, 2, 3
        assert [r.rank for r in results] == [1, 2, 3]

        # Order must strictly match original input order
        assert [r.chunk.chunk_id for r in results] == [
            "chunk_first",
            "chunk_second",
            "chunk_third",
        ]

    def test_top_k_truncation(self) -> None:
        chunks = [_make_chunk(f"c{i}", f"common term unique_{i}") for i in range(10)]
        retriever = BM25Retriever()
        retriever.index(chunks)

        assert len(retriever.retrieve("common", top_k=3)) == 3
        assert len(retriever.retrieve("common", top_k=1)) == 1
        assert len(retriever.retrieve("common", top_k=15)) == 10

    def test_top_k_zero_returns_empty_list(self) -> None:
        chunks = [_make_chunk("c1", "data retrieval")]
        retriever = BM25Retriever()
        retriever.index(chunks)
        assert retriever.retrieve("data", top_k=0) == []

    @pytest.mark.parametrize("invalid_top_k", [-1, -10])
    def test_reject_negative_top_k(self, invalid_top_k: int) -> None:
        chunks = [_make_chunk("c1", "data retrieval")]
        retriever = BM25Retriever()
        retriever.index(chunks)
        with pytest.raises(ValueError, match="non-negative integer"):
            retriever.retrieve("data", top_k=invalid_top_k)

    @pytest.mark.parametrize("invalid_top_k_type", ["5", None, True, False, 2.5])
    def test_reject_invalid_top_k_types(self, invalid_top_k_type: Any) -> None:
        chunks = [_make_chunk("c1", "data retrieval")]
        retriever = BM25Retriever()
        retriever.index(chunks)
        with pytest.raises(TypeError, match="top_k must be an integer"):
            retriever.retrieve("data", top_k=invalid_top_k_type)


class TestTokenizationAndQuerySemantics:
    """Test tokenization, case-normalization, Unicode, and query term semantics."""

    def test_case_insensitive_matching(self) -> None:
        chunks = [_make_chunk("c1", "Neural Network Architecture")]
        retriever = BM25Retriever()
        retriever.index(chunks)

        assert len(retriever.retrieve("neural", top_k=1)) == 1
        assert len(retriever.retrieve("NEURAL", top_k=1)) == 1
        assert len(retriever.retrieve("NeUrAl", top_k=1)) == 1

    def test_punctuation_and_whitespace_handling(self) -> None:
        chunks = [_make_chunk("c1", "Reliability, safety & fault-tolerance (v2.0)!")]
        retriever = BM25Retriever()
        retriever.index(chunks)

        assert len(retriever.retrieve("reliability", top_k=1)) == 1
        assert len(retriever.retrieve("fault", top_k=1)) == 1
        assert len(retriever.retrieve("tolerance", top_k=1)) == 1
        assert len(retriever.retrieve("v2", top_k=1)) == 1
        assert len(retriever.retrieve("0", top_k=1)) == 1

    def test_unicode_alphanumeric_support(self) -> None:
        chunks = [
            _make_chunk("c1", "Hôtel résumé café"),
            _make_chunk("c2", "Проверка надежности систем"),
            _make_chunk("c3", "自然言語処理モデル"),
        ]
        retriever = BM25Retriever()
        retriever.index(chunks)

        assert len(retriever.retrieve("hôtel", top_k=1)) == 1
        assert len(retriever.retrieve("проверка", top_k=1)) == 1
        assert len(retriever.retrieve("自然言語処理モデル", top_k=1)) == 1

    def test_repeated_query_tokens_scored_once(self) -> None:
        """Ensure duplicate query tokens do not inflate the BM25 score."""
        chunks = [_make_chunk("c1", "single instance of token")]
        retriever = BM25Retriever()
        retriever.index(chunks)

        score_single = retriever.retrieve("token", top_k=1)[0].score
        score_repeated = retriever.retrieve("token token token", top_k=1)[0].score

        assert math.isclose(score_single, score_repeated, rel_tol=1e-9)

    def test_mixed_known_and_unknown_query_tokens(self) -> None:
        chunks = [_make_chunk("c1", "vector database indexing")]
        retriever = BM25Retriever()
        retriever.index(chunks)

        results = retriever.retrieve("vector unknown_word_xyz", top_k=1)
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c1"
        assert results[0].metadata["matched_terms"] == ["vector"]

    def test_purely_out_of_vocabulary_query(self) -> None:
        chunks = [_make_chunk("c1", "vector database indexing")]
        retriever = BM25Retriever()
        retriever.index(chunks)

        results = retriever.retrieve("quantum mechanics physics", top_k=5)
        assert results == []

    def test_empty_or_punctuation_only_query(self) -> None:
        chunks = [_make_chunk("c1", "content here")]
        retriever = BM25Retriever()
        retriever.index(chunks)

        assert retriever.retrieve("", top_k=5) == []
        assert retriever.retrieve("   ", top_k=5) == []
        assert retriever.retrieve("!@#$%^&*()", top_k=5) == []


class TestIndexLifecycleAndEdgeCases:
    """Test lifecycle states, empty collections, and re-indexing."""

    def test_retrieve_before_indexing_raises_runtime_error(self) -> None:
        retriever = BM25Retriever()
        with pytest.raises(RuntimeError, match="before index\\(\\) was executed"):
            retriever.retrieve("test query")

    def test_empty_corpus_indexing(self) -> None:
        retriever = BM25Retriever()
        retriever.index([])

        assert retriever.retrieve("any query") == []

    def test_reindexing_replaces_old_state(self) -> None:
        c_old = _make_chunk("old", "legacy information")
        c_new = _make_chunk("new", "modern architecture")

        retriever = BM25Retriever()
        retriever.index([c_old])
        assert len(retriever.retrieve("legacy")) == 1

        # Re-index with new corpus
        retriever.index([c_new])
        assert retriever.retrieve("legacy") == []
        assert len(retriever.retrieve("modern")) == 1
        assert retriever.retrieve("modern")[0].chunk.chunk_id == "new"

    def test_reindexing_empty_clears_state(self) -> None:
        c1 = _make_chunk("c1", "some content")
        retriever = BM25Retriever()
        retriever.index([c1])
        assert len(retriever.retrieve("content")) == 1

        retriever.index([])
        assert retriever.retrieve("content") == []

    def test_single_chunk_corpus(self) -> None:
        c1 = _make_chunk("solo", "unique solitary chunk")
        retriever = BM25Retriever()
        retriever.index([c1])

        results = retriever.retrieve("solitary")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "solo"
        assert results[0].score > 0
        assert math.isfinite(results[0].score)

    def test_chunks_with_only_non_words(self) -> None:
        c1 = _make_chunk("c1", "... --- ...")
        c2 = _make_chunk("c2", "valid words here")
        retriever = BM25Retriever()
        retriever.index([c1, c2])

        results = retriever.retrieve("words")
        assert len(results) == 1
        assert results[0].chunk.chunk_id == "c2"

    def test_result_contract_and_metadata(self) -> None:
        chunk = _make_chunk("target_chunk", "precision grounded citations", doc_id="parent_doc")
        retriever = BM25Retriever(k1=1.6, b=0.7)
        retriever.index([chunk])

        results = retriever.retrieve("grounded")
        assert len(results) == 1
        res = results[0]

        assert isinstance(res, RetrievalResult)
        assert res.chunk is chunk
        assert res.chunk.doc_id == "parent_doc"
        assert res.rank == 1
        assert res.retriever_name == "bm25"
        assert res.metadata["k1"] == 1.6
        assert res.metadata["b"] == 0.7
        assert res.metadata["matched_terms"] == ["grounded"]
        assert isinstance(res.to_dict(), dict)
        assert isinstance(res.to_json(), str)
