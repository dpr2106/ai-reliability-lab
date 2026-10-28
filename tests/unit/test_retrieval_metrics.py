"""Unit tests for retrieval evaluation metrics: set-based and positional (EVAL-001 & EVAL-002).

Validates Recall@K, Precision@K, HitRate@K, MRR@K, and NDCG@K against hand-calculated values,
logarithmic discounting, edge-case semantics, type constraints, and real benchmark data.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.retrieval_metrics import (
    BaseSetRetrievalMetric,
    HitRateAtK,
    MRRAtK,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
)


def _make_chunk(chunk_id: str) -> Chunk:
    return Chunk(chunk_id=chunk_id, doc_id="doc-1", content=f"Content for {chunk_id}")


def _make_result(chunk_id: str, rank: int, score: float = 1.0) -> RetrievalResult:
    return RetrievalResult(
        chunk=_make_chunk(chunk_id),
        score=score,
        rank=rank,
        retriever_name="test-retriever",
    )


class TestMetricConstructionAndValidation:
    """Tests parameter validation and metadata properties on metric classes."""

    @pytest.mark.parametrize(
        ("metric_cls", "prefix"),
        [
            (RecallAtK, "recall"),
            (PrecisionAtK, "precision"),
            (HitRateAtK, "hit_rate"),
            (MRRAtK, "mrr"),
            (NDCGAtK, "ndcg"),
        ],
    )
    def test_default_construction_and_name(
        self, metric_cls: type[BaseSetRetrievalMetric], prefix: str
    ) -> None:
        """Verify default k=5, inheritance from BaseMetric, and stable name property."""
        metric = metric_cls()
        assert isinstance(metric, BaseMetric)
        assert isinstance(metric, BaseSetRetrievalMetric)
        assert metric.k == 5
        assert metric.name == f"{prefix}@5"

    @pytest.mark.parametrize(
        ("metric_cls", "prefix", "k"),
        [
            (RecallAtK, "recall", 1),
            (PrecisionAtK, "precision", 10),
            (HitRateAtK, "hit_rate", 20),
            (MRRAtK, "mrr", 3),
            (NDCGAtK, "ndcg", 15),
        ],
    )
    def test_custom_positive_k(
        self, metric_cls: type[BaseSetRetrievalMetric], prefix: str, k: int
    ) -> None:
        """Verify custom positive k values reflect in property and name."""
        metric = metric_cls(k=k)
        assert metric.k == k
        assert metric.name == f"{prefix}@{k}"

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK, MRRAtK, NDCGAtK])
    @pytest.mark.parametrize("invalid_k", [0, -1, -10])
    def test_rejects_non_positive_k(
        self, metric_cls: type[BaseSetRetrievalMetric], invalid_k: int
    ) -> None:
        """Verify ValueError is raised for k <= 0."""
        with pytest.raises(ValueError, match="positive integer"):
            metric_cls(k=invalid_k)

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK, MRRAtK, NDCGAtK])
    @pytest.mark.parametrize(
        "invalid_type_k",
        [True, False, 3.14, "5", [5], None, {"k": 5}],
    )
    def test_rejects_invalid_types_for_k(
        self, metric_cls: type[BaseSetRetrievalMetric], invalid_type_k: object
    ) -> None:
        """Verify TypeError is raised for booleans, floats, strings, and non-ints."""
        with pytest.raises(TypeError, match="integer"):
            metric_cls(k=invalid_type_k)  # type: ignore[arg-type]


class TestStandardMetricCalculations:
    """Validates set-based metrics against hand-calculated ground-truth values."""

    def test_normal_retrieval_scenario(self) -> None:
        """Query with 4 ground truth items; top-5 has 2 relevant and 3 irrelevant."""
        query = Query(
            query_id="q1",
            text="query text",
            relevant_chunk_ids=["c1", "c2", "c3", "c4"],
        )
        retrieved = [
            _make_result("c1", rank=1, score=0.9),
            _make_result("irr1", rank=2, score=0.8),
            _make_result("c2", rank=3, score=0.7),
            _make_result("irr2", rank=4, score=0.6),
            _make_result("irr3", rank=5, score=0.5),
            _make_result("c3", rank=6, score=0.4),  # beyond k=5
        ]

        # In top-5: hits are {c1, c2} (2 hits)
        # Ground truth has 4 items
        # Recall@5 = 2 / 4 = 0.5
        assert RecallAtK(k=5).compute(retrieved, query) == pytest.approx(0.5)

        # Precision@5 = 2 / 5 = 0.4
        assert PrecisionAtK(k=5).compute(retrieved, query) == pytest.approx(0.4)

        # HitRate@5 = 1.0 (since hits >= 1)
        assert HitRateAtK(k=5).compute(retrieved, query) == 1.0

    def test_k1_cutoff(self) -> None:
        """Check behavior at k=1 cutoff."""
        query = Query(
            query_id="q1",
            text="query text",
            relevant_chunk_ids=["c1", "c2"],
        )
        # First position is a hit
        retrieved_hit = [_make_result("c1", rank=1)]
        assert RecallAtK(k=1).compute(retrieved_hit, query) == pytest.approx(0.5)
        assert PrecisionAtK(k=1).compute(retrieved_hit, query) == 1.0
        assert HitRateAtK(k=1).compute(retrieved_hit, query) == 1.0

        # First position is a miss
        retrieved_miss = [_make_result("irr1", rank=1), _make_result("c1", rank=2)]
        assert RecallAtK(k=1).compute(retrieved_miss, query) == 0.0
        assert PrecisionAtK(k=1).compute(retrieved_miss, query) == 0.0
        assert HitRateAtK(k=1).compute(retrieved_miss, query) == 0.0

    def test_grades_and_explicit_overrides(self) -> None:
        """Grade 1 and 2 both count as relevant; grade 0 explicitly overrides relevant_chunk_ids."""
        query = Query(
            query_id="q1",
            text="query text",
            relevant_chunk_ids=["c1", "c-overridden"],
            relevance_grades={
                "c-overridden": 0,  # Explicitly overridden to irrelevant
                "c-grade2": 2,  # Critical evidence (not in relevant_chunk_ids)
                "c-grade0": 0,  # Distractor
            },
        )
        # Ground truth: c1 (grade 1 via relevant_chunk_ids), c-grade2 (grade 2 via relevance_grades)
        # Total ground truth items = 2
        retrieved = [
            _make_result("c-overridden", rank=1),  # grade 0 -> NOT a hit
            _make_result("c-grade2", rank=2),  # grade 2 -> HIT
            _make_result("c-grade0", rank=3),  # grade 0 -> NOT a hit
            _make_result("c1", rank=4),  # grade 1 -> HIT
            _make_result("unmentioned", rank=5),  # unjudged -> NOT a hit
        ]

        # Top 3: only c-grade2 is a hit (1 hit)
        assert RecallAtK(k=3).compute(retrieved, query) == pytest.approx(0.5)
        assert PrecisionAtK(k=3).compute(retrieved, query) == pytest.approx(1.0 / 3.0)
        assert HitRateAtK(k=3).compute(retrieved, query) == 1.0

        # Top 5: c-grade2 and c1 are hits (2 hits out of 2 ground-truth items)
        assert RecallAtK(k=5).compute(retrieved, query) == 1.0
        assert PrecisionAtK(k=5).compute(retrieved, query) == pytest.approx(2.0 / 5.0)
        assert HitRateAtK(k=5).compute(retrieved, query) == 1.0

    def test_duplicate_retrieved_chunk_ids_do_not_inflate_numerator(self) -> None:
        """Repeated retrieved chunk IDs within top-k must count at most once."""
        query = Query(
            query_id="q1",
            text="query text",
            relevant_chunk_ids=["c1", "c2"],
        )
        # c1 repeated 3 times in top 5
        retrieved = [
            _make_result("c1", rank=1),
            _make_result("c1", rank=2),
            _make_result("c1", rank=3),
            _make_result("irr1", rank=4),
            _make_result("irr2", rank=5),
        ]

        # Unique hits in top 5: {c1} (1 hit)
        # Ground truth: {c1, c2} (2 items)
        assert RecallAtK(k=5).compute(retrieved, query) == pytest.approx(0.5)
        # Precision denominator is 5; numerator is 1 (not 3)
        assert PrecisionAtK(k=5).compute(retrieved, query) == pytest.approx(0.2)
        assert HitRateAtK(k=5).compute(retrieved, query) == 1.0


class TestEdgeCaseSemantics:
    """Validates deterministic edge case semantics specified in DATA-001 & EVAL-001."""

    def test_empty_retrieval_results(self) -> None:
        """Empty retrieval list produces 0.0 for all metrics."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        assert RecallAtK(k=5).compute([], query) == 0.0
        assert PrecisionAtK(k=5).compute([], query) == 0.0
        assert HitRateAtK(k=5).compute([], query) == 0.0

    def test_empty_ground_truth(self) -> None:
        """Query with zero relevant chunks returns 0.0 for all metrics (defined convention)."""
        query_empty = Query(query_id="q1", text="query", relevant_chunk_ids=[])
        query_all_zeros = Query(
            query_id="q2",
            text="query",
            relevance_grades={"c1": 0, "c2": 0},
        )
        retrieved = [_make_result("c1", rank=1), _make_result("c2", rank=2)]

        for q in [query_empty, query_all_zeros]:
            assert RecallAtK(k=5).compute(retrieved, q) == 0.0
            assert PrecisionAtK(k=5).compute(retrieved, q) == 0.0
            assert HitRateAtK(k=5).compute(retrieved, q) == 0.0

    def test_fewer_results_than_k(self) -> None:
        """When fewer than k results are returned, Precision denominator remains k."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=["c1", "c2", "c3"],
        )
        # Only 2 results returned, k=5
        retrieved = [_make_result("c1", rank=1), _make_result("irr", rank=2)]

        # Hits = 1, ground truth = 3
        assert RecallAtK(k=5).compute(retrieved, query) == pytest.approx(1.0 / 3.0)
        # Precision: 1 hit / 5 = 0.2 (NOT 1 / 2)
        assert PrecisionAtK(k=5).compute(retrieved, query) == pytest.approx(0.2)
        assert HitRateAtK(k=5).compute(retrieved, query) == 1.0

    def test_k_exceeds_available_results(self) -> None:
        """Processing proceeds without error when k > len(retrieved)."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=["c1"],
        )
        retrieved = [_make_result("c1", rank=1)]

        assert RecallAtK(k=100).compute(retrieved, query) == 1.0
        assert PrecisionAtK(k=100).compute(retrieved, query) == pytest.approx(0.01)
        assert HitRateAtK(k=100).compute(retrieved, query) == 1.0

    def test_no_relevant_chunks_retrieved(self) -> None:
        """When top-k contains only non-relevant items, all metrics return 0.0."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=["c1", "c2"],
        )
        retrieved = [_make_result("x1", rank=1), _make_result("x2", rank=2)]

        assert RecallAtK(k=5).compute(retrieved, query) == 0.0
        assert PrecisionAtK(k=5).compute(retrieved, query) == 0.0
        assert HitRateAtK(k=5).compute(retrieved, query) == 0.0

    def test_relevant_chunks_beyond_k_ignored(self) -> None:
        """Relevant chunks at rank > k are ignored."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [
            _make_result("irr1", rank=1),
            _make_result("irr2", rank=2),
            _make_result("c1", rank=3),  # rank 3 > k=2
        ]

        assert RecallAtK(k=2).compute(retrieved, query) == 0.0
        assert PrecisionAtK(k=2).compute(retrieved, query) == 0.0
        assert HitRateAtK(k=2).compute(retrieved, query) == 0.0


class TestMRRMetricCalculations:
    """Validates MRR@K positional reciprocal rank calculations (EVAL-002)."""

    def test_first_result_relevant_returns_1(self) -> None:
        """First retrieved result relevant -> reciprocal rank is 1.0."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [_make_result("c1", rank=1), _make_result("irr", rank=2)]
        assert MRRAtK(k=5).compute(retrieved, query) == 1.0

    def test_first_relevant_at_rank_2_returns_half(self) -> None:
        """First relevant chunk at rank 2 -> reciprocal rank is 0.5."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [_make_result("irr", rank=1), _make_result("c1", rank=2)]
        assert MRRAtK(k=5).compute(retrieved, query) == 0.5

    def test_first_relevant_at_rank_3_returns_third(self) -> None:
        """First relevant chunk at rank 3 -> reciprocal rank is 1/3."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [
            _make_result("irr1", rank=1),
            _make_result("irr2", rank=2),
            _make_result("c1", rank=3),
        ]
        assert MRRAtK(k=5).compute(retrieved, query) == pytest.approx(1.0 / 3.0)

    def test_relevant_result_just_outside_k_returns_zero(self) -> None:
        """Relevant result at rank 6 with k=5 returns 0.0."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [_make_result(f"irr{i}", rank=i) for i in range(1, 6)] + [
            _make_result("c1", rank=6)
        ]
        assert MRRAtK(k=5).compute(retrieved, query) == 0.0

    def test_no_relevant_retrieved_candidates_returns_zero(self) -> None:
        """All retrieved items irrelevant -> returns 0.0."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [_make_result(f"irr{i}", rank=i) for i in range(1, 6)]
        assert MRRAtK(k=5).compute(retrieved, query) == 0.0

    def test_empty_retrieval_returns_zero(self) -> None:
        """Empty retrieval sequence returns 0.0."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        assert MRRAtK(k=5).compute([], query) == 0.0

    def test_empty_ground_truth_returns_zero(self) -> None:
        """Zero relevant items in query returns 0.0."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=[])
        retrieved = [_make_result("c1", rank=1)]
        assert MRRAtK(k=5).compute(retrieved, query) == 0.0

    def test_grade1_and_grade2_both_count_as_relevant(self) -> None:
        """Grade 1 and grade 2 both satisfy the relevance condition for reciprocal rank."""
        query_g1 = Query(query_id="q1", text="query", relevance_grades={"c-g1": 1, "c-g2": 2})
        # Grade 1 at rank 1 -> returns 1.0
        retrieved_g1_first = [_make_result("c-g1", rank=1), _make_result("c-g2", rank=2)]
        assert MRRAtK(k=5).compute(retrieved_g1_first, query_g1) == 1.0

        # Grade 2 at rank 2 (first hit) -> returns 0.5
        retrieved_g2_second = [_make_result("irr", rank=1), _make_result("c-g2", rank=2)]
        assert MRRAtK(k=5).compute(retrieved_g2_second, query_g1) == 0.5

    def test_explicit_grade0_override(self) -> None:
        """Explicit grade 0 override in relevance_grades negates relevant_chunk_ids."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=["c-overridden", "c-real"],
            relevance_grades={"c-overridden": 0, "c-real": 1},
        )
        # c-overridden at rank 1 is NOT a hit; c-real at rank 2 IS the first hit
        retrieved = [_make_result("c-overridden", rank=1), _make_result("c-real", rank=2)]
        assert MRRAtK(k=5).compute(retrieved, query) == 0.5

    def test_explicitly_graded_chunk_absent_from_relevant_chunk_ids(self) -> None:
        """Chunk with explicit grade counts even if absent from relevant_chunk_ids."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=[],
            relevance_grades={"c-explicit": 2},
        )
        retrieved = [_make_result("c-explicit", rank=1)]
        assert MRRAtK(k=5).compute(retrieved, query) == 1.0

    def test_different_cutoffs(self) -> None:
        """Cutoff k=1 captures only rank 1 hits; k > len(retrieved) processes available items."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [_make_result("irr", rank=1), _make_result("c1", rank=2)]

        # k=1 misses rank 2 hit
        assert MRRAtK(k=1).compute(retrieved, query) == 0.0
        # k=10 finds rank 2 hit
        assert MRRAtK(k=10).compute(retrieved, query) == 0.5

    def test_per_query_reciprocal_rank_contract(self) -> None:
        """Confirm MRRAtK.compute evaluates a single query, not a cross-query mean."""
        query = Query(query_id="q1", text="query", relevant_chunk_ids=["c1"])
        retrieved = [_make_result("irr", rank=1), _make_result("c1", rank=2)]
        score = MRRAtK(k=5).compute(retrieved, query)
        assert isinstance(score, float)
        assert score == 0.5


class TestNDCGMetricCalculations:
    """Validates NDCG@K graded ranking and logarithmic discounting (EVAL-002)."""

    def test_perfect_ranking_returns_one(self) -> None:
        """When retrieved candidates match the ideal ranking order, NDCG is exactly 1.0."""
        query = Query(
            query_id="q1",
            text="query",
            relevance_grades={"c_g2": 2, "c_g1": 1},
        )
        # Ideal order: c_g2 (rank 1), c_g1 (rank 2)
        perfect_retrieved = [_make_result("c_g2", rank=1), _make_result("c_g1", rank=2)]
        assert NDCGAtK(k=5).compute(perfect_retrieved, query) == pytest.approx(1.0)

    def test_reversed_or_degraded_ranking_produces_lower_score(self) -> None:
        """Reversed relevance grades (lower grade first) produce lower score than 1.0."""
        query = Query(
            query_id="q1",
            text="query",
            relevance_grades={"c_g2": 2, "c_g1": 1},
        )
        # Reversed order: c_g1 (rank 1, gain 1), c_g2 (rank 2, gain 3)
        reversed_retrieved = [_make_result("c_g1", rank=1), _make_result("c_g2", rank=2)]

        # Ideal: (3 / log2(2)) + (1 / log2(3)) = 3 + 1/1.5849625 ≈ 3.63092975
        # DCG: (1 / log2(2)) + (3 / log2(3)) = 1 + 3/1.5849625 ≈ 2.89278926
        # NDCG = 2.89278926 / 3.63092975 ≈ 0.796707
        expected_idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_dcg = 1.0 / math.log2(2) + 3.0 / math.log2(3)
        expected_ndcg = expected_dcg / expected_idcg

        ndcg = NDCGAtK(k=5).compute(reversed_retrieved, query)
        assert ndcg == pytest.approx(expected_ndcg)
        assert 0.0 < ndcg < 1.0

    def test_relevant_chunk_ranked_lower_is_discounted(self) -> None:
        """A relevant chunk ranked at lower positions receives greater logarithmic discount."""
        query = Query(query_id="q1", text="query", relevance_grades={"c1": 2})

        # Rank 1: DCG = 3 / log2(2) = 3.0, IDCG = 3.0 -> NDCG = 1.0
        retrieved_rank1 = [_make_result("c1", rank=1)]
        assert NDCGAtK(k=5).compute(retrieved_rank1, query) == 1.0

        # Rank 2: DCG = 3 / log2(3) ≈ 1.892789, IDCG = 3.0 -> NDCG ≈ 0.63092975
        retrieved_rank2 = [_make_result("irr", rank=1), _make_result("c1", rank=2)]
        expected_ndcg_rank2 = (3.0 / math.log2(3)) / (3.0 / math.log2(2))
        assert NDCGAtK(k=5).compute(retrieved_rank2, query) == pytest.approx(expected_ndcg_rank2)

        # Rank 3: DCG = 3 / log2(4) = 1.5, IDCG = 3.0 -> NDCG = 0.5
        retrieved_rank3 = [
            _make_result("irr1", rank=1),
            _make_result("irr2", rank=2),
            _make_result("c1", rank=3),
        ]
        assert NDCGAtK(k=5).compute(retrieved_rank3, query) == pytest.approx(0.5)

    def test_grade2_weighted_more_than_grade1(self) -> None:
        """Grade 2 contributes exponential gain 3 (2**2 - 1) vs grade 1 gain 1 (2**1 - 1)."""
        query_g2 = Query(query_id="q1", text="query", relevance_grades={"c1": 2, "other": 2})
        query_g1 = Query(query_id="q2", text="query", relevance_grades={"c1": 1, "other": 2})

        retrieved = [_make_result("c1", rank=1)]
        # DCG for g2 is 3.0 / log2(2) = 3.0
        # DCG for g1 is 1.0 / log2(2) = 1.0
        score_g2 = NDCGAtK(k=5).compute(retrieved, query_g2)
        score_g1 = NDCGAtK(k=5).compute(retrieved, query_g1)
        assert score_g2 > score_g1

    def test_hand_calculated_mixed_grade_ranking(self) -> None:
        """Verify exact hand-calculated NDCG@3 with mixed grades and distractors."""
        query = Query(
            query_id="q1",
            text="query",
            relevance_grades={"c_high": 2, "c_med": 1, "c_extra": 2},
        )
        # Ground truth has two grade 2s and one grade 1.
        # Ideal top 3 grades: [2, 2, 1]
        # IDCG@3 = (2**2 - 1)/log2(2) + (2**2 - 1)/log2(3) + (2**1 - 1)/log2(4)
        #        = 3/1 + 3/1.584962500721156 + 1/2
        #        = 3.0 + 1.8927892607143721 + 0.5 = 5.392789260714372
        idcg_3 = 3.0 / math.log2(2) + 3.0 / math.log2(3) + 1.0 / math.log2(4)

        # Retrieved ranking at top 3:
        # rank 1: irr (grade 0) -> gain 0
        # rank 2: c_high (grade 2) -> gain 3, discount log2(3)
        # rank 3: c_med (grade 1) -> gain 1, discount log2(4)
        # DCG@3 = 0 + 3/log2(3) + 1/log2(4) = 1.8927892607143721 + 0.5 = 2.392789260714372
        dccg_3 = 3.0 / math.log2(3) + 1.0 / math.log2(4)
        expected_ndcg = dccg_3 / idcg_3

        retrieved = [
            _make_result("irr", rank=1),
            _make_result("c_high", rank=2),
            _make_result("c_med", rank=3),
            _make_result("c_extra", rank=4),  # beyond k=3
        ]
        assert NDCGAtK(k=3).compute(retrieved, query) == pytest.approx(expected_ndcg)

    def test_no_relevant_retrieved_candidates_returns_zero(self) -> None:
        """When no relevant candidates are in top-k, NDCG is 0.0."""
        query = Query(query_id="q1", text="query", relevance_grades={"c1": 2})
        retrieved = [_make_result("irr1", rank=1), _make_result("irr2", rank=2)]
        assert NDCGAtK(k=5).compute(retrieved, query) == 0.0

    def test_empty_retrieval_and_empty_ground_truth_returns_zero(self) -> None:
        """Empty retrieved or empty ground truth safely returns 0.0 without division by zero."""
        query_normal = Query(query_id="q1", text="query", relevance_grades={"c1": 2})
        query_empty = Query(query_id="q2", text="query", relevance_grades={})

        assert NDCGAtK(k=5).compute([], query_normal) == 0.0
        assert NDCGAtK(k=5).compute([_make_result("c1", rank=1)], query_empty) == 0.0
        assert NDCGAtK(k=5).compute([], query_empty) == 0.0

    def test_all_zero_relevance_grades_returns_zero(self) -> None:
        """When all ground truth items have grade 0, IDCG is 0 and metric returns 0.0."""
        query = Query(query_id="q1", text="query", relevance_grades={"c1": 0, "c2": 0})
        retrieved = [_make_result("c1", rank=1)]
        assert NDCGAtK(k=5).compute(retrieved, query) == 0.0

    def test_explicit_grade0_override(self) -> None:
        """Grade 0 override ensures chunk contributes zero gain and is excluded from IDCG."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=["c-overridden", "c-real"],
            relevance_grades={"c-overridden": 0, "c-real": 2},
        )
        # IDCG should only have c-real (gain 3): IDCG = 3 / log2(2) = 3.0
        # If retrieved is c-overridden at rank 1, gain is 0 -> NDCG = 0.0
        retrieved_override = [_make_result("c-overridden", rank=1)]
        assert NDCGAtK(k=5).compute(retrieved_override, query) == 0.0

        # If retrieved is c-real at rank 1, NDCG is 1.0
        retrieved_real = [_make_result("c-real", rank=1)]
        assert NDCGAtK(k=5).compute(retrieved_real, query) == 1.0

    def test_explicitly_graded_chunk_absent_from_relevant_chunk_ids(self) -> None:
        """Chunk with explicit grade in relevance_grades contributes to IDCG and DCG."""
        query = Query(
            query_id="q1",
            text="query",
            relevant_chunk_ids=[],
            relevance_grades={"c-exp": 2},
        )
        retrieved = [_make_result("c-exp", rank=1)]
        assert NDCGAtK(k=5).compute(retrieved, query) == 1.0

    def test_k_greater_than_retrieved_result_count(self) -> None:
        """Processing proceeds normally when k exceeds available retrieved candidates."""
        query = Query(query_id="q1", text="query", relevance_grades={"c1": 2})
        retrieved = [_make_result("c1", rank=1)]
        assert NDCGAtK(k=100).compute(retrieved, query) == 1.0

    def test_relevant_candidates_outside_k_ignored_in_dcg(self) -> None:
        """Relevant candidates beyond cutoff k do not contribute to DCG."""
        query = Query(query_id="q1", text="query", relevance_grades={"c1": 2})
        retrieved = [_make_result("irr", rank=1), _make_result("c1", rank=2)]
        assert NDCGAtK(k=1).compute(retrieved, query) == 0.0

    def test_duplicate_retrieved_ids_cannot_inflate_dcg(self) -> None:
        """Duplicate retrieved chunk IDs earn gain at most once, preserving their rank discount."""
        query = Query(
            query_id="q1",
            text="query",
            relevance_grades={"c1": 2, "c2": 1},
        )
        # Ideal: c1 (gain 3) at rank 1, c2 (gain 1) at rank 2
        # IDCG = 3/log2(2) + 1/log2(3) ≈ 3.0 + 0.63092975 = 3.63092975
        idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)

        # Retrieved:
        # rank 1: c1 (gain 3)
        # rank 2: c1 (duplicate! occupies position 2, but gains 0)
        # rank 3: c2 (gain 1, discounted at log2(4) = 2.0)
        # DCG = 3/log2(2) + 0 + 1/log2(4) = 3.0 + 0.5 = 3.5
        dcg = 3.0 / math.log2(2) + 1.0 / math.log2(4)
        expected_ndcg = dcg / idcg

        retrieved = [
            _make_result("c1", rank=1),
            _make_result("c1", rank=2),
            _make_result("c2", rank=3),
        ]
        assert NDCGAtK(k=3).compute(retrieved, query) == pytest.approx(expected_ndcg)

    def test_ndcg_bounded_between_zero_and_one(self) -> None:
        """NDCG values are strictly within [0.0, 1.0] across diverse configurations."""
        query = Query(
            query_id="q1",
            text="query",
            relevance_grades={"c1": 2, "c2": 1, "c3": 1},
        )
        scenarios = [
            [_make_result("c1", rank=1), _make_result("c2", rank=2), _make_result("c3", rank=3)],
            [_make_result("c3", rank=1), _make_result("c2", rank=2), _make_result("c1", rank=3)],
            [_make_result("irr", rank=1), _make_result("c3", rank=2)],
            [_make_result("c2", rank=1)],
            [],
        ]
        for retrieved in scenarios:
            score = NDCGAtK(k=5).compute(retrieved, query)
            assert 0.0 <= score <= 1.0
            assert math.isfinite(score)


class TestComputeInputValidation:
    """Validates type checks on compute() arguments."""

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK, MRRAtK, NDCGAtK])
    def test_rejects_invalid_query_type(self, metric_cls: type[BaseSetRetrievalMetric]) -> None:
        """Reject non-Query instances."""
        metric = metric_cls(k=5)
        with pytest.raises(TypeError, match="query must be an instance of Query"):
            metric.compute([], "not_a_query")  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK, MRRAtK, NDCGAtK])
    def test_rejects_invalid_retrieved_type(self, metric_cls: type[BaseSetRetrievalMetric]) -> None:
        """Reject non-Sequence instances for retrieved."""
        metric = metric_cls(k=5)
        query = Query(query_id="q1", text="query")
        with pytest.raises(TypeError, match="retrieved must be a Sequence"):
            metric.compute(12345, query)  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK, MRRAtK, NDCGAtK])
    def test_rejects_invalid_element_in_retrieved(
        self, metric_cls: type[BaseSetRetrievalMetric]
    ) -> None:
        """Reject elements in retrieved that are not RetrievalResult instances."""
        metric = metric_cls(k=5)
        query = Query(query_id="q1", text="query")
        with pytest.raises(TypeError, match="RetrievalResult"):
            metric.compute(["not_a_result"], query)  # type: ignore[list-item]


class TestIntegrationWithGoldenBenchmarkDataset:
    """Validates metrics with actual sample benchmark dataset from DATA-001."""

    def test_evaluation_with_sample_benchmark_record(self) -> None:
        """Evaluate a known query record from sample_benchmark.jsonl across all 5 metrics."""
        benchmark_file = (
            Path(__file__).resolve().parents[2] / "data/benchmarks/sample_benchmark.jsonl"
        )
        first_line = benchmark_file.read_text(encoding="utf-8").strip().splitlines()[0]
        record = json.loads(first_line)

        # Q-001 has CHK-HYB-001 (grade 2) and CHK-HYB-002 (grade 1)
        relevance_grades = {j["chunk_id"]: j["grade"] for j in record["relevance_judgments"]}
        query = Query(
            query_id=record["query_id"],
            text=record["query"],
            relevance_grades=relevance_grades,
            ground_truth_answer=record["ground_truth_answer"],
            required_citations=[c["chunk_id"] for c in record["required_citations"]],
        )

        # Simulated perfect retrieval (both relevant chunks in top 2 in ideal order)
        perfect_retrieved = [
            _make_result("CHK-HYB-001", rank=1, score=4.5),
            _make_result("CHK-HYB-002", rank=2, score=3.2),
            _make_result("CHK-LAT-001", rank=3, score=1.1),
        ]
        assert RecallAtK(k=2).compute(perfect_retrieved, query) == 1.0
        assert PrecisionAtK(k=2).compute(perfect_retrieved, query) == 1.0
        assert HitRateAtK(k=2).compute(perfect_retrieved, query) == 1.0
        assert MRRAtK(k=2).compute(perfect_retrieved, query) == 1.0
        assert NDCGAtK(k=2).compute(perfect_retrieved, query) == pytest.approx(1.0)

        # Simulated partial retrieval (distractor at rank 1, grade 2 at rank 2)
        partial_retrieved = [
            _make_result("CHK-LAT-001", rank=1, score=4.5),
            _make_result("CHK-HYB-001", rank=2, score=3.2),
            _make_result("CHK-IR-001", rank=3, score=2.1),
            _make_result("CHK-IR-002", rank=4, score=1.5),
            _make_result("CHK-IR-003", rank=5, score=0.9),
        ]
        # 1 hit out of 2 ground-truth items
        assert RecallAtK(k=5).compute(partial_retrieved, query) == pytest.approx(0.5)
        # 1 hit in top 5
        assert PrecisionAtK(k=5).compute(partial_retrieved, query) == pytest.approx(0.2)
        assert HitRateAtK(k=5).compute(partial_retrieved, query) == 1.0
        # First relevant hit at rank 2 -> MRR = 0.5
        assert MRRAtK(k=5).compute(partial_retrieved, query) == 0.5

        # NDCG@5 calculation:
        # Ideal: CHK-HYB-001 (grade 2, gain 3) at pos 1, CHK-HYB-002 (grade 1, gain 1) at pos 2
        # IDCG = 3/log2(2) + 1/log2(3) = 3 + 1/1.5849625 ≈ 3.63092975
        # Actual DCG: pos 1 is distractor (0), pos 2 is CHK-HYB-001 (gain 3, discount log2(3))
        # DCG = 3/log2(3) ≈ 1.89278926
        # NDCG = 1.89278926 / 3.63092975 ≈ 0.521296
        expected_idcg = 3.0 / math.log2(2) + 1.0 / math.log2(3)
        expected_dcg = 3.0 / math.log2(3)
        assert NDCGAtK(k=5).compute(partial_retrieved, query) == pytest.approx(
            expected_dcg / expected_idcg
        )
