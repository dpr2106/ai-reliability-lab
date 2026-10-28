"""Unit tests for set-based retrieval evaluation metrics (EVAL-001).

Validates Recall@K, Precision@K, and HitRate@K against hand-calculated values,
edge-case semantics, type constraints, and real benchmark data.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.retrieval_metrics import (
    BaseSetRetrievalMetric,
    HitRateAtK,
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
        ],
    )
    def test_custom_positive_k(
        self, metric_cls: type[BaseSetRetrievalMetric], prefix: str, k: int
    ) -> None:
        """Verify custom positive k values reflect in property and name."""
        metric = metric_cls(k=k)
        assert metric.k == k
        assert metric.name == f"{prefix}@{k}"

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK])
    @pytest.mark.parametrize("invalid_k", [0, -1, -10])
    def test_rejects_non_positive_k(
        self, metric_cls: type[BaseSetRetrievalMetric], invalid_k: int
    ) -> None:
        """Verify ValueError is raised for k <= 0."""
        with pytest.raises(ValueError, match="positive integer"):
            metric_cls(k=invalid_k)

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK])
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
    """Validates metrics against hand-calculated ground-truth values."""

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


class TestComputeInputValidation:
    """Validates type checks on compute() arguments."""

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK])
    def test_rejects_invalid_query_type(self, metric_cls: type[BaseSetRetrievalMetric]) -> None:
        """Reject non-Query instances."""
        metric = metric_cls(k=5)
        with pytest.raises(TypeError, match="query must be an instance of Query"):
            metric.compute([], "not_a_query")  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK])
    def test_rejects_invalid_retrieved_type(self, metric_cls: type[BaseSetRetrievalMetric]) -> None:
        """Reject non-Sequence instances for retrieved."""
        metric = metric_cls(k=5)
        query = Query(query_id="q1", text="query")
        with pytest.raises(TypeError, match="retrieved must be a Sequence"):
            metric.compute(12345, query)  # type: ignore[arg-type]

    @pytest.mark.parametrize("metric_cls", [RecallAtK, PrecisionAtK, HitRateAtK])
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
        """Evaluate a known query record from sample_benchmark.jsonl."""
        benchmark_file = (
            Path(__file__).resolve().parents[2] / "data/benchmarks/sample_benchmark.jsonl"
        )
        first_line = benchmark_file.read_text(encoding="utf-8").strip().splitlines()[0]
        record = json.loads(first_line)

        # q-bm25-k1-tuning has chunk-index-hybrid-001 (grade 2) and chunk-index-hybrid-004 (grade 1)
        relevance_grades = {j["chunk_id"]: j["grade"] for j in record["relevance_judgments"]}
        query = Query(
            query_id=record["query_id"],
            text=record["query"],
            relevance_grades=relevance_grades,
            ground_truth_answer=record["ground_truth_answer"],
            required_citations=[c["chunk_id"] for c in record["required_citations"]],
        )

        # Simulated perfect retrieval (both relevant chunks in top 2)
        perfect_retrieved = [
            _make_result("CHK-HYB-001", rank=1, score=4.5),
            _make_result("CHK-HYB-002", rank=2, score=3.2),
            _make_result("CHK-LAT-001", rank=3, score=1.1),
        ]
        # Ground truth has 2 relevant items (grades 2 and 1)
        assert RecallAtK(k=2).compute(perfect_retrieved, query) == 1.0
        assert PrecisionAtK(k=2).compute(perfect_retrieved, query) == 1.0
        assert HitRateAtK(k=2).compute(perfect_retrieved, query) == 1.0

        # Simulated partial retrieval (only grade 2 chunk in top 5, plus distractors)
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
