"""Retrieval evaluation metrics: set-based (Recall, Precision, HitRate) and positional (MRR, NDCG).

Implemented using Python standard library only, per ADR-005.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Query, RetrievalResult


class BaseSetRetrievalMetric(BaseMetric):
    """Base class for set-based and rank-aware retrieval metrics evaluated at cutoff k.

    Attributes:
        k: Positive integer cutoff position (defaults to 5).
    """

    def __init__(self, k: int = 5) -> None:
        """Initialize metric with cutoff parameter k.

        Args:
            k: Top-k ranking cutoff position (must be a positive integer >= 1).

        Raises:
            TypeError: If k is not an integer or is a boolean.
            ValueError: If k is not strictly positive (<= 0).
        """
        if isinstance(k, bool) or not isinstance(k, int):
            raise TypeError(f"k must be an integer, got {type(k).__name__}.")
        if k <= 0:
            raise ValueError(f"k must be a positive integer >= 1, got {k}.")
        self._k = k

    @property
    def k(self) -> int:
        """Return the ranking cutoff position k."""
        return self._k

    def _extract_ground_truth(self, query: Query) -> set[str]:
        """Extract the set of distinct relevant chunk IDs from the query ground truth.

        Constructs candidate identifiers from `query.relevant_chunk_ids` and
        `query.relevance_grades`, using `query.get_grade(chunk_id)` as the source
        of truth. Any chunk with an explicit or implicit grade > 0 is considered
        relevant. Chunks with grade 0 (including explicit overrides) are excluded.
        """
        candidate_ids = set(query.relevant_chunk_ids) | set(query.relevance_grades.keys())
        return {cid for cid in candidate_ids if query.get_grade(cid) > 0}

    def _extract_retrieved_hits(
        self, retrieved: Sequence[RetrievalResult], query: Query
    ) -> set[str]:
        """Extract the distinct relevant chunk IDs retrieved in the first k positions.

        Treats the incoming sequence as already ranked. Missing positions (if len < k)
        are ignored without raising an error. Duplicate chunk IDs in top-k results
        are deduplicated and count at most once.
        """
        hits: set[str] = set()
        for res in retrieved[: self._k]:
            cid = res.chunk.chunk_id
            if query.get_grade(cid) > 0:
                hits.add(cid)
        return hits

    def _validate_inputs(self, retrieved: Sequence[RetrievalResult], query: Query) -> None:
        """Validate input types for compute()."""
        if not isinstance(query, Query):
            raise TypeError(f"query must be an instance of Query, got {type(query).__name__}.")
        if not isinstance(retrieved, Sequence):
            raise TypeError(
                f"retrieved must be a Sequence of RetrievalResult, got {type(retrieved).__name__}."
            )
        for res in retrieved[: self._k]:
            if not isinstance(res, RetrievalResult):
                raise TypeError(
                    "Elements of retrieved must be RetrievalResult instances, "
                    f"got {type(res).__name__}."
                )


class RecallAtK(BaseSetRetrievalMetric):
    """Computes Recall@K: proportion of ground-truth relevant items retrieved in top-k.

    Definition:
        Recall@K = |Retrieved_K ∩ Relevant| / |Relevant|

    Edge cases:
        - When query has no ground-truth relevant chunks (|Relevant| == 0), returns 0.0.
        - When retrieved list is empty, returns 0.0.
        - When k exceeds the number of retrieved items, evaluates available items without error.
        - Repeated retrieved chunk IDs are counted at most once toward relevant hits.
    """

    @property
    def name(self) -> str:
        """Return the stable metric identifier."""
        return f"recall@{self.k}"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute Recall@K for the given retrieval candidates and query.

        Args:
            retrieved: Ranked sequence of RetrievalResult candidates (first is rank 1).
            query: Query instance with ground-truth relevant chunk IDs and grades.

        Returns:
            Proportion of relevant items retrieved in top-k as a float in [0.0, 1.0].
        """
        self._validate_inputs(retrieved, query)

        ground_truth = self._extract_ground_truth(query)
        if not ground_truth or not retrieved:
            return 0.0

        hits = self._extract_retrieved_hits(retrieved, query)
        return len(hits) / float(len(ground_truth))


class PrecisionAtK(BaseSetRetrievalMetric):
    """Computes Precision@K: proportion of top-k retrieved items that are relevant.

    Definition:
        Precision@K = |Retrieved_K ∩ Relevant| / K

    Edge cases:
        - The denominator is strictly K. If fewer than K items are retrieved,
          unreturned positions are non-hits and the denominator remains K.
        - When query has no ground-truth relevant chunks (|Relevant| == 0), returns 0.0.
        - When retrieved list is empty, returns 0.0.
        - Repeated retrieved chunk IDs are counted at most once toward hits.
    """

    @property
    def name(self) -> str:
        """Return the stable metric identifier."""
        return f"precision@{self.k}"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute Precision@K for the given retrieval candidates and query.

        Args:
            retrieved: Ranked sequence of RetrievalResult candidates (first is rank 1).
            query: Query instance with ground-truth relevant chunk IDs and grades.

        Returns:
            Precision in top-k as a float in [0.0, 1.0].
        """
        self._validate_inputs(retrieved, query)

        ground_truth = self._extract_ground_truth(query)
        if not ground_truth or not retrieved:
            return 0.0

        hits = self._extract_retrieved_hits(retrieved, query)
        return len(hits) / float(self.k)


class HitRateAtK(BaseSetRetrievalMetric):
    """Computes HitRate@K: binary indicator whether at least one hit is retrieved in top-k.

    Definition:
        HitRate@K = 1.0 if |Retrieved_K ∩ Relevant| >= 1 else 0.0

    Edge cases:
        - When query has no ground-truth relevant chunks (|Relevant| == 0), returns 0.0.
        - When retrieved list is empty, returns 0.0.
        - When k exceeds the number of retrieved items, evaluates available items without error.
    """

    @property
    def name(self) -> str:
        """Return the stable metric identifier."""
        return f"hit_rate@{self.k}"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute HitRate@K for the given retrieval candidates and query.

        Args:
            retrieved: Ranked sequence of RetrievalResult candidates (first is rank 1).
            query: Query instance with ground-truth relevant chunk IDs and grades.

        Returns:
            1.0 if at least one relevant chunk is retrieved in top-k, else 0.0.
        """
        self._validate_inputs(retrieved, query)

        ground_truth = self._extract_ground_truth(query)
        if not ground_truth or not retrieved:
            return 0.0

        hits = self._extract_retrieved_hits(retrieved, query)
        return 1.0 if len(hits) > 0 else 0.0


class MRRAtK(BaseSetRetrievalMetric):
    """Computes Mean Reciprocal Rank at cutoff K (MRR@K) for a single query.

    Evaluates the reciprocal rank (1 / rank) of the first relevant chunk retrieved
    within the top-k positions. Note: as an implementation of BaseMetric.compute,
    this evaluates a single query. Aggregate Mean Reciprocal Rank across queries
    is computed by the benchmark runner.

    Definition:
        RR@K = 1 / min({rank_i : query.get_grade(chunk_i) > 0, 1 <= rank_i <= K})
        If no relevant item is found in top-k, returns 0.0.

    Edge cases:
        - When query has no ground-truth relevant chunks, returns 0.0.
        - When retrieved list is empty, returns 0.0.
        - When no relevant chunk is retrieved within top-k, returns 0.0.
        - Relevant items at positions > K do not contribute.
        - Both Grade 1 and Grade 2 count as relevant.
        - Grade 0 overrides strictly exclude that chunk.
    """

    @property
    def name(self) -> str:
        """Return the stable metric identifier."""
        return f"mrr@{self.k}"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute reciprocal rank at cutoff K for a single query.

        Args:
            retrieved: Ranked sequence of RetrievalResult candidates (first is rank 1).
            query: Query instance with ground-truth relevant chunk IDs and grades.

        Returns:
            Reciprocal rank as a float in [0.0, 1.0].
        """
        self._validate_inputs(retrieved, query)

        ground_truth = self._extract_ground_truth(query)
        if not ground_truth or not retrieved:
            return 0.0

        for rank, res in enumerate(retrieved[: self.k], start=1):
            cid = res.chunk.chunk_id
            if query.get_grade(cid) > 0:
                return 1.0 / float(rank)

        return 0.0


class NDCGAtK(BaseSetRetrievalMetric):
    """Computes Normalized Discounted Cumulative Gain at cutoff K (NDCG@K).

    Uses standard logarithmic discounting and exponential gain formulation:
        gain(rel) = 2**rel - 1
        discount(i) = log2(i + 1)
        DCG@K = sum_{i=1}^{min(K, |retrieved|)} (2**rel_i - 1) / log2(i + 1)
        IDCG@K = sum_{i=1}^{min(K, |ideal|)} (2**ideal_rel_i - 1) / log2(i + 1)
        NDCG@K = DCG@K / IDCG@K (or 0.0 if IDCG@K == 0.0)

    Duplicate handling:
        When a retrieved chunk ID is duplicated in top-k, it preserves its rank position
        but contributes zero gain for duplicate occurrences (gains are earned at most once).
        The ideal ranking (IDCG) considers only distinct ground-truth chunk IDs.

    Edge cases:
        - When query has no ground-truth relevant chunks (IDCG == 0), returns 0.0.
        - When retrieved list is empty, returns 0.0.
        - When all relevance grades are 0, returns 0.0.
        - When K exceeds the number of retrieved items, processes available items without error.
        - Value is strictly bounded in [0.0, 1.0].
    """

    @property
    def name(self) -> str:
        """Return the stable metric identifier."""
        return f"ndcg@{self.k}"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute NDCG@K for the given retrieval candidates and query.

        Args:
            retrieved: Ranked sequence of RetrievalResult candidates (first is rank 1).
            query: Query instance with ground-truth relevant chunk IDs and grades.

        Returns:
            Normalized discounted cumulative gain as a float in [0.0, 1.0].
        """
        self._validate_inputs(retrieved, query)

        # Distinct ground-truth relevant grades in descending order
        candidate_ids = set(query.relevant_chunk_ids) | set(query.relevance_grades.keys())
        ground_truth_grades = [
            query.get_grade(cid) for cid in candidate_ids if query.get_grade(cid) > 0
        ]
        if not ground_truth_grades or not retrieved:
            return 0.0

        sorted_ideal_grades = sorted(ground_truth_grades, reverse=True)[: self.k]
        idcg = sum(
            ((2.0**grade) - 1.0) / math.log2(i + 1)
            for i, grade in enumerate(sorted_ideal_grades, start=1)
        )
        if idcg <= 0.0:
            return 0.0

        dcg = 0.0
        seen_chunk_ids: set[str] = set()
        for i, res in enumerate(retrieved[: self.k], start=1):
            cid = res.chunk.chunk_id
            if cid in seen_chunk_ids:
                # Duplicate occupies rank position i but contributes 0 gain
                continue
            seen_chunk_ids.add(cid)
            grade = query.get_grade(cid)
            if grade > 0:
                gain = (2.0**grade) - 1.0
                dcg += gain / math.log2(i + 1)

        ndcg = dcg / idcg
        return max(0.0, min(1.0, ndcg))
