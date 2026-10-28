"""Set-based retrieval evaluation metrics: Recall@K, Precision@K, HitRate@K.

Implemented using Python standard library only, per ADR-005.
"""

from __future__ import annotations

from collections.abc import Sequence

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Query, RetrievalResult


class BaseSetRetrievalMetric(BaseMetric):
    """Base class for set-based retrieval metrics evaluated at rank cutoff k.

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
