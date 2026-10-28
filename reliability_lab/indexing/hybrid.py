"""Hybrid sparse-dense retrieval engine combining BM25 and FAISS using Reciprocal Rank Fusion (RRF).

Implements unweighted Reciprocal Rank Fusion (RRF) over injected BaseRetriever instances,
eliminating score-scale discrepancies between lexical and semantic algorithms while
strictly preserving full Chunk provenance and deterministic tie-breaking.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, RetrievalResult

DEFAULT_RRF_K: int = 60
DEFAULT_HYBRID_NAME: str = "hybrid_rrf"


@dataclass(frozen=True)
class _FusedCandidate:
    """Internal helper tracking merged candidate properties prior to result creation."""

    chunk: Chunk
    fused_score: float
    sparse_rank: int | None
    sparse_score: float | None
    dense_rank: int | None
    dense_score: float | None
    contributions: dict[str, float]
    retriever_count: int
    best_rank: int


def _validate_child_results(
    results: list[RetrievalResult],
    retriever_name: str,
) -> None:
    """Validate structure and integrity of candidate list returned by a child retriever.

    Raises:
        TypeError: If results contains items that are not RetrievalResult instances.
        ValueError: If scores are non-finite, ranks are invalid, or duplicate IDs/ranks exist.
    """
    seen_ids: set[str] = set()
    seen_ranks: set[int] = set()

    for idx, res in enumerate(results):
        if not isinstance(res, RetrievalResult):
            raise TypeError(
                f"Retriever '{retriever_name}' returned non-RetrievalResult item at index {idx}: "
                f"{type(res).__name__}."
            )
        if not math.isfinite(res.score):
            raise ValueError(
                f"Retriever '{retriever_name}' returned non-finite score {res.score} "
                f"for chunk ID '{res.chunk.chunk_id}'."
            )
        if res.rank < 1:
            raise ValueError(
                f"Retriever '{retriever_name}' returned invalid non-positive rank {res.rank}."
            )
        if res.chunk.chunk_id in seen_ids:
            raise ValueError(
                f"Retriever '{retriever_name}' returned duplicate chunk ID "
                f"'{res.chunk.chunk_id}' within a single ranking."
            )
        seen_ids.add(res.chunk.chunk_id)

        if res.rank in seen_ranks:
            raise ValueError(
                f"Retriever '{retriever_name}' returned duplicate rank position {res.rank}."
            )
        seen_ranks.add(res.rank)


class HybridRetriever(BaseRetriever):
    """Hybrid sparse-dense retrieval engine powered by Reciprocal Rank Fusion (RRF).

    Mathematical Formulation:
        For a set of retrievers M and a candidate chunk d:

            RRF(d) = sum_{m in M} ( 1.0 / (k + r_m(d)) )

        where:
        - r_m(d) is the 1-based rank of chunk d returned by retriever m.
        - k is the rank-smoothing constant (default 60).
        - A retriever that did not retrieve chunk d contributes 0.0 to its score.
        - Raw BM25 and dense scores are never summed; they are preserved in metadata only.

    Tie-Breaking Policy:
        In the event of identical fused RRF scores, ties are resolved deterministically:
        1. Higher consensus: items retrieved by both retrievers rank ahead of items from one.
        2. Lower best rank: minimum individual rank across retrievers (min(r_sparse, r_dense)).
        3. Lexicographical chunk ID: ascending chunk_id string for absolute reproducibility.
    """

    def __init__(
        self,
        sparse_retriever: BaseRetriever,
        dense_retriever: BaseRetriever,
        rrf_k: int = DEFAULT_RRF_K,
        name: str = DEFAULT_HYBRID_NAME,
        candidate_depth: int | None = None,
    ) -> None:
        """Initialize HybridRetriever with injected sparse and dense retrievers.

        Args:
            sparse_retriever: BaseRetriever instance for lexical retrieval (e.g. BM25Retriever).
            dense_retriever: BaseRetriever instance for semantic retrieval
                (e.g. FaissVectorRetriever).
            rrf_k: Rank-smoothing constant k (must be a positive integer, defaults to 60).
            name: Identifier for the hybrid retriever (defaults to 'hybrid_rrf').
            candidate_depth: Optional candidate fetch depth requested from child retrievers.
                If None, defaults dynamically to max(top_k, 60).

        Raises:
            TypeError: If arguments have invalid types (including booleans for integers).
            ValueError: If parameters are invalid or out of allowed bounds.
        """
        if not isinstance(sparse_retriever, BaseRetriever):
            raise TypeError(
                f"sparse_retriever must be an instance of BaseRetriever, "
                f"got {type(sparse_retriever).__name__}."
            )
        if not isinstance(dense_retriever, BaseRetriever):
            raise TypeError(
                f"dense_retriever must be an instance of BaseRetriever, "
                f"got {type(dense_retriever).__name__}."
            )

        if isinstance(rrf_k, bool) or not isinstance(rrf_k, int):
            raise TypeError(f"rrf_k must be an integer, got {type(rrf_k).__name__}.")
        if rrf_k < 1:
            raise ValueError(f"rrf_k must be a strictly positive integer >= 1, got {rrf_k}.")

        if not isinstance(name, str):
            raise TypeError(f"name must be a string, got {type(name).__name__}.")
        clean_name = name.strip()
        if not clean_name:
            raise ValueError("name must be a non-empty string.")

        if candidate_depth is not None:
            if isinstance(candidate_depth, bool) or not isinstance(candidate_depth, int):
                raise TypeError(
                    f"candidate_depth must be an integer or None, "
                    f"got {type(candidate_depth).__name__}."
                )
            if candidate_depth < 1:
                raise ValueError(
                    f"candidate_depth must be a positive integer >= 1, got {candidate_depth}."
                )

        self._sparse_retriever = sparse_retriever
        self._dense_retriever = dense_retriever
        self._rrf_k = rrf_k
        self._name = clean_name
        self._candidate_depth = candidate_depth
        self._is_indexed: bool = False

    @property
    def name(self) -> str:
        """Return the unique name identifying this hybrid retriever."""
        return self._name

    @property
    def sparse_retriever(self) -> BaseRetriever:
        """Return the injected sparse retriever."""
        return self._sparse_retriever

    @property
    def dense_retriever(self) -> BaseRetriever:
        """Return the injected dense retriever."""
        return self._dense_retriever

    @property
    def rrf_k(self) -> int:
        """Return the RRF rank-smoothing constant k."""
        return self._rrf_k

    @property
    def candidate_depth(self) -> int | None:
        """Return the configured candidate over-fetching depth, if set."""
        return self._candidate_depth

    @property
    def is_indexed(self) -> bool:
        """Return whether both child retrievers have been successfully indexed."""
        return self._is_indexed

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index a collection of chunks into both child retrievers.

        Both sparse and dense child retrievers are indexed with the identical corpus.
        If either child indexing operation fails, the hybrid retriever is not marked
        as indexed.

        Args:
            chunks: Sequence of Chunk objects to populate into both child indices.

        Raises:
            TypeError: If chunks is not a Sequence or contains items that are not Chunk instances.
        """
        if not isinstance(chunks, Sequence):
            raise TypeError(
                f"chunks must be a Sequence of Chunk objects, got {type(chunks).__name__}."
            )

        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(
                    f"Item at index {idx} in chunks is not a Chunk instance: "
                    f"{type(chunk).__name__}."
                )

        # Reset indexing state before mutating children
        self._is_indexed = False

        self._sparse_retriever.index(chunks)
        self._dense_retriever.index(chunks)

        self._is_indexed = True

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-K ranked chunks for the given query text using RRF.

        Args:
            query: Query string.
            top_k: Maximum number of ranked results to return (must be > 0).

        Returns:
            List of RetrievalResult objects ordered by descending fused score.

        Raises:
            RuntimeError: If retrieve is called before index() has completed successfully.
            TypeError: If query or top_k have invalid types.
            ValueError: If child results contain malformed ranks, non-finite scores,
                internal duplicate IDs, or conflicting chunk provenance.
        """
        if not self._is_indexed:
            raise RuntimeError(
                f"{self.__class__.__name__} has not been indexed yet. "
                "Call index(chunks) before attempting retrieval."
            )

        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")

        if top_k <= 0 or not query.strip():
            return []

        # Determine over-fetching candidate depth
        fetch_k = (
            max(top_k, self._candidate_depth)
            if self._candidate_depth is not None
            else max(top_k, 60)
        )

        sparse_candidates = self._sparse_retriever.retrieve(query, top_k=fetch_k)
        dense_candidates = self._dense_retriever.retrieve(query, top_k=fetch_k)

        _validate_child_results(sparse_candidates, self._sparse_retriever.name)
        _validate_child_results(dense_candidates, self._dense_retriever.name)

        if not sparse_candidates and not dense_candidates:
            return []

        # Build fast lookup maps by chunk_id
        sparse_by_id = {res.chunk.chunk_id: res for res in sparse_candidates}
        dense_by_id = {res.chunk.chunk_id: res for res in dense_candidates}

        # Check for conflicting provenance across common chunk IDs
        common_ids = set(sparse_by_id.keys()) & set(dense_by_id.keys())
        for chunk_id in common_ids:
            c_s = sparse_by_id[chunk_id].chunk
            c_d = dense_by_id[chunk_id].chunk
            if (
                c_s.doc_id != c_d.doc_id
                or c_s.content != c_d.content
                or c_s.page_number != c_d.page_number
                or c_s.char_start != c_d.char_start
                or c_s.char_end != c_d.char_end
                or c_s.metadata != c_d.metadata
            ):
                raise ValueError(
                    f"Conflicting chunk provenance detected for chunk ID '{chunk_id}' across "
                    f"sparse retriever '{self._sparse_retriever.name}' and dense retriever "
                    f"'{self._dense_retriever.name}'."
                )

        # Merge candidates and compute RRF scores
        all_ids: list[str] = list(
            dict.fromkeys(
                [res.chunk.chunk_id for res in sparse_candidates]
                + [res.chunk.chunk_id for res in dense_candidates]
            )
        )

        fused_candidates: list[_FusedCandidate] = []
        for chunk_id in all_ids:
            s_res = sparse_by_id.get(chunk_id)
            d_res = dense_by_id.get(chunk_id)

            canonical_chunk = (s_res or d_res).chunk  # type: ignore[union-attr]

            fused_score = 0.0
            contributions: dict[str, float] = {}

            s_rank = s_res.rank if s_res is not None else None
            s_score = s_res.score if s_res is not None else None
            if s_rank is not None:
                contrib_s = 1.0 / (self._rrf_k + s_rank)
                fused_score += contrib_s
                contributions[self._sparse_retriever.name] = contrib_s

            d_rank = d_res.rank if d_res is not None else None
            d_score = d_res.score if d_res is not None else None
            if d_rank is not None:
                contrib_d = 1.0 / (self._rrf_k + d_rank)
                fused_score += contrib_d
                contributions[self._dense_retriever.name] = contrib_d

            retriever_count = int(s_res is not None) + int(d_res is not None)
            best_rank = min(
                s_rank if s_rank is not None else math.inf,
                d_rank if d_rank is not None else math.inf,
            )

            fused_candidates.append(
                _FusedCandidate(
                    chunk=canonical_chunk,
                    fused_score=fused_score,
                    sparse_rank=s_rank,
                    sparse_score=s_score,
                    dense_rank=d_rank,
                    dense_score=d_score,
                    contributions=contributions,
                    retriever_count=retriever_count,
                    best_rank=int(best_rank),
                )
            )

        # Deterministic sorting:
        # 1. Higher fused score first (-fused_score)
        # 2. Consensus: retrieved by both retrievers first (-retriever_count)
        # 3. Better single best rank (best_rank ascending)
        # 4. Lexicographical chunk_id string ascending
        fused_candidates.sort(
            key=lambda c: (
                -c.fused_score,
                -c.retriever_count,
                c.best_rank,
                c.chunk.chunk_id,
            )
        )

        top_candidates = fused_candidates[:top_k]

        results: list[RetrievalResult] = []
        for rank_idx, cand in enumerate(top_candidates, start=1):
            metadata: dict[str, Any] = {
                "rrf_k": self._rrf_k,
                "sparse_retriever_name": self._sparse_retriever.name,
                "dense_retriever_name": self._dense_retriever.name,
                "sparse_rank": cand.sparse_rank,
                "sparse_score": cand.sparse_score,
                "dense_rank": cand.dense_rank,
                "dense_score": cand.dense_score,
                "rrf_contributions": cand.contributions,
            }
            results.append(
                RetrievalResult(
                    chunk=cand.chunk,
                    score=float(cand.fused_score),
                    rank=rank_idx,
                    retriever_name=self.name,
                    metadata=metadata,
                )
            )

        return results
