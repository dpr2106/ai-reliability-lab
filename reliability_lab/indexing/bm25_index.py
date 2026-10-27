"""In-memory Okapi BM25 lexical retrieval engine.

Implements Okapi BM25 ranking over citation-aware Chunk domain entities
using the Python standard library only, per repository architecture and ADR-005.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from collections.abc import Sequence
from typing import Any

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, RetrievalResult

_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


class BM25Retriever(BaseRetriever):
    """In-memory Okapi BM25 lexical retriever.

    Formulation:
        For a collection of N chunks, the BM25 score of chunk D for query Q is:

            BM25(D, Q) = sum_{t in Q_distinct} IDF(t) * (f(t, D) * (k1 + 1))
                         / (f(t, D) + k1 * (1 - b + b * (|D| / avgdl)))

        where IDF(t) uses the standard non-negative Robertson-Spärck Jones formulation:

            IDF(t) = ln(1 + (N - df(t) + 0.5) / (df(t) + 0.5))

        - N: total number of indexed chunks
        - df(t): number of chunks containing term t
        - f(t, D): term frequency of t in chunk D
        - |D|: token count of chunk D
        - avgdl: mean token count across all indexed chunks in the corpus

    Attributes:
        k1: Term frequency saturation parameter (k1 > 0). Defaults to 1.5.
        b: Document length normalization parameter (0 <= b <= 1). Defaults to 0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        """Initialize BM25Retriever with configurable k1 and b parameters.

        Args:
            k1: Term frequency saturation constant (> 0).
            b: Document length normalization constant (0.0 to 1.0).

        Raises:
            TypeError: If k1 or b are not numeric or are booleans.
            ValueError: If k1 <= 0, b < 0, b > 1, or values are non-finite.
        """
        if isinstance(k1, bool) or not isinstance(k1, (int, float)):
            raise TypeError(f"k1 must be a finite float or int, got {type(k1).__name__}.")
        if not math.isfinite(k1) or k1 <= 0:
            raise ValueError(f"k1 must be a strictly positive finite number, got {k1}.")

        if isinstance(b, bool) or not isinstance(b, (int, float)):
            raise TypeError(f"b must be a finite float or int, got {type(b).__name__}.")
        if not math.isfinite(b) or b < 0 or b > 1:
            raise ValueError(f"b must be a finite number in the range [0.0, 1.0], got {b}.")

        self.k1 = float(k1)
        self.b = float(b)

        self._is_indexed: bool = False
        self._chunks: list[Chunk] = []
        self._doc_lengths: list[int] = []
        self._avgdl: float = 0.0
        self._doc_frequencies: dict[str, int] = {}
        self._idfs: dict[str, float] = {}
        # Inverted index: term -> list of (doc_index, term_frequency)
        self._inverted_index: dict[str, list[tuple[int, int]]] = defaultdict(list)
        # Fast lookup set of doc indices per term for metadata reporting
        self._postings_by_doc: dict[str, set[int]] = defaultdict(set)

    @property
    def name(self) -> str:
        """Return unique algorithm identifier."""
        return "bm25"

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Normalize and tokenize text into words using Unicode-aware matching.

        Performs casefold normalization without stemming or arbitrary stopword
        filtering to keep lexical evaluation transparent and reproducible.

        Args:
            text: Input string to tokenize.

        Returns:
            List of normalized token strings.
        """
        if not text:
            return []
        return _TOKEN_PATTERN.findall(text.casefold())

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index a sequence of Chunk entities replacing any prior index state.

        Args:
            chunks: Sequence of Chunk domain objects to index.

        Raises:
            TypeError: If chunks is not a Sequence or contains non-Chunk items.
        """
        if not isinstance(chunks, Sequence):
            raise TypeError("chunks must be a sequence of Chunk objects.")

        validated_chunks: list[Chunk] = []
        for idx, chunk in enumerate(chunks):
            if not isinstance(chunk, Chunk):
                raise TypeError(f"Item at index {idx} is not an instance of Chunk: {type(chunk)}")
            validated_chunks.append(chunk)

        # Reset index state
        self._chunks = validated_chunks
        self._doc_lengths = []
        self._doc_frequencies = {}
        self._idfs = {}
        self._inverted_index = defaultdict(list)
        self._postings_by_doc = defaultdict(set)
        self._is_indexed = True

        num_docs = len(self._chunks)
        if num_docs == 0:
            self._avgdl = 0.0
            return

        total_tokens = 0
        doc_freq_counts: Counter[str] = Counter()

        for doc_idx, chunk in enumerate(self._chunks):
            tokens = self.tokenize(chunk.content)
            doc_len = len(tokens)
            self._doc_lengths.append(doc_len)
            total_tokens += doc_len

            term_counts = Counter(tokens)
            for term, tf in term_counts.items():
                self._inverted_index[term].append((doc_idx, tf))
                self._postings_by_doc[term].add(doc_idx)
                doc_freq_counts[term] += 1

        self._avgdl = total_tokens / num_docs if num_docs > 0 else 0.0
        self._doc_frequencies = dict(doc_freq_counts)

        # Precompute Robertson-Spärck Jones IDF for all vocabulary terms
        for term, df in self._doc_frequencies.items():
            self._idfs[term] = math.log(1.0 + (num_docs - df + 0.5) / (df + 0.5))

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve and rank the top_k chunks matching query using Okapi BM25.

        Args:
            query: Query string to evaluate.
            top_k: Maximum number of ranked results to return (default 5).

        Returns:
            Ordered list of RetrievalResult entities ranked by descending score.
            Ties in score are deterministically broken using original corpus order.

        Raises:
            RuntimeError: If retrieve() is called before index() was executed.
            TypeError: If query is not a string or top_k is not an integer.
            ValueError: If top_k is negative.
        """
        if not self._is_indexed:
            raise RuntimeError("retrieve() called before index() was executed.")

        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")

        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")

        if top_k < 0:
            raise ValueError(f"top_k must be a non-negative integer, got {top_k}.")

        if top_k == 0 or len(self._chunks) == 0:
            return []

        query_tokens = self.tokenize(query)
        if not query_tokens:
            return []

        # Distinct normalized query terms present in the vocabulary
        distinct_terms = [
            term for term in dict.fromkeys(query_tokens) if term in self._inverted_index
        ]
        if not distinct_terms:
            # Out-of-vocabulary query returns empty list
            return []

        doc_scores: dict[int, float] = defaultdict(float)

        for term in distinct_terms:
            idf = self._idfs[term]
            postings = self._inverted_index[term]

            for doc_idx, tf in postings:
                doc_len = self._doc_lengths[doc_idx]
                len_ratio = (doc_len / self._avgdl) if self._avgdl > 0.0 else 1.0
                denominator = tf + self.k1 * (1.0 - self.b + self.b * len_ratio)
                numerator = tf * (self.k1 + 1.0)
                term_score = idf * (numerator / denominator)
                doc_scores[doc_idx] += term_score

        # Only return documents with strictly positive scores (at least one term match)
        positive_scored_docs = [
            (doc_idx, score) for doc_idx, score in doc_scores.items() if score > 0.0
        ]
        if not positive_scored_docs:
            return []

        # Sort descending by score; tie-break deterministically ascending by doc_idx
        sorted_candidates = sorted(positive_scored_docs, key=lambda item: (-item[1], item[0]))
        top_candidates = sorted_candidates[:top_k]

        results: list[RetrievalResult] = []
        for rank_idx, (doc_idx, score) in enumerate(top_candidates, start=1):
            chunk = self._chunks[doc_idx]
            matched_terms = [
                term for term in distinct_terms if doc_idx in self._postings_by_doc[term]
            ]
            metadata: dict[str, Any] = {
                "doc_length": self._doc_lengths[doc_idx],
                "matched_terms": matched_terms,
                "k1": self.k1,
                "b": self.b,
            }
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(score),
                    rank=rank_idx,
                    retriever_name=self.name,
                    metadata=metadata,
                )
            )

        return results
