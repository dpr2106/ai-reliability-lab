"""Core abstract interfaces and protocols for AI Reliability Lab.

Implemented using Python standard library only, per ADR-005.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from reliability_lab.core.models import Chunk, Document, Query, RetrievalResult


class BaseRetriever(ABC):
    """Abstract interface for all sparse, dense, and hybrid retrieval engines."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifying the retriever algorithm or index."""
        ...

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index a collection of chunks into the retrieval engine."""
        ...

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-K ranked chunks for the given query text."""
        ...


class BaseChunker(ABC):
    """Abstract interface for document segmenters and chunking strategies."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique name identifying the chunking strategy."""
        ...

    @abstractmethod
    def chunk(self, document: Document) -> list[Chunk]:
        """Split a source document into an ordered sequence of citation-aware Chunks."""
        ...


class BaseMetric(ABC):
    """Abstract interface for evaluation metrics (retrieval ranking, groundedness, citation)."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Return the unique identifier of the metric (e.g. 'recall@5', 'mrr@10')."""
        ...

    @abstractmethod
    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute the evaluation score for a single query and its retrieved candidates."""
        ...
