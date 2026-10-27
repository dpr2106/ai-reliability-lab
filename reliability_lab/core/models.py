"""Core domain models for AI Reliability Lab.

Implemented using Python standard library only, per ADR-005.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class Document:
    """Represents a source document prior to chunking."""

    doc_id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    page_count: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")
        if not isinstance(self.content, str):
            raise TypeError("content must be a string.")
        if self.page_count is not None:
            if not isinstance(self.page_count, int) or self.page_count < 1:
                raise ValueError("page_count must be an integer >= 1 when provided.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize document to a dictionary."""
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": dict(self.metadata),
            "page_count": self.page_count,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Document:
        """Construct Document from a dictionary."""
        return cls(
            doc_id=str(data["doc_id"]),
            content=str(data["content"]),
            metadata=dict(data.get("metadata", {})),
            page_count=data.get("page_count"),
        )

    def to_json(self) -> str:
        """Serialize document to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Document:
        """Construct Document from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class Chunk:
    """Represents an atomic text chunk with provenance tracking for citation auditing."""

    chunk_id: str
    doc_id: str
    content: str
    page_number: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk_id, str) or not self.chunk_id.strip():
            raise ValueError("chunk_id must be a non-empty string.")
        if not isinstance(self.doc_id, str) or not self.doc_id.strip():
            raise ValueError("doc_id must be a non-empty string.")
        if not isinstance(self.content, str) or not self.content.strip():
            raise ValueError("content must be a non-empty string.")
        if self.page_number is not None:
            if not isinstance(self.page_number, int) or self.page_number < 1:
                raise ValueError("page_number must be an integer >= 1 when provided.")
        if self.char_start is not None:
            if not isinstance(self.char_start, int) or self.char_start < 0:
                raise ValueError("char_start must be an integer >= 0 when provided.")
        if self.char_end is not None:
            if not isinstance(self.char_end, int) or self.char_end < 0:
                raise ValueError("char_end must be an integer >= 0 when provided.")
        if self.char_start is not None and self.char_end is not None:
            if self.char_end < self.char_start:
                raise ValueError(
                    f"char_end ({self.char_end}) must be >= char_start ({self.char_start})."
                )
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize chunk to a dictionary."""
        return {
            "chunk_id": self.chunk_id,
            "doc_id": self.doc_id,
            "content": self.content,
            "page_number": self.page_number,
            "char_start": self.char_start,
            "char_end": self.char_end,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Chunk:
        """Construct Chunk from a dictionary."""
        return cls(
            chunk_id=str(data["chunk_id"]),
            doc_id=str(data["doc_id"]),
            content=str(data["content"]),
            page_number=data.get("page_number"),
            char_start=data.get("char_start"),
            char_end=data.get("char_end"),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        """Serialize chunk to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Chunk:
        """Construct Chunk from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class Query:
    """Represents an evaluation query with ground-truth relevance annotations."""

    query_id: str
    text: str
    relevant_chunk_ids: list[str] = field(default_factory=list)
    relevance_grades: dict[str, int] = field(default_factory=dict)
    ground_truth_answer: str | None = None
    required_citations: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.query_id, str) or not self.query_id.strip():
            raise ValueError("query_id must be a non-empty string.")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string.")
        if not isinstance(self.relevant_chunk_ids, list):
            raise TypeError("relevant_chunk_ids must be a list of strings.")
        for cid in self.relevant_chunk_ids:
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("All relevant_chunk_ids must be non-empty strings.")
        if not isinstance(self.relevance_grades, dict):
            raise TypeError("relevance_grades must be a dictionary mapping chunk_id to int.")
        for cid, grade in self.relevance_grades.items():
            if not isinstance(cid, str) or not cid.strip():
                raise ValueError("relevance_grades keys must be non-empty strings.")
            if not isinstance(grade, int) or grade < 0:
                raise ValueError(f"relevance grade for {cid} must be a non-negative integer.")
        if self.ground_truth_answer is not None and not isinstance(self.ground_truth_answer, str):
            raise TypeError("ground_truth_answer must be a string or None.")
        if not isinstance(self.required_citations, list):
            raise TypeError("required_citations must be a list of strings.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    def get_grade(self, chunk_id: str) -> int:
        """Return the relevance grade of a chunk (default 1 if in relevant_chunk_ids, else 0)."""
        if chunk_id in self.relevance_grades:
            return self.relevance_grades[chunk_id]
        if chunk_id in self.relevant_chunk_ids:
            return 1
        return 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize query to a dictionary."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "relevant_chunk_ids": list(self.relevant_chunk_ids),
            "relevance_grades": dict(self.relevance_grades),
            "ground_truth_answer": self.ground_truth_answer,
            "required_citations": list(self.required_citations),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Query:
        """Construct Query from a dictionary."""
        return cls(
            query_id=str(data["query_id"]),
            text=str(data["text"]),
            relevant_chunk_ids=list(data.get("relevant_chunk_ids", [])),
            relevance_grades={str(k): int(v) for k, v in data.get("relevance_grades", {}).items()},
            ground_truth_answer=data.get("ground_truth_answer"),
            required_citations=list(data.get("required_citations", [])),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        """Serialize query to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> Query:
        """Construct Query from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass(frozen=True)
class RetrievalResult:
    """Represents a retrieved chunk candidate with score and ranking metadata."""

    chunk: Chunk
    score: float
    rank: int
    retriever_name: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.chunk, Chunk):
            raise TypeError("chunk must be an instance of Chunk.")
        if not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError(f"score must be a finite float, got: {self.score}")
        if not isinstance(self.rank, int) or self.rank < 1:
            raise ValueError(f"rank must be a positive 1-indexed integer >= 1, got: {self.rank}")
        if not isinstance(self.retriever_name, str) or not self.retriever_name.strip():
            raise ValueError("retriever_name must be a non-empty string.")
        if not isinstance(self.metadata, dict):
            raise TypeError("metadata must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize retrieval result to a dictionary."""
        return {
            "chunk": self.chunk.to_dict(),
            "score": float(self.score),
            "rank": self.rank,
            "retriever_name": self.retriever_name,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> RetrievalResult:
        """Construct RetrievalResult from a dictionary, reconstructing the Chunk."""
        chunk_data = data["chunk"]
        chunk = chunk_data if isinstance(chunk_data, Chunk) else Chunk.from_dict(chunk_data)
        return cls(
            chunk=chunk,
            score=float(data["score"]),
            rank=int(data["rank"]),
            retriever_name=str(data["retriever_name"]),
            metadata=dict(data.get("metadata", {})),
        )

    def to_json(self) -> str:
        """Serialize retrieval result to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> RetrievalResult:
        """Construct RetrievalResult from a JSON string."""
        return cls.from_dict(json.loads(json_str))


@dataclass
class EvaluationReport:
    """Represents the outcome of a benchmark evaluation run."""

    experiment_name: str
    metrics: dict[str, float]
    report_id: str = field(default_factory=lambda: str(uuid4()))
    per_query_results: list[dict[str, Any]] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    config: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.report_id, str) or not self.report_id.strip():
            raise ValueError("report_id must be a non-empty string.")
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string.")
        if not isinstance(self.metrics, dict):
            raise TypeError("metrics must be a dictionary of metric names to finite floats.")
        for name, value in self.metrics.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("Metric names must be non-empty strings.")
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(
                    f"Metric '{name}' must have a finite numerical value, got: {value}"
                )
        if not isinstance(self.per_query_results, list):
            raise TypeError("per_query_results must be a list.")
        if not isinstance(self.config, dict):
            raise TypeError("config must be a dictionary.")

    def to_dict(self) -> dict[str, Any]:
        """Serialize evaluation report to a dictionary."""
        return {
            "report_id": self.report_id,
            "experiment_name": self.experiment_name,
            "metrics": {k: float(v) for k, v in self.metrics.items()},
            "per_query_results": [dict(r) for r in self.per_query_results],
            "timestamp": self.timestamp,
            "config": dict(self.config),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationReport:
        """Construct EvaluationReport from a dictionary."""
        return cls(
            report_id=str(data.get("report_id", str(uuid4()))),
            experiment_name=str(data["experiment_name"]),
            metrics={str(k): float(v) for k, v in data["metrics"].items()},
            per_query_results=list(data.get("per_query_results", [])),
            timestamp=str(data.get("timestamp", datetime.now(timezone.utc).isoformat())),
            config=dict(data.get("config", {})),
        )

    def to_json(self) -> str:
        """Serialize evaluation report to a JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @classmethod
    def from_json(cls, json_str: str) -> EvaluationReport:
        """Construct EvaluationReport from a JSON string."""
        return cls.from_dict(json.loads(json_str))
