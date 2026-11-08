"""Domain models for RAG Evaluation Datasets and Benchmarks."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from reliability_lab.core.models import Query


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class BenchmarkDatasetRecord:
    """Header record summarizing a benchmark evaluation dataset."""

    dataset_id: str
    name: str
    description: str | None = None
    document_scope: list[str] = field(default_factory=list)
    query_count: int = 0
    is_synthetic: bool = False
    created_at: str = field(default_factory=_utc_now_iso)
    updated_at: str = field(default_factory=_utc_now_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_id": self.dataset_id,
            "name": self.name,
            "description": self.description,
            "document_scope": list(self.document_scope),
            "query_count": self.query_count,
            "is_synthetic": self.is_synthetic,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> BenchmarkDatasetRecord:
        return cls(
            dataset_id=str(data["dataset_id"]),
            name=str(data["name"]),
            description=data.get("description"),
            document_scope=list(data.get("document_scope", [])),
            query_count=int(data.get("query_count", 0)),
            is_synthetic=bool(data.get("is_synthetic", False)),
            created_at=str(data.get("created_at") or _utc_now_iso()),
            updated_at=str(data.get("updated_at") or _utc_now_iso()),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class BenchmarkDataset:
    """Complete evaluation dataset including its constituent queries."""

    record: BenchmarkDatasetRecord
    queries: list[Query] = field(default_factory=list)

    @property
    def dataset_id(self) -> str:
        return self.record.dataset_id

    @property
    def name(self) -> str:
        return self.record.name

    @property
    def query_count(self) -> int:
        return len(self.queries)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.record.to_dict(),
            "query_count": len(self.queries),
            "queries": [q.to_dict() for q in self.queries],
        }

    def to_jsonl(self) -> str:
        """Serialize benchmark dataset to standard JSON Lines format."""
        lines: list[str] = []
        for q in self.queries:
            judgments: list[dict[str, Any]] = []
            for cid in q.relevant_chunk_ids:
                grade = q.get_grade(cid)
                judgments.append({"chunk_id": cid, "grade": grade})
            for cid, grade in q.relevance_grades.items():
                if cid not in q.relevant_chunk_ids:
                    judgments.append({"chunk_id": cid, "grade": grade})

            required_citations: list[dict[str, Any]] | list[str]
            citation_records = q.metadata.get("citation_records")
            if isinstance(citation_records, list):
                required_citations = citation_records
            elif q.metadata.get("citation_to_chunk_id"):
                c_map = q.metadata["citation_to_chunk_id"]
                required_citations = [
                    {"citation_id": cit_id, "chunk_id": c_map.get(cit_id, cit_id)}
                    for cit_id in q.required_citations
                ]
            else:
                required_citations = list(q.required_citations)

            q_dict = {
                "query_id": q.query_id,
                "query": q.text,
                "text": q.text,
                "ground_truth_answer": q.ground_truth_answer,
                "relevant_chunk_ids": list(q.relevant_chunk_ids),
                "relevance_grades": dict(q.relevance_grades),
                "relevance_judgments": judgments,
                "required_citations": required_citations,
                "metadata": {
                    k: v
                    for k, v in q.metadata.items()
                    if k not in ("citation_records", "citation_to_chunk_id")
                },
            }
            lines.append(json.dumps(q_dict, ensure_ascii=False))
        return "\n".join(lines) + ("\n" if lines else "")
