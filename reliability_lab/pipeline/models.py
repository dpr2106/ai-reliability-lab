"""Domain models, configurations, and result containers for the RAG evaluation pipeline.

Implemented using Python standard library only, per ADR-005 and ADR-011.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from reliability_lab.core.models import EvaluationReport
from reliability_lab.llm.judge import CorrelationReport


@dataclass(frozen=True)
class GenerationProvenance:
    """Provenance tracking for LLM answer generation."""

    provider: str
    model: str
    is_mock: bool
    prompt_version: str
    latency_ms: float
    token_usage: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        """Convert provenance metadata to a JSON-serializable dictionary."""
        return {
            "provider": self.provider,
            "model": self.model,
            "is_mock": self.is_mock,
            "prompt_version": self.prompt_version,
            "latency_ms": float(self.latency_ms),
            "token_usage": dict(self.token_usage),
        }


@dataclass(frozen=True)
class RAGPipelineConfig:
    """Configuration options for RAGEvaluationRunner execution."""

    top_k: int = 5
    retriever_name: str = "BM25Retriever"
    prompt_version: str = "v1"
    temperature: float = 0.0
    max_output_tokens: int = 1024
    enable_groundedness: bool = True
    enable_citations: bool = True
    enable_nli: bool = True
    enable_judge: bool = True
    experiment_name: str = "rag_evaluation"
    groundedness_containment_threshold: float = 0.5
    groundedness_lcs_threshold: float = 0.3

    def __post_init__(self) -> None:
        if isinstance(self.top_k, bool) or not isinstance(self.top_k, int) or self.top_k <= 0:
            raise ValueError(f"top_k must be a positive integer >= 1, got {self.top_k}")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(self.temperature)
            or self.temperature < 0.0
        ):
            raise ValueError(
                f"temperature must be a finite non-negative float, got {self.temperature}"
            )
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens <= 0
        ):
            raise ValueError(
                f"max_output_tokens must be a positive integer, got {self.max_output_tokens}"
            )
        if not isinstance(self.experiment_name, str) or not self.experiment_name.strip():
            raise ValueError("experiment_name must be a non-empty string.")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to a JSON-serializable dictionary."""
        return {
            "top_k": self.top_k,
            "retriever_name": self.retriever_name,
            "prompt_version": self.prompt_version,
            "temperature": self.temperature,
            "max_output_tokens": self.max_output_tokens,
            "enable_groundedness": self.enable_groundedness,
            "enable_citations": self.enable_citations,
            "enable_nli": self.enable_nli,
            "enable_judge": self.enable_judge,
            "experiment_name": self.experiment_name,
            "groundedness_containment_threshold": self.groundedness_containment_threshold,
            "groundedness_lcs_threshold": self.groundedness_lcs_threshold,
        }


@dataclass(frozen=True)
class PerQueryRAGResult:
    """Complete evaluation vertical slice for a single query."""

    query_id: str
    text: str
    retrieved: list[dict[str, Any]]
    retrieval_metrics: dict[str, float]
    generated_answer: str
    generation_provenance: GenerationProvenance
    groundedness: dict[str, Any] | None = None
    citation_validation: dict[str, Any] | None = None
    nli_result: dict[str, Any] | None = None
    judge_result: dict[str, Any] | None = None
    latency_ms: float = 0.0
    metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert per-query result into a dictionary suitable for storage diagnostics."""
        return {
            "query_id": self.query_id,
            "text": self.text,
            "retrieved": list(self.retrieved),
            "retrieval_metrics": dict(self.retrieval_metrics),
            "generated_answer": self.generated_answer,
            "generation_provenance": self.generation_provenance.to_dict(),
            "groundedness": dict(self.groundedness) if self.groundedness else None,
            "citation_validation": (
                dict(self.citation_validation) if self.citation_validation else None
            ),
            "nli_result": dict(self.nli_result) if self.nli_result else None,
            "judge_result": dict(self.judge_result) if self.judge_result else None,
            "latency_ms": float(self.latency_ms),
            "metrics": {str(k): float(v) for k, v in self.metrics.items()},
        }


@dataclass(frozen=True)
class RAGPipelineExecutionResult:
    """Execution wrapper containing canonical EvaluationReport and granular details."""

    report: EvaluationReport
    correlation: CorrelationReport | None
    is_mock: bool
    total_queries: int
    queries_evaluated: list[PerQueryRAGResult]
    metrics: dict[str, float]
    config: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert execution result into a JSON-serializable dictionary."""
        return {
            "report_id": self.report.report_id,
            "experiment_name": self.report.experiment_name,
            "is_mock": self.is_mock,
            "total_queries": self.total_queries,
            "metrics": dict(self.metrics),
            "correlation": self.correlation.to_dict() if self.correlation else None,
            "config": dict(self.config),
            "queries": [q.to_dict() for q in self.queries_evaluated],
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Serialize result to a formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)
