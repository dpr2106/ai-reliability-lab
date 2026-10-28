"""Evaluation metrics, diagnostic runners, and benchmark reporting package."""

from reliability_lab.evaluation.benchmark_runner import (
    BenchmarkRunner,
    RetrievalBenchmarkRunner,
    calculate_latency_percentiles,
    load_benchmark_dataset,
    load_corpus,
)
from reliability_lab.evaluation.citation_metrics import (
    BaseCitationMetric,
    CitationEvaluator,
    CitationMarker,
    CitationParser,
    CitationPrecision,
    CitationRecall,
    CitationValidationResult,
    calculate_citation_precision,
    calculate_citation_recall,
    extract_citation_ids,
    parse_citation_markers,
    resolve_citation_reference,
)
from reliability_lab.evaluation.retrieval_metrics import (
    BaseSetRetrievalMetric,
    HitRateAtK,
    MRRAtK,
    NDCGAtK,
    PrecisionAtK,
    RecallAtK,
)

__all__ = [
    "BaseCitationMetric",
    "BaseSetRetrievalMetric",
    "BenchmarkRunner",
    "CitationEvaluator",
    "CitationMarker",
    "CitationParser",
    "CitationPrecision",
    "CitationRecall",
    "CitationValidationResult",
    "HitRateAtK",
    "MRRAtK",
    "NDCGAtK",
    "PrecisionAtK",
    "RecallAtK",
    "RetrievalBenchmarkRunner",
    "calculate_citation_precision",
    "calculate_citation_recall",
    "calculate_latency_percentiles",
    "extract_citation_ids",
    "load_benchmark_dataset",
    "load_corpus",
    "parse_citation_markers",
    "resolve_citation_reference",
]
