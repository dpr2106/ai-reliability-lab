"""Metadata, definitions, and limitations for STING evaluation metrics."""

from __future__ import annotations

from typing import Any

METRIC_DEFINITIONS: dict[str, dict[str, Any]] = {
    "Recall@K": {
        "name": "Recall@K",
        "category": "retrieval",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "retrieval",
        "method": "Set-based fraction of gold-standard relevant chunks present in top-K.",
        "description": "Measures what fraction of required evidence chunks were retrieved.",
        "limitations": "Does not account for ranking order of chunks within the top-K cutoff.",
    },
    "Precision@K": {
        "name": "Precision@K",
        "category": "retrieval",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "retrieval",
        "method": "Set-based fraction of retrieved top-K chunks that are relevant.",
        "description": "Measures retrieval precision; penalizes irrelevant distractor chunks.",
        "limitations": "Can be artificially low if gold annotations are sparse or incomplete.",
    },
    "HitRate@K": {
        "name": "HitRate@K",
        "category": "retrieval",
        "direction": "higher",
        "range": "{0.0, 1.0}",
        "source": "retrieval",
        "method": (
            "Binary indicator: 1.0 if at least one relevant chunk is in top-K, 0.0 otherwise."
        ),
        "description": "Measures whether the retriever surfaced any relevant evidence at all.",
        "limitations": "Insensitive to multi-hop evidence requirements where all chunks needed.",
    },
    "MRR@K": {
        "name": "MRR@K",
        "category": "retrieval",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "retrieval",
        "method": "Reciprocal rank: 1 / rank of first relevant chunk, or 0.0 if none retrieved.",
        "description": "Evaluates how high the first relevant chunk appears in the ranking.",
        "limitations": "Only evaluates the position of the single highest-ranked relevant chunk.",
    },
    "NDCG@K": {
        "name": "NDCG@K",
        "category": "retrieval",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "retrieval",
        "method": ("Discounted Cumulative Gain normalized by ideal ranking over graded relevance."),
        "description": (
            "Evaluates graded relevance ranking quality with logarithmic position discounting."
        ),
        "limitations": "Requires multi-grade relevance annotations to distinguish quality.",
    },
    "citation_precision": {
        "name": "Citation Precision",
        "category": "citation",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "citation",
        "method": "Fraction of generated citations matching required gold-standard citations.",
        "description": "Measures whether citations in the answer point to genuine evidence.",
        "limitations": "Penalizes citations to alternate chunks with equivalent information.",
    },
    "citation_recall": {
        "name": "Citation Recall",
        "category": "citation",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "citation",
        "method": ("Fraction of required gold-standard citations generated in the answer."),
        "description": "Measures whether all necessary source citations were attributed.",
        "limitations": "Cannot verify if the cited chunk is syntactically formatted.",
    },
    "lexical_groundedness": {
        "name": "Lexical Groundedness",
        "category": "groundedness",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "groundedness",
        "method": "N-gram containment and longest common subsequence overlap against evidence.",
        "description": "Fast deterministic lexical diagnostic measuring passage grounding.",
        "limitations": "Cannot detect semantic grounding when answer uses synonyms.",
    },
    "nli_entailment": {
        "name": "NLI Entailment",
        "category": "nli",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "nli",
        "method": "DeBERTa natural language inference entailment probability score.",
        "description": "Deep cross-encoder verifying answer logically follows from evidence.",
        "limitations": "Computationally intensive; falls back to lexical diagnostics if offline.",
    },
    "judge_faithfulness": {
        "name": "Judge Faithfulness",
        "category": "judge",
        "direction": "higher",
        "range": "[0.0, 1.0]",
        "source": "judge",
        "method": "Structured LLM-as-a-judge prompt assessing claim-by-claim groundedness.",
        "description": "Semantic audit checking whether generated claims are supported.",
        "limitations": "Dependent on judge model reasoning capabilities and temperature.",
    },
    "retrieval_latency_ms": {
        "name": "Retrieval Latency",
        "category": "performance",
        "direction": "lower",
        "range": "[0, inf) ms",
        "source": "performance",
        "method": "Wall-clock time in milliseconds elapsed during candidate retrieval.",
        "description": "Measures how fast the retrieval stage locates top-K candidates.",
        "limitations": "Subject to OS caching and hardware load variances.",
    },
    "generation_latency_ms": {
        "name": "Generation Latency",
        "category": "performance",
        "direction": "lower",
        "range": "[0, inf) ms",
        "source": "performance",
        "method": "Wall-clock time in milliseconds elapsed during LLM answer generation.",
        "description": "Measures how fast the LLM produces the conversational answer.",
        "limitations": "Correlates directly with output token length and provider queue latency.",
    },
    "total_latency_ms": {
        "name": "Total Latency",
        "category": "performance",
        "direction": "lower",
        "range": "[0, inf) ms",
        "source": "performance",
        "method": "End-to-end elapsed time per query turn in milliseconds.",
        "description": "Total user-perceived turnaround time for retrieval and generation.",
        "limitations": "Includes evaluation diagnostic calculations if run synchronously.",
    },
}


def classify_query_failure(q_rec: dict[str, Any]) -> dict[str, Any]:
    """Classify the primary failure mode and diagnostic explanation of an evaluated query."""
    metrics = q_rec.get("metrics", {}) or {}

    hit_rate = metrics.get("HitRate@K", metrics.get("hit_rate", 1.0))
    recall = metrics.get("Recall@K", metrics.get("recall", 1.0))
    cit_prec = metrics.get("citation_precision")
    cit_rec = metrics.get("citation_recall")
    groundedness = metrics.get("lexical_groundedness")
    faithfulness = metrics.get("judge_faithfulness")

    classification = "passed"
    severity = "none"
    explanation = "All measured evaluation criteria passed without significant defect."

    if hit_rate is not None and float(hit_rate) == 0.0:
        classification = "retrieval_miss"
        severity = "critical"
        explanation = "Retriever failed to surface any gold-standard relevant chunks in top-K."
    elif recall is not None and float(recall) < 1.0:
        classification = "partial_retrieval"
        severity = "moderate"
        explanation = f"Retriever missed relevant evidence (Recall@K: {float(recall):.2f})."
    elif cit_prec is not None and float(cit_prec) < 1.0:
        classification = "citation_hallucination"
        severity = "high"
        explanation = "Generated answer cited irrelevant or ungrounded chunk IDs."
    elif cit_rec is not None and float(cit_rec) < 1.0:
        classification = "citation_omission"
        severity = "moderate"
        explanation = "Generated answer omitted required evidence citations."
    elif (groundedness is not None and float(groundedness) < 0.5) or (
        faithfulness is not None and float(faithfulness) < 0.5
    ):
        classification = "ungrounded_generation"
        severity = "high"
        explanation = "Generated answer claims lack sufficient grounding in retrieved passages."

    return {
        "classification": classification,
        "severity": severity,
        "explanation": explanation,
        "has_failure": classification != "passed",
    }
