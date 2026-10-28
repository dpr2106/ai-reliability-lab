"""Unit tests for local NLI entailment evaluator and hallucination risk scorer.

Validates GROUND-003 requirements:
- Injected backend inference with deterministic probability distributions
- Dynamic id2label mapping across varied label orders and casing
- Rejection of invalid/ambiguous model label mappings and non-finite outputs
- Independent (premise, hypothesis) sequence-pair construction with correct ordering
- Exact claim span reuse from groundedness segmenter
- Maximum-entailment evidence selection and deterministic multi-key tie-breaking
- Hand-calculated unsupportedness risks and aggregate Hallucination Score
- Empty answer and empty evidence contracts
- Conflicting chunk provenance rejection and defensive input validation
- JSON serializability and round-trip consistency
- Lazy loading: importing evaluation package does not initialize neural models
- Integration with BaseMetric via NLIHallucinationMetric adapter
"""

import json
from collections.abc import Sequence
from typing import Any

import pytest

from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.citation_metrics import CitationEvaluator
from reliability_lab.evaluation.groundedness import GroundednessEvaluator
from reliability_lab.evaluation.nli_evaluator import (
    BaseNLIBackend,
    ClaimNLIResult,
    HuggingFaceNLIBackend,
    NLIEntailmentEvaluator,
    NLIHallucinationMetric,
    NLILabel,
    NLIResult,
    validate_nli_distribution,
)


class MockNLIBackend(BaseNLIBackend):
    """Deterministic test backend with configurable pair distributions."""

    def __init__(
        self,
        rule_fn: Any | None = None,
        default_dist: dict[NLILabel, float] | None = None,
    ) -> None:
        self.recorded_pairs: list[tuple[str, str]] = []
        self.rule_fn = rule_fn
        self.default_dist = default_dist or {
            NLILabel.ENTAILMENT: 0.8,
            NLILabel.NEUTRAL: 0.15,
            NLILabel.CONTRADICTION: 0.05,
        }

    def predict_batch(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[dict[NLILabel, float]]:
        self.recorded_pairs.extend(pairs)
        results: list[dict[NLILabel, float]] = []
        for premise, hypothesis in pairs:
            if self.rule_fn is not None:
                dist = self.rule_fn(premise, hypothesis)
            else:
                dist = dict(self.default_dist)
            results.append(dist)
        return results


def _make_result(
    chunk_id: str,
    content: str,
    doc_id: str | None = None,
    score: float = 1.0,
    rank: int = 1,
) -> RetrievalResult:
    """Helper to construct RetrievalResult with specific chunk content."""
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id or f"doc-{chunk_id}",
        content=content,
    )
    return RetrievalResult(
        chunk=chunk,
        score=score,
        rank=rank,
        retriever_name="bm25",
    )


# ---------------------------------------------------------------------------
# Label Mapping & Resolution Tests
# ---------------------------------------------------------------------------


def test_resolve_id2label_deberta_order() -> None:
    # Standard DeBERTa MNLI order: 0: contradiction, 1: neutral, 2: entailment
    id2label = {0: "contradiction", 1: "neutral", 2: "entailment"}
    resolved = HuggingFaceNLIBackend.resolve_id2label_mapping(id2label)
    assert resolved == {
        0: NLILabel.CONTRADICTION,
        1: NLILabel.NEUTRAL,
        2: NLILabel.ENTAILMENT,
    }


def test_resolve_id2label_roberta_order_uppercase() -> None:
    # RoBERTa / BART MNLI order: 0: CONTRADICTION, 1: NEUTRAL, 2: ENTAILMENT
    id2label = {"0": "CONTRADICTION", "1": "NEUTRAL", "2": "ENTAILMENT"}
    resolved = HuggingFaceNLIBackend.resolve_id2label_mapping(id2label)
    assert resolved == {
        0: NLILabel.CONTRADICTION,
        1: NLILabel.NEUTRAL,
        2: NLILabel.ENTAILMENT,
    }


def test_resolve_id2label_alternative_ordering() -> None:
    # 0: entailment, 1: neutral, 2: contradiction
    id2label = {0: "LABEL_ENTAILMENT", 1: "LABEL_NEUTRAL", 2: "LABEL_CONTRADICTION"}
    resolved = HuggingFaceNLIBackend.resolve_id2label_mapping(id2label)
    assert resolved == {
        0: NLILabel.ENTAILMENT,
        1: NLILabel.NEUTRAL,
        2: NLILabel.CONTRADICTION,
    }


def test_resolve_id2label_missing_label_raises() -> None:
    # Missing neutral
    id2label = {0: "entailment", 1: "contradiction"}
    with pytest.raises(ValueError, match="Unable to resolve required NLI labels"):
        HuggingFaceNLIBackend.resolve_id2label_mapping(id2label)


def test_resolve_id2label_invalid_input_raises() -> None:
    with pytest.raises(ValueError, match="invalid or missing id2label"):
        HuggingFaceNLIBackend.resolve_id2label_mapping(None)

    with pytest.raises(ValueError, match="Non-integer class index"):
        HuggingFaceNLIBackend.resolve_id2label_mapping({"invalid_idx": "entailment"})


# ---------------------------------------------------------------------------
# Distribution Validation Tests
# ---------------------------------------------------------------------------


def test_validate_nli_distribution_valid() -> None:
    valid_dist = {
        NLILabel.ENTAILMENT: 0.7,
        NLILabel.NEUTRAL: 0.2,
        NLILabel.CONTRADICTION: 0.1,
    }
    validate_nli_distribution(valid_dist)  # Should not raise


def test_validate_nli_distribution_missing_key() -> None:
    dist = {NLILabel.ENTAILMENT: 0.8, NLILabel.NEUTRAL: 0.2}
    with pytest.raises(ValueError, match="missing required label: 'contradiction'"):
        validate_nli_distribution(dist)  # type: ignore[arg-type]


def test_validate_nli_distribution_out_of_bounds() -> None:
    dist = {
        NLILabel.ENTAILMENT: 1.2,
        NLILabel.NEUTRAL: -0.1,
        NLILabel.CONTRADICTION: -0.1,
    }
    with pytest.raises(ValueError, match="must be in"):
        validate_nli_distribution(dist)


def test_validate_nli_distribution_sum_error() -> None:
    dist = {
        NLILabel.ENTAILMENT: 0.5,
        NLILabel.NEUTRAL: 0.2,
        NLILabel.CONTRADICTION: 0.1,
    }
    with pytest.raises(ValueError, match=r"must sum to 1\.0"):
        validate_nli_distribution(dist)


def test_validate_nli_distribution_nan_or_inf() -> None:
    dist_nan = {
        NLILabel.ENTAILMENT: float("nan"),
        NLILabel.NEUTRAL: 0.5,
        NLILabel.CONTRADICTION: 0.5,
    }
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_nli_distribution(dist_nan)

    dist_inf = {
        NLILabel.ENTAILMENT: float("inf"),
        NLILabel.NEUTRAL: 0.0,
        NLILabel.CONTRADICTION: 0.0,
    }
    with pytest.raises(ValueError, match="must be a finite number"):
        validate_nli_distribution(dist_inf)


# ---------------------------------------------------------------------------
# Pair Construction & Claim Extraction Tests
# ---------------------------------------------------------------------------


def test_premise_hypothesis_pair_construction() -> None:
    chunk1 = _make_result("CHK-001", "Premise content from chunk 1.")
    chunk2 = _make_result("CHK-002", "Premise content from chunk 2.")
    answer = "Claim one. Claim two."

    mock_backend = MockNLIBackend()
    evaluator = NLIEntailmentEvaluator(backend=mock_backend)
    evaluator.evaluate(answer, [chunk1, chunk2])

    # 2 claims * 2 chunks = 4 pairs
    assert len(mock_backend.recorded_pairs) == 4
    # First pair: (chunk1.content, claim1)
    assert mock_backend.recorded_pairs[0] == ("Premise content from chunk 1.", "Claim one.")
    # Second pair: (chunk2.content, claim1)
    assert mock_backend.recorded_pairs[1] == ("Premise content from chunk 2.", "Claim one.")
    # Third pair: (chunk1.content, claim2)
    assert mock_backend.recorded_pairs[2] == ("Premise content from chunk 1.", "Claim two.")
    # Fourth pair: (chunk2.content, claim2)
    assert mock_backend.recorded_pairs[3] == ("Premise content from chunk 2.", "Claim two.")


def test_claim_span_exact_offsets_preserved() -> None:
    answer = "  First claim with whitespace.   Second claim!  "
    chunk = _make_result("CHK-001", "Evidence text.")
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    result = evaluator.evaluate(answer, [chunk])

    assert result.total_claims == 2
    c1, c2 = result.claims
    assert answer[c1.span[0] : c1.span[1]] == c1.claim_text
    assert c1.claim_text == "First claim with whitespace."
    assert answer[c2.span[0] : c2.span[1]] == c2.claim_text
    assert c2.claim_text == "Second claim!"


# ---------------------------------------------------------------------------
# Maximum-Entailment Selection & Hand-Calculated Hallucination Score
# ---------------------------------------------------------------------------


def test_hand_calculated_hallucination_score() -> None:
    """Explicit hand-calculated test scenario.

    Answer:
    - Claim 1: "BM25 default k1 is 1.5."
    - Claim 2: "Sliding window step size is at least 1."

    Retrieved Chunks:
    - CHK-001: discusses BM25 k1=1.5
    - CHK-002: discusses ingestion window step size

    Inference Rules:
    - Claim 1 vs CHK-001: E=0.90, N=0.08, C=0.02 (best for Claim 1)
    - Claim 1 vs CHK-002: E=0.10, N=0.80, C=0.10
    - Claim 2 vs CHK-001: E=0.05, N=0.85, C=0.10
    - Claim 2 vs CHK-002: E=0.80, N=0.15, C=0.05 (best for Claim 2)

    Calculation:
    - Claim 1: s_1 = max(0.90, 0.10) = 0.90, h_1 = 1 - 0.90 = 0.10. Best: CHK-001.
      Predicted label: ENTAILMENT (argmax(0.90, 0.08, 0.02)).
    - Claim 2: s_2 = max(0.05, 0.80) = 0.80, h_2 = 1 - 0.80 = 0.20. Best: CHK-002.
      Predicted label: ENTAILMENT (argmax(0.80, 0.15, 0.05)).

    Aggregate Hallucination Score:
    H_answer = (h_1 + h_2) / 2 = (0.10 + 0.20) / 2 = 0.15.
    """

    def mock_rules(premise: str, hypothesis: str) -> dict[NLILabel, float]:
        if "BM25" in hypothesis:
            if "CHK-001" in premise:
                return {
                    NLILabel.ENTAILMENT: 0.90,
                    NLILabel.NEUTRAL: 0.08,
                    NLILabel.CONTRADICTION: 0.02,
                }
            return {
                NLILabel.ENTAILMENT: 0.10,
                NLILabel.NEUTRAL: 0.80,
                NLILabel.CONTRADICTION: 0.10,
            }
        else:  # Sliding window hypothesis
            if "CHK-002" in premise:
                return {
                    NLILabel.ENTAILMENT: 0.80,
                    NLILabel.NEUTRAL: 0.15,
                    NLILabel.CONTRADICTION: 0.05,
                }
            return {
                NLILabel.ENTAILMENT: 0.05,
                NLILabel.NEUTRAL: 0.85,
                NLILabel.CONTRADICTION: 0.10,
            }

    chunk1 = _make_result("CHK-001", "CHK-001 content about BM25 k1 parameter.")
    chunk2 = _make_result("CHK-002", "CHK-002 content about sliding window chunking.")
    answer = "BM25 default k1 is 1.5. Sliding window step size is at least 1."

    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend(rule_fn=mock_rules))
    res = evaluator.evaluate(answer, [chunk1, chunk2])

    assert isinstance(res, NLIResult)
    assert res.total_claims == 2
    assert res.has_claims is True
    assert res.hallucination_score == pytest.approx(0.15, abs=1e-6)
    assert res.entailed_claims_count == 2
    assert res.neutral_claims_count == 0
    assert res.contradicted_claims_count == 0
    assert res.no_evidence_claims_count == 0

    c1 = res.claims[0]
    assert isinstance(c1, ClaimNLIResult)
    assert c1.best_chunk_id == "CHK-001"
    assert c1.entailment_prob == pytest.approx(0.90)
    assert c1.unsupported_risk == pytest.approx(0.10)
    assert c1.predicted_label == NLILabel.ENTAILMENT

    c2 = res.claims[1]
    assert c2.best_chunk_id == "CHK-002"
    assert c2.entailment_prob == pytest.approx(0.80)
    assert c2.unsupported_risk == pytest.approx(0.20)
    assert c2.predicted_label == NLILabel.ENTAILMENT


def test_contradiction_label_and_score() -> None:
    # A claim contradicted by evidence: E=0.05, N=0.10, C=0.85
    # Risk h = 1 - 0.05 = 0.95 (very high risk)
    chunk = _make_result("CHK-001", "Default k1 is 1.5.")
    answer = "Default k1 is 999.0."

    def contradict_rule(premise: str, hyp: str) -> dict[NLILabel, float]:
        return {NLILabel.ENTAILMENT: 0.05, NLILabel.NEUTRAL: 0.10, NLILabel.CONTRADICTION: 0.85}

    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend(rule_fn=contradict_rule))
    res = evaluator.evaluate(answer, [chunk])

    assert res.total_claims == 1
    assert res.hallucination_score == pytest.approx(0.95)
    assert res.contradicted_claims_count == 1
    assert res.claims[0].predicted_label == NLILabel.CONTRADICTION


def test_deterministic_tie_breaking_for_equal_entailment() -> None:
    # Two chunks giving identical entailment (0.7), neutral (0.2),
    # and contradiction (0.1)
    chunk_b = _make_result("CHK-B", "Content B")
    chunk_a = _make_result("CHK-A", "Content A")

    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    res = evaluator.evaluate("Single claim.", [chunk_b, chunk_a])

    # Tie-breaking selects lexicographically smallest chunk_id ("CHK-A" < "CHK-B")
    assert res.claims[0].best_chunk_id == "CHK-A"


# ---------------------------------------------------------------------------
# Empty Inputs & No Evidence Contracts
# ---------------------------------------------------------------------------


def test_empty_answer_contract() -> None:
    chunk = _make_result("CHK-001", "Evidence text.")
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    res = evaluator.evaluate("", [chunk])

    assert res.total_claims == 0
    assert res.has_claims is False
    assert res.hallucination_score == 0.0
    assert res.claims == []
    assert res.entailed_claims_count == 0
    assert res.no_evidence_claims_count == 0


def test_whitespace_only_answer_contract() -> None:
    chunk = _make_result("CHK-001", "Evidence text.")
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    res = evaluator.evaluate("   \n\t  \r  ", [chunk])

    assert res.total_claims == 0
    assert res.has_claims is False
    assert res.hallucination_score == 0.0


def test_claims_with_no_retrieved_evidence_contract() -> None:
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    res = evaluator.evaluate("First claim. Second claim.", [])

    assert res.total_claims == 2
    assert res.has_claims is True
    # Without evidence, unsupportedness risk is 1.0 for every claim
    assert res.hallucination_score == 1.0
    assert res.no_evidence_claims_count == 2
    assert res.claims[0].best_chunk_id is None
    assert res.claims[0].has_evidence is False
    assert res.claims[0].entailment_prob == 0.0
    assert res.claims[0].unsupported_risk == 1.0
    assert res.claims[0].predicted_label is None


# ---------------------------------------------------------------------------
# Input Validation & Provenance Conflicts
# ---------------------------------------------------------------------------


def test_evaluator_input_type_validation() -> None:
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())

    with pytest.raises(TypeError, match="answer_text must be a string"):
        evaluator.evaluate(123, [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="retrieved must be a Sequence"):
        evaluator.evaluate("text", "not_a_sequence")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="must be RetrievalResult or Chunk"):
        evaluator.evaluate("text", ["invalid_element"])  # type: ignore[list-item]


def test_conflicting_chunk_provenance_rejected() -> None:
    chunk1 = _make_result("CHK-001", "Content A", doc_id="doc-A")
    chunk2 = _make_result("CHK-001", "Different Content B", doc_id="doc-B")

    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    with pytest.raises(ValueError, match="Conflicting chunk content or provenance detected"):
        evaluator.evaluate("Some claim.", [chunk1, chunk2])


def test_direct_chunk_sequence_support() -> None:
    chunk = Chunk(chunk_id="CHK-001", doc_id="doc-01", content="Direct chunk content.")
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    res = evaluator.evaluate("Some claim.", [chunk])

    assert res.total_claims == 1
    assert res.claims[0].best_chunk_id == "CHK-001"
    assert res.claims[0].has_evidence is True


# ---------------------------------------------------------------------------
# Backend & Packaging Tests
# ---------------------------------------------------------------------------


def test_huggingface_backend_missing_optional_dependency() -> None:
    # When torch/transformers is not installed, loading triggers actionable ImportError
    backend = HuggingFaceNLIBackend(model_name_or_path="nonexistent-model")
    with pytest.raises(ImportError, match="Hugging Face transformers and PyTorch are required"):
        backend.predict_batch([("premise", "hypothesis")])


def test_lazy_loading_import_safety() -> None:
    # Importing evaluator and initializing with default HuggingFaceBackend does not initialize model
    evaluator = NLIEntailmentEvaluator(model_name="cross-encoder/nli-deberta-v3-small")
    assert isinstance(evaluator.backend, HuggingFaceNLIBackend)
    assert evaluator.backend._model is None
    assert evaluator.backend._tokenizer is None


def test_huggingface_backend_batch_size_validation() -> None:
    with pytest.raises(TypeError, match="batch_size must be an integer"):
        HuggingFaceNLIBackend(batch_size="16")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="batch_size must be an integer"):
        HuggingFaceNLIBackend(batch_size=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="batch_size must be a positive integer"):
        HuggingFaceNLIBackend(batch_size=0)


def test_backend_mismatched_output_length_rejected() -> None:
    class BadBackend(BaseNLIBackend):
        def predict_batch(self, pairs: Sequence[tuple[str, str]]) -> list[dict[NLILabel, float]]:
            return []  # Return 0 distributions for 2 pairs

    evaluator = NLIEntailmentEvaluator(backend=BadBackend())
    chunk = _make_result("CHK-001", "Premise.")
    with pytest.raises(ValueError, match="NLI backend returned 0 distributions for 1 input pairs"):
        evaluator.evaluate("Hypothesis claim.", [chunk])


# ---------------------------------------------------------------------------
# JSON Serialization Tests
# ---------------------------------------------------------------------------


def test_json_serializability() -> None:
    chunk = _make_result("CHK-001", "Default parameters are k1 = 1.5.")
    evaluator = NLIEntailmentEvaluator(backend=MockNLIBackend())
    res = evaluator.evaluate("Default parameters are k1 = 1.5.", [chunk])

    res_dict = res.to_dict()
    assert isinstance(res_dict, dict)
    assert "hallucination_score" in res_dict
    assert len(res_dict["claims"]) == 1

    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["hallucination_score"] == pytest.approx(0.2)  # 1 - 0.8 = 0.2
    assert parsed["claims"][0]["best_chunk_id"] == "CHK-001"
    assert parsed["claims"][0]["predicted_label"] == "entailment"


# ---------------------------------------------------------------------------
# Metric Adapter Tests (NLIHallucinationMetric)
# ---------------------------------------------------------------------------


def test_nli_hallucination_metric_adapter() -> None:
    metric = NLIHallucinationMetric(backend=MockNLIBackend())
    assert metric.name == "nli_hallucination_score"

    chunk = _make_result("CHK-001", "Default parameters are k1 = 1.5.")
    query = Query(
        query_id="Q-001",
        text="What are BM25 defaults?",
        ground_truth_answer="Default parameters are k1 = 1.5.",
        metadata={"generated_answer": "Default parameters are k1 = 1.5."},
    )

    score = metric.compute([chunk], query)
    assert score == pytest.approx(0.2)

    # Test direct evaluate_text method
    text_score = metric.evaluate_text("Default parameters are k1 = 1.5.", [chunk])
    assert text_score == pytest.approx(0.2)


# ---------------------------------------------------------------------------
# Preservation of Existing Evaluators
# ---------------------------------------------------------------------------


def test_existing_groundedness_and_citation_evaluators_unaffected() -> None:
    g_eval = GroundednessEvaluator()
    c_eval = CitationEvaluator()
    assert g_eval.containment_threshold == 0.5
    assert c_eval.parser is not None
