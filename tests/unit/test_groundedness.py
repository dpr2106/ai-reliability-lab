"""Unit tests for deterministic claim span extraction and lexical overlap groundedness.

Validates GROUND-002 requirements:
- Deterministic sentence-level claim segmentation with exact character offsets
- Protection of decimal numbers (1.5, 0.75, 25.5ms) from erroneous splitting
- Unicode text, punctuation, quotes, and newline handling
- Case-insensitive Unicode tokenization matching repository conventions
- Memory-efficient longest common subsequence (LCS) calculation preserving token order
- Token containment calculation with explicit repeated-token semantics
- Deterministic independent evidence evaluation and tie-breaking across multiple chunks
- Configurable dual-threshold decision rule (containment and LCS)
- Aggregate answer-level scoring, empty-input contracts, and has_claims distinction
- Input validation, conflicting provenance detection, and JSON serializability
- LexicalGroundednessMetric adapter integration with Query and BaseMetric
"""

import json

import pytest

from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.groundedness import (
    ClaimGroundingResult,
    GroundednessEvaluator,
    GroundednessResult,
    LexicalGroundednessMetric,
    compute_lcs_length,
    compute_token_containment,
    compute_token_lcs_score,
    extract_claim_spans,
    tokenize,
)


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
# Claim Span Extraction Tests
# ---------------------------------------------------------------------------


def test_extract_single_sentence_span() -> None:
    text = "In Okapi BM25, k1 controls term frequency saturation."
    spans = extract_claim_spans(text)
    assert len(spans) == 1
    start, end = spans[0]
    assert (start, end) == (0, len(text))
    assert text[start:end] == text


def test_extract_multiple_sentences_exact_offsets() -> None:
    text = "First claim is here. Second claim follows! Third claim is a question?"
    spans = extract_claim_spans(text)
    assert len(spans) == 3
    claims = [text[start:end] for start, end in spans]
    assert claims == [
        "First claim is here.",
        "Second claim follows!",
        "Third claim is a question?",
    ]
    for start, end in spans:
        assert text[start:end] == text[start:end].strip()


def test_protect_decimal_numbers_from_splitting() -> None:
    text = "Parameter k1 is 1.5 and b is 0.75. Query p99 latency target is 25.5ms."
    spans = extract_claim_spans(text)
    assert len(spans) == 2
    claims = [text[start:end] for start, end in spans]
    assert claims == [
        "Parameter k1 is 1.5 and b is 0.75.",
        "Query p99 latency target is 25.5ms.",
    ]


def test_unicode_and_punctuation_sentence_extraction() -> None:
    text = "文書一の記述です\u3002 文書二の記述です\uff01 文書三でしょうか\uff1f"
    spans = extract_claim_spans(text)
    assert len(spans) == 3
    claims = [text[start:end] for start, end in spans]
    assert claims == [
        "文書一の記述です\u3002",
        "文書二の記述です\uff01",
        "文書三でしょうか\uff1f",
    ]


def test_line_breaks_and_blank_lines() -> None:
    text = "Line one.\nLine two without period\n\nLine three after blank line."
    spans = extract_claim_spans(text)
    assert len(spans) == 3
    claims = [text[start:end] for start, end in spans]
    assert claims == [
        "Line one.",
        "Line two without period",
        "Line three after blank line.",
    ]


def test_quotes_and_brackets_after_punctuation() -> None:
    text = '"Quoted assertion." Next assertion (see [CIT-001]). Final point.'
    spans = extract_claim_spans(text)
    assert len(spans) == 3
    claims = [text[start:end] for start, end in spans]
    assert claims[0] == '"Quoted assertion."'
    assert claims[1] == "Next assertion (see [CIT-001])."
    assert claims[2] == "Final point."


def test_empty_and_whitespace_only_claim_extraction() -> None:
    assert extract_claim_spans("") == []
    assert extract_claim_spans("   \t  \n  \n\r  ") == []


def test_claim_extraction_type_validation() -> None:
    with pytest.raises(TypeError, match="text must be a string"):
        extract_claim_spans(123)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Tokenization & Normalization Tests
# ---------------------------------------------------------------------------


def test_tokenize_unicode_and_casefold() -> None:
    tokens = tokenize("In Okapi BM25, k1 = 1.5 and b = 0.75.")
    assert "bm25" in tokens
    assert "k1" in tokens
    assert "1" in tokens
    assert "5" in tokens
    assert "0" in tokens
    assert "75" in tokens

    unicode_tokens = tokenize("日本語の文書 Spärck Jones")
    assert unicode_tokens == ["日本語の文書", "spärck", "jones"]


def test_tokenize_empty_and_whitespace() -> None:
    assert tokenize("") == []
    assert tokenize("   \t\n  ") == []


# ---------------------------------------------------------------------------
# Token Containment & LCS Unit Tests
# ---------------------------------------------------------------------------


def test_token_containment_exact_match() -> None:
    tokens = ["bm25", "k1", "parameter", "is", "1", "5"]
    assert compute_token_containment(tokens, tokens) == 1.0


def test_token_containment_unrelated_is_zero() -> None:
    claim_tokens = ["alpha", "beta", "gamma"]
    evidence_tokens = ["delta", "epsilon", "zeta"]
    assert compute_token_containment(claim_tokens, evidence_tokens) == 0.0


def test_token_containment_hand_calculated() -> None:
    # 5 claim tokens, 3 occurrences present in evidence ('the', 'parameter', 'the')
    claim_tokens = ["the", "parameter", "is", "the", "key"]
    evidence_tokens = ["the", "parameter", "fast", "engine"]
    # covered: 'the' (yes), 'parameter' (yes), 'is' (no), 'the' (yes), 'key' (no) => 3 / 5 = 0.6
    assert compute_token_containment(claim_tokens, evidence_tokens) == pytest.approx(0.6)


def test_token_containment_empty_claim() -> None:
    assert compute_token_containment([], ["some", "tokens"]) == 0.0
    assert compute_token_containment(["some"], []) == 0.0


def test_lcs_length_and_order_sensitivity() -> None:
    claim = ["a", "b", "c", "d"]
    evidence = ["x", "a", "y", "b", "z", "d"]
    # common subsequence: a, b, d (len 3)
    assert compute_lcs_length(claim, evidence) == 3
    assert compute_token_lcs_score(claim, evidence) == pytest.approx(0.75)

    # Reversed order produces lower LCS than containment
    reversed_evidence = ["d", "c", "b", "a"]
    # LCS of ["a", "b", "c", "d"] and ["d", "c", "b", "a"] is 1
    assert compute_lcs_length(claim, reversed_evidence) == 1
    assert compute_token_lcs_score(claim, reversed_evidence) == pytest.approx(0.25)
    # Notice containment is 100% (4/4) but LCS is only 25% (1/4)
    assert compute_token_containment(claim, reversed_evidence) == 1.0


def test_lcs_empty_sequences() -> None:
    assert compute_lcs_length([], ["a", "b"]) == 0
    assert compute_lcs_length(["a", "b"], []) == 0
    assert compute_token_lcs_score([], ["a", "b"]) == 0.0


# ---------------------------------------------------------------------------
# GroundednessEvaluator Tests
# ---------------------------------------------------------------------------


def test_hand_calculated_groundedness_evaluation() -> None:
    """Explicit hand-calculated test scenario.

    Chunk 1 (CHK-001):
    "In the Okapi BM25 implementation, default parameter values are k1 = 1.5 and b = 0.75."

    Chunk 2 (CHK-002):
    "IndexFlatIP performs exhaustive inner product comparison with 100% recall guarantee."

    Answer Text:
    "Claim 1: The default parameter values are k1 = 1.5 and b = 0.75 in Okapi BM25.
     Claim 2: IndexFlatIP provides approximate nearest neighbor search with logarithmic latency.
     Claim 3: Completely unsupported statement about quantum computers."

    Claim 1 matches CHK-001 with high containment (100%) and high LCS (near 100%). -> GROUNDED
    Claim 2 overlaps with CHK-002 on IndexFlatIP, but states approximate search
    (contradicting CHK-002, which states exhaustive comparison). Overlap is low.
    Claim 3 has zero overlap with any chunk. -> UNSUPPORTED

    Thresholds: containment >= 0.6, lcs >= 0.4
    Expected:
    - Claim 1: Grounded (best chunk CHK-001)
    - Claim 2: Unsupported (best chunk CHK-002 or CHK-001, below thresholds)
    - Claim 3: Unsupported (best chunk CHK-001 or CHK-002, 0.0)
    Grounded ratio = 1 / 3 = 0.333333
    """
    chunk1 = _make_result(
        "CHK-001",
        "In the Okapi BM25 implementation, default parameter values are k1 = 1.5 and b = 0.75.",
    )
    chunk2 = _make_result(
        "CHK-002",
        "IndexFlatIP performs exhaustive inner product comparison with 100% recall guarantee.",
    )

    answer = (
        "The default parameter values are k1 = 1.5 and b = 0.75 in Okapi BM25. "
        "IndexFlatIP provides approximate nearest neighbor search with logarithmic latency. "
        "Completely unsupported statement about quantum computers."
    )

    evaluator = GroundednessEvaluator(containment_threshold=0.6, lcs_threshold=0.4)
    result = evaluator.evaluate(answer, [chunk1, chunk2])

    assert isinstance(result, GroundednessResult)
    assert result.total_claims == 3
    assert result.has_claims is True
    assert result.grounded_claims_count == 1
    assert result.unsupported_claims_count == 2
    assert result.no_evidence_claims_count == 0
    assert result.aggregate_score == pytest.approx(1.0 / 3.0, abs=1e-6)

    # Claim 1 assertions
    c1 = result.claims[0]
    assert isinstance(c1, ClaimGroundingResult)
    assert c1.best_chunk_id == "CHK-001"
    assert c1.token_containment >= 0.9
    assert c1.lcs_score >= 0.6
    assert c1.is_grounded is True

    # Claim 3 assertions
    c3 = result.claims[2]
    assert c3.token_containment == 0.0
    assert c3.lcs_score == 0.0
    assert c3.is_grounded is False


def test_independent_evidence_evaluation_not_concatenated() -> None:
    """Verifies that evidence chunks are evaluated independently, not concatenated.

    If chunks were concatenated, a claim consisting of half words from Chunk A and half
    words from Chunk B might achieve 100% containment. But against independent chunks,
    neither chunk provides full containment.
    """
    chunk_a = _make_result("CHK-A", "cats and dogs are domestic animals.")
    chunk_b = _make_result("CHK-B", "quantum computers use superconducting qubits.")

    # Claim borrows terms from both unrelated chunks
    synthetic_claim = "Domestic animals use superconducting qubits."

    evaluator = GroundednessEvaluator(containment_threshold=0.8, lcs_threshold=0.5)
    res = evaluator.evaluate(synthetic_claim, [chunk_a, chunk_b])

    assert res.total_claims == 1
    claim = res.claims[0]
    # Against chunk_a alone or chunk_b alone, containment cannot reach 0.8
    assert claim.is_grounded is False
    assert res.aggregate_score == 0.0


def test_deterministic_tie_breaking_between_chunks() -> None:
    # Two identical content chunks with different IDs
    chunk1 = _make_result("CHK-002", "Exact matching text content.")
    chunk2 = _make_result("CHK-001", "Exact matching text content.")

    evaluator = GroundednessEvaluator()
    res = evaluator.evaluate("Exact matching text content.", [chunk1, chunk2])

    # Tie-breaking rule prefers lexicographically smaller chunk_id ("CHK-001" < "CHK-002")
    assert res.claims[0].best_chunk_id == "CHK-001"


def test_no_evidence_candidates_contract() -> None:
    evaluator = GroundednessEvaluator()
    res = evaluator.evaluate("Some generated claim.", [])

    assert res.total_claims == 1
    assert res.has_claims is True
    assert res.grounded_claims_count == 0
    assert res.unsupported_claims_count == 1
    assert res.no_evidence_claims_count == 1
    assert res.aggregate_score == 0.0

    claim = res.claims[0]
    assert claim.best_chunk_id is None
    assert claim.token_containment == 0.0
    assert claim.lcs_score == 0.0
    assert claim.is_grounded is False
    assert claim.candidate_scores == {}


def test_empty_answer_contract() -> None:
    chunk = _make_result("CHK-001", "Some content.")
    evaluator = GroundednessEvaluator()
    res = evaluator.evaluate("", [chunk])

    assert res.total_claims == 0
    assert res.has_claims is False
    assert res.grounded_claims_count == 0
    assert res.unsupported_claims_count == 0
    assert res.aggregate_score == 0.0
    assert res.claims == []


def test_direct_chunk_sequence_support() -> None:
    # Test that evaluator accepts Sequence[Chunk] directly in addition to Sequence[RetrievalResult]
    chunk = Chunk(chunk_id="CHK-DIRECT-01", doc_id="doc-01", content="Direct chunk content.")
    evaluator = GroundednessEvaluator()
    res = evaluator.evaluate("Direct chunk content.", [chunk])

    assert res.total_claims == 1
    assert res.claims[0].best_chunk_id == "CHK-DIRECT-01"
    assert res.claims[0].is_grounded is True


def test_threshold_parameter_validation() -> None:
    with pytest.raises(TypeError, match="containment_threshold must be a float or int"):
        GroundednessEvaluator(containment_threshold="0.5")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="containment_threshold must be a float or int"):
        GroundednessEvaluator(containment_threshold=True)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="containment_threshold must be within"):
        GroundednessEvaluator(containment_threshold=-0.1)

    with pytest.raises(ValueError, match="containment_threshold must be within"):
        GroundednessEvaluator(containment_threshold=1.1)

    with pytest.raises(ValueError, match="lcs_threshold must be a finite number"):
        GroundednessEvaluator(lcs_threshold=float("nan"))

    with pytest.raises(ValueError, match="lcs_threshold must be a finite number"):
        GroundednessEvaluator(lcs_threshold=float("inf"))


def test_evidence_input_validation() -> None:
    evaluator = GroundednessEvaluator()

    with pytest.raises(TypeError, match="answer_text must be a string"):
        evaluator.evaluate(123, [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="retrieved must be a Sequence"):
        evaluator.evaluate("text", "not_a_sequence")  # type: ignore[arg-type]

    with pytest.raises(
        TypeError, match="retrieved item at index 0 must be RetrievalResult or Chunk"
    ):
        evaluator.evaluate("text", ["invalid_item"])  # type: ignore[list-item]


def test_conflicting_chunk_provenance_rejected() -> None:
    chunk1 = _make_result("CHK-001", "Content A", doc_id="doc-A")
    chunk2 = _make_result("CHK-001", "Different Content B", doc_id="doc-B")

    evaluator = GroundednessEvaluator()
    with pytest.raises(ValueError, match="Conflicting chunk content or provenance detected"):
        evaluator.evaluate("Some claim.", [chunk1, chunk2])


def test_json_serializability() -> None:
    chunk = _make_result("CHK-001", "Default parameters are k1 = 1.5.")
    evaluator = GroundednessEvaluator()
    res = evaluator.evaluate("Default parameters are k1 = 1.5.", [chunk])

    res_dict = res.to_dict()
    assert isinstance(res_dict, dict)
    assert res_dict["aggregate_score"] == 1.0
    assert len(res_dict["claims"]) == 1

    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["grounded_claims_count"] == 1
    assert parsed["claims"][0]["best_chunk_id"] == "CHK-001"


# ---------------------------------------------------------------------------
# LexicalGroundednessMetric Adapter Tests
# ---------------------------------------------------------------------------


def test_lexical_groundedness_metric_adapter() -> None:
    metric = LexicalGroundednessMetric(containment_threshold=0.5, lcs_threshold=0.3)
    assert metric.name == "lexical_groundedness"

    chunk = _make_result("CHK-001", "Default parameters are k1 = 1.5 and b = 0.75.")
    query = Query(
        query_id="Q-001",
        text="What are BM25 parameter defaults?",
        ground_truth_answer="Default parameters are k1 = 1.5 and b = 0.75.",
        metadata={"generated_answer": "Default parameters are k1 = 1.5 and b = 0.75."},
    )

    score = metric.compute([chunk], query)
    assert score == 1.0

    # Test evaluate_text direct API
    score_text = metric.evaluate_text("Default parameters are k1 = 1.5 and b = 0.75.", [chunk])
    assert score_text == 1.0
