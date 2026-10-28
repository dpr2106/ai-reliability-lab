"""Unit tests for citation marker parsing, reference link validation, and citation metrics.

Validates GROUND-001 requirements:
- Deterministic parsing of citation markers ([CIT-xxx], [CHK-xxx], composite [CIT-001, CIT-002])
- Ignoring unrelated bracketed prose ([see above], [1], [note])
- Malformed citation-like marker detection ([CIT-], [citation: 1])
- Reference link resolution to source chunk IDs
- Retrieved evidence presence checks against Sequence[RetrievalResult]
- Exact mathematical calculation of citation precision and citation recall
- Deduplication semantics (no metric inflation from repeated citations)
- Empty text, empty retrieved list, and zero-denominator contracts
- JSON serialization and diagnostics completeness
- Benchmark dataset loader citation mapping preservation and Query compatibility
"""

import json
from pathlib import Path

import pytest

from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.benchmark_runner import load_benchmark_dataset
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


def _make_result(
    chunk_id: str,
    score: float = 1.0,
    rank: int = 1,
    retriever_name: str = "bm25",
) -> RetrievalResult:
    """Helper to construct dummy RetrievalResult with specified chunk ID."""
    chunk = Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        content=f"Content for {chunk_id}",
    )
    return RetrievalResult(
        chunk=chunk,
        score=score,
        rank=rank,
        retriever_name=retriever_name,
    )


# ---------------------------------------------------------------------------
# CitationParser Tests
# ---------------------------------------------------------------------------


def test_parse_single_citation_marker() -> None:
    text = "In Okapi BM25, the default parameter k1 is 1.5 [CIT-001]."
    markers = parse_citation_markers(text)
    assert len(markers) == 1
    assert isinstance(markers[0], CitationMarker)
    assert markers[0].marker_id == "CIT-001"
    assert markers[0].marker_type == "citation"
    assert markers[0].raw == "[CIT-001]"
    assert markers[0].span == (47, 56)


def test_parse_multiple_markers_preserving_order() -> None:
    text = "Point one [CIT-001] leads to point two [CIT-002] and chunk [CHK-HYB-001]."
    markers = parse_citation_markers(text)
    assert len(markers) == 3
    assert [m.marker_id for m in markers] == ["CIT-001", "CIT-002", "CHK-HYB-001"]
    assert [m.marker_type for m in markers] == ["citation", "citation", "chunk"]


def test_parse_composite_bracket_markers() -> None:
    text = "Combined facts are supported by [CIT-001, CIT-002; CHK-ING-003]."
    markers = parse_citation_markers(text)
    assert len(markers) == 3
    assert [m.marker_id for m in markers] == ["CIT-001", "CIT-002", "CHK-ING-003"]
    assert markers[0].marker_type == "citation"
    assert markers[1].marker_type == "citation"
    assert markers[2].marker_type == "chunk"


def test_ignore_unrelated_bracketed_prose() -> None:
    text = "See [figure 1] and [table 2] or [note] or [123] for further background."
    markers = parse_citation_markers(text)
    assert markers == []
    ids = extract_citation_ids(text)
    assert ids == []


def test_detect_malformed_citation_like_markers() -> None:
    parser = CitationParser()
    text = "Valid [CIT-001], malformed [CIT-], [chk-], [citation: 1], and ignored [see above]."
    markers = parser.parse(text)
    assert len(markers) == 1
    assert markers[0].marker_id == "CIT-001"

    malformed = parser.find_malformed(text)
    assert "[CIT-]" in malformed
    assert "[chk-]" in malformed
    assert "[citation: 1]" in malformed
    assert "[see above]" not in malformed
    assert "[CIT-001]" not in malformed


def test_handle_punctuation_whitespace_and_unicode() -> None:
    text = "日本語の解説 [CIT-001] 、詳細 ([CIT-002]) と引用 [  CIT-003  ] です。"
    markers = parse_citation_markers(text)
    assert len(markers) == 3
    assert [m.marker_id for m in markers] == ["CIT-001", "CIT-002", "CIT-003"]


def test_parser_input_validation() -> None:
    parser = CitationParser()
    with pytest.raises(TypeError, match="text must be a string"):
        parser.parse(123)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="text must be a string"):
        parser.find_malformed(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Reference Resolution Tests
# ---------------------------------------------------------------------------


def test_resolve_citation_reference_via_mapping() -> None:
    mapping = {"CIT-001": "CHK-HYB-001", "CIT-002": "CHK-ING-003"}
    assert resolve_citation_reference("CIT-001", mapping) == "CHK-HYB-001"
    assert resolve_citation_reference("cit-002", mapping) == "CHK-ING-003"
    assert resolve_citation_reference("CIT-999", mapping) is None


def test_resolve_direct_chunk_reference() -> None:
    mapping = {"CIT-001": "CHK-HYB-001"}
    assert resolve_citation_reference("CHK-HYB-002", mapping) == "CHK-HYB-002"
    assert resolve_citation_reference("CHK-CRAN-003", None) == "CHK-CRAN-003"


def test_resolve_reference_validation() -> None:
    with pytest.raises(TypeError, match="marker_id must be a string"):
        resolve_citation_reference(None)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="marker_id must be a non-empty string"):
        resolve_citation_reference("   ")


# ---------------------------------------------------------------------------
# Calculation Helpers Tests
# ---------------------------------------------------------------------------


def test_calculate_citation_precision_bounds() -> None:
    assert calculate_citation_precision(0, 0) == 0.0
    assert calculate_citation_precision(0, 5) == 0.0
    assert calculate_citation_precision(3, 4) == 0.75
    assert calculate_citation_precision(5, 5) == 1.0
    assert calculate_citation_precision(-1, 5) == 0.0


def test_calculate_citation_recall_bounds() -> None:
    assert calculate_citation_recall(0, 0) == 0.0
    assert calculate_citation_recall(0, 5) == 0.0
    assert calculate_citation_recall(2, 4) == 0.5
    assert calculate_citation_recall(4, 4) == 1.0
    assert calculate_citation_recall(5, 0) == 0.0


# ---------------------------------------------------------------------------
# CitationEvaluator Tests & Hand-Calculated Scenarios
# ---------------------------------------------------------------------------


def test_hand_calculated_citation_metrics() -> None:
    """Explicit hand-calculated test scenario.

    Ground truth:
    - Required citations: ["CIT-001", "CIT-002", "CIT-003"] (3 required)
    - Mapping:
        CIT-001 -> CHK-001
        CIT-002 -> CHK-002
        CIT-003 -> CHK-003
        CIT-004 -> CHK-004

    Retrieved evidence:
    - CHK-001
    - CHK-004
    (CHK-002 and CHK-003 were NOT retrieved)

    Answer text emits:
    - [CIT-001] -> CHK-001 (retrieved -> VALID)
    - [CIT-002] -> CHK-002 (not retrieved -> MISSING EVIDENCE)
    - [CIT-004] -> CHK-004 (retrieved -> VALID)
    - [CIT-999] -> unknown mapping -> UNRESOLVED
    - [CIT-001] -> duplicate repetition
    - [see above] -> ignored prose

    Distinct parsed: {CIT-001, CIT-002, CIT-004, CIT-999} (4 distinct)
    Distinct valid: {CIT-001, CIT-004} (2 valid)
    Precision = 2 / 4 = 0.50

    Distinct required: {CIT-001, CIT-002, CIT-003} (3 distinct)
    Satisfied required: {CIT-001} (only CIT-001 is validly cited; CIT-002 chunk not retrieved)
    Recall = 1 / 3 = 0.3333333333333333
    """
    evaluator = CitationEvaluator()
    mapping = {
        "CIT-001": "CHK-001",
        "CIT-002": "CHK-002",
        "CIT-003": "CHK-003",
        "CIT-004": "CHK-004",
    }
    retrieved = [_make_result("CHK-001"), _make_result("CHK-004")]
    answer = (
        "Fact A [CIT-001]. Fact B [CIT-002]. Context [CIT-004]. "
        "Invented [CIT-999]. Repetition [CIT-001]. Note [see above]."
    )

    res = evaluator.evaluate(
        answer_text=answer,
        retrieved=retrieved,
        citation_mapping=mapping,
        required_citations=["CIT-001", "CIT-002", "CIT-003"],
    )

    assert res.citation_precision == pytest.approx(0.5, abs=1e-6)
    assert res.citation_recall == pytest.approx(1.0 / 3.0, abs=1e-6)
    assert res.has_required_citations is True

    assert res.distinct_citations == ["CIT-001", "CIT-002", "CIT-004", "CIT-999"]
    assert res.valid_citations == ["CIT-001", "CIT-004"]
    assert res.invalid_citations == ["CIT-999", "CIT-002"]
    assert res.unresolved_citations == ["CIT-999"]
    assert res.missing_evidence_citations == ["CIT-002"]
    assert res.duplicate_citations == ["CIT-001"]
    assert res.unreferenced_required_citations == ["CIT-002", "CIT-003"]
    assert res.retrieved_chunk_ids == ["CHK-001", "CHK-004"]


def test_duplicate_references_do_not_inflate_metrics() -> None:
    evaluator = CitationEvaluator()
    mapping = {"CIT-001": "CHK-001"}
    retrieved = [_make_result("CHK-001")]
    # Emits same citation 5 times
    answer = "[CIT-001] [CIT-001] [CIT-001] [CIT-001] [CIT-001]"

    res = evaluator.evaluate(
        answer_text=answer,
        retrieved=retrieved,
        citation_mapping=mapping,
        required_citations=["CIT-001"],
    )

    assert res.citation_precision == 1.0
    assert res.citation_recall == 1.0
    assert res.duplicate_citations == ["CIT-001"]
    assert len(res.distinct_citations) == 1
    assert len(res.parsed_citations) == 5


def test_empty_answer_and_empty_retrieved() -> None:
    evaluator = CitationEvaluator()
    res = evaluator.evaluate(
        answer_text="",
        retrieved=[],
        citation_mapping={"CIT-001": "CHK-001"},
        required_citations=["CIT-001"],
    )

    assert res.citation_precision == 0.0
    assert res.citation_recall == 0.0
    assert res.has_required_citations is True
    assert res.parsed_citations == []
    assert res.unreferenced_required_citations == ["CIT-001"]


def test_no_required_citations_contract() -> None:
    evaluator = CitationEvaluator()
    retrieved = [_make_result("CHK-001")]
    res = evaluator.evaluate(
        answer_text="Fact [CIT-001].",
        retrieved=retrieved,
        citation_mapping={"CIT-001": "CHK-001"},
        required_citations=[],
    )

    assert res.citation_precision == 1.0
    assert res.citation_recall == 0.0
    assert res.has_required_citations is False
    assert res.unreferenced_required_citations == []


def test_direct_chunk_reference_satisfies_required_citation() -> None:
    evaluator = CitationEvaluator()
    mapping = {"CIT-001": "CHK-HYB-001"}
    retrieved = [_make_result("CHK-HYB-001")]
    # Answer references chunk directly instead of citation ID
    answer = "The fact is stated in [CHK-HYB-001]."

    res = evaluator.evaluate(
        answer_text=answer,
        retrieved=retrieved,
        citation_mapping=mapping,
        required_citations=["CIT-001"],
    )

    assert res.citation_precision == 1.0
    assert res.citation_recall == 1.0
    assert "CHK-HYB-001" in res.valid_citations


def test_ambiguous_direct_chunk_does_not_inflate_multi_required() -> None:
    evaluator = CitationEvaluator()
    # Two citations mapping to same chunk
    mapping = {"CIT-001": "CHK-001", "CIT-002": "CHK-001"}
    retrieved = [_make_result("CHK-001")]
    answer = "Claim [CHK-001]."

    res = evaluator.evaluate(
        answer_text=answer,
        retrieved=retrieved,
        citation_mapping=mapping,
        required_citations=["CIT-001", "CIT-002"],
    )

    # Because CHK-001 is ambiguous between CIT-001 and CIT-002, neither is satisfied by proxy
    assert res.citation_precision == 1.0
    assert res.citation_recall == 0.0
    assert res.unreferenced_required_citations == ["CIT-001", "CIT-002"]


def test_evaluation_with_query_object() -> None:
    query = Query(
        query_id="Q-TEST-001",
        text="Test query?",
        required_citations=["CIT-001"],
        metadata={
            "citation_to_chunk_id": {"CIT-001": "CHK-001"},
            "generated_answer": "Answer with [CIT-001].",
        },
    )
    retrieved = [_make_result("CHK-001")]
    evaluator = CitationEvaluator()

    res = evaluator.evaluate(
        answer_text="Answer with [CIT-001].",
        retrieved=retrieved,
        query=query,
    )

    assert res.citation_precision == 1.0
    assert res.citation_recall == 1.0
    assert res.citation_to_chunk_map == {"CIT-001": "CHK-001"}


def test_conflicting_mappings_rejected() -> None:
    evaluator = CitationEvaluator()
    with pytest.raises(ValueError, match="citation_mapping keys must be non-empty strings"):
        evaluator.evaluate("text", [], citation_mapping={"": "CHK-001"})

    with pytest.raises(ValueError, match="citation_mapping values must be non-empty strings"):
        evaluator.evaluate("text", [], citation_mapping={"CIT-001": "  "})

    with pytest.raises(TypeError, match="citation_mapping must be a Mapping"):
        evaluator.evaluate("text", [], citation_mapping="invalid")  # type: ignore[arg-type]


def test_evaluator_input_type_validation() -> None:
    evaluator = CitationEvaluator()
    with pytest.raises(TypeError, match="answer_text must be a string"):
        evaluator.evaluate(123, [])  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="retrieved must be a Sequence of RetrievalResult"):
        evaluator.evaluate("text", "not_a_sequence")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="retrieved element at index 0 must be RetrievalResult"):
        evaluator.evaluate("text", ["not_retrieval_result"])  # type: ignore[list-item]

    with pytest.raises(TypeError, match="query must be a Query instance or None"):
        evaluator.evaluate("text", [], query="not_a_query")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="required_citations must be a Sequence of strings"):
        evaluator.evaluate("text", [], required_citations=123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="All required_citations must be non-empty strings"):
        evaluator.evaluate("text", [], required_citations=[""])


def test_json_serializability() -> None:
    evaluator = CitationEvaluator()
    res = evaluator.evaluate(
        answer_text="Test [CIT-001] and [CIT-999].",
        retrieved=[_make_result("CHK-001")],
        citation_mapping={"CIT-001": "CHK-001"},
        required_citations=["CIT-001", "CIT-002"],
    )

    assert isinstance(res, CitationValidationResult)
    res_dict = res.to_dict()
    assert isinstance(res_dict, dict)
    json_str = res.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["citation_precision"] == 0.5
    assert parsed["citation_recall"] == 0.5
    assert parsed["diagnostics"]["valid_count"] == 1
    assert parsed["diagnostics"]["unresolved_count"] == 1


# ---------------------------------------------------------------------------
# Metric Classes Tests (CitationPrecision & CitationRecall)
# ---------------------------------------------------------------------------


def test_citation_precision_and_recall_classes() -> None:
    prec = CitationPrecision()
    rec = CitationRecall()

    assert prec.name == "citation_precision"
    assert rec.name == "citation_recall"
    assert isinstance(prec, BaseCitationMetric)
    assert isinstance(rec, BaseCitationMetric)

    mapping = {"CIT-001": "CHK-001"}
    retrieved = [_make_result("CHK-001")]

    p_score = prec.evaluate_text("Fact [CIT-001].", retrieved, citation_mapping=mapping)
    assert p_score == 1.0

    r_score = rec.evaluate_text(
        "Fact [CIT-001].",
        retrieved,
        required_citations=["CIT-001"],
        citation_mapping=mapping,
    )
    assert r_score == 1.0

    # Test compute with Query
    query = Query(
        query_id="Q-001",
        text="Sample query",
        required_citations=["CIT-001"],
        metadata={
            "citation_to_chunk_id": mapping,
            "generated_answer": "Answer referencing [CIT-001].",
        },
    )
    assert prec.compute(retrieved, query) == 1.0
    assert rec.compute(retrieved, query) == 1.0


# ---------------------------------------------------------------------------
# Benchmark Loader Compatibility & Query Integration
# ---------------------------------------------------------------------------


def test_benchmark_loader_preserves_citation_mapping(tmp_path: Path) -> None:
    benchmark_file = tmp_path / "test_benchmark.jsonl"
    record = {
        "query_id": "Q-001",
        "query": "What is the BM25 k1 default?",
        "ground_truth_answer": "Default k1 is 1.5.",
        "relevance_judgments": [
            {"chunk_id": "CHK-HYB-001", "grade": 2, "rationale": "Direct answer"}
        ],
        "required_citations": [
            {
                "citation_id": "CIT-001",
                "chunk_id": "CHK-HYB-001",
                "doc_id": "doc-01",
                "page_number": 1,
                "quote": "default is k1 = 1.5",
            }
        ],
        "metadata": {"domain": "search"},
    }
    benchmark_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    queries = load_benchmark_dataset(benchmark_file)
    assert len(queries) == 1
    query = queries[0]

    # Required citations preserved as backward-compatible string list
    assert query.required_citations == ["CIT-001"]

    # Structured mapping preserved in metadata and accessible via property
    assert query.citation_to_chunk_map == {"CIT-001": "CHK-HYB-001"}
    assert "citation_records" in query.metadata
    assert query.metadata["citation_records"][0]["citation_id"] == "CIT-001"
    assert query.metadata["citation_records"][0]["chunk_id"] == "CHK-HYB-001"

    # Query serialization round-trip preservation
    q_dict = query.to_dict()
    restored_query = Query.from_dict(q_dict)
    assert restored_query.citation_to_chunk_map == {"CIT-001": "CHK-HYB-001"}

    q_json = query.to_json()
    json_restored = Query.from_json(q_json)
    assert json_restored.citation_to_chunk_map == {"CIT-001": "CHK-HYB-001"}


def test_benchmark_loader_detects_conflicting_citation_mappings(tmp_path: Path) -> None:
    benchmark_file = tmp_path / "conflicting_benchmark.jsonl"
    record = {
        "query_id": "Q-CONFLICT",
        "query": "Sample question?",
        "ground_truth_answer": "Sample answer.",
        "relevance_judgments": [{"chunk_id": "CHK-001", "grade": 2, "rationale": "R"}],
        "required_citations": [
            {
                "citation_id": "CIT-001",
                "chunk_id": "CHK-001",
                "doc_id": "doc-01",
                "page_number": 1,
                "quote": "Q1",
            },
            {
                "citation_id": "CIT-001",
                "chunk_id": "CHK-002",  # Conflicting chunk ID for same CIT-001
                "doc_id": "doc-01",
                "page_number": 1,
                "quote": "Q2",
            },
        ],
    }
    benchmark_file.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="has conflicting chunk_id mapping for citation 'CIT-001'"):
        load_benchmark_dataset(benchmark_file)


def test_end_to_end_evaluation_with_real_sample_benchmark() -> None:
    sample_file = Path("data/benchmarks/sample_benchmark.jsonl")
    if not sample_file.exists():
        pytest.skip("sample_benchmark.jsonl not present")

    queries = load_benchmark_dataset(sample_file)
    q1 = next(q for q in queries if q.query_id == "Q-001")

    assert "CIT-001" in q1.required_citations
    assert q1.citation_to_chunk_map.get("CIT-001") == "CHK-HYB-001"

    evaluator = CitationEvaluator()
    retrieved = [_make_result("CHK-HYB-001"), _make_result("CHK-HYB-002")]
    answer = "In Okapi BM25, the default parameter values are k1 = 1.5 and b = 0.75 [CIT-001]."

    result = evaluator.evaluate(answer, retrieved, q1)
    assert result.citation_precision == 1.0
    assert result.citation_recall == 1.0
    assert result.valid_citations == ["CIT-001"]
