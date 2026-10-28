"""Unit tests for retrieval benchmark dataset schema and integrity (DATA-001).

Validates schema definitions, corpus integrity, relevance grade distributions,
verbatim citation quotation, and domain model compatibility without external dependencies.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from reliability_lab.core.models import Chunk, Query

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "benchmarks"
SCHEMA_FILE = DATA_DIR / "schema.json"
BENCHMARK_FILE = DATA_DIR / "sample_benchmark.jsonl"
CORPUS_FILE = DATA_DIR / "sample_corpus.jsonl"


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Helper to parse a JSONL file, ensuring no malformed or blank lines."""
    assert path.is_file(), f"File does not exist: {path}"
    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8")
    assert text.strip(), f"File is empty: {path}"

    lines = text.splitlines()
    for line_num, line in enumerate(lines, start=1):
        stripped = line.strip()
        assert stripped, f"Unexpected empty or blank line at {path.name}:{line_num}"
        try:
            records.append(json.loads(stripped))
        except json.JSONDecodeError as exc:
            raise AssertionError(
                f"Malformed JSON on line {line_num} of {path.name}: {exc}"
            ) from exc
    return records


def _validate_benchmark_record(
    record: dict[str, Any], corpus_by_id: dict[str, dict[str, Any]]
) -> None:
    """Validate a single benchmark query record against schema rules and corpus constraints."""
    # Required top-level fields
    for field_name in [
        "query_id",
        "query",
        "ground_truth_answer",
        "relevance_judgments",
        "required_citations",
    ]:
        assert field_name in record, f"Missing required field '{field_name}' in record"

    query_id = record["query_id"]
    assert isinstance(query_id, str) and query_id.startswith("Q-"), (
        f"Invalid query_id format: {query_id}"
    )

    query = record["query"]
    assert isinstance(query, str) and len(query.strip()) >= 5, (
        f"Query text must be non-empty string in {query_id}"
    )

    answer = record["ground_truth_answer"]
    assert isinstance(answer, str) and len(answer.strip()) >= 5, (
        f"ground_truth_answer must be non-empty in {query_id}"
    )

    judgments = record["relevance_judgments"]
    assert isinstance(judgments, list) and len(judgments) >= 1, (
        f"relevance_judgments must be non-empty list in {query_id}"
    )

    judged_chunk_ids: set[str] = set()
    has_critical_evidence = False

    for j_idx, judgment in enumerate(judgments):
        for req_key in ["chunk_id", "grade", "rationale"]:
            assert req_key in judgment, f"Missing key '{req_key}' in judgment {j_idx} of {query_id}"

        cid = judgment["chunk_id"]
        assert isinstance(cid, str) and cid, (
            f"chunk_id must be non-empty string in judgment {j_idx} of {query_id}"
        )
        assert cid in corpus_by_id, (
            f"Judgment references unknown corpus chunk '{cid}' in {query_id}"
        )
        assert cid not in judged_chunk_ids, f"Duplicate judgment for chunk '{cid}' in {query_id}"
        judged_chunk_ids.add(cid)

        grade = judgment["grade"]
        # Strict int check: reject booleans because bool subclasses int in Python
        assert isinstance(grade, int) and not isinstance(grade, bool), (
            f"Grade must be integer in {query_id}, got {type(grade).__name__}"
        )
        assert grade in {0, 1, 2}, f"Grade must be 0, 1, or 2 in {query_id}, got {grade}"

        if grade == 2:
            has_critical_evidence = True

        rationale = judgment["rationale"]
        assert isinstance(rationale, str) and len(rationale.strip()) >= 5, (
            f"Rationale must be non-empty in judgment {j_idx} of {query_id}"
        )

    assert has_critical_evidence, f"Query {query_id} must have at least one Grade 2 judgment"

    citations = record["required_citations"]
    assert isinstance(citations, list) and len(citations) >= 1, (
        f"required_citations must be non-empty list in {query_id}"
    )

    citation_ids: set[str] = set()
    for c_idx, citation in enumerate(citations):
        for req_key in ["citation_id", "chunk_id", "doc_id", "page_number", "quote"]:
            assert req_key in citation, f"Missing key '{req_key}' in citation {c_idx} of {query_id}"

        cit_id = citation["citation_id"]
        assert isinstance(cit_id, str) and cit_id.startswith("CIT-"), (
            f"Invalid citation_id format: {cit_id}"
        )
        assert cit_id not in citation_ids, f"Duplicate citation_id '{cit_id}' in {query_id}"
        citation_ids.add(cit_id)

        c_chunk_id = citation["chunk_id"]
        assert c_chunk_id in corpus_by_id, (
            f"Citation references unknown chunk '{c_chunk_id}' in {query_id}"
        )

        # Citations must reference chunks with grade 2 judgments
        chunk_judgment = next((j for j in judgments if j["chunk_id"] == c_chunk_id), None)
        assert chunk_judgment is not None and chunk_judgment["grade"] == 2, (
            f"Citation {cit_id} references chunk '{c_chunk_id}' without Grade 2 in {query_id}"
        )

        corpus_chunk = corpus_by_id[c_chunk_id]
        assert citation["doc_id"] == corpus_chunk["doc_id"], (
            f"Citation doc_id mismatch for {cit_id}: "
            f"{citation['doc_id']} vs {corpus_chunk['doc_id']}"
        )

        if corpus_chunk["page_number"] is not None:
            assert citation["page_number"] == corpus_chunk["page_number"], (
                f"Citation page_number mismatch for {cit_id}"
            )

        quote = citation["quote"]
        assert isinstance(quote, str) and len(quote.strip()) >= 5, (
            f"Citation quote must be non-empty in {cit_id}"
        )
        assert quote in corpus_chunk["content"], (
            f"Citation quote for {cit_id} does not appear verbatim in chunk {c_chunk_id}"
        )


class TestBenchmarkSchemaAndIntegrity:
    """Validate JSON Schema definition, corpus structure, and benchmark dataset."""

    def test_schema_file_valid_draft_2020_12(self) -> None:
        assert SCHEMA_FILE.is_file(), f"Missing schema file: {SCHEMA_FILE}"
        schema_data = json.loads(SCHEMA_FILE.read_text(encoding="utf-8"))

        assert schema_data.get("$schema") == "https://json-schema.org/draft/2020-12/schema"
        assert schema_data.get("type") == "object"
        assert "query_id" in schema_data["properties"]
        assert "relevance_judgments" in schema_data["properties"]
        assert "required_citations" in schema_data["properties"]

    def test_corpus_file_structure_and_unique_chunks(self) -> None:
        corpus_records = _load_jsonl(CORPUS_FILE)
        assert len(corpus_records) >= 15, "Corpus must contain at least 15 chunks"

        seen_chunk_ids: set[str] = set()
        for idx, chunk_dict in enumerate(corpus_records):
            # Verify compatibility with domain Chunk model
            chunk = Chunk.from_dict(chunk_dict)
            assert chunk.chunk_id not in seen_chunk_ids, f"Duplicate chunk_id: {chunk.chunk_id}"
            seen_chunk_ids.add(chunk.chunk_id)
            assert chunk.content.strip(), f"Empty chunk content at index {idx}"
            assert chunk.doc_id.strip(), f"Empty doc_id at index {idx}"

    def test_sample_benchmark_size_and_schema_validation(self) -> None:
        corpus_records = _load_jsonl(CORPUS_FILE)
        corpus_by_id = {c["chunk_id"]: c for c in corpus_records}

        benchmark_records = _load_jsonl(BENCHMARK_FILE)
        assert len(benchmark_records) >= 15, "Benchmark must contain at least 15 queries"

        seen_query_ids: set[str] = set()
        all_grades: set[int] = set()
        scenarios: set[str] = set()

        for record in benchmark_records:
            qid = record["query_id"]
            assert qid not in seen_query_ids, f"Duplicate query_id: {qid}"
            seen_query_ids.add(qid)

            _validate_benchmark_record(record, corpus_by_id)

            for j in record["relevance_judgments"]:
                all_grades.add(j["grade"])

            if "metadata" in record and "scenario" in record["metadata"]:
                scenarios.add(record["metadata"]["scenario"])

        # Check full grade coverage: 0, 1, and 2
        assert all_grades == {0, 1, 2}, f"All grades 0, 1, 2 must appear, found: {all_grades}"
        # Check scenario diversity
        assert len(scenarios) >= 4, f"Must have diverse scenarios, found: {scenarios}"

    def test_verbatim_citation_quote_exactness(self) -> None:
        corpus_records = _load_jsonl(CORPUS_FILE)
        corpus_by_id = {c["chunk_id"]: c for c in corpus_records}
        benchmark_records = _load_jsonl(BENCHMARK_FILE)

        total_citations = 0
        for record in benchmark_records:
            for citation in record["required_citations"]:
                total_citations += 1
                chunk = corpus_by_id[citation["chunk_id"]]
                quote = citation["quote"]
                # Must be exact substring match
                assert quote in chunk["content"]
                # Verify non-trivial length
                assert len(quote) >= 10

        assert total_citations >= 15, f"Expected at least 15 citations, found {total_citations}"

    def test_domain_query_model_round_trip(self) -> None:
        """Verify benchmark records map cleanly into the project's Query domain model."""
        benchmark_records = _load_jsonl(BENCHMARK_FILE)
        for record in benchmark_records:
            query_obj = Query(
                query_id=record["query_id"],
                text=record["query"],
                relevant_chunk_ids=[
                    j["chunk_id"] for j in record["relevance_judgments"] if j["grade"] > 0
                ],
                relevance_grades={j["chunk_id"]: j["grade"] for j in record["relevance_judgments"]},
                ground_truth_answer=record["ground_truth_answer"],
                required_citations=[c["citation_id"] for c in record["required_citations"]],
                metadata=record.get("metadata", {}),
            )
            assert query_obj.query_id == record["query_id"]
            assert query_obj.text == record["query"]
            assert len(query_obj.relevant_chunk_ids) > 0
            assert query_obj.to_dict()["query_id"] == record["query_id"]
            assert json.loads(query_obj.to_json())["text"] == record["query"]


class TestDatasetValidationFailureModes:
    """Mutation tests ensuring dataset validation catches defects and corruptions."""

    @pytest.fixture
    def valid_state(self) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
        corpus_records = _load_jsonl(CORPUS_FILE)
        corpus_by_id = {c["chunk_id"]: c for c in corpus_records}
        benchmark_records = _load_jsonl(BENCHMARK_FILE)
        # Deep copy first record
        rec = json.loads(json.dumps(benchmark_records[0]))
        return rec, corpus_by_id

    def test_reject_unknown_chunk_in_judgment(
        self, valid_state: tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ) -> None:
        rec, corpus_by_id = valid_state
        rec["relevance_judgments"].append(
            {"chunk_id": "CHK-NONEXISTENT", "grade": 1, "rationale": "Ghost chunk"}
        )
        with pytest.raises(AssertionError, match="references unknown corpus chunk"):
            _validate_benchmark_record(rec, corpus_by_id)

    def test_reject_boolean_grade(
        self, valid_state: tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ) -> None:
        rec, corpus_by_id = valid_state
        rec["relevance_judgments"][0]["grade"] = True  # bool subclasses int
        with pytest.raises(AssertionError, match="Grade must be integer"):
            _validate_benchmark_record(rec, corpus_by_id)

    def test_reject_out_of_range_grade(
        self, valid_state: tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ) -> None:
        rec, corpus_by_id = valid_state
        rec["relevance_judgments"][0]["grade"] = 3
        with pytest.raises(AssertionError, match="Grade must be 0, 1, or 2"):
            _validate_benchmark_record(rec, corpus_by_id)

    def test_reject_broken_citation_quote(
        self, valid_state: tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ) -> None:
        rec, corpus_by_id = valid_state
        rec["required_citations"][0]["quote"] = (
            "This phrase definitely does not exist in the source."
        )
        with pytest.raises(AssertionError, match="does not appear verbatim"):
            _validate_benchmark_record(rec, corpus_by_id)

    def test_reject_duplicate_judgment_chunk(
        self, valid_state: tuple[dict[str, Any], dict[str, dict[str, Any]]]
    ) -> None:
        rec, corpus_by_id = valid_state
        first_cid = rec["relevance_judgments"][0]["chunk_id"]
        rec["relevance_judgments"].append(
            {"chunk_id": first_cid, "grade": 0, "rationale": "Duplicate entry"}
        )
        with pytest.raises(AssertionError, match="Duplicate judgment"):
            _validate_benchmark_record(rec, corpus_by_id)
