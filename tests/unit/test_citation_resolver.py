"""Unit tests for CitationAliasMap and CitationResolver (STING V2-003)."""

from __future__ import annotations

from reliability_lab.core.models import Chunk
from reliability_lab.query.citation import CitationAliasMap, CitationResolver
from reliability_lab.query.models import (
    MalformedCitation,
    ResolvedCitation,
    UnresolvedCitation,
)


def _make_test_chunks() -> list[Chunk]:
    return [
        Chunk(
            chunk_id="doc_1:chunk_0",
            doc_id="doc_1",
            content="Evidence chunk one explaining retrieval observability.",
            page_number=1,
            char_start=0,
            char_end=55,
        ),
        Chunk(
            chunk_id="doc_1:chunk_1",
            doc_id="doc_1",
            content="Evidence chunk two explaining reciprocal rank fusion.",
            page_number=2,
            char_start=56,
            char_end=110,
        ),
    ]


class TestCitationAliasMap:
    """Tests for sequential alias assignment and prompt evidence formatting."""

    def test_alias_mapping_and_retrieval(self) -> None:
        chunks = _make_test_chunks()
        fn_map = {"doc_1": "architecture.pdf"}
        alias_map, traces = CitationAliasMap.from_chunks(chunks, filename_map=fn_map)

        assert len(traces) == 2
        assert traces[0].alias == "CIT-001"
        assert traces[0].filename == "architecture.pdf"
        assert traces[0].chunk_id == "doc_1:chunk_0"
        assert traces[1].alias == "CIT-002"
        assert traces[1].page_number == 2

        assert alias_map.contains("CIT-001") is True
        assert alias_map.contains("cit-001") is True
        assert alias_map.contains("CIT-999") is False

        assert alias_map.get_by_alias("CIT-001") == traces[0]
        assert alias_map.get_by_alias("cit-002") == traces[1]
        assert alias_map.get_by_alias("CIT-999") is None

    def test_prompt_evidence_formatting(self) -> None:
        chunks = _make_test_chunks()
        alias_map, _ = CitationAliasMap.from_chunks(chunks)
        evidence_dicts = alias_map.to_prompt_evidence()

        assert len(evidence_dicts) == 2
        assert evidence_dicts[0] == {
            "chunk_id": "CIT-001",
            "content": "Evidence chunk one explaining retrieval observability.",
        }


class TestCitationResolver:
    """Tests for categorizing valid, unresolved, and malformed citations."""

    def test_resolved_citations_bind_to_chunk_metadata(self) -> None:
        chunks = _make_test_chunks()
        fn_map = {"doc_1": "paper.pdf"}
        alias_map, _ = CitationAliasMap.from_chunks(chunks, filename_map=fn_map)
        resolver = CitationResolver(alias_map)

        text = "Retrieval observability is critical [CIT-001]. Fusion improves ranking [CIT-002]."
        resolved, unresolved, malformed = resolver.resolve(text)

        assert len(resolved) == 2
        assert len(unresolved) == 0
        assert len(malformed) == 0

        assert isinstance(resolved[0], ResolvedCitation)
        assert resolved[0].marker == "[CIT-001]"
        assert resolved[0].alias == "CIT-001"
        assert resolved[0].chunk_id == "doc_1:chunk_0"
        assert resolved[0].filename == "paper.pdf"
        assert resolved[0].page_number == 1
        assert "retrieval observability" in resolved[0].snippet
        assert resolved[0].to_dict()["is_valid"] is True

        assert resolved[1].alias == "CIT-002"
        assert resolved[1].page_number == 2

    def test_unknown_alias_recorded_as_unresolved_without_fabricated_metadata(self) -> None:
        chunks = _make_test_chunks()
        alias_map, _ = CitationAliasMap.from_chunks(chunks)
        resolver = CitationResolver(alias_map)

        text = "Grounded claim [CIT-001] followed by hallucinated citation [CIT-999]."
        resolved, unresolved, malformed = resolver.resolve(text)

        assert len(resolved) == 1
        assert len(unresolved) == 1
        assert len(malformed) == 0

        unres = unresolved[0]
        assert isinstance(unres, UnresolvedCitation)
        assert unres.marker == "[CIT-999]"
        assert unres.alias == "CIT-999"
        assert unres.reason == "unknown_alias_not_in_evidence"
        data = unres.to_dict()
        assert data["is_valid"] is False
        assert "chunk_id" not in data  # Never fabricate chunk IDs!

    def test_malformed_citations_detected_with_span(self) -> None:
        chunks = _make_test_chunks()
        alias_map, _ = CitationAliasMap.from_chunks(chunks)
        resolver = CitationResolver(alias_map)

        text = "Valid [CIT-001], non-numeric [CIT-ABC], and verbose [citation: 1]."
        resolved, unresolved, malformed = resolver.resolve(text)

        assert len(resolved) == 1
        assert len(unresolved) == 0
        assert len(malformed) == 2

        assert isinstance(malformed[0], MalformedCitation)
        assert malformed[0].raw_text == "[CIT-ABC]"
        assert malformed[0].span == (text.index("[CIT-ABC]"), text.index("[CIT-ABC]") + 9)
        assert malformed[0].to_dict()["is_valid"] is False

        assert malformed[1].raw_text == "[citation: 1]"

    def test_generation_abstention_directive_ignored_by_resolver(self) -> None:
        chunks = _make_test_chunks()
        alias_map, _ = CitationAliasMap.from_chunks(chunks)
        resolver = CitationResolver(alias_map)

        text = "[ABSTAIN: INSUFFICIENT_EVIDENCE] The documents do not contain the answer."
        resolved, unresolved, malformed = resolver.resolve(text)

        assert len(resolved) == 0
        assert len(unresolved) == 0
        assert len(malformed) == 0

    def test_citation_deduplication_preserves_first_order(self) -> None:
        chunks = _make_test_chunks()
        alias_map, _ = CitationAliasMap.from_chunks(chunks)
        resolver = CitationResolver(alias_map)

        text = "Fact A [CIT-002]. Fact B [CIT-001]. Fact C [CIT-002]."
        resolved, _, _ = resolver.resolve(text)

        assert len(resolved) == 2
        assert resolved[0].alias == "CIT-002"
        assert resolved[1].alias == "CIT-001"

    def test_chk_alias_unresolved_and_never_fabricated(self) -> None:
        """CHK aliases are not query aliases and must not be rewritten to CIT aliases."""
        chunks = _make_test_chunks()
        alias_map, _ = CitationAliasMap.from_chunks(chunks)
        resolver = CitationResolver(alias_map)

        text = "Model used old format [CHK-001] and cross-query citation [CIT-005]."
        resolved, unresolved, malformed = resolver.resolve(text)

        assert len(resolved) == 0
        assert len(unresolved) == 2
        assert len(malformed) == 0

        # Verify CHK-001 was NOT rewritten to CIT-001
        assert unresolved[0].marker == "[CHK-001]"
        assert unresolved[0].alias == "CHK-001"
        assert unresolved[0].reason == "unsupplied_chk_alias_not_in_query"

        # Verify cross-query unknown citation
        assert unresolved[1].marker == "[CIT-005]"
        assert unresolved[1].alias == "CIT-005"
        assert unresolved[1].reason == "unknown_alias_not_in_evidence"

    def test_supplied_chk_alias_resolves_when_in_query_map(self) -> None:
        """When CHK aliases are explicitly supplied to the query, they resolve cleanly."""
        from reliability_lab.query.models import EvidenceChunkTrace

        chk_trace = EvidenceChunkTrace(
            alias="CHK-001",
            chunk_id="doc_x:chunk_0",
            doc_id="doc_x",
            filename="doc_x.txt",
            page_number=1,
            char_start=0,
            char_end=50,
            content="Explicitly supplied CHK chunk content.",
        )
        alias_map = CitationAliasMap((chk_trace,))
        resolver = CitationResolver(alias_map)

        text = "Answer with explicitly supplied [CHK-001]."
        resolved, unresolved, _malformed = resolver.resolve(text)

        assert len(resolved) == 1
        assert len(unresolved) == 0
        assert resolved[0].alias == "CHK-001"
        assert resolved[0].chunk_id == "doc_x:chunk_0"
