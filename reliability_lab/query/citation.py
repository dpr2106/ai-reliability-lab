"""Citation aliasing, evidence mapping, and response citation resolution."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Any

from reliability_lab.core.models import Chunk
from reliability_lab.query.models import (
    EvidenceChunkTrace,
    MalformedCitation,
    ResolvedCitation,
    UnresolvedCitation,
)


class CitationAliasMap:
    """Manages sequential alias assignment ([CIT-001] ... [CIT-00K]) for prompt evidence chunks."""

    def __init__(self, evidence_chunks: Sequence[EvidenceChunkTrace]) -> None:
        self._traces = list(evidence_chunks)
        self._by_alias: dict[str, EvidenceChunkTrace] = {t.alias.upper(): t for t in self._traces}
        self._by_chunk_id: dict[str, EvidenceChunkTrace] = {t.chunk_id: t for t in self._traces}

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[Chunk],
        filename_map: dict[str, str] | None = None,
    ) -> tuple[CitationAliasMap, list[EvidenceChunkTrace]]:
        """Construct sequential alias mappings from an ordered list of retrieved chunks."""
        fn_map = filename_map or {}
        traces: list[EvidenceChunkTrace] = []
        for idx, chunk in enumerate(chunks, start=1):
            alias = f"CIT-{idx:03d}"
            filename = fn_map.get(chunk.doc_id, f"doc_{chunk.doc_id}")
            traces.append(
                EvidenceChunkTrace(
                    alias=alias,
                    chunk_id=chunk.chunk_id,
                    doc_id=chunk.doc_id,
                    filename=filename,
                    page_number=chunk.page_number,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    content=chunk.content,
                )
            )
        return cls(traces), traces

    def get_by_alias(self, alias: str) -> EvidenceChunkTrace | None:
        """Lookup evidence trace by alias (case-insensitive)."""
        return self._by_alias.get(alias.strip().upper())

    def contains(self, alias: str) -> bool:
        """Check whether alias is registered in active evidence."""
        return alias.strip().upper() in self._by_alias

    def to_prompt_evidence(self) -> list[dict[str, Any]]:
        """Produce dictionary representations formatted for build_rag_prompt()."""
        return [{"chunk_id": t.alias, "content": t.content} for t in self._traces]


class CitationResolver:
    """Parses answer text, categorizing citations into resolved, unresolved, and malformed."""

    # Matches bracketed expressions
    _BRACKET_PATTERN = re.compile(r"\[([^\]]+)\]")
    # Matches valid citation alias formats: CIT-001 or CHK-001
    _VALID_ALIAS_PATTERN = re.compile(r"^(?:CIT|CHK)-(\d+)$", re.IGNORECASE)
    # Matches citation-like markers that fail standard syntax
    _MALFORMED_CANDIDATE_PATTERN = re.compile(
        r"^(?:CIT|CHK)-[A-Za-z0-9_-]+$|^(?:citation|cite|source|ref|chunk)[\s:-]*\w+$",
        re.IGNORECASE,
    )

    def __init__(self, alias_map: CitationAliasMap) -> None:
        self._alias_map = alias_map

    def resolve(
        self,
        answer_text: str,
    ) -> tuple[
        tuple[ResolvedCitation, ...],
        tuple[UnresolvedCitation, ...],
        tuple[MalformedCitation, ...],
    ]:
        """Extract and categorize all citation occurrences in the generated answer text.

        Args:
            answer_text: The complete generated text returned by the language model.

        Returns:
            Tuple of (resolved_citations, unresolved_citations, malformed_citations).
        """
        if not answer_text:
            return (), (), ()

        resolved: list[ResolvedCitation] = []
        unresolved: list[UnresolvedCitation] = []
        malformed: list[MalformedCitation] = []

        seen_resolved: set[str] = set()
        seen_unresolved: set[str] = set()
        seen_malformed: set[str] = set()

        for match in self._BRACKET_PATTERN.finditer(answer_text):
            inner = match.group(1).strip()
            raw_marker = match.group(0)

            # Skip generation abstention directives
            if inner.upper().startswith("ABSTAIN:"):
                continue

            valid_alias_match = self._VALID_ALIAS_PATTERN.match(inner)
            if valid_alias_match:
                alias_num = int(valid_alias_match.group(1))
                canonical_alias = f"CIT-{alias_num:03d}"
                trace = self._alias_map.get_by_alias(canonical_alias)

                if trace is not None:
                    if canonical_alias not in seen_resolved:
                        seen_resolved.add(canonical_alias)
                        snippet = trace.content[:200]
                        resolved.append(
                            ResolvedCitation(
                                marker=raw_marker,
                                alias=canonical_alias,
                                chunk_id=trace.chunk_id,
                                doc_id=trace.doc_id,
                                filename=trace.filename,
                                page_number=trace.page_number,
                                char_start=trace.char_start,
                                char_end=trace.char_end,
                                snippet=snippet,
                            )
                        )
                else:
                    if canonical_alias not in seen_unresolved:
                        seen_unresolved.add(canonical_alias)
                        unresolved.append(
                            UnresolvedCitation(
                                marker=raw_marker,
                                alias=canonical_alias,
                                reason="unknown_alias_not_in_evidence",
                            )
                        )
                continue

            # Check if it looks like a malformed citation marker
            if self._MALFORMED_CANDIDATE_PATTERN.match(inner):
                if raw_marker not in seen_malformed:
                    seen_malformed.add(raw_marker)
                    malformed.append(
                        MalformedCitation(
                            raw_text=raw_marker,
                            span=(match.start(), match.end()),
                            reason="invalid_citation_syntax",
                        )
                    )

        return tuple(resolved), tuple(unresolved), tuple(malformed)
