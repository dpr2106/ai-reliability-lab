"""Citation evaluation metrics, citation marker parser, and reference link validator.

Implemented using Python standard library only, per ADR-005.
Supports deterministic, 100% local citation precision and recall evaluation
against golden benchmarks and actual retrieved evidence.
"""

from __future__ import annotations

import json
import re
from abc import ABC
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Query, RetrievalResult

# Regex patterns for citation markers
_BRACKETED_REGEX = re.compile(r"\[(?P<inner>[^\]]+)\]")
_VALID_CITATION_ID_REGEX = re.compile(r"^CIT-\d+$", re.IGNORECASE)
_VALID_CHUNK_ID_REGEX = re.compile(r"^CHK-[A-Z0-9]+-\d+$|^CHK-[A-Z0-9_]+$", re.IGNORECASE)
_MALFORMED_CANDIDATE_REGEX = re.compile(r"^(?:cit|chk|citation)[\s\-_:]?.*$", re.IGNORECASE)


@dataclass(frozen=True)
class CitationMarker:
    """Represents a recognized citation marker extracted from text.

    Attributes:
        raw: Full raw text of the citation marker as appeared in text (e.g. '[CIT-001]').
        marker_id: Normalized identifier of the citation or chunk (e.g. 'CIT-001', 'CHK-HYB-001').
        marker_type: Type classification: 'citation' for citation IDs, 'chunk' for direct chunk IDs.
        span: Character start and end offsets (start, end) within the parsed text.
    """

    raw: str
    marker_id: str
    marker_type: str
    span: tuple[int, int]


@dataclass(frozen=True)
class CitationValidationResult:
    """Structured, JSON-serializable diagnostic result from citation evaluation.

    Attributes:
        citation_precision: Ratio of distinct valid citations to distinct parsed citations.
        citation_recall: Ratio of distinct satisfied citations to distinct required citations.
        has_required_citations: True if at least one required ground-truth citation was specified.
        parsed_citations: All recognized citation marker IDs in emission order.
        distinct_citations: Deduplicated citation marker IDs in first-appearance order.
        valid_citations: Distinct citations whose chunk was present in retrieved evidence.
        invalid_citations: Distinct citations that were unresolved or lacked retrieved evidence.
        unresolved_citations: Distinct citations whose ID was unknown in the benchmark mapping.
        missing_evidence_citations: Distinct citations whose chunk was not in retrieved evidence.
        duplicate_citations: Distinct citation IDs that appeared more than once in the text.
        unreferenced_required_citations: Distinct required citations that were not validly cited.
        malformed_citations: Detected malformed citation-like expressions.
        citation_to_chunk_map: Mapping from resolved citation ID to source chunk ID.
        retrieved_chunk_ids: List of distinct chunk IDs present in the retrieved evidence set.
        diagnostics: Additional serializable diagnostic details.
    """

    citation_precision: float
    citation_recall: float
    has_required_citations: bool
    parsed_citations: list[str]
    distinct_citations: list[str]
    valid_citations: list[str]
    invalid_citations: list[str]
    unresolved_citations: list[str]
    missing_evidence_citations: list[str]
    duplicate_citations: list[str]
    unreferenced_required_citations: list[str]
    malformed_citations: list[str]
    citation_to_chunk_map: dict[str, str]
    retrieved_chunk_ids: list[str]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert validation result into a JSON-serializable dictionary."""
        return {
            "citation_precision": self.citation_precision,
            "citation_recall": self.citation_recall,
            "has_required_citations": self.has_required_citations,
            "parsed_citations": list(self.parsed_citations),
            "distinct_citations": list(self.distinct_citations),
            "valid_citations": list(self.valid_citations),
            "invalid_citations": list(self.invalid_citations),
            "unresolved_citations": list(self.unresolved_citations),
            "missing_evidence_citations": list(self.missing_evidence_citations),
            "duplicate_citations": list(self.duplicate_citations),
            "unreferenced_required_citations": list(self.unreferenced_required_citations),
            "malformed_citations": list(self.malformed_citations),
            "citation_to_chunk_map": dict(self.citation_to_chunk_map),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Convert validation result into a formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class CitationParser:
    """Deterministic citation marker parser for generated answers.

    Extracts citation markers formatted as:
    - [CIT-001] (citation IDs)
    - [CHK-HYB-001] (direct chunk IDs)
    - [CIT-001, CIT-002] (multiple comma- or semicolon-delimited markers)

    Ignores non-citation square brackets (e.g. '[see above]', '[1]', '[note]').
    Detects and reports malformed citation-like markers (e.g. '[CIT-]', '[citation: 1]').
    """

    def parse(self, text: str) -> list[CitationMarker]:
        """Extract all recognized citation markers from text in appearance order.

        Args:
            text: Input string containing generated answer text.

        Returns:
            List of CitationMarker instances.

        Raises:
            TypeError: If text is not a string.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")

        markers: list[CitationMarker] = []
        for match in _BRACKETED_REGEX.finditer(text):
            inner = match.group("inner").strip()
            if not inner:
                continue

            # Split on commas or semicolons for composite bracketed citations: [CIT-001, CIT-002]
            parts = re.split(r"[,;]", inner)
            span_start = match.start()
            span_end = match.end()

            for part in parts:
                token = part.strip()
                if not token:
                    continue

                token_upper = token.upper()
                if _VALID_CITATION_ID_REGEX.match(token_upper):
                    markers.append(
                        CitationMarker(
                            raw=match.group(0),
                            marker_id=token_upper,
                            marker_type="citation",
                            span=(span_start, span_end),
                        )
                    )
                elif _VALID_CHUNK_ID_REGEX.match(token_upper):
                    markers.append(
                        CitationMarker(
                            raw=match.group(0),
                            marker_id=token_upper,
                            marker_type="chunk",
                            span=(span_start, span_end),
                        )
                    )

        return markers

    def extract_marker_ids(self, text: str) -> list[str]:
        """Extract ordered list of recognized citation marker IDs from text.

        Args:
            text: Input answer text.

        Returns:
            List of marker ID strings.
        """
        return [m.marker_id for m in self.parse(text)]

    def find_malformed(self, text: str) -> list[str]:
        """Detect malformed citation-like expressions enclosed in brackets.

        Finds bracketed substrings that appear intended as citations (e.g. starting
        with 'CIT', 'CHK', or 'citation') but fail the formal grammar.

        Args:
            text: Input answer text.

        Returns:
            List of malformed citation strings as they appeared in text.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")

        malformed: list[str] = []
        for match in _BRACKETED_REGEX.finditer(text):
            inner = match.group("inner").strip()
            if not inner:
                continue

            parts = re.split(r"[,;]", inner)
            for part in parts:
                token = part.strip()
                if not token:
                    continue

                token_upper = token.upper()
                is_valid = bool(
                    _VALID_CITATION_ID_REGEX.match(token_upper)
                    or _VALID_CHUNK_ID_REGEX.match(token_upper)
                )
                if not is_valid and _MALFORMED_CANDIDATE_REGEX.match(token):
                    malformed.append(match.group(0))
                    break

        return malformed


def parse_citation_markers(text: str) -> list[CitationMarker]:
    """Convenience function to parse citation markers using default CitationParser."""
    return CitationParser().parse(text)


def extract_citation_ids(text: str) -> list[str]:
    """Convenience function to extract ordered marker IDs using default CitationParser."""
    return CitationParser().extract_marker_ids(text)


def resolve_citation_reference(
    marker_id: str,
    citation_mapping: Mapping[str, str] | None = None,
) -> str | None:
    """Resolve a citation marker ID to its underlying chunk ID.

    Args:
        marker_id: Citation ID (e.g. 'CIT-001') or direct chunk ID (e.g. 'CHK-HYB-001').
        citation_mapping: Optional dictionary mapping citation IDs to chunk IDs.

    Returns:
        Source chunk ID string if resolvable, or None if unresolved.

    Raises:
        TypeError: If marker_id is not a string.
        ValueError: If marker_id is empty.
    """
    if not isinstance(marker_id, str):
        raise TypeError(f"marker_id must be a string, got {type(marker_id).__name__}.")
    clean_id = marker_id.strip().upper()
    if not clean_id:
        raise ValueError("marker_id must be a non-empty string.")

    if citation_mapping is not None:
        if clean_id in citation_mapping:
            return str(citation_mapping[clean_id]).strip()
        # Case-insensitive lookup fallback
        upper_map = {str(k).upper(): str(v).strip() for k, v in citation_mapping.items()}
        if clean_id in upper_map:
            return upper_map[clean_id]

    # Direct chunk ID reference fallback
    if _VALID_CHUNK_ID_REGEX.match(clean_id):
        return clean_id

    return None


def calculate_citation_precision(valid_count: int, total_emitted_count: int) -> float:
    """Calculate citation precision score in [0.0, 1.0].

    Precision is the proportion of distinct cited references that are valid
    (resolve to source chunks present in the retrieved evidence).
    Zero emitted citations yields 0.0.

    Args:
        valid_count: Number of distinct valid citations emitted.
        total_emitted_count: Number of distinct citations emitted in total.

    Returns:
        Float value between 0.0 and 1.0 inclusive.
    """
    if total_emitted_count <= 0 or valid_count <= 0:
        return 0.0
    return min(1.0, max(0.0, float(valid_count) / float(total_emitted_count)))


def calculate_citation_recall(satisfied_count: int, required_count: int) -> float:
    """Calculate citation recall score in [0.0, 1.0].

    Recall is the proportion of distinct required ground-truth citations that
    were correctly referenced in the answer by valid references.
    Zero required citations yields 0.0.

    Args:
        satisfied_count: Number of distinct required citations correctly referenced.
        required_count: Number of distinct required citations in ground truth.

    Returns:
        Float value between 0.0 and 1.0 inclusive.
    """
    if required_count <= 0 or satisfied_count <= 0:
        return 0.0
    return min(1.0, max(0.0, float(satisfied_count) / float(required_count)))


class CitationEvaluator:
    """Evaluates citation validity, precision, and recall for generated answers.

    Validates citations against:
    - Ground-truth citation-to-chunk mapping.
    - Actual retrieved evidence candidates (Sequence[RetrievalResult]).
    - Required citations list from the evaluation benchmark.
    """

    def __init__(self, parser: CitationParser | None = None) -> None:
        """Initialize evaluator with optional customized CitationParser."""
        self._parser = parser or CitationParser()

    @property
    def parser(self) -> CitationParser:
        """Return the underlying CitationParser instance."""
        return self._parser

    def evaluate(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult],
        query: Query | None = None,
        *,
        citation_mapping: Mapping[str, str] | None = None,
        required_citations: Sequence[str] | None = None,
    ) -> CitationValidationResult:
        """Evaluate citations in generated answer text against evidence and ground truth.

        Args:
            answer_text: Generated response containing citation markers.
            retrieved: Sequence of RetrievalResult instances containing retrieved chunks.
            query: Optional Query instance supplying required_citations and metadata mapping.
            citation_mapping: Optional explicit mapping from citation ID to source chunk ID.
            required_citations: Optional explicit sequence of required citation IDs.

        Returns:
            CitationValidationResult with precision, recall, and comprehensive diagnostics.

        Raises:
            TypeError: If input arguments have invalid types.
            ValueError: If citation_mapping contains conflicting or malformed entries.
        """
        if not isinstance(answer_text, str):
            raise TypeError(f"answer_text must be a string, got {type(answer_text).__name__}.")

        if not isinstance(retrieved, Sequence) or isinstance(retrieved, (str, bytes)):
            raise TypeError(
                f"retrieved must be a Sequence of RetrievalResult, got {type(retrieved).__name__}."
            )
        for i, item in enumerate(retrieved):
            if not isinstance(item, RetrievalResult):
                raise TypeError(
                    f"retrieved element at index {i} must be RetrievalResult, "
                    f"got {type(item).__name__}."
                )

        if query is not None and not isinstance(query, Query):
            raise TypeError(f"query must be a Query instance or None, got {type(query).__name__}.")

        # Determine effective citation mapping
        active_mapping: dict[str, str] = {}
        if citation_mapping is not None:
            if not isinstance(citation_mapping, Mapping):
                raise TypeError("citation_mapping must be a Mapping[str, str].")
            for k, v in citation_mapping.items():
                if not isinstance(k, str) or not k.strip():
                    raise ValueError("citation_mapping keys must be non-empty strings.")
                if not isinstance(v, str) or not v.strip():
                    raise ValueError("citation_mapping values must be non-empty strings.")
                norm_k = k.strip().upper()
                norm_v = v.strip()
                if norm_k in active_mapping and active_mapping[norm_k] != norm_v:
                    raise ValueError(
                        f"Conflicting mapping for citation '{norm_k}': "
                        f"'{active_mapping[norm_k]}' vs '{norm_v}'."
                    )
                active_mapping[norm_k] = norm_v
        elif query is not None:
            active_mapping = {
                str(k).upper(): str(v).strip() for k, v in query.citation_to_chunk_map.items()
            }

        # Determine effective required citations
        active_required: list[str] = []
        if required_citations is not None:
            if not isinstance(required_citations, Sequence) or isinstance(
                required_citations, (str, bytes)
            ):
                raise TypeError("required_citations must be a Sequence of strings.")
            for r in required_citations:
                if not isinstance(r, str) or not r.strip():
                    raise ValueError("All required_citations must be non-empty strings.")
                active_required.append(r.strip().upper())
        elif query is not None:
            active_required = [
                str(r).strip().upper() for r in query.required_citations if str(r).strip()
            ]

        # Build retrieved chunk ID set
        retrieved_ids = [r.chunk.chunk_id for r in retrieved]
        retrieved_set = set(retrieved_ids)

        # Parse markers from answer text
        markers = self._parser.parse(answer_text)
        malformed = self._parser.find_malformed(answer_text)
        parsed_ids = [m.marker_id for m in markers]

        # Track duplicates preserving order
        seen: set[str] = set()
        duplicate_ids: list[str] = []
        for mid in parsed_ids:
            if mid in seen and mid not in duplicate_ids:
                duplicate_ids.append(mid)
            seen.add(mid)

        distinct_parsed = list(dict.fromkeys(parsed_ids))

        # Reference resolution and evidence validation
        valid_citations: list[str] = []
        unresolved_citations: list[str] = []
        missing_evidence: list[str] = []
        citation_to_chunk: dict[str, str] = {}

        for mid in distinct_parsed:
            resolved_chunk_id = resolve_citation_reference(mid, active_mapping)
            if resolved_chunk_id is None:
                unresolved_citations.append(mid)
            else:
                citation_to_chunk[mid] = resolved_chunk_id
                if resolved_chunk_id in retrieved_set:
                    valid_citations.append(mid)
                else:
                    missing_evidence.append(mid)

        invalid_citations = unresolved_citations + missing_evidence

        # Citation precision
        precision = calculate_citation_precision(len(valid_citations), len(distinct_parsed))

        # Citation recall
        distinct_required = list(dict.fromkeys(active_required))
        has_required = len(distinct_required) > 0

        satisfied_required: list[str] = []
        unreferenced_required: list[str] = []

        valid_set = set(valid_citations)

        for req_id in distinct_required:
            if req_id in valid_set:
                satisfied_required.append(req_id)
            else:
                # Check if a direct chunk reference satisfied this required citation
                mapped_chunk = active_mapping.get(req_id)
                if mapped_chunk and mapped_chunk in valid_set:
                    # Verify unambiguous mapping among required citations
                    matching_reqs = [
                        r for r in distinct_required if active_mapping.get(r) == mapped_chunk
                    ]
                    if len(matching_reqs) == 1:
                        satisfied_required.append(req_id)
                    else:
                        unreferenced_required.append(req_id)
                else:
                    unreferenced_required.append(req_id)

        recall = calculate_citation_recall(len(satisfied_required), len(distinct_required))

        diagnostics = {
            "total_parsed_count": len(parsed_ids),
            "distinct_parsed_count": len(distinct_parsed),
            "valid_count": len(valid_citations),
            "invalid_count": len(invalid_citations),
            "unresolved_count": len(unresolved_citations),
            "missing_evidence_count": len(missing_evidence),
            "duplicate_count": len(duplicate_ids),
            "malformed_count": len(malformed),
            "required_count": len(distinct_required),
            "satisfied_required_count": len(satisfied_required),
            "unreferenced_required_count": len(unreferenced_required),
        }

        return CitationValidationResult(
            citation_precision=precision,
            citation_recall=recall,
            has_required_citations=has_required,
            parsed_citations=parsed_ids,
            distinct_citations=distinct_parsed,
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            unresolved_citations=unresolved_citations,
            missing_evidence_citations=missing_evidence,
            duplicate_citations=duplicate_ids,
            unreferenced_required_citations=unreferenced_required,
            malformed_citations=malformed,
            citation_to_chunk_map=citation_to_chunk,
            retrieved_chunk_ids=list(dict.fromkeys(retrieved_ids)),
            diagnostics=diagnostics,
        )


class BaseCitationMetric(BaseMetric, ABC):
    """Abstract base class for standalone citation evaluation metrics conforming to BaseMetric."""

    def __init__(self, evaluator: CitationEvaluator | None = None) -> None:
        """Initialize metric with optional CitationEvaluator."""
        self._evaluator = evaluator or CitationEvaluator()

    @property
    def evaluator(self) -> CitationEvaluator:
        """Return the underlying CitationEvaluator."""
        return self._evaluator

    def _extract_answer(self, query: Query) -> str:
        """Extract answer text from query metadata or ground_truth_answer."""
        ans = (
            query.metadata.get("generated_answer")
            or query.metadata.get("answer")
            or query.ground_truth_answer
            or ""
        )
        return str(ans)


class CitationPrecision(BaseCitationMetric):
    """Citation Precision metric: proportion of distinct cited references that are valid.

    A citation is valid if and only if it resolves to a chunk present in the retrieved set.
    """

    @property
    def name(self) -> str:
        """Return metric identifier."""
        return "citation_precision"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute citation precision for a query and retrieved results."""
        answer = self._extract_answer(query)
        result = self._evaluator.evaluate(answer, retrieved, query)
        return result.citation_precision

    def evaluate_text(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult],
        citation_mapping: Mapping[str, str] | None = None,
    ) -> float:
        """Direct computation from answer text and retrieved results."""
        result = self._evaluator.evaluate(
            answer_text,
            retrieved,
            citation_mapping=citation_mapping,
        )
        return result.citation_precision


class CitationRecall(BaseCitationMetric):
    """Citation Recall metric: proportion of required ground-truth citations correctly referenced.

    A required citation is satisfied when it is referenced by a valid citation marker
    whose source chunk was actually retrieved.
    """

    @property
    def name(self) -> str:
        """Return metric identifier."""
        return "citation_recall"

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute citation recall for a query and retrieved results."""
        answer = self._extract_answer(query)
        result = self._evaluator.evaluate(answer, retrieved, query)
        return result.citation_recall

    def evaluate_text(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult],
        required_citations: Sequence[str],
        citation_mapping: Mapping[str, str] | None = None,
    ) -> float:
        """Direct computation from answer text, retrieved results, and required citations."""
        result = self._evaluator.evaluate(
            answer_text,
            retrieved,
            citation_mapping=citation_mapping,
            required_citations=required_citations,
        )
        return result.citation_recall
