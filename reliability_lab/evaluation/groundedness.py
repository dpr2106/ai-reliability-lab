"""Deterministic claim span extraction and lexical overlap groundedness evaluation.

Implemented using Python standard library only, per ADR-005.
Measures token containment and longest common subsequence (LCS) between answer
claims and retrieved evidence chunks.

NOTE: Lexical grounding measures surface token overlap and sequence similarity;
it is a fast heuristic and does NOT determine semantic entailment or contradiction.
Semantic NLI evaluation is addressed in subsequent tasks (GROUND-003).
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Chunk, Query, RetrievalResult

# Unicode-aware word token pattern matching the BM25Retriever convention
_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)

# Sentence boundary terminators (ASCII and full-width CJK punctuation)
_TERMINATORS = {".", "!", "?", "\u3002", "\uff01", "\uff1f"}
_CLOSING_PUNCT = {'"', "'", "\u201d", "\u2019", "\u00bb", ")", "]", "}", "\u203a"}


def tokenize(text: str) -> list[str]:
    """Normalize and tokenize text into words using Unicode-aware matching.

    Performs casefold normalization without stemming or arbitrary stopword
    filtering to keep lexical evaluation transparent, deterministic, and reproducible.

    Args:
        text: Input string to tokenize.

    Returns:
        List of normalized token strings.
    """
    if not text:
        return []
    return _TOKEN_PATTERN.findall(text.casefold())


def extract_claim_spans(text: str) -> list[tuple[int, int]]:
    """Deterministically segment text into sentence-level claim spans with exact offsets.

    Boundary rules:
    - Segments at sentence terminators ('.', '!', '?', CJK full-width) followed by
      whitespace, newline, or end-of-string.
    - Preserves closing quotation marks and brackets following terminal punctuation.
    - Protects decimal numbers (e.g. '1.5', '0.75', '25.5ms') from being split.
    - Segments at newline boundaries ('\\n', '\\r\\n') even without terminal punctuation.
    - Strips leading and trailing whitespace while adjusting (start, end) offsets to
      guarantee that text[start:end] exactly equals the trimmed claim text.
    - Does not return empty or whitespace-only spans.

    Known limitations:
    - Abbreviations containing internal periods followed by whitespace (e.g. 'e.g. ', 'i.e. ')
      will be treated as sentence boundaries under this deterministic rule.

    Args:
        text: Input text string.

    Returns:
        List of half-open character offset tuples (start, end) into text.

    Raises:
        TypeError: If text is not a string.
    """
    if not isinstance(text, str):
        raise TypeError(f"text must be a string, got {type(text).__name__}.")

    if not text or not text.strip():
        return []

    n = len(text)
    spans: list[tuple[int, int]] = []
    seg_start = 0
    i = 0

    while i < n:
        c = text[i]

        # Check for newline boundary
        if c in ("\n", "\r"):
            seg_end = i
            _add_trimmed_span(text, seg_start, seg_end, spans)
            # Advance past consecutive newline characters
            while i < n and text[i] in ("\n", "\r"):
                i += 1
            seg_start = i
            continue

        # Check for sentence terminator
        if c in _TERMINATORS:
            # Protect decimal numbers: digit before '.' and digit after '.'
            if (
                c == "."
                and i > 0
                and (i + 1) < n
                and text[i - 1].isdigit()
                and text[i + 1].isdigit()
            ):
                i += 1
                continue

            # Advance past any immediately trailing closing quotes or brackets
            j = i + 1
            while j < n and text[j] in _CLOSING_PUNCT:
                j += 1

            # Check if this terminator is followed by whitespace, newline, or EOF
            if j >= n or text[j].isspace() or text[j] in ("\n", "\r"):
                seg_end = j
                _add_trimmed_span(text, seg_start, seg_end, spans)
                # Advance past whitespace following the sentence boundary
                while j < n and text[j].isspace() and text[j] not in ("\n", "\r"):
                    j += 1
                seg_start = j
                i = j
                continue

        i += 1

    # Add any remaining tail segment
    if seg_start < n:
        _add_trimmed_span(text, seg_start, n, spans)

    return spans


def _add_trimmed_span(
    text: str,
    raw_start: int,
    raw_end: int,
    spans: list[tuple[int, int]],
) -> None:
    """Trim whitespace from raw segment and append non-empty adjusted span."""
    segment = text[raw_start:raw_end]
    if not segment or not segment.strip():
        return

    leading_ws = len(segment) - len(segment.lstrip())
    trailing_ws = len(segment) - len(segment.rstrip())

    start = raw_start + leading_ws
    end = raw_end - trailing_ws

    if end > start:
        spans.append((start, end))


def compute_token_containment(
    claim_tokens: Sequence[str],
    evidence_tokens: Sequence[str],
) -> float:
    """Calculate the token containment ratio of claim tokens in evidence tokens.

    Containment is defined as the fraction of claim token occurrences whose
    word type appears anywhere in the evidence token set:
        C_token = (claim tokens present in evidence) / (total claim tokens)

    Args:
        claim_tokens: Sequence of normalized claim tokens.
        evidence_tokens: Sequence of normalized evidence chunk tokens.

    Returns:
        Float value bounded in [0.0, 1.0]. Returns 0.0 if claim_tokens is empty.
    """
    if not claim_tokens:
        return 0.0
    evidence_set = set(evidence_tokens)
    matched = sum(1 for tok in claim_tokens if tok in evidence_set)
    return min(1.0, max(0.0, float(matched) / float(len(claim_tokens))))


def compute_lcs_length(seq_a: Sequence[str], seq_b: Sequence[str]) -> int:
    """Compute the length of the longest common subsequence (LCS) of two sequences.

    Uses a deterministic, memory-efficient dynamic programming algorithm with
    O(len_a * len_b) time and O(min(len_a, len_b)) extra space.

    Args:
        seq_a: First token sequence.
        seq_b: Second token sequence.

    Returns:
        Integer length of the longest common subsequence.
    """
    if not seq_a or not seq_b:
        return 0

    # Ensure shorter sequence is used for DP row to minimize extra memory
    if len(seq_a) < len(seq_b):
        short_seq, long_seq = seq_a, seq_b
    else:
        short_seq, long_seq = seq_b, seq_a

    k = len(short_seq)
    dp = [0] * (k + 1)

    for a_elem in long_seq:
        prev = 0
        for j in range(1, k + 1):
            temp = dp[j]
            if a_elem == short_seq[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp

    return dp[k]


def compute_token_lcs_score(
    claim_tokens: Sequence[str],
    evidence_tokens: Sequence[str],
) -> float:
    """Calculate normalized token-level LCS score relative to claim length.

    LCS score is defined as:
        C_LCS = LCS(claim_tokens, evidence_tokens) / |claim_tokens|

    Args:
        claim_tokens: Sequence of normalized claim tokens.
        evidence_tokens: Sequence of normalized evidence chunk tokens.

    Returns:
        Float value bounded in [0.0, 1.0]. Returns 0.0 if claim_tokens is empty.
    """
    if not claim_tokens:
        return 0.0
    lcs_len = compute_lcs_length(claim_tokens, evidence_tokens)
    return min(1.0, max(0.0, float(lcs_len) / float(len(claim_tokens))))


@dataclass(frozen=True)
class ClaimGroundingResult:
    """Evaluation result for an individual answer claim span against evidence.

    Attributes:
        claim_text: Exact substring of the claim extracted from the answer.
        span: Character offset tuple (start, end) indexing the original answer text.
        tokens: Normalized token sequence for this claim.
        best_chunk_id: Chunk ID of the highest-scoring evidence candidate, or None.
        token_containment: Highest token containment score against evidence chunks.
        lcs_score: Highest token-level LCS score against evidence chunks.
        is_grounded: True if both containment and LCS meet configured thresholds.
        candidate_scores: Mapping from candidate chunk IDs to their containment and LCS scores.
        diagnostics: Additional diagnostic details.
    """

    claim_text: str
    span: tuple[int, int]
    tokens: list[str]
    best_chunk_id: str | None
    token_containment: float
    lcs_score: float
    is_grounded: bool
    candidate_scores: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert claim result into a JSON-serializable dictionary."""
        return {
            "claim_text": self.claim_text,
            "span": list(self.span),
            "tokens": list(self.tokens),
            "best_chunk_id": self.best_chunk_id,
            "token_containment": self.token_containment,
            "lcs_score": self.lcs_score,
            "is_grounded": self.is_grounded,
            "candidate_scores": dict(self.candidate_scores),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class GroundednessResult:
    """Aggregate answer-level groundedness evaluation result.

    Attributes:
        aggregate_score: Fraction of claims meeting both lexical grounding thresholds.
        grounded_claims_count: Number of claims classified as grounded.
        unsupported_claims_count: Number of claims failing at least one threshold.
        no_evidence_claims_count: Number of claims with no candidate evidence.
        total_claims: Total number of claim spans extracted from the answer.
        has_claims: True if at least one non-empty claim span was extracted.
        claims: Detailed per-claim grounding evaluation results.
        thresholds: Dictionary of configured threshold parameters.
        retrieved_chunk_ids: List of unique chunk IDs supplied as evidence.
        diagnostics: Summary diagnostics dictionary.
    """

    aggregate_score: float
    grounded_claims_count: int
    unsupported_claims_count: int
    no_evidence_claims_count: int
    total_claims: int
    has_claims: bool
    claims: list[ClaimGroundingResult]
    thresholds: dict[str, float]
    retrieved_chunk_ids: list[str]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate result into a JSON-serializable dictionary."""
        return {
            "aggregate_score": self.aggregate_score,
            "grounded_claims_count": self.grounded_claims_count,
            "unsupported_claims_count": self.unsupported_claims_count,
            "no_evidence_claims_count": self.no_evidence_claims_count,
            "total_claims": self.total_claims,
            "has_claims": self.has_claims,
            "claims": [c.to_dict() for c in self.claims],
            "thresholds": dict(self.thresholds),
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Convert aggregate result into a formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class GroundednessEvaluator:
    """Deterministic lexical overlap groundedness evaluator.

    Extracts claim spans from generated answer text and evaluates token containment
    and token-level LCS against retrieved evidence candidates.

    Attributes:
        containment_threshold: Minimum token containment required for a claim to pass.
        lcs_threshold: Minimum normalized LCS score required for a claim to pass.
    """

    def __init__(
        self,
        containment_threshold: float = 0.5,
        lcs_threshold: float = 0.3,
    ) -> None:
        """Initialize evaluator with configurable lexical thresholds.

        Args:
            containment_threshold: Minimum token containment threshold in [0.0, 1.0].
            lcs_threshold: Minimum token LCS score threshold in [0.0, 1.0].

        Raises:
            TypeError: If thresholds are not finite numbers.
            ValueError: If thresholds are outside the valid [0.0, 1.0] interval.
        """
        self._validate_threshold("containment_threshold", containment_threshold)
        self._validate_threshold("lcs_threshold", lcs_threshold)

        self._containment_threshold = float(containment_threshold)
        self._lcs_threshold = float(lcs_threshold)

    @property
    def containment_threshold(self) -> float:
        """Return configured token containment threshold."""
        return self._containment_threshold

    @property
    def lcs_threshold(self) -> float:
        """Return configured LCS score threshold."""
        return self._lcs_threshold

    @staticmethod
    def _validate_threshold(name: str, value: Any) -> None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{name} must be a float or int, got {type(value).__name__}.")
        if math.isnan(value) or math.isinf(value):
            raise ValueError(f"{name} must be a finite number, got {value}.")
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be within [0.0, 1.0], got {value}.")

    def evaluate(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult | Chunk],
    ) -> GroundednessResult:
        """Evaluate lexical groundedness of answer text against retrieved evidence.

        Args:
            answer_text: Generated answer string to evaluate.
            retrieved: Sequence of RetrievalResult or Chunk objects providing evidence.

        Returns:
            GroundednessResult containing per-claim breakdowns and aggregate score.

        Raises:
            TypeError: If input parameters or elements are of invalid types.
            ValueError: If conflicting chunk content or provenance is detected.
        """
        if not isinstance(answer_text, str):
            raise TypeError(f"answer_text must be a string, got {type(answer_text).__name__}.")

        if not isinstance(retrieved, Sequence) or isinstance(retrieved, (str, bytes)):
            raise TypeError(
                f"retrieved must be a Sequence of RetrievalResult or Chunk, "
                f"got {type(retrieved).__name__}."
            )

        # Validate evidence chunks and detect conflicting provenance
        evidence_chunks: list[Chunk] = []
        seen_chunks: dict[str, Chunk] = {}

        for idx, item in enumerate(retrieved):
            if isinstance(item, RetrievalResult):
                chunk = item.chunk
            elif isinstance(item, Chunk):
                chunk = item
            else:
                raise TypeError(
                    f"retrieved item at index {idx} must be RetrievalResult or Chunk, "
                    f"got {type(item).__name__}."
                )

            cid = chunk.chunk_id
            if cid in seen_chunks:
                existing = seen_chunks[cid]
                if existing.content != chunk.content or existing.doc_id != chunk.doc_id:
                    raise ValueError(
                        f"Conflicting chunk content or provenance detected for chunk '{cid}': "
                        f"doc_id '{existing.doc_id}' vs '{chunk.doc_id}'."
                    )
            else:
                seen_chunks[cid] = chunk
                evidence_chunks.append(chunk)

        retrieved_ids = [c.chunk_id for c in evidence_chunks]

        # Extract sentence claim spans
        spans = extract_claim_spans(answer_text)
        has_claims = len(spans) > 0

        # Pre-tokenize evidence chunks
        evidence_tokens_map = {c.chunk_id: tokenize(c.content) for c in evidence_chunks}

        claims_results: list[ClaimGroundingResult] = []

        for start, end in spans:
            claim_text = answer_text[start:end]
            claim_tokens = tokenize(claim_text)

            candidate_scores: dict[str, dict[str, float]] = {}
            best_chunk_id: str | None = None
            best_containment: float = 0.0
            best_lcs: float = 0.0

            if evidence_chunks and claim_tokens:
                # Rank evidence candidates: (-containment, -lcs, chunk_id)
                candidates: list[tuple[float, float, str]] = []
                for chunk in evidence_chunks:
                    c_tokens = evidence_tokens_map[chunk.chunk_id]
                    containment = compute_token_containment(claim_tokens, c_tokens)
                    lcs = compute_token_lcs_score(claim_tokens, c_tokens)
                    candidate_scores[chunk.chunk_id] = {
                        "token_containment": containment,
                        "lcs_score": lcs,
                    }
                    candidates.append((containment, lcs, chunk.chunk_id))

                # Deterministic selection with tie-breaker
                candidates.sort(key=lambda x: (-x[0], -x[1], x[2]))
                best_containment, best_lcs, best_chunk_id = candidates[0]

            is_grounded = bool(
                best_chunk_id is not None
                and best_containment >= self._containment_threshold
                and best_lcs >= self._lcs_threshold
            )

            claims_results.append(
                ClaimGroundingResult(
                    claim_text=claim_text,
                    span=(start, end),
                    tokens=claim_tokens,
                    best_chunk_id=best_chunk_id,
                    token_containment=best_containment,
                    lcs_score=best_lcs,
                    is_grounded=is_grounded,
                    candidate_scores=candidate_scores,
                    diagnostics={
                        "token_count": len(claim_tokens),
                        "passed_containment": best_containment >= self._containment_threshold,
                        "passed_lcs": best_lcs >= self._lcs_threshold,
                    },
                )
            )

        grounded_count = sum(1 for c in claims_results if c.is_grounded)
        unsupported_count = sum(1 for c in claims_results if not c.is_grounded)
        no_evidence_count = sum(1 for c in claims_results if c.best_chunk_id is None)

        aggregate_score = (
            float(grounded_count) / float(len(claims_results)) if claims_results else 0.0
        )

        return GroundednessResult(
            aggregate_score=aggregate_score,
            grounded_claims_count=grounded_count,
            unsupported_claims_count=unsupported_count,
            no_evidence_claims_count=no_evidence_count,
            total_claims=len(claims_results),
            has_claims=has_claims,
            claims=claims_results,
            thresholds={
                "containment_threshold": self._containment_threshold,
                "lcs_threshold": self._lcs_threshold,
            },
            retrieved_chunk_ids=retrieved_ids,
            diagnostics={
                "evidence_chunks_count": len(evidence_chunks),
                "answer_char_length": len(answer_text),
            },
        )


class LexicalGroundednessMetric(BaseMetric):
    """Adapter metric conforming to BaseMetric for retrieval and groundedness evaluation."""

    def __init__(
        self,
        evaluator: GroundednessEvaluator | None = None,
        containment_threshold: float = 0.5,
        lcs_threshold: float = 0.3,
    ) -> None:
        """Initialize metric adapter."""
        if evaluator is not None:
            self._evaluator = evaluator
        else:
            self._evaluator = GroundednessEvaluator(
                containment_threshold=containment_threshold,
                lcs_threshold=lcs_threshold,
            )

    @property
    def name(self) -> str:
        """Return metric identifier."""
        return "lexical_groundedness"

    @property
    def evaluator(self) -> GroundednessEvaluator:
        """Return the underlying GroundednessEvaluator."""
        return self._evaluator

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute aggregate lexical groundedness for a Query and retrieved evidence."""
        answer = (
            query.metadata.get("generated_answer")
            or query.metadata.get("answer")
            or query.ground_truth_answer
            or ""
        )
        result = self._evaluator.evaluate(str(answer), retrieved)
        return result.aggregate_score

    def evaluate_text(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult | Chunk],
    ) -> float:
        """Direct computation from answer text and retrieved evidence."""
        result = self._evaluator.evaluate(answer_text, retrieved)
        return result.aggregate_score
