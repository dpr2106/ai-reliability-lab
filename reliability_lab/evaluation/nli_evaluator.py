"""Local Natural Language Inference (NLI) entailment evaluator and hallucination risk scorer.

Implemented using Python standard library interfaces and optional neural inference backends
(Hugging Face Transformers / PyTorch), per ADR-005.

Evaluates answer claims against retrieved source chunks to produce calibrated probabilities for:
- Entailment: the evidence logically supports the claim.
- Neutral: the evidence neither sufficiently supports nor contradicts the claim.
- Contradiction: the evidence directly conflicts with the claim.

Computes an aggregate answer-level Hallucination Score representing unsupportedness risk.

NOTE: NLI predictions are model-estimated inference signals conditioned on the underlying
pretrained model; they serve as an automated unsupportedness proxy, not a proof of real-world truth.
"""

from __future__ import annotations

import json
import math
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from reliability_lab.core.interfaces import BaseMetric
from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.evaluation.groundedness import extract_claim_spans


class NLILabel(str, Enum):
    """Canonical Natural Language Inference classification labels."""

    ENTAILMENT = "entailment"
    NEUTRAL = "neutral"
    CONTRADICTION = "contradiction"


class BaseNLIBackend(ABC):
    """Abstract interface for NLI sequence-pair inference backends."""

    @abstractmethod
    def predict_batch(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[dict[NLILabel, float]]:
        """Compute NLI probability distributions for a sequence of (premise, hypothesis) pairs.

        Args:
            pairs: Sequence of tuples where each element is (premise_text, hypothesis_text).

        Returns:
            List of dictionaries mapping each NLILabel to its normalized probability in [0.0, 1.0].

        Raises:
            ValueError: If predictions are invalid, non-finite, or malformed.
        """
        ...


class HuggingFaceNLIBackend(BaseNLIBackend):
    """Neural NLI backend using Hugging Face Transformers and PyTorch.

    Loads pretrained sequence-classification models lazily on demand.
    Supports CPU execution and dynamic label mapping inspection.
    """

    def __init__(
        self,
        model_name_or_path: str = "cross-encoder/nli-deberta-v3-small",
        batch_size: int = 16,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        """Initialize HuggingFace NLI backend configuration.

        Args:
            model_name_or_path: Hugging Face model identifier or local directory path.
            batch_size: Positive integer batch size for model inference.
            device: Target torch device ('cpu', 'cuda', etc.). Defaults to 'cpu' if None.
            local_files_only: Whether to restrict loading to locally cached files only.

        Raises:
            TypeError: If batch_size or local_files_only has invalid type.
            ValueError: If batch_size <= 0.
        """
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer, got {batch_size}.")

        self._model_name = str(model_name_or_path).strip()
        self._batch_size = batch_size
        self._device_str = device or "cpu"
        self._local_files_only = bool(local_files_only)

        self._tokenizer: Any = None
        self._model: Any = None
        self._id2canonical: dict[int, NLILabel] | None = None

    @property
    def model_name(self) -> str:
        """Return configured model identifier or directory path."""
        return self._model_name

    def _ensure_loaded(self) -> None:
        """Load tokenizer and model lazily with explicit error handling."""
        if self._model is not None and self._tokenizer is not None:
            return

        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Hugging Face transformers and PyTorch are required for neural NLI evaluation. "
                "Install them via: pip install 'reliability-lab[nli]'"
            ) from exc

        try:
            self._tokenizer = AutoTokenizer.from_pretrained(
                self._model_name,
                local_files_only=self._local_files_only,
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name,
                local_files_only=self._local_files_only,
            )
            device = torch.device(self._device_str)
            self._model.to(device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load NLI model or tokenizer from '{self._model_name}' "
                f"(local_files_only={self._local_files_only}): {exc}"
            ) from exc

        # Dynamically inspect and resolve model id2label mapping
        id2label = getattr(self._model.config, "id2label", None)
        self._id2canonical = self.resolve_id2label_mapping(id2label)

    @staticmethod
    def resolve_id2label_mapping(
        id2label: Mapping[int | str, str] | None,
    ) -> dict[int, NLILabel]:
        """Map model config id2label into canonical NLILabel enum members.

        Recognizes common variations across NLI architectures (e.g. MNLI, SNLI, ANLI):
        - Entailment: 'entailment', 'entail', 'positive'
        - Neutral: 'neutral', 'neither'
        - Contradiction: 'contradiction', 'contradict', 'negative'

        Args:
            id2label: Mapping from class integer index to model label string.

        Returns:
            Dictionary mapping integer class index to NLILabel.

        Raises:
            ValueError: If mapping cannot unambiguously map all three canonical labels.
        """
        if not id2label or not isinstance(id2label, Mapping):
            raise ValueError(f"Model config has invalid or missing id2label mapping: {id2label}")

        resolved: dict[int, NLILabel] = {}
        found_labels: set[NLILabel] = set()

        for idx_key, raw_label in id2label.items():
            try:
                idx = int(idx_key)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Non-integer class index in id2label: {idx_key}") from exc

            label_str = str(raw_label).strip().lower()
            canonical: NLILabel | None = None

            if "entail" in label_str:
                canonical = NLILabel.ENTAILMENT
            elif "contra" in label_str:
                canonical = NLILabel.CONTRADICTION
            elif "neut" in label_str:
                canonical = NLILabel.NEUTRAL

            if canonical is not None:
                resolved[idx] = canonical
                found_labels.add(canonical)

        required = {NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION}
        missing = required - found_labels
        if missing:
            missing_names = ", ".join(m.value for m in sorted(missing, key=lambda x: x.value))
            raise ValueError(
                f"Unable to resolve required NLI labels ({missing_names}) from model "
                f"id2label: {dict(id2label)}"
            )

        return resolved

    def predict_batch(
        self,
        pairs: Sequence[tuple[str, str]],
    ) -> list[dict[NLILabel, float]]:
        """Compute NLI probabilities for premise-hypothesis pairs in batched inference."""
        if not pairs:
            return []

        self._ensure_loaded()
        assert self._tokenizer is not None
        assert self._model is not None
        assert self._id2canonical is not None

        import torch

        results: list[dict[NLILabel, float]] = []
        n_pairs = len(pairs)

        for batch_start in range(0, n_pairs, self._batch_size):
            batch_pairs = pairs[batch_start : batch_start + self._batch_size]
            premises = [p[0] for p in batch_pairs]
            hypotheses = [p[1] for p in batch_pairs]

            inputs = self._tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            device = next(self._model.parameters()).device
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row in probs:
                row_dist: dict[NLILabel, float] = {}
                for idx, val in enumerate(row):
                    if idx in self._id2canonical:
                        row_dist[self._id2canonical[idx]] = float(val)

                # Validate distribution
                validate_nli_distribution(row_dist)
                results.append(row_dist)

        return results


def validate_nli_distribution(dist: Mapping[NLILabel, float], tolerance: float = 1e-3) -> None:
    """Validate that probability distribution covers all NLILabels and sums to ~1.0.

    Args:
        dist: Mapping from NLILabel to float probability.
        tolerance: Absolute tolerance for probability sum check.

    Raises:
        ValueError: If probabilities are invalid, non-finite, out of [0, 1], or sum is off.
    """
    for label in (NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION):
        if label not in dist:
            raise ValueError(f"NLI distribution missing required label: '{label.value}'.")
        val = dist[label]
        if not isinstance(val, (int, float)) or math.isnan(val) or math.isinf(val):
            raise ValueError(f"Probability for '{label.value}' must be a finite number, got {val}.")
        if not (0.0 <= val <= 1.0):
            raise ValueError(f"Probability for '{label.value}' must be in [0.0, 1.0], got {val}.")

    prob_sum = sum(
        dist[lbl] for lbl in (NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION)
    )
    if abs(prob_sum - 1.0) > tolerance:
        raise ValueError(
            f"NLI probabilities must sum to 1.0 within tolerance {tolerance}, got sum {prob_sum}."
        )


@dataclass(frozen=True)
class ClaimNLIResult:
    """Diagnostics and evaluation result for an individual answer claim span.

    Attributes:
        claim_text: Exact substring of the claim extracted from the answer.
        span: Character offset tuple (start, end) indexing the original answer text.
        best_chunk_id: Chunk ID of the evidence candidate providing maximum entailment.
        entailment_prob: Maximum entailment probability across evidence chunks, or 0.0.
        neutral_prob: Neutral probability from the selected candidate chunk, or 0.0.
        contradiction_prob: Contradiction probability from selected candidate, or 0.0.
        unsupported_risk: Claim unsupportedness risk h_i = 1 - s_i (1.0 if no evidence).
        predicted_label: Argmax probability label for the selected chunk, or None.
        has_evidence: True if at least one candidate chunk was evaluated for this claim.
        candidate_scores: Mapping from candidate chunk IDs to (E, N, C) probabilities.
        diagnostics: Additional diagnostic metadata.
    """

    claim_text: str
    span: tuple[int, int]
    best_chunk_id: str | None
    entailment_prob: float
    neutral_prob: float
    contradiction_prob: float
    unsupported_risk: float
    predicted_label: NLILabel | None
    has_evidence: bool
    candidate_scores: dict[str, dict[str, float]]
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert claim result into a JSON-serializable dictionary."""
        return {
            "claim_text": self.claim_text,
            "span": list(self.span),
            "best_chunk_id": self.best_chunk_id,
            "entailment_prob": self.entailment_prob,
            "neutral_prob": self.neutral_prob,
            "contradiction_prob": self.contradiction_prob,
            "unsupported_risk": self.unsupported_risk,
            "predicted_label": self.predicted_label.value if self.predicted_label else None,
            "has_evidence": self.has_evidence,
            "candidate_scores": dict(self.candidate_scores),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class NLIResult:
    """Aggregate answer-level NLI entailment and hallucination risk result.

    Attributes:
        hallucination_score: Average unsupportedness risk across claims, H_answer in [0.0, 1.0].
        total_claims: Total number of claim spans extracted from the answer.
        has_claims: True if at least one claim span was extracted from the answer.
        claims: Detailed per-claim NLI evaluation results.
        entailed_claims_count: Number of claims where ENTAILMENT was the argmax label.
        neutral_claims_count: Number of claims where NEUTRAL was the argmax label.
        contradicted_claims_count: Number of claims where CONTRADICTION was the argmax label.
        no_evidence_claims_count: Number of claims with no candidate evidence available.
        retrieved_chunk_ids: List of unique chunk IDs evaluated as candidate evidence.
        model_name: Identifier of the NLI model or backend used for inference.
        diagnostics: Summary diagnostics dictionary.
    """

    hallucination_score: float
    total_claims: int
    has_claims: bool
    claims: list[ClaimNLIResult]
    entailed_claims_count: int
    neutral_claims_count: int
    contradicted_claims_count: int
    no_evidence_claims_count: int
    retrieved_chunk_ids: list[str]
    model_name: str
    diagnostics: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Convert aggregate result into a JSON-serializable dictionary."""
        return {
            "hallucination_score": self.hallucination_score,
            "total_claims": self.total_claims,
            "has_claims": self.has_claims,
            "claims": [c.to_dict() for c in self.claims],
            "entailed_claims_count": self.entailed_claims_count,
            "neutral_claims_count": self.neutral_claims_count,
            "contradicted_claims_count": self.contradicted_claims_count,
            "no_evidence_claims_count": self.no_evidence_claims_count,
            "retrieved_chunk_ids": list(self.retrieved_chunk_ids),
            "model_name": self.model_name,
            "diagnostics": dict(self.diagnostics),
        }

    def to_json(self, indent: int | None = 2) -> str:
        """Convert aggregate result into a formatted JSON string."""
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=indent)


class NLIEntailmentEvaluator:
    """Evaluates answer claims against retrieved evidence using Natural Language Inference.

    Extracts claims via exact sentence segmentation, constructs independent (premise, hypothesis)
    pairs, and selects the maximum entailment candidate with deterministic tie-breaking.
    """

    def __init__(
        self,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        backend: BaseNLIBackend | None = None,
        batch_size: int = 16,
        device: str | None = None,
        local_files_only: bool = True,
    ) -> None:
        """Initialize NLI evaluator with optional custom backend or model configuration.

        Args:
            model_name: Pretrained model identifier or local directory.
            backend: Injected BaseNLIBackend implementation (used for unit tests or custom engines).
            batch_size: Inference batch size.
            device: Torch device string ('cpu', 'cuda'). Defaults to 'cpu'.
            local_files_only: Whether to restrict Hugging Face loading to local cache only.
        """
        self._model_name = str(model_name).strip()
        if backend is not None:
            if not isinstance(backend, BaseNLIBackend):
                raise TypeError(
                    f"backend must implement BaseNLIBackend, got {type(backend).__name__}."
                )
            self._backend = backend
        else:
            self._backend = HuggingFaceNLIBackend(
                model_name_or_path=self._model_name,
                batch_size=batch_size,
                device=device,
                local_files_only=local_files_only,
            )

    @property
    def model_name(self) -> str:
        """Return identifier of the model or backend."""
        return self._model_name

    @property
    def backend(self) -> BaseNLIBackend:
        """Return the underlying inference backend."""
        return self._backend

    def evaluate(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult | Chunk],
    ) -> NLIResult:
        """Evaluate NLI entailment and hallucination score for generated answer text.

        Args:
            answer_text: Generated answer string to evaluate.
            retrieved: Sequence of RetrievalResult or Chunk instances supplying evidence.

        Returns:
            NLIResult containing per-claim breakdowns and aggregate Hallucination Score.

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

        # Validate evidence chunks and reject conflicting duplicate provenance
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

        # Extract sentence claim spans using shared segmentation contract
        spans = extract_claim_spans(answer_text)
        has_claims = len(spans) > 0

        # Handle empty answer contract
        if not has_claims:
            return NLIResult(
                hallucination_score=0.0,
                total_claims=0,
                has_claims=False,
                claims=[],
                entailed_claims_count=0,
                neutral_claims_count=0,
                contradicted_claims_count=0,
                no_evidence_claims_count=0,
                retrieved_chunk_ids=retrieved_ids,
                model_name=self._model_name,
                diagnostics={
                    "evidence_chunks_count": len(evidence_chunks),
                    "answer_char_length": len(answer_text),
                },
            )

        # Handle case where claims exist but evidence set is empty
        if not evidence_chunks:
            claim_results: list[ClaimNLIResult] = []
            for start, end in spans:
                claim_text = answer_text[start:end]
                claim_results.append(
                    ClaimNLIResult(
                        claim_text=claim_text,
                        span=(start, end),
                        best_chunk_id=None,
                        entailment_prob=0.0,
                        neutral_prob=0.0,
                        contradiction_prob=0.0,
                        unsupported_risk=1.0,
                        predicted_label=None,
                        has_evidence=False,
                        candidate_scores={},
                        diagnostics={"no_evidence_available": True},
                    )
                )

            return NLIResult(
                hallucination_score=1.0,
                total_claims=len(claim_results),
                has_claims=True,
                claims=claim_results,
                entailed_claims_count=0,
                neutral_claims_count=0,
                contradicted_claims_count=0,
                no_evidence_claims_count=len(claim_results),
                retrieved_chunk_ids=[],
                model_name=self._model_name,
                diagnostics={
                    "evidence_chunks_count": 0,
                    "answer_char_length": len(answer_text),
                },
            )

        # Build all independent (premise, hypothesis) pairs: (chunk.content, claim_text)
        pair_metadata: list[tuple[int, str]] = []  # (claim_index, chunk_id)
        inference_pairs: list[tuple[str, str]] = []

        for c_idx, (start, end) in enumerate(spans):
            claim_text = answer_text[start:end]
            for chunk in evidence_chunks:
                pair_metadata.append((c_idx, chunk.chunk_id))
                inference_pairs.append((chunk.content, claim_text))

        # Run batch prediction through configured backend
        distributions = self._backend.predict_batch(inference_pairs)
        if len(distributions) != len(inference_pairs):
            raise ValueError(
                f"NLI backend returned {len(distributions)} distributions for "
                f"{len(inference_pairs)} input pairs."
            )

        # Organize predictions per claim
        claim_candidate_map: dict[int, dict[str, dict[NLILabel, float]]] = {
            i: {} for i in range(len(spans))
        }
        for (c_idx, chunk_id), dist in zip(pair_metadata, distributions, strict=True):
            validate_nli_distribution(dist)
            claim_candidate_map[c_idx][chunk_id] = dist

        claim_results = []
        entailed_count = 0
        neutral_count = 0
        contradicted_count = 0

        for c_idx, (start, end) in enumerate(spans):
            claim_text = answer_text[start:end]
            candidates = claim_candidate_map[c_idx]

            # Deterministic selection key: (-entailment, -neutral, contradiction, chunk_id)
            ranked_candidates = []
            serialized_scores: dict[str, dict[str, float]] = {}

            for chunk_id, dist in candidates.items():
                e_prob = dist[NLILabel.ENTAILMENT]
                n_prob = dist[NLILabel.NEUTRAL]
                c_prob = dist[NLILabel.CONTRADICTION]

                serialized_scores[chunk_id] = {
                    NLILabel.ENTAILMENT.value: e_prob,
                    NLILabel.NEUTRAL.value: n_prob,
                    NLILabel.CONTRADICTION.value: c_prob,
                }
                ranked_candidates.append((-e_prob, -n_prob, c_prob, chunk_id, dist))

            ranked_candidates.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
            best_neg_e, best_neg_n, best_c, best_cid, best_dist = ranked_candidates[0]

            best_e_prob = -best_neg_e
            best_n_prob = -best_neg_n
            best_c_prob = best_c
            unsupported_risk = max(0.0, min(1.0, 1.0 - best_e_prob))

            # Determine predicted label for the selected best candidate
            argmax_label = max(
                (NLILabel.ENTAILMENT, NLILabel.NEUTRAL, NLILabel.CONTRADICTION),
                key=lambda lbl: best_dist[lbl],
            )

            if argmax_label == NLILabel.ENTAILMENT:
                entailed_count += 1
            elif argmax_label == NLILabel.NEUTRAL:
                neutral_count += 1
            elif argmax_label == NLILabel.CONTRADICTION:
                contradicted_count += 1

            claim_results.append(
                ClaimNLIResult(
                    claim_text=claim_text,
                    span=(start, end),
                    best_chunk_id=best_cid,
                    entailment_prob=best_e_prob,
                    neutral_prob=best_n_prob,
                    contradiction_prob=best_c_prob,
                    unsupported_risk=unsupported_risk,
                    predicted_label=argmax_label,
                    has_evidence=True,
                    candidate_scores=serialized_scores,
                    diagnostics={
                        "candidate_chunks_evaluated": len(candidates),
                        "argmax_label": argmax_label.value,
                    },
                )
            )

        # Aggregate answer Hallucination Score: H_answer = mean(h_i)
        mean_hallucination_score = sum(c.unsupported_risk for c in claim_results) / float(
            len(claim_results)
        )
        bounded_score = max(0.0, min(1.0, float(mean_hallucination_score)))

        return NLIResult(
            hallucination_score=bounded_score,
            total_claims=len(claim_results),
            has_claims=True,
            claims=claim_results,
            entailed_claims_count=entailed_count,
            neutral_claims_count=neutral_count,
            contradicted_claims_count=contradicted_count,
            no_evidence_claims_count=0,
            retrieved_chunk_ids=retrieved_ids,
            model_name=self._model_name,
            diagnostics={
                "evidence_chunks_count": len(evidence_chunks),
                "answer_char_length": len(answer_text),
                "pairs_inferred_count": len(inference_pairs),
            },
        )


class NLIHallucinationMetric(BaseMetric):
    """Adapter metric conforming to BaseMetric evaluating Hallucination Score via NLI."""

    def __init__(
        self,
        evaluator: NLIEntailmentEvaluator | None = None,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
        backend: BaseNLIBackend | None = None,
    ) -> None:
        """Initialize metric adapter with optional evaluator or backend."""
        if evaluator is not None:
            self._evaluator = evaluator
        else:
            self._evaluator = NLIEntailmentEvaluator(model_name=model_name, backend=backend)

    @property
    def name(self) -> str:
        """Return metric identifier."""
        return "nli_hallucination_score"

    @property
    def evaluator(self) -> NLIEntailmentEvaluator:
        """Return the underlying NLIEntailmentEvaluator."""
        return self._evaluator

    def compute(self, retrieved: Sequence[RetrievalResult], query: Query) -> float:
        """Compute hallucination score for a Query and retrieved evidence."""
        answer = (
            query.metadata.get("generated_answer")
            or query.metadata.get("answer")
            or query.ground_truth_answer
            or ""
        )
        result = self._evaluator.evaluate(str(answer), retrieved)
        return result.hallucination_score

    def evaluate_text(
        self,
        answer_text: str,
        retrieved: Sequence[RetrievalResult | Chunk],
    ) -> float:
        """Direct computation from answer text and retrieved evidence."""
        result = self._evaluator.evaluate(answer_text, retrieved)
        return result.hallucination_score
