"""Local Sentence-Transformers embedding adapter and vector generation pipeline.

Provides reproducible, CPU-friendly dense vector embeddings using all-MiniLM-L6-v2,
strict 384-dimensional shape validation, deterministic L2 normalization, device fallback,
and offline operation support.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_EMBEDDING_DIM = 384
DEFAULT_MODEL_REVISION = "e4ce9877abf3ed59ced85d2278ed52520f93874b"


def compute_model_artifact_fingerprint(
    model_dir_or_path: Path | str,
    require_complete: bool = False,
) -> str:
    """Build a deterministic SHA-256 manifest over loaded model and tokenizer assets.

    Never includes absolute filesystem paths in the manifest or fingerprint.
    Sorts files deterministically by relative POSIX path.
    When require_complete is True, enforces that the directory contains execution-relevant
    assets for weights, configuration, and tokenizer, failing closed if any are missing.
    """
    model_dir = Path(model_dir_or_path)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model artifact path '{model_dir.name}' does not exist.")

    manifest: list[dict[str, Any]] = []

    if model_dir.is_file():
        if require_complete:
            raise ValueError(
                f"Model snapshot in '{model_dir.name}' is a single file, "
                "not a complete model directory."
            )
        content_hash = hashlib.sha256(model_dir.read_bytes()).hexdigest()
        manifest.append(
            {
                "path": model_dir.name,
                "sha256": content_hash,
                "size": model_dir.stat().st_size,
            }
        )
    else:
        for path in sorted(model_dir.rglob("*")):
            if not path.is_file():
                continue
            # Ignore hidden files, documentation, and VCS metadata
            if path.name.startswith(".") or path.name.lower() in (
                "readme.md",
                "license.txt",
                "notice",
            ):
                continue
            if ".git" in path.parts:
                continue

            rel_posix = path.relative_to(model_dir).as_posix()
            content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
            manifest.append(
                {
                    "path": rel_posix,
                    "sha256": content_hash,
                    "size": path.stat().st_size,
                }
            )

    if not manifest:
        raise ValueError(f"No valid model assets found in '{model_dir.name}'.")

    if require_complete:
        weight_exts = (".safetensors", ".bin", ".pt", ".onnx", ".ckpt", ".h5", ".ot")
        has_weights = any(p["path"].lower().endswith(weight_exts) for p in manifest)
        config_names = {"config.json", "modules.json", "sentence_bert_config.json"}
        has_config = any(Path(p["path"]).name.lower() in config_names for p in manifest)
        tokenizer_names = {
            "tokenizer.json",
            "vocab.txt",
            "tokenizer_config.json",
            "sentencepiece.bpe.model",
            "spiece.model",
            "vocab.json",
            "merges.txt",
        }
        has_tokenizer = any(Path(p["path"]).name.lower() in tokenizer_names for p in manifest)

        missing = []
        if not has_weights:
            missing.append("weights")
        if not has_config:
            missing.append("configuration")
        if not has_tokenizer:
            missing.append("tokenizer")

        if missing:
            raise ValueError(
                f"Incomplete model artifact directory '{model_dir.name}': "
                f"missing execution-relevant assets ({', '.join(missing)})."
            )

    manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(manifest_bytes).hexdigest()
    return f"sha256:{digest}"


def compute_pipeline_fingerprint(
    canonical_model_id: str,
    dimension: int,
    artifact_fingerprint: str,
    model_revision: str | None = None,
    distance_metric: str = "ip",
    document_instruction: str = "",
    faiss_normalize: bool = True,
    max_seq_length: int = 256,
    pooling_strategy: str = "mean",
    query_instruction: str = "",
    sentence_transformer_normalize: bool = True,
    truncation_side: str = "right",
    truncation_strategy: str = "longest_first",
) -> str:
    """Compute deterministic SHA-256 pipeline fingerprint over canonical parameters."""
    pipeline_payload = {
        "artifact_fingerprint": artifact_fingerprint,
        "canonical_model_id": canonical_model_id,
        "dimension": dimension,
        "distance_metric": distance_metric,
        "document_instruction": document_instruction or "",
        "faiss_normalize": faiss_normalize,
        "max_seq_length": max_seq_length,
        "model_revision": model_revision or "",
        "pooling_strategy": pooling_strategy,
        "query_instruction": query_instruction or "",
        "sentence_transformer_normalize": sentence_transformer_normalize,
        "truncation_side": truncation_side,
        "truncation_strategy": truncation_strategy,
    }
    digest = hashlib.sha256(
        json.dumps(pipeline_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


@dataclass(frozen=True)
class EmbeddingIdentity:
    """Cryptographic and architectural identity specification for dense embeddings."""

    canonical_model_id: str
    display_label: str
    model_revision: str | None
    artifact_fingerprint: str
    pipeline_fingerprint: str
    dimension: int
    sentence_transformer_normalize: bool
    faiss_normalize: bool
    distance_metric: str = "ip"
    pooling_strategy: str = "mean"
    max_seq_length: int = 256
    truncation_side: str = "right"
    truncation_strategy: str = "longest_first"
    query_instruction: str = ""
    document_instruction: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize embedding identity to dictionary representation."""
        return {
            "canonical_model_id": self.canonical_model_id,
            "display_label": self.display_label,
            "model_revision": self.model_revision,
            "artifact_fingerprint": self.artifact_fingerprint,
            "pipeline_fingerprint": self.pipeline_fingerprint,
            "dimension": self.dimension,
            "sentence_transformer_normalize": self.sentence_transformer_normalize,
            "faiss_normalize": self.faiss_normalize,
            "distance_metric": self.distance_metric,
            "pooling_strategy": self.pooling_strategy,
            "max_seq_length": self.max_seq_length,
            "truncation_side": self.truncation_side,
            "truncation_strategy": self.truncation_strategy,
            "query_instruction": self.query_instruction,
            "document_instruction": self.document_instruction,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EmbeddingIdentity:
        """Instantiate and strictly validate EmbeddingIdentity from dictionary representation.

        Rejects missing fields, invalid types, malformed fingerprints, and internally
        inconsistent pipeline fingerprints.
        """
        import re

        if not isinstance(data, dict):
            raise TypeError("Embedding identity data must be a dictionary.")

        required_keys = (
            "canonical_model_id",
            "display_label",
            "artifact_fingerprint",
            "pipeline_fingerprint",
            "dimension",
            "sentence_transformer_normalize",
            "faiss_normalize",
            "distance_metric",
            "pooling_strategy",
            "max_seq_length",
            "truncation_side",
            "truncation_strategy",
            "query_instruction",
            "document_instruction",
        )
        for k in required_keys:
            if k not in data:
                raise ValueError(f"Missing required embedding identity field: '{k}'")

        dim = data["dimension"]
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"Invalid dimension in embedding identity: {dim!r}")

        max_seq = data["max_seq_length"]
        if isinstance(max_seq, bool) or not isinstance(max_seq, int) or max_seq <= 0:
            raise ValueError(f"Invalid max_seq_length in embedding identity: {max_seq!r}")

        st_norm = data["sentence_transformer_normalize"]
        if not isinstance(st_norm, bool):
            raise ValueError(
                f"Invalid sentence_transformer_normalize in embedding identity: {st_norm!r}"
            )

        faiss_norm = data["faiss_normalize"]
        if not isinstance(faiss_norm, bool):
            raise ValueError(f"Invalid faiss_normalize in embedding identity: {faiss_norm!r}")

        art_fp = str(data["artifact_fingerprint"])
        if not re.match(r"^sha256:[0-9a-f]{64}$", art_fp):
            raise ValueError(
                f"Invalid or unverified artifact_fingerprint in embedding identity: '{art_fp}'"
            )

        pipe_fp = str(data["pipeline_fingerprint"])
        if not re.match(r"^sha256:[0-9a-f]{64}$", pipe_fp):
            raise ValueError(
                f"Invalid pipeline_fingerprint format in embedding identity: '{pipe_fp}'"
            )

        model_rev = str(data["model_revision"]) if data.get("model_revision") is not None else None

        recomputed_fp = compute_pipeline_fingerprint(
            canonical_model_id=str(data["canonical_model_id"]),
            dimension=dim,
            artifact_fingerprint=art_fp,
            model_revision=model_rev,
            distance_metric=str(data["distance_metric"]),
            document_instruction=str(data["document_instruction"]),
            faiss_normalize=faiss_norm,
            max_seq_length=max_seq,
            pooling_strategy=str(data["pooling_strategy"]),
            query_instruction=str(data["query_instruction"]),
            sentence_transformer_normalize=st_norm,
            truncation_side=str(data["truncation_side"]),
            truncation_strategy=str(data["truncation_strategy"]),
        )
        if recomputed_fp != pipe_fp:
            raise ValueError(
                f"Internally inconsistent or tampered embedding identity: declared pipeline "
                f"fingerprint '{pipe_fp}' does not match recomputed '{recomputed_fp}'."
            )

        return cls(
            canonical_model_id=str(data["canonical_model_id"]),
            display_label=str(data["display_label"]),
            model_revision=model_rev,
            artifact_fingerprint=art_fp,
            pipeline_fingerprint=pipe_fp,
            dimension=dim,
            sentence_transformer_normalize=st_norm,
            faiss_normalize=faiss_norm,
            distance_metric=str(data["distance_metric"]),
            pooling_strategy=str(data["pooling_strategy"]),
            max_seq_length=max_seq,
            truncation_side=str(data["truncation_side"]),
            truncation_strategy=str(data["truncation_strategy"]),
            query_instruction=str(data["query_instruction"]),
            document_instruction=str(data["document_instruction"]),
        )


class BaseEmbedder(ABC):
    """Abstract interface for dense text embedding models."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier or path."""
        ...

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Return the output vector dimension (e.g. 384)."""
        ...

    @property
    def canonical_model_id(self) -> str:
        """Return the public canonical identifier for the embedding model."""
        name = self.model_name
        if name in (DEFAULT_MODEL_NAME, f"sentence-transformers/{DEFAULT_MODEL_NAME}"):
            return f"sentence-transformers/{DEFAULT_MODEL_NAME}"
        return name

    @property
    def display_label(self) -> str:
        """Return user-facing sanitized label without local paths."""
        name = self.model_name
        if "/" in name:
            return name.split("/")[-1]
        return name

    @property
    def model_revision(self) -> str | None:
        """Return git commit hash or revision if established."""
        return None

    @property
    def artifact_fingerprint(self) -> str:
        """Return SHA-256 digest of loaded model assets, or 'unavailable' if unverified."""
        return "unavailable:unverified_model_artifacts"

    @property
    def distance_metric(self) -> str:
        """Return distance metric, default 'ip' (inner product)."""
        return "ip"

    @property
    def pooling_strategy(self) -> str:
        """Return pooling strategy, default 'mean'."""
        return "mean"

    @property
    def max_seq_length(self) -> int:
        """Return maximum sequence length in tokens."""
        return 256

    @property
    def truncation_side(self) -> str:
        """Return truncation side, default 'right'."""
        return "right"

    @property
    def truncation_strategy(self) -> str:
        """Return truncation strategy, default 'longest_first'."""
        return "longest_first"

    @property
    def query_instruction(self) -> str:
        """Return query instruction prefix, default empty."""
        return ""

    @property
    def document_instruction(self) -> str:
        """Return document instruction prefix, default empty."""
        return ""

    @property
    def normalize_embeddings(self) -> bool:
        """Return whether embeddings are normalized."""
        return True

    def get_pipeline_fingerprint(self, faiss_normalize: bool = True) -> str:
        """Return SHA-256 over 13 canonical parameters with the specified FAISS normalization."""
        return compute_pipeline_fingerprint(
            canonical_model_id=self.canonical_model_id,
            dimension=self.embedding_dim,
            artifact_fingerprint=self.artifact_fingerprint,
            model_revision=self.model_revision,
            distance_metric=self.distance_metric,
            document_instruction=self.document_instruction,
            faiss_normalize=faiss_normalize,
            max_seq_length=self.max_seq_length,
            pooling_strategy=self.pooling_strategy,
            query_instruction=self.query_instruction,
            sentence_transformer_normalize=self.normalize_embeddings,
            truncation_side=self.truncation_side,
            truncation_strategy=self.truncation_strategy,
        )

    def get_embedding_identity(self, faiss_normalize: bool = True) -> EmbeddingIdentity:
        """Return complete EmbeddingIdentity dataclass for the specified FAISS normalization."""
        return EmbeddingIdentity(
            canonical_model_id=self.canonical_model_id,
            display_label=self.display_label,
            model_revision=self.model_revision,
            artifact_fingerprint=self.artifact_fingerprint,
            pipeline_fingerprint=self.get_pipeline_fingerprint(faiss_normalize=faiss_normalize),
            dimension=self.embedding_dim,
            sentence_transformer_normalize=self.normalize_embeddings,
            faiss_normalize=faiss_normalize,
            distance_metric=self.distance_metric,
            pooling_strategy=self.pooling_strategy,
            max_seq_length=self.max_seq_length,
            truncation_side=self.truncation_side,
            truncation_strategy=self.truncation_strategy,
            query_instruction=self.query_instruction,
            document_instruction=self.document_instruction,
        )

    @property
    def pipeline_fingerprint(self) -> str:
        """Return default SHA-256 over 13 canonical parameters (faiss_normalize=True)."""
        return self.get_pipeline_fingerprint(faiss_normalize=True)

    @property
    def embedding_identity(self) -> EmbeddingIdentity:
        """Return default complete EmbeddingIdentity dataclass (faiss_normalize=True)."""
        return self.get_embedding_identity(faiss_normalize=True)

    @abstractmethod
    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Embed a sequence of text strings into a 2D float32 numpy array.

        Args:
            texts: Sequence of string inputs to encode.
            batch_size: Number of texts per inference batch (must be >= 1).

        Returns:
            Contiguous 2D float32 numpy array with shape (len(texts), embedding_dim).
        """
        ...

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a single text string into a 1D float32 vector of shape (embedding_dim,).

        Args:
            text: Single text string to encode.

        Returns:
            1D float32 numpy array of length embedding_dim.
        """
        if not isinstance(text, str):
            raise TypeError(f"text must be a string, got {type(text).__name__}.")
        batch = self.embed_texts([text], batch_size=1)
        return np.asarray(batch[0], dtype=np.float32)


def _resolve_device(requested_device: str | None) -> tuple[str, str]:
    """Resolve requested compute device to an available effective device.

    Args:
        requested_device: Desired device string (e.g. 'cpu', 'cuda', 'cuda:0', 'mps').

    Returns:
        Tuple of (requested_device, effective_device).
    """
    if requested_device is None:
        return "cpu", "cpu"

    req = str(requested_device).strip().lower()
    if req == "cpu":
        return "cpu", "cpu"

    if req.startswith("cuda"):
        try:
            import torch

            if torch.cuda.is_available():
                return req, req
            logger.warning("Requested CUDA device '%s' is unavailable; falling back to CPU.", req)
            return req, "cpu"
        except ImportError:
            logger.warning("Torch is not installed; falling back to CPU.")
            return req, "cpu"

    if req.startswith("mps"):
        try:
            import torch

            if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return req, "mps"
            logger.warning("Requested MPS device '%s' is unavailable; falling back to CPU.", req)
            return req, "cpu"
        except ImportError:
            logger.warning("Torch is not installed; falling back to CPU.")
            return req, "cpu"

    return req, "cpu"


class SentenceTransformerEmbedder(BaseEmbedder):
    """Dense embedding adapter wrapping Sentence-Transformers models.

    Produces contiguous, 2D float32 numpy arrays of shape (N, D). Defaults to
    all-MiniLM-L6-v2 producing 384-dimensional, L2-normalized vectors suitable
    for inner-product similarity search in FAISS.
    """

    def __init__(
        self,
        model_name_or_path: str = DEFAULT_MODEL_NAME,
        expected_dim: int = DEFAULT_EMBEDDING_DIM,
        device: str | None = None,
        normalize_embeddings: bool = True,
        local_files_only: bool = False,
        cache_folder: str | Path | None = None,
        model: Any = None,
        model_factory: Callable[..., Any] | None = None,
        model_revision: str | None = None,
        artifact_fingerprint: str | None = None,
        max_seq_length: int | None = None,
        pooling_strategy: str = "mean",
        query_instruction: str = "",
        document_instruction: str = "",
    ) -> None:
        """Initialize the SentenceTransformerEmbedder.

        Args:
            model_name_or_path: Sentence-Transformers model ID or local directory path.
            expected_dim: Declared vector dimensionality to enforce (must be >= 1).
            device: Requested device ('cpu', 'cuda', etc.). Defaults to CPU fallback.
            normalize_embeddings: If True, vectors are L2-normalized to unit norm (1.0).
            local_files_only: If True, prevents network downloads and requires cached weights.
            cache_folder: Optional local path for caching Hugging Face model files.
            model: Optional pre-instantiated model instance for testing and dependency injection.
            model_factory: Optional factory callable to instantiate the model.
            model_revision: Optional immutable model revision git commit hash.
            artifact_fingerprint: Optional SHA-256 digest of model artifact.
            max_seq_length: Optional max sequence length in tokens
                (defaults to model setting or 256).
            pooling_strategy: Pooling strategy ('mean', 'cls', etc.). Defaults to 'mean'.
            query_instruction: Optional instruction prefix for queries.
            document_instruction: Optional instruction prefix for documents.

        Raises:
            TypeError: If arguments are of invalid types.
            ValueError: If expected_dim is <= 0 or model_name_or_path is empty.
            ImportError: If sentence-transformers is not installed when loading a model.
            FileNotFoundError: If local_files_only=True and model weights are not cached locally.
        """
        if not isinstance(model_name_or_path, str) or not model_name_or_path.strip():
            raise ValueError("model_name_or_path must be a non-empty string.")
        if isinstance(expected_dim, bool) or not isinstance(expected_dim, int):
            raise TypeError(f"expected_dim must be an integer, got {type(expected_dim).__name__}.")
        if expected_dim <= 0:
            raise ValueError(f"expected_dim must be a positive integer >= 1, got {expected_dim}.")
        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be a boolean.")
        if not isinstance(local_files_only, bool):
            raise TypeError("local_files_only must be a boolean.")

        self._model_name = model_name_or_path.strip()
        self._expected_dim = expected_dim
        self._normalize_embeddings = normalize_embeddings
        self._local_files_only = local_files_only
        self._cache_folder = str(cache_folder) if cache_folder is not None else None

        req_device, eff_device = _resolve_device(device)
        self._requested_device = req_device
        self._effective_device = eff_device

        # Initialize or inject model
        if model is not None:
            self._model = model
        elif model_factory is not None:
            self._model = model_factory(
                self._model_name,
                device=self._effective_device,
                cache_folder=self._cache_folder,
                local_files_only=self._local_files_only,
            )
        else:
            self._model = self._load_sentence_transformer()

        # Resolve canonical identity without exposing local host paths
        is_path = (
            (":\\" in self._model_name or ":/" in self._model_name)
            or self._model_name.startswith(("/", "\\", "./", ".\\"))
            or ("\\" in self._model_name)
        )
        if is_path:
            path_hash = hashlib.sha256(self._model_name.encode("utf-8")).hexdigest()[:16]
            self._canonical_model_id = f"local-model:{path_hash}"
            self._display_label = "local-model"
        elif self._model_name in (
            DEFAULT_MODEL_NAME,
            f"sentence-transformers/{DEFAULT_MODEL_NAME}",
        ):
            self._canonical_model_id = f"sentence-transformers/{DEFAULT_MODEL_NAME}"
            self._display_label = DEFAULT_MODEL_NAME
        else:
            self._canonical_model_id = self._model_name
            self._display_label = self._model_name.split("/")[-1]

        # Discover underlying model artifact directory for cryptographic asset fingerprinting
        discovered_dir: Path | None = None
        ambiguous_snapshot = False

        model_weights_dir: Path | None = None
        tokenizer_dir: Path | None = None

        if Path(self._model_name).is_dir():
            discovered_dir = Path(self._model_name).resolve()
        elif hasattr(self._model, "_first_module"):
            try:
                first_module = self._model._first_module()
                if hasattr(first_module, "auto_model") and getattr(
                    first_module.auto_model, "name_or_path", None
                ):
                    mpath = Path(first_module.auto_model.name_or_path)
                    if mpath.is_dir():
                        model_weights_dir = mpath.resolve()

                if hasattr(first_module, "tokenizer"):
                    if getattr(first_module.tokenizer, "vocab_file", None):
                        vpath = Path(first_module.tokenizer.vocab_file)
                        if vpath.is_file():
                            tokenizer_dir = vpath.parent.resolve()
                    elif getattr(first_module.tokenizer, "name_or_path", None):
                        tpath = Path(first_module.tokenizer.name_or_path)
                        if tpath.is_dir():
                            tokenizer_dir = tpath.resolve()

                if model_weights_dir is not None and tokenizer_dir is not None:
                    if model_weights_dir != tokenizer_dir:
                        logger.warning(
                            "Model weights dir (%s) and tokenizer dir (%s) differ; "
                            "snapshot is ambiguous.",
                            model_weights_dir,
                            tokenizer_dir,
                        )
                        ambiguous_snapshot = True
                    else:
                        discovered_dir = model_weights_dir
                elif model_weights_dir is not None:
                    discovered_dir = model_weights_dir
                elif tokenizer_dir is not None:
                    discovered_dir = tokenizer_dir
            except Exception as e:
                logger.debug("Could not inspect tokenizer/auto_model for artifact dir: %s", e)

        if model_revision is not None:
            self._model_revision: str | None = model_revision
        elif hasattr(self._model, "config") and getattr(self._model.config, "_commit_hash", None):
            self._model_revision = self._model.config._commit_hash
        elif (
            discovered_dir is not None
            and len(discovered_dir.name) == 40
            and all(c in "0123456789abcdefABCDEF" for c in discovered_dir.name)
        ):
            self._model_revision = discovered_dir.name
        elif self._canonical_model_id == f"sentence-transformers/{DEFAULT_MODEL_NAME}":
            self._model_revision = DEFAULT_MODEL_REVISION
        else:
            self._model_revision = None

        if artifact_fingerprint is not None:
            if not isinstance(artifact_fingerprint, str) or not re.match(
                r"^sha256:[0-9a-f]{64}$", artifact_fingerprint
            ):
                raise ValueError(f"Invalid artifact_fingerprint format: '{artifact_fingerprint}'")
            if ambiguous_snapshot:
                self._artifact_fingerprint = "unavailable:incomplete_or_ambiguous_assets"
            elif discovered_dir is not None:
                try:
                    computed_fp = compute_model_artifact_fingerprint(
                        discovered_dir, require_complete=True
                    )
                    if artifact_fingerprint != computed_fp:
                        raise ValueError(
                            f"Supplied artifact_fingerprint '{artifact_fingerprint}' does not "
                            f"match computed fingerprint '{computed_fp}' for "
                            f"model directory '{discovered_dir}'."
                        )
                    self._artifact_fingerprint = artifact_fingerprint
                except Exception as exc:
                    if isinstance(exc, ValueError) and "does not match" in str(exc):
                        raise
                    logger.warning(
                        "Failed computing model artifact fingerprint for %s: %s",
                        self._model_name,
                        exc,
                    )
                    self._artifact_fingerprint = "unavailable:incomplete_or_ambiguous_assets"
            else:
                self._artifact_fingerprint = artifact_fingerprint
        elif ambiguous_snapshot:
            self._artifact_fingerprint = "unavailable:incomplete_or_ambiguous_assets"
        elif discovered_dir is not None:
            try:
                self._artifact_fingerprint = compute_model_artifact_fingerprint(
                    discovered_dir, require_complete=True
                )
            except Exception as exc:
                logger.warning(
                    "Failed computing model artifact fingerprint for %s: %s",
                    self._model_name,
                    exc,
                )
                self._artifact_fingerprint = "unavailable:incomplete_or_ambiguous_assets"
        else:
            self._artifact_fingerprint = "unavailable:unverified_model_artifacts"

        if max_seq_length is not None:
            self._max_seq_length = max_seq_length
        else:
            self._max_seq_length = int(getattr(self._model, "max_seq_length", 256))

        self._pooling_strategy = pooling_strategy
        self._query_instruction = query_instruction
        self._document_instruction = document_instruction

        # Validate declared dimension if supported by the model interface
        if hasattr(self._model, "get_sentence_embedding_dimension"):
            try:
                declared_dim = self._model.get_sentence_embedding_dimension()
                if (
                    declared_dim is not None
                    and isinstance(declared_dim, int)
                    and declared_dim != self._expected_dim
                ):
                    raise ValueError(
                        f"Model '{self._model_name}' has declared embedding dimension "
                        f"{declared_dim}, which does not match expected_dim {self._expected_dim}."
                    )
            except Exception as exc:
                if isinstance(exc, ValueError):
                    raise
                logger.debug("Could not verify declared model dimension: %s", exc)

    def _load_sentence_transformer(self) -> Any:
        """Load SentenceTransformer model with actionable missing-dependency messaging."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is required for dense vector embeddings. "
                "Install with 'pip install reliability-lab[dense]' or "
                "'pip install sentence-transformers'."
            ) from exc

        try:
            return SentenceTransformer(
                model_name_or_path=self._model_name,
                device=self._effective_device,
                cache_folder=self._cache_folder,
                local_files_only=self._local_files_only,
            )
        except Exception as exc:
            # Handle local_files_only / offline weight missing errors with actionable guidance
            if self._local_files_only or "local_files_only" in str(exc).lower():
                raise FileNotFoundError(
                    f"Model weights for '{self._model_name}' were not found locally in cache "
                    f"and local_files_only=True. Pre-download the model weights or disable "
                    f"local_files_only to allow retrieval."
                ) from exc
            raise

    @property
    def model_name(self) -> str:
        """Return the model identifier or path."""
        return self._model_name

    @property
    def embedding_dim(self) -> int:
        """Return the expected embedding dimension."""
        return self._expected_dim

    @property
    def device(self) -> str:
        """Return the requested device string."""
        return self._requested_device

    @property
    def effective_device(self) -> str:
        """Return the actual active device string."""
        return self._effective_device

    @property
    def normalize_embeddings(self) -> bool:
        """Return whether embeddings are normalized."""
        return self._normalize_embeddings

    @property
    def local_files_only(self) -> bool:
        """Return whether offline local-only operation is enforced."""
        return self._local_files_only

    @property
    def canonical_model_id(self) -> str:
        """Return canonical model identifier."""
        return self._canonical_model_id

    @property
    def display_label(self) -> str:
        """Return sanitized display label."""
        return self._display_label

    @property
    def model_revision(self) -> str | None:
        """Return model revision hash."""
        return self._model_revision

    @property
    def artifact_fingerprint(self) -> str:
        """Return artifact fingerprint."""
        return self._artifact_fingerprint

    @property
    def max_seq_length(self) -> int:
        """Return maximum sequence length."""
        return self._max_seq_length

    @property
    def pooling_strategy(self) -> str:
        """Return pooling strategy."""
        return self._pooling_strategy

    @property
    def query_instruction(self) -> str:
        """Return query instruction prefix."""
        return self._query_instruction

    @property
    def document_instruction(self) -> str:
        """Return document instruction prefix."""
        return self._document_instruction

    def embed_texts(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        """Encode a sequence of text strings into an L2-normalized 2D float32 numpy array.

        Args:
            texts: Sequence of strings to encode.
            batch_size: Number of texts per batch (positive integer >= 1).

        Returns:
            Contiguous 2D float32 numpy array with shape (len(texts), embedding_dim).

        Raises:
            TypeError: If texts is not a sequence, contains non-strings, or invalid batch_size.
            ValueError: If batch_size < 1, or output shape/dimensions/values are invalid.
        """
        if isinstance(texts, (str, bytes)) or not isinstance(texts, Sequence):
            raise TypeError(f"texts must be a Sequence of strings, got {type(texts).__name__}.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError(f"batch_size must be an integer, got {type(batch_size).__name__}.")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be a positive integer >= 1, got {batch_size}.")

        # Handle empty input sequence deterministically
        if len(texts) == 0:
            return np.empty((0, self._expected_dim), dtype=np.float32)

        # Validate all items are strings
        for idx, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"All elements must be strings, got {type(text).__name__} at index {idx}."
                )

        text_list = list(texts)

        # Execute encoding
        try:
            # We enforce and validate normalization in this adapter, so request raw vectors
            # from underlying model if supported, or handle pre-normalized outputs gracefully
            raw_output = self._model.encode(
                text_list,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=False,
            )
        except TypeError:
            # Fallback if mock/custom model encode signature doesn't take all optional args
            raw_output = self._model.encode(text_list, batch_size=batch_size)

        arr = np.asarray(raw_output, dtype=np.float32)

        # Validate 2D shape
        if arr.ndim != 2:
            raise ValueError(
                f"Model output must be a 2D array of shape (N, {self._expected_dim}), "
                f"got array with {arr.ndim} dimensions: {arr.shape}."
            )

        # Validate row count matches input count exactly
        if arr.shape[0] != len(text_list):
            raise ValueError(
                f"Output row count ({arr.shape[0]}) does not match input count ({len(text_list)})."
            )

        # Validate vector dimension matches expected_dim
        if arr.shape[1] != self._expected_dim:
            raise ValueError(
                f"Output vector dimension ({arr.shape[1]}) does not match expected_dim "
                f"({self._expected_dim})."
            )

        # Validate all numerical values are finite (reject NaN, +inf, -inf)
        if not np.isfinite(arr).all():
            raise ValueError("Encoded embeddings contain non-finite values (NaN or Inf).")

        # Apply L2 normalization if enabled
        if self._normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            # Safe zero-norm handling: do not divide zero-norm vectors to avoid NaNs
            is_zero_norm = (norms < 1e-12).flatten()
            safe_norms = np.where(norms < 1e-12, 1.0, norms)
            normalized = arr / safe_norms
            normalized[is_zero_norm] = 0.0

            # Verify non-zero vectors are unit norm within tolerance
            if np.any(~is_zero_norm):
                unit_norms = np.linalg.norm(normalized[~is_zero_norm], axis=1)
                if not np.allclose(unit_norms, 1.0, atol=1e-4):
                    raise ValueError(
                        "L2 normalization check failed: output vectors are not unit norm."
                    )

            return np.ascontiguousarray(normalized, dtype=np.float32)

        return np.ascontiguousarray(arr, dtype=np.float32)
