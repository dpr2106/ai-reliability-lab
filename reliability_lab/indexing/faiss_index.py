"""FAISS dense vector retrieval engine supporting exact and approximate search.

Provides exact inner product (IndexFlatIP) and approximate graph (IndexHNSWFlat)
similarity search wrapped in IndexIDMap2 for reliable integer-to-Chunk mapping,
incremental additions, corpus replacement, and atomic disk serialization.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

import numpy as np

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, RetrievalResult
from reliability_lab.indexing.embeddings import BaseEmbedder, EmbeddingIdentity

logger = logging.getLogger(__name__)

SUPPORTED_INDEX_TYPES = ("flat", "hnsw")
SUPPORTED_METRICS = ("ip",)
SUPPORTED_INDEX_FORMAT_VERSIONS: frozenset[str] = frozenset({"2.0"})


class DenseIndexCompatibilityError(ValueError):
    """Base class for dense index compatibility and format validation failures."""


class UnsupportedIndexFormatError(DenseIndexCompatibilityError):
    """Raised when index format version is unrecognized or unsupported."""


class IncompatibleEmbeddingIdentityError(DenseIndexCompatibilityError):
    """Raised when index embedding identity does not match runtime embedder."""


class LegacyIndexIncompatibilityError(DenseIndexCompatibilityError):
    """Raised when legacy format 1.0 index lacks trusted provenance or digests mismatch."""


def compute_chunk_manifest_hash(
    chunks: Sequence[Chunk] | dict[Any, Any] | list[dict[str, Any]],
) -> str:
    """Compute deterministic SHA-256 manifest hash over chunks."""
    items: list[dict[str, str]] = []
    if isinstance(chunks, dict):
        for chunk_val in chunks.values():
            if isinstance(chunk_val, Chunk):
                c_id = chunk_val.chunk_id
                c_text = chunk_val.content
            elif isinstance(chunk_val, dict):
                c_id = str(chunk_val.get("chunk_id", ""))
                c_text = str(chunk_val.get("content", ""))
            else:
                continue
            c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
            items.append({"chunk_id": c_id, "content_hash": c_hash})
    else:
        for c in chunks:
            if isinstance(c, Chunk):
                c_id = c.chunk_id
                c_text = c.content
            elif isinstance(c, dict):
                c_id = str(c.get("chunk_id", ""))
                c_text = str(c.get("content", ""))
            else:
                continue
            c_hash = hashlib.sha256(c_text.encode("utf-8")).hexdigest()
            items.append({"chunk_id": c_id, "content_hash": c_hash})
    items.sort(key=lambda x: x["chunk_id"])
    digest = hashlib.sha256(
        json.dumps(items, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"sha256:{digest}"


def _require_faiss() -> Any:
    """Import and return the faiss module, or raise a clear actionable ImportError."""
    try:
        import faiss

        return faiss
    except ImportError as err:
        raise ImportError(
            "faiss is required for FaissVectorRetriever. "
            "Install with 'pip install reliability-lab[faiss]' or 'pip install faiss-cpu'."
        ) from err


class FaissVectorRetriever(BaseRetriever):
    """Dense vector retriever powered by FAISS.

    Supports:
        - Exact inner-product search (IndexFlatIP wrapped in IndexIDMap2).
        - Approximate nearest neighbor graph search (IndexHNSWFlat wrapped in IndexIDMap2).
        - Provenance-preserving integer ID to Chunk mapping.
        - Incremental chunk additions with duplicate detection.
        - Corpus replacement (index reset).
        - Safe disk serialization and deserialization.
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        index_type: Literal["flat", "hnsw"] | str = "flat",
        metric: Literal["ip"] | str = "ip",
        m: int = 32,
        ef_construction: int = 64,
        ef_search: int = 32,
        normalize: bool = True,
    ) -> None:
        """Initialize FaissVectorRetriever.

        Args:
            embedder: BaseEmbedder instance for generating text vector embeddings.
            index_type: Index topology, either 'flat' (exact) or 'hnsw' (approximate).
            metric: Distance metric, currently 'ip' (inner product / cosine similarity).
            m: Number of bi-directional links per node for HNSW (>= 4).
            ef_construction: Size of dynamic candidate list during HNSW construction (> 0).
            ef_search: Size of dynamic candidate list during HNSW retrieval (> 0).
            normalize: Whether to L2-normalize vectors prior to indexing and querying.

        Raises:
            TypeError: If arguments have invalid types.
            ValueError: If parameters are invalid or out of allowed ranges.
            ImportError: If faiss is not installed.
        """
        if not isinstance(embedder, BaseEmbedder):
            raise TypeError(
                f"embedder must be an instance of BaseEmbedder, got {type(embedder).__name__}."
            )
        if not isinstance(embedder.embedding_dim, int) or embedder.embedding_dim <= 0:
            raise ValueError(
                f"embedder.embedding_dim must be a positive integer, got {embedder.embedding_dim}."
            )

        if not isinstance(index_type, str):
            raise TypeError(f"index_type must be a string, got {type(index_type).__name__}.")
        normalized_index_type = index_type.strip().lower()
        if normalized_index_type not in SUPPORTED_INDEX_TYPES:
            raise ValueError(
                f"Invalid index_type '{index_type}'. Must be one of: {SUPPORTED_INDEX_TYPES}."
            )

        if not isinstance(metric, str):
            raise TypeError(f"metric must be a string, got {type(metric).__name__}.")
        normalized_metric = metric.strip().lower()
        if normalized_metric not in SUPPORTED_METRICS:
            raise ValueError(
                f"Unsupported metric '{metric}'. Only {SUPPORTED_METRICS} are supported."
            )

        if isinstance(m, bool) or not isinstance(m, int):
            raise TypeError(f"m must be an integer, got {type(m).__name__}.")
        if m < 4:
            raise ValueError(f"m must be an integer >= 4, got {m}.")

        if isinstance(ef_construction, bool) or not isinstance(ef_construction, int):
            raise TypeError(
                f"ef_construction must be an integer, got {type(ef_construction).__name__}."
            )
        if ef_construction <= 0:
            raise ValueError(f"ef_construction must be a positive integer, got {ef_construction}.")

        if isinstance(ef_search, bool) or not isinstance(ef_search, int):
            raise TypeError(f"ef_search must be an integer, got {type(ef_search).__name__}.")
        if ef_search <= 0:
            raise ValueError(f"ef_search must be a positive integer, got {ef_search}.")

        if not isinstance(normalize, bool):
            raise TypeError(f"normalize must be a boolean, got {type(normalize).__name__}.")

        self._faiss = _require_faiss()

        self._embedder = embedder
        self._dimension = embedder.embedding_dim
        self._index_type = normalized_index_type
        self._metric = normalized_metric
        self._m = m
        self._ef_construction = ef_construction
        self._ef_search = ef_search
        self._normalize = normalize

        self._id_to_chunk: dict[int, Chunk] = {}
        self._chunk_id_to_id: dict[str, int] = {}
        self._next_id: int = 0
        self._is_legacy: bool = False

        self._index = self._build_empty_index()

    def _build_empty_index(self) -> Any:
        """Construct a fresh empty FAISS index wrapped in IndexIDMap2."""
        if self._index_type == "flat":
            sub_index = self._faiss.IndexFlatIP(self._dimension)
            return self._faiss.IndexIDMap2(sub_index)
        elif self._index_type == "hnsw":
            sub_index = self._faiss.IndexHNSWFlat(
                self._dimension, self._m, self._faiss.METRIC_INNER_PRODUCT
            )
            sub_index.hnsw.efConstruction = self._ef_construction
            sub_index.hnsw.efSearch = self._ef_search
            return self._faiss.IndexIDMap2(sub_index)
        else:
            raise ValueError(f"Unknown index_type: {self._index_type}")

    @property
    def name(self) -> str:
        """Return the unique retriever name identifying this engine."""
        return f"faiss-{self._index_type}"

    @property
    def embedder(self) -> BaseEmbedder:
        """Return the underlying embedder instance."""
        return self._embedder

    @property
    def index_type(self) -> str:
        """Return the index topology type ('flat' or 'hnsw')."""
        return self._index_type

    @property
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        return self._dimension

    @property
    def metric(self) -> str:
        """Return the similarity metric ('ip')."""
        return self._metric

    @property
    def m(self) -> int:
        """Return the HNSW graph link parameter M."""
        return self._m

    @property
    def ef_construction(self) -> int:
        """Return the HNSW efConstruction parameter."""
        return self._ef_construction

    @property
    def ef_search(self) -> int:
        """Return the HNSW efSearch parameter."""
        return self._ef_search

    @property
    def normalize(self) -> bool:
        """Return whether vectors are L2-normalized."""
        return self._normalize

    @property
    def embedding_identity(self) -> EmbeddingIdentity:
        """Return the complete EmbeddingIdentity bound to this retriever's normalization."""
        return self._embedder.get_embedding_identity(faiss_normalize=self._normalize)

    @property
    def size(self) -> int:
        """Return total number of chunks currently indexed."""
        return len(self._id_to_chunk)

    @property
    def chunk_ids(self) -> list[str]:
        """Return list of chunk IDs in indexed order."""
        return list(self._chunk_id_to_id.keys())

    def __len__(self) -> int:
        """Return total number of indexed chunks."""
        return self.size

    def __contains__(self, chunk_id: str) -> bool:
        """Return whether chunk_id exists in the index."""
        return chunk_id in self._chunk_id_to_id

    def get_chunk(self, chunk_id: str) -> Chunk | None:
        """Retrieve a Chunk by its chunk_id, or None if not found."""
        int_id = self._chunk_id_to_id.get(chunk_id)
        if int_id is None:
            return None
        return self._id_to_chunk.get(int_id)

    def index(self, chunks: Sequence[Chunk]) -> None:
        """Index a collection of chunks, replacing any existing index.

        Args:
            chunks: Sequence of Chunk objects to populate into the index.
        """
        self._index = self._build_empty_index()
        self._id_to_chunk.clear()
        self._chunk_id_to_id.clear()
        self._next_id = 0

        if not chunks:
            return

        self.add(chunks)

    def add(self, chunks: Sequence[Chunk]) -> None:
        """Incrementally add chunks into the index without discarding existing entries.

        Args:
            chunks: Sequence of Chunk objects to add.

        Raises:
            TypeError: If input contains elements that are not Chunk instances.
            ValueError: If duplicate chunk IDs are detected or embedding shape is invalid.
        """
        if not chunks:
            return

        # Phase 1: Strict validation before state mutation
        seen_batch_ids: set[str] = set()
        texts: list[str] = []
        for chunk in chunks:
            if not isinstance(chunk, Chunk):
                raise TypeError(
                    f"All elements in chunks must be Chunk instances, got {type(chunk).__name__}."
                )
            if chunk.chunk_id in self._chunk_id_to_id:
                raise ValueError(
                    f"Chunk ID '{chunk.chunk_id}' already exists in index. "
                    "Incremental additions must have unique chunk IDs."
                )
            if chunk.chunk_id in seen_batch_ids:
                raise ValueError(f"Duplicate chunk ID '{chunk.chunk_id}' found in the input batch.")
            seen_batch_ids.add(chunk.chunk_id)
            texts.append(chunk.content)

        # Phase 2: Vector embedding generation and validation
        embeddings = self._embedder.embed_texts(texts)
        if not isinstance(embeddings, np.ndarray):
            embeddings = np.asarray(embeddings, dtype=np.float32)

        if embeddings.ndim != 2 or embeddings.shape[1] != self._dimension:
            raise ValueError(
                f"Embedding shape {embeddings.shape} does not match expected dimension "
                f"(*, {self._dimension})."
            )
        if embeddings.shape[0] != len(chunks):
            raise ValueError(
                f"Embedder returned {embeddings.shape[0]} embeddings for {len(chunks)} chunks."
            )

        vecs = np.ascontiguousarray(embeddings, dtype=np.float32)
        if not np.all(np.isfinite(vecs)):
            raise ValueError("Embeddings contain non-finite values (NaN or Inf).")

        if self._normalize:
            self._faiss.normalize_L2(vecs)

        # Phase 3: Allocate integer IDs and update FAISS index
        start_id = self._next_id
        ids = np.arange(start_id, start_id + len(chunks), dtype=np.int64)
        self._index.add_with_ids(vecs, ids)

        # Phase 4: Update provenance mappings
        for int_id, chunk in zip(ids, chunks, strict=True):
            int_id_int = int(int_id)
            self._id_to_chunk[int_id_int] = chunk
            self._chunk_id_to_id[chunk.chunk_id] = int_id_int
        self._next_id = start_id + len(chunks)

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve the top-K ranked chunks for the given query text.

        Args:
            query: Query string.
            top_k: Maximum number of ranked results to return (must be > 0).

        Returns:
            List of RetrievalResult objects ordered by descending score.

        Raises:
            TypeError: If query or top_k have invalid types.
        """
        if not isinstance(query, str):
            raise TypeError(f"query must be a string, got {type(query).__name__}.")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise TypeError(f"top_k must be an integer, got {type(top_k).__name__}.")

        if top_k <= 0 or not query.strip() or self.size == 0:
            return []

        k = min(top_k, self.size)

        # Generate and validate query embedding
        q_emb = self._embedder.embed_text(query)
        q_vec = np.ascontiguousarray(q_emb.reshape(1, -1), dtype=np.float32)
        if q_vec.shape[1] != self._dimension:
            raise ValueError(
                f"Query embedding dimension {q_vec.shape[1]} does not match "
                f"retriever dimension {self._dimension}."
            )
        if not np.all(np.isfinite(q_vec)):
            raise ValueError("Query embedding contains non-finite values (NaN or Inf).")

        if self._normalize:
            self._faiss.normalize_L2(q_vec)

        # Apply HNSW search parameters if applicable
        if self._index_type == "hnsw":
            base_index = self._faiss.downcast_index(self._index.index)
            base_index.hnsw.efSearch = self._ef_search

        distances, indices = self._index.search(q_vec, k)

        results: list[RetrievalResult] = []
        rank = 1
        for dist, idx in zip(distances[0], indices[0], strict=True):
            if idx == -1 or int(idx) not in self._id_to_chunk:
                continue
            int_id = int(idx)
            chunk = self._id_to_chunk[int_id]
            metadata: dict[str, Any] = {
                "faiss_id": int_id,
                "index_type": self._index_type,
                "metric": self._metric,
                "normalized": self._normalize,
            }
            if self._index_type == "hnsw":
                metadata["ef_search"] = self._ef_search
                metadata["m"] = self._m

            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=float(dist),
                    rank=rank,
                    retriever_name=self.name,
                    metadata=metadata,
                )
            )
            rank += 1

        return results

    @property
    def is_legacy(self) -> bool:
        """Return whether index was loaded through legacy compatibility path."""
        return getattr(self, "_is_legacy", False)

    def save(self, path: str | Path) -> None:
        """Persist the index and chunk metadata durably to disk using format 2.0.

        Args:
            path: Directory path where 'index.faiss' and 'metadata.json' will be written.

        Raises:
            OSError: If directory creation or file writing fails.
        """
        target_dir = Path(path)
        target_dir.mkdir(parents=True, exist_ok=True)

        saved_identity = self.embedding_identity
        if self._normalize != saved_identity.faiss_normalize:
            raise ValueError(
                f"Retriever normalize={self._normalize} contradicts embedding identity "
                f"faiss_normalize={saved_identity.faiss_normalize}."
            )

        meta_payload: dict[str, Any] = {
            "format_version": "2.0",
            "embedding_identity": saved_identity.to_dict(),
            "index_type": self._index_type,
            "dimension": self._dimension,
            "metric": self._metric,
            "m": self._m,
            "ef_construction": self._ef_construction,
            "ef_search": self._ef_search,
            "normalize": self._normalize,
            "next_id": self._next_id,
            "chunks": {str(int_id): chunk.to_dict() for int_id, chunk in self._id_to_chunk.items()},
        }

        # Write to temporary files first to guarantee atomic write safety
        with tempfile.TemporaryDirectory(dir=target_dir) as tmp_dir:
            tmp_index = Path(tmp_dir) / "index.faiss"
            tmp_meta = Path(tmp_dir) / "metadata.json"

            self._faiss.write_index(self._index, str(tmp_index))
            artifact_digest = f"sha256:{hashlib.sha256(tmp_index.read_bytes()).hexdigest()}"
            chunk_manifest_hash = compute_chunk_manifest_hash(list(self._id_to_chunk.values()))
            meta_payload["artifact_digest"] = artifact_digest
            meta_payload["chunk_manifest_hash"] = chunk_manifest_hash

            tmp_meta.write_text(json.dumps(meta_payload, indent=2), encoding="utf-8")

            # Move files to target directory
            final_index = target_dir / "index.faiss"
            final_meta = target_dir / "metadata.json"

            tmp_index.replace(final_index)
            tmp_meta.replace(final_meta)

        logger.info(
            "Saved %s index with %d chunks to %s (format 2.0)",
            self.name,
            len(self._id_to_chunk),
            target_dir,
        )

    @classmethod
    def load(cls, path: str | Path, embedder: BaseEmbedder) -> FaissVectorRetriever:
        """Load a persisted format 2.0 FAISS index and chunk metadata from disk.

        Enforces SUPPORTED_INDEX_FORMAT_VERSIONS and validates cryptographic
        pipeline and artifact fingerprints against the active runtime embedder.

        Args:
            path: Directory containing 'index.faiss' and 'metadata.json'.
            embedder: BaseEmbedder instance to attach for query encoding.

        Returns:
            Restored FaissVectorRetriever instance.

        Raises:
            FileNotFoundError: If the directory or expected files do not exist.
            UnsupportedIndexFormatError: If format_version is not supported (e.g. 1.0).
            IncompatibleEmbeddingIdentityError: If embedding identity is missing or mismatched.
            ValueError: If metadata is invalid or dimension mismatches embedder.
        """
        faiss_mod = _require_faiss()

        source_dir = Path(path)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"FAISS index directory not found: '{source_dir.name}'.")

        index_file = source_dir / "index.faiss"
        meta_file = source_dir / "metadata.json"

        if not index_file.is_file():
            raise FileNotFoundError("Index binary not found: 'index.faiss'.")
        if not meta_file.is_file():
            raise FileNotFoundError("Index metadata not found: 'metadata.json'.")

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Failed to parse metadata file 'metadata.json': {err}") from err

        if not isinstance(meta, dict):
            raise ValueError("Corrupt metadata: root JSON structure must be an object.")

        # 1. Enforce Format Allowlist
        format_version = str(meta.get("format_version", "1.0"))
        if format_version not in SUPPORTED_INDEX_FORMAT_VERSIONS:
            raise UnsupportedIndexFormatError(
                f"Unsupported index format version '{format_version}'. "
                f"Supported versions: {sorted(SUPPORTED_INDEX_FORMAT_VERSIONS)}."
            )

        # 2. Enforce Embedding Identity & Fingerprints
        if "embedding_identity" not in meta or not isinstance(meta["embedding_identity"], dict):
            raise IncompatibleEmbeddingIdentityError(
                "Format 2.0 index metadata missing required 'embedding_identity' payload."
            )

        try:
            saved_identity = EmbeddingIdentity.from_dict(meta["embedding_identity"])
        except Exception as exc:
            raise IncompatibleEmbeddingIdentityError(
                f"Malformed or internally inconsistent embedding identity: {exc}"
            ) from exc

        # 3. Validate Dimension
        if "dimension" not in meta:
            raise ValueError("Corrupt metadata: missing required 'dimension' key.")

        dim_val = meta["dimension"]
        if isinstance(dim_val, bool) or not isinstance(dim_val, int) or dim_val <= 0:
            raise ValueError(
                f"Corrupt metadata: 'dimension' must be a positive integer, got {dim_val!r}."
            )
        expected_dimension = dim_val

        if saved_identity.dimension != expected_dimension:
            raise IncompatibleEmbeddingIdentityError(
                f"Dimension mismatch in saved metadata: identity declared "
                f"{saved_identity.dimension} but metadata root has {expected_dimension}."
            )

        if embedder.embedding_dim != expected_dimension:
            raise ValueError(
                f"Saved index dimension ({expected_dimension}) does not match "
                f"embedder dimension ({embedder.embedding_dim})."
            )

        # 4. Validate Normalization Setting
        if "normalize" not in meta:
            raise ValueError("Corrupt metadata: missing required 'normalize' key.")
        if not isinstance(meta["normalize"], bool):
            raise ValueError(
                f"Corrupt metadata: 'normalize' must be a boolean, got {meta['normalize']!r}."
            )
        index_normalize = meta["normalize"]

        if index_normalize != saved_identity.faiss_normalize:
            raise IncompatibleEmbeddingIdentityError(
                f"Normalization setting mismatch: metadata root has normalize={index_normalize} "
                f"but embedding identity declares faiss_normalize={saved_identity.faiss_normalize}."
            )

        # Fail closed if active embedder's artifact identity is unavailable
        if embedder.artifact_fingerprint.startswith("unavailable"):
            raise IncompatibleEmbeddingIdentityError(
                f"Active embedder artifact identity cannot be cryptographically verified "
                f"({embedder.artifact_fingerprint}); failing closed for dense retrieval."
            )

        # Compare pipeline and artifact fingerprints against active embedder
        active_pipeline_fp = embedder.get_pipeline_fingerprint(
            faiss_normalize=saved_identity.faiss_normalize
        )
        if saved_identity.pipeline_fingerprint != active_pipeline_fp:
            raise IncompatibleEmbeddingIdentityError(
                f"Pipeline fingerprint mismatch: index has {saved_identity.pipeline_fingerprint}, "
                f"active embedder has {active_pipeline_fp}."
            )

        if saved_identity.artifact_fingerprint != embedder.artifact_fingerprint:
            raise IncompatibleEmbeddingIdentityError(
                f"Artifact fingerprint mismatch: index has {saved_identity.artifact_fingerprint}, "
                f"active embedder has {embedder.artifact_fingerprint}."
            )

        # 5. Validate Binary Artifact Digest and Chunk Manifest Hash (mandatory in format 2.0)
        if "artifact_digest" not in meta:
            raise IncompatibleEmbeddingIdentityError(
                "Format 2.0 index metadata missing required 'artifact_digest'."
            )
        expected_digest = meta["artifact_digest"]
        if not isinstance(expected_digest, str) or not re.match(
            r"^sha256:[0-9a-f]{64}$", expected_digest
        ):
            raise IncompatibleEmbeddingIdentityError(
                f"Invalid artifact_digest format in index metadata: {expected_digest!r}."
            )
        actual_digest = f"sha256:{hashlib.sha256(index_file.read_bytes()).hexdigest()}"
        if actual_digest != expected_digest:
            raise IncompatibleEmbeddingIdentityError(
                f"FAISS index binary artifact digest mismatch: expected '{expected_digest}', "
                f"got '{actual_digest}'."
            )

        if "chunk_manifest_hash" not in meta:
            raise IncompatibleEmbeddingIdentityError(
                "Format 2.0 index metadata missing required 'chunk_manifest_hash'."
            )
        expected_chunk_manifest = meta["chunk_manifest_hash"]
        if not isinstance(expected_chunk_manifest, str) or not re.match(
            r"^sha256:[0-9a-f]{64}$", expected_chunk_manifest
        ):
            raise IncompatibleEmbeddingIdentityError(
                "Invalid chunk_manifest_hash format in index metadata: "
                f"{expected_chunk_manifest!r}."
            )
        chunks_dict = meta.get("chunks", {})
        try:
            chunks_list = [Chunk.from_dict(c) for c in chunks_dict.values()]
            actual_chunk_manifest = compute_chunk_manifest_hash(chunks_list)
        except Exception as exc:
            raise IncompatibleEmbeddingIdentityError(
                f"Failed computing chunk manifest hash from saved metadata: {exc}"
            ) from exc
        if actual_chunk_manifest != expected_chunk_manifest:
            raise IncompatibleEmbeddingIdentityError(
                f"Chunk manifest hash mismatch in saved index metadata: "
                f"expected '{expected_chunk_manifest}', got '{actual_chunk_manifest}'."
            )

        try:
            loaded_index = faiss_mod.read_index(str(index_file))
        except Exception as err:
            logger.error("Failed to read FAISS index binary '%s': %s", index_file.name, err)
            raise ValueError(f"Failed to read FAISS index binary '{index_file.name}'.") from err

        retriever = cls(
            embedder=embedder,
            index_type=meta["index_type"],
            metric=meta.get("metric", "ip"),
            m=int(meta.get("m", 32)),
            ef_construction=int(meta.get("ef_construction", 64)),
            ef_search=int(meta.get("ef_search", 32)),
            normalize=index_normalize,
        )

        id_to_chunk: dict[int, Chunk] = {}
        chunk_id_to_id: dict[str, int] = {}
        for int_id_str, chunk_dict in meta.get("chunks", {}).items():
            int_id = int(int_id_str)
            chunk = Chunk.from_dict(chunk_dict)
            id_to_chunk[int_id] = chunk
            chunk_id_to_id[chunk.chunk_id] = int_id

        retriever._index = loaded_index
        retriever._id_to_chunk = id_to_chunk
        retriever._chunk_id_to_id = chunk_id_to_id
        retriever._next_id = int(meta.get("next_id", len(id_to_chunk)))
        retriever._is_legacy = False

        if retriever._index_type == "hnsw":
            base_index = faiss_mod.downcast_index(retriever._index.index)
            base_index.hnsw.efSearch = retriever._ef_search

        logger.info(
            "Loaded %s index with %d chunks from %s (format 2.0)",
            retriever.name,
            len(id_to_chunk),
            source_dir,
        )
        return retriever

    @classmethod
    def load_legacy_verified(
        cls,
        path: str | Path,
        embedder: BaseEmbedder,
        trusted_provenance: dict[str, Any] | None,
    ) -> FaissVectorRetriever:
        """Load a legacy format 1.0 FAISS index strictly with trusted server-controlled provenance.

        Args:
            path: Directory containing legacy 'index.faiss' and 'metadata.json'.
            embedder: Active BaseEmbedder instance.
            trusted_provenance: Provenance payload stored in SQLite metadata.

        Returns:
            Restored FaissVectorRetriever marked with is_legacy=True.

        Raises:
            FileNotFoundError: If index directory or expected files do not exist.
            LegacyIndexIncompatibilityError: If provenance is missing, invalid, or digests mismatch.
            IncompatibleEmbeddingIdentityError: If provenance embedding identity
                mismatches active embedder.
            ValueError: If metadata or index is corrupt.
        """
        source_dir = Path(path)
        if not source_dir.is_dir():
            raise FileNotFoundError(f"FAISS index directory not found: '{source_dir.name}'.")

        index_file = source_dir / "index.faiss"
        meta_file = source_dir / "metadata.json"

        if not index_file.is_file():
            raise FileNotFoundError("Index binary not found: 'index.faiss'.")
        if not meta_file.is_file():
            raise FileNotFoundError("Index metadata not found: 'metadata.json'.")

        if not trusted_provenance or not isinstance(trusted_provenance, dict):
            raise LegacyIndexIncompatibilityError(
                "Legacy index format 1.0 lacks trusted embedding provenance in SQLite metadata. "
                "Reindexing is required before dense retrieval can proceed."
            )

        try:
            meta = json.loads(meta_file.read_text(encoding="utf-8"))
        except Exception as err:
            raise ValueError(f"Failed to parse metadata file 'metadata.json': {err}") from err

        if not isinstance(meta, dict):
            raise ValueError("Corrupt metadata: root JSON structure must be an object.")

        format_version = str(meta.get("format_version", "1.0"))
        if format_version != "1.0":
            raise LegacyIndexIncompatibilityError(
                f"Expected legacy format 1.0 but index metadata indicates '{format_version}'."
            )

        # 1. Cryptographic Artifact Binding: compare SHA-256 of index.faiss
        artifact_bytes = index_file.read_bytes()
        actual_artifact_digest = f"sha256:{hashlib.sha256(artifact_bytes).hexdigest()}"
        expected_artifact_digest = trusted_provenance.get("artifact_digest")
        if not expected_artifact_digest or actual_artifact_digest != expected_artifact_digest:
            raise LegacyIndexIncompatibilityError(
                f"Legacy index artifact digest mismatch: disk has {actual_artifact_digest}, "
                f"trusted provenance recorded {expected_artifact_digest}. "
                "Reindexing is required."
            )

        # 2. Cryptographic Chunk Manifest Binding
        chunks_dict = meta.get("chunks", {})
        actual_chunk_manifest = compute_chunk_manifest_hash(chunks_dict)
        expected_chunk_manifest = trusted_provenance.get("chunk_manifest_hash")
        if not expected_chunk_manifest or actual_chunk_manifest != expected_chunk_manifest:
            raise LegacyIndexIncompatibilityError(
                f"Legacy chunk manifest mismatch: actual is {actual_chunk_manifest}, "
                f"trusted provenance recorded {expected_chunk_manifest}. "
                "Reindexing is required."
            )

        # 3. Provenance Identity Verification against active embedder
        provenance_identity = trusted_provenance.get("embedding_identity")
        if not isinstance(provenance_identity, dict):
            raise LegacyIndexIncompatibilityError(
                "Trusted provenance is missing 'embedding_identity'. Reindexing is required."
            )

        if provenance_identity.get("pipeline_fingerprint") != embedder.pipeline_fingerprint:
            raise IncompatibleEmbeddingIdentityError(
                f"Pipeline fingerprint mismatch: legacy provenance has "
                f"{provenance_identity.get('pipeline_fingerprint')}, "
                f"active embedder has {embedder.pipeline_fingerprint}."
            )

        # Dimension validation
        expected_dimension = int(meta.get("dimension", 0))
        if embedder.embedding_dim != expected_dimension:
            raise ValueError(
                f"Legacy index dimension ({expected_dimension}) does not match "
                f"embedder dimension ({embedder.embedding_dim})."
            )

        faiss_mod = _require_faiss()
        try:
            loaded_index = faiss_mod.read_index(str(index_file))
        except Exception as err:
            logger.error("Failed to read FAISS index binary '%s': %s", index_file.name, err)
            raise ValueError(f"Failed to read FAISS index binary '{index_file.name}'.") from err

        retriever = cls(
            embedder=embedder,
            index_type=meta.get("index_type", "flat"),
            metric=meta.get("metric", "ip"),
            m=int(meta.get("m", 32)),
            ef_construction=int(meta.get("ef_construction", 64)),
            ef_search=int(meta.get("ef_search", 32)),
            normalize=bool(meta.get("normalize", True)),
        )

        id_to_chunk: dict[int, Chunk] = {}
        chunk_id_to_id: dict[str, int] = {}
        for int_id_str, chunk_dict in chunks_dict.items():
            int_id = int(int_id_str)
            chunk = Chunk.from_dict(chunk_dict)
            id_to_chunk[int_id] = chunk
            chunk_id_to_id[chunk.chunk_id] = int_id

        retriever._index = loaded_index
        retriever._id_to_chunk = id_to_chunk
        retriever._chunk_id_to_id = chunk_id_to_id
        retriever._next_id = int(meta.get("next_id", len(id_to_chunk)))
        retriever._is_legacy = True

        if retriever._index_type == "hnsw":
            base_index = faiss_mod.downcast_index(retriever._index.index)
            base_index.hnsw.efSearch = retriever._ef_search

        logger.info(
            "Loaded %s index with %d chunks from %s (legacy format 1.0 verified)",
            retriever.name,
            len(id_to_chunk),
            source_dir,
        )
        return retriever
