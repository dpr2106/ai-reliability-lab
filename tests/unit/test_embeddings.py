"""Unit tests for the Sentence-Transformers dense embedding adapter (DENSE-001).

Tests vector generation, batching, dimension validation, L2 normalization,
device selection/fallback, offline operation, zero-norm handling, and edge cases
without downloading external model weights.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from reliability_lab.indexing.embeddings import (
    DEFAULT_EMBEDDING_DIM,
    DEFAULT_MODEL_NAME,
    BaseEmbedder,
    SentenceTransformerEmbedder,
    _resolve_device,
)


class FakeSentenceTransformer:
    """Mock SentenceTransformer model for deterministic testing without external weights."""

    def __init__(
        self,
        embedding_dim: int = 384,
        return_val: np.ndarray | None = None,
    ) -> None:
        self._dim = embedding_dim
        self._return_val = return_val
        self.encode_calls: list[dict[str, Any]] = []

    def get_sentence_embedding_dimension(self) -> int:
        return self._dim

    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> np.ndarray:
        self.encode_calls.append(
            {
                "sentences": list(sentences),
                "batch_size": batch_size,
                "normalize_embeddings": normalize_embeddings,
            }
        )
        if self._return_val is not None:
            return self._return_val

        # Deterministic vector generation based on sentence length and index
        n = len(sentences)
        rng = np.random.default_rng(seed=12345)
        # Generate non-zero random normal vectors
        arr = rng.standard_normal((n, self._dim)).astype(np.float32)
        return arr


class TestSentenceTransformerEmbedder:
    """Core functionality tests for SentenceTransformerEmbedder."""

    def test_cannot_instantiate_base_embedder(self) -> None:
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseEmbedder()  # type: ignore[abstract]

    def test_default_configuration_and_properties(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        assert embedder.model_name == DEFAULT_MODEL_NAME
        assert embedder.embedding_dim == DEFAULT_EMBEDDING_DIM
        assert embedder.device == "cpu"
        assert embedder.effective_device == "cpu"
        assert embedder.normalize_embeddings is True
        assert embedder.local_files_only is False

    def test_batch_encoding_and_output_shape(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        texts = ["First document text", "Second query text", "Third document text"]
        vectors = embedder.embed_texts(texts, batch_size=2)

        assert isinstance(vectors, np.ndarray)
        assert vectors.dtype == np.float32
        assert vectors.shape == (3, 384)
        assert len(fake_model.encode_calls) == 1
        assert fake_model.encode_calls[0]["sentences"] == texts
        assert fake_model.encode_calls[0]["batch_size"] == 2

    def test_input_ordering_preservation(self) -> None:
        """Verify that the order of output vectors corresponds exactly to input texts."""
        dim = 384
        # Return distinctly structured vectors for each input row
        v0 = np.full((1, dim), 1.0, dtype=np.float32)
        v1 = np.full((1, dim), 2.0, dtype=np.float32)
        v2 = np.full((1, dim), 3.0, dtype=np.float32)
        stacked = np.vstack([v0, v1, v2])

        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=stacked)
        embedder = SentenceTransformerEmbedder(model=fake_model, normalize_embeddings=False)

        texts = ["doc_A", "doc_B", "doc_C"]
        out = embedder.embed_texts(texts)

        assert np.allclose(out[0], 1.0)
        assert np.allclose(out[1], 2.0)
        assert np.allclose(out[2], 3.0)

    def test_empty_sequence_behavior(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        out = embedder.embed_texts([])
        assert isinstance(out, np.ndarray)
        assert out.dtype == np.float32
        assert out.shape == (0, 384)
        # Should not call encode on model for empty input
        assert len(fake_model.encode_calls) == 0

    def test_blank_and_whitespace_texts(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        texts = ["", "   ", "\t\n"]
        out = embedder.embed_texts(texts)
        assert out.shape == (3, 384)
        assert np.isfinite(out).all()
        # Non-zero vectors should have unit norm
        norms = np.linalg.norm(out, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_embed_text_single_string_helper(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        vec = embedder.embed_text("Single search query")
        assert isinstance(vec, np.ndarray)
        assert vec.ndim == 1
        assert vec.shape == (384,)
        assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-4)

    def test_embed_text_rejects_non_string(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)
        with pytest.raises(TypeError, match="text must be a string"):
            embedder.embed_text(12345)  # type: ignore[arg-type]

    def test_batch_size_validation(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        with pytest.raises(ValueError, match="positive integer >= 1"):
            embedder.embed_texts(["test"], batch_size=0)
        with pytest.raises(ValueError, match="positive integer >= 1"):
            embedder.embed_texts(["test"], batch_size=-5)
        with pytest.raises(TypeError, match="batch_size must be an integer"):
            embedder.embed_texts(["test"], batch_size=True)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="batch_size must be an integer"):
            embedder.embed_texts(["test"], batch_size="16")  # type: ignore[arg-type]

    def test_texts_type_validation(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        with pytest.raises(TypeError, match="texts must be a Sequence"):
            embedder.embed_texts("single string instead of list")  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="All elements must be strings"):
            embedder.embed_texts(["valid", 999, "valid"])  # type: ignore[list-item]

    def test_expected_dimension_mismatch_at_init(self) -> None:
        # Fake model reports 768 dimensions (like BERT base)
        fake_model = FakeSentenceTransformer(embedding_dim=768)
        with pytest.raises(ValueError, match="declared embedding dimension 768"):
            SentenceTransformerEmbedder(
                model_name_or_path="bert-base",
                expected_dim=384,
                model=fake_model,
            )

    def test_expected_dimension_mismatch_at_encoding(self) -> None:
        # Model returns shape (2, 512) when 384 was expected
        wrong_shape = np.ones((2, 512), dtype=np.float32)
        fake_model = FakeSentenceTransformer(embedding_dim=384, return_val=wrong_shape)
        # Avoid init check by mocking get_sentence_embedding_dimension returning 384
        embedder = SentenceTransformerEmbedder(expected_dim=384, model=fake_model)

        with pytest.raises(ValueError, match=r"Output vector dimension .* does not match"):
            embedder.embed_texts(["query 1", "query 2"])

    def test_l2_normalization_and_numerical_tolerance(self) -> None:
        dim = 384
        # Create vectors with arbitrary distinct magnitudes
        v0 = np.full((1, dim), 3.0, dtype=np.float32)
        v1 = np.full((1, dim), 0.5, dtype=np.float32)
        v2 = np.linspace(0.1, 10.0, num=dim, dtype=np.float32).reshape(1, dim)
        stacked = np.vstack([v0, v1, v2])

        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=stacked)
        embedder = SentenceTransformerEmbedder(model=fake_model, normalize_embeddings=True)

        normalized = embedder.embed_texts(["t1", "t2", "t3"])

        norms = np.linalg.norm(normalized, axis=1)
        for idx, norm_val in enumerate(norms):
            assert np.isclose(norm_val, 1.0, atol=1e-4), f"Row {idx} norm ({norm_val}) is not 1.0."

    def test_zero_norm_vector_handling_never_produces_nans(self) -> None:
        dim = 384
        v_normal = np.ones((1, dim), dtype=np.float32)
        v_zero = np.zeros((1, dim), dtype=np.float32)
        stacked = np.vstack([v_normal, v_zero])

        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=stacked)
        embedder = SentenceTransformerEmbedder(model=fake_model, normalize_embeddings=True)

        res = embedder.embed_texts(["normal", "zero"])

        assert not np.isnan(res).any(), "Result must not contain NaN."
        # First vector should be unit norm
        assert np.isclose(np.linalg.norm(res[0]), 1.0, atol=1e-4)
        # Zero vector should remain all-zeros without crashing
        assert np.allclose(res[1], 0.0)

    def test_nan_values_rejected(self) -> None:
        dim = 384
        bad_val = np.ones((2, dim), dtype=np.float32)
        bad_val[0, 5] = np.nan

        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=bad_val)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        with pytest.raises(ValueError, match="non-finite values"):
            embedder.embed_texts(["text1", "text2"])

    def test_infinite_values_rejected(self) -> None:
        dim = 384
        bad_val = np.ones((2, dim), dtype=np.float32)
        bad_val[1, 10] = np.inf

        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=bad_val)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        with pytest.raises(ValueError, match="non-finite values"):
            embedder.embed_texts(["text1", "text2"])

    def test_malformed_1d_array_rejected(self) -> None:
        dim = 384
        arr_1d = np.ones(dim, dtype=np.float32)
        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=arr_1d)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        with pytest.raises(ValueError, match="must be a 2D array"):
            embedder.embed_texts(["query"])

    def test_row_count_mismatch_rejected(self) -> None:
        dim = 384
        # Model returns 2 rows for 3 inputs
        arr = np.ones((2, dim), dtype=np.float32)
        fake_model = FakeSentenceTransformer(embedding_dim=dim, return_val=arr)
        embedder = SentenceTransformerEmbedder(model=fake_model)

        with pytest.raises(ValueError, match=r"Output row count .* does not match"):
            embedder.embed_texts(["query 1", "query 2", "query 3"])


class TestDeviceResolutionAndFallback:
    """Tests for device detection and fallback behavior."""

    def test_cpu_device_resolution(self) -> None:
        req, eff = _resolve_device("cpu")
        assert req == "cpu"
        assert eff == "cpu"

    def test_none_device_resolves_to_cpu(self) -> None:
        req, eff = _resolve_device(None)
        assert req == "cpu"
        assert eff == "cpu"

    def test_cuda_unavailable_falls_back_to_cpu(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            req, eff = _resolve_device("cuda")
            assert req == "cuda"
            assert eff == "cpu"

    def test_cuda_available_preserved(self) -> None:
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            req, eff = _resolve_device("cuda")
            assert req == "cuda"
            assert eff == "cuda"

    def test_torch_uninstalled_falls_back_to_cpu(self) -> None:
        with patch.dict("sys.modules", {"torch": None}):
            req, eff = _resolve_device("cuda")
            assert req == "cuda"
            assert eff == "cpu"

    def test_mps_resolution(self) -> None:
        mock_torch = MagicMock()
        mock_torch.backends.mps.is_available.return_value = True
        with patch.dict("sys.modules", {"torch": mock_torch}):
            req, eff = _resolve_device("mps")
            assert req == "mps"
            assert eff == "mps"

        mock_torch.backends.mps.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            req, eff = _resolve_device("mps")
            assert req == "mps"
            assert eff == "cpu"

    def test_device_fallback_in_embedder(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        mock_torch = MagicMock()
        mock_torch.cuda.is_available.return_value = False
        with patch.dict("sys.modules", {"torch": mock_torch}):
            embedder = SentenceTransformerEmbedder(device="cuda", model=fake_model)
            assert embedder.device == "cuda"
            assert embedder.effective_device == "cpu"


class TestDependencyAndOfflineHandling:
    """Tests for missing sentence-transformers import and offline/local weight errors."""

    def test_missing_sentence_transformers_raises_actionable_import_error(self) -> None:
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            with pytest.raises(ImportError, match="reliability-lab\\[dense\\]"):
                SentenceTransformerEmbedder(model_name_or_path="all-MiniLM-L6-v2")

    def test_offline_local_files_only_missing_weights_raises_file_not_found(self) -> None:
        mock_st = MagicMock(side_effect=OSError("Cannot find files in offline mode"))
        with patch.dict(
            "sys.modules", {"sentence_transformers": MagicMock(SentenceTransformer=mock_st)}
        ):
            with pytest.raises(FileNotFoundError, match="local_files_only=True"):
                SentenceTransformerEmbedder(
                    model_name_or_path="nonexistent-local-model",
                    local_files_only=True,
                )

    def test_custom_model_factory_injection(self) -> None:
        fake_model = FakeSentenceTransformer(embedding_dim=384)
        factory_called = []

        def factory(model_name: str, **kwargs: Any) -> Any:
            factory_called.append((model_name, kwargs))
            return fake_model

        embedder = SentenceTransformerEmbedder(
            model_name_or_path="my-custom-model",
            model_factory=factory,
        )

        assert len(factory_called) == 1
        assert factory_called[0][0] == "my-custom-model"
        assert embedder.model_name == "my-custom-model"
        res = embedder.embed_texts(["hello"])
        assert res.shape == (1, 384)


class TestConstructorValidations:
    """Validation checks on SentenceTransformerEmbedder constructor parameters."""

    def test_invalid_model_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty string"):
            SentenceTransformerEmbedder(model_name_or_path="")
        with pytest.raises(ValueError, match="non-empty string"):
            SentenceTransformerEmbedder(model_name_or_path="   ")

    def test_invalid_expected_dim(self) -> None:
        with pytest.raises(ValueError, match="positive integer >= 1"):
            SentenceTransformerEmbedder(expected_dim=0, model=FakeSentenceTransformer(384))
        with pytest.raises(ValueError, match="positive integer >= 1"):
            SentenceTransformerEmbedder(expected_dim=-384, model=FakeSentenceTransformer(384))
        with pytest.raises(TypeError, match="expected_dim must be an integer"):
            SentenceTransformerEmbedder(expected_dim=True, model=FakeSentenceTransformer(384))  # type: ignore[arg-type]

    def test_invalid_booleans(self) -> None:
        fake_model = FakeSentenceTransformer(384)
        with pytest.raises(TypeError, match="normalize_embeddings must be a boolean"):
            SentenceTransformerEmbedder(normalize_embeddings="yes", model=fake_model)  # type: ignore[arg-type]
        with pytest.raises(TypeError, match="local_files_only must be a boolean"):
            SentenceTransformerEmbedder(local_files_only=1, model=fake_model)  # type: ignore[arg-type]
