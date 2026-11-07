"""Unit tests for LLM streaming capabilities (STING V2-003 Phase C).

Verifies incremental token streaming, usage accounting, and error handling
for both MockLLMClient and OllamaLLMClient without external dependencies.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock

import httpx
import pytest

from reliability_lab.llm.client import MockLLMClient, OllamaLLMClient
from reliability_lab.llm.exceptions import (
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
)
from reliability_lab.llm.models import LLMMessage, LLMStreamChunk, TokenUsage


class TestMockLLMClientStreaming:
    """Test suite for MockLLMClient streaming implementation."""

    def test_mock_streaming_progressive_tokens(self) -> None:
        """Verify mock yields incremental tokens that reconstruct the complete response."""
        client = MockLLMClient(default_response="Alpha beta gamma.")
        messages = [LLMMessage(role="user", content="Hello")]

        chunks: list[LLMStreamChunk] = list(client.generate_stream(messages))

        assert len(chunks) > 1, (
            "Expected multiple progressive chunks, not a single monolithic chunk."
        )
        reconstructed = "".join(c.delta for c in chunks)
        assert reconstructed == "Alpha beta gamma."

        # The last chunk must signal done with token usage
        last_chunk = chunks[-1]
        assert last_chunk.done is True
        assert last_chunk.usage is not None
        assert last_chunk.usage.total_tokens > 0
        assert last_chunk.finish_reason == "STOP"

        # Intermediate chunks must not be done
        for c in chunks[:-1]:
            assert c.done is False

    def test_mock_streaming_custom_chunks(self) -> None:
        """Verify custom stream_chunks list is yielded sequentially."""
        custom_tokens = ["Step 1: ", "retrieval ", "done.", " [CIT-001]"]
        client = MockLLMClient(stream_chunks=custom_tokens)
        messages = [LLMMessage(role="user", content="Explain steps")]

        chunks = list(client.generate_stream(messages))
        deltas = [c.delta for c in chunks if c.delta]
        assert deltas == custom_tokens

        last_chunk = chunks[-1]
        assert last_chunk.done is True

    def test_mock_streaming_mid_stream_exception(self) -> None:
        """Verify exception raised mid-stream terminates generator and propagates."""
        custom_chunks = ["First token ", RuntimeError("Network connection dropped mid-stream")]
        client = MockLLMClient(stream_chunks=custom_chunks)
        messages = [LLMMessage(role="user", content="Test error")]

        gen = client.generate_stream(messages)
        first_chunk = next(gen)
        assert first_chunk.delta == "First token "

        with pytest.raises(RuntimeError, match="Network connection dropped mid-stream"):
            next(gen)

    def test_mock_streaming_records_history(self) -> None:
        """Verify streaming generation is recorded in client history."""
        client = MockLLMClient(default_response="Trace test")
        list(client.generate_stream("Direct prompt"))

        assert client.call_count == 1
        record = client.history[0]
        assert record["streaming"] is True
        assert record["response_text"] == "Trace test"


class TestOllamaLLMClientStreaming:
    """Test suite for OllamaLLMClient streaming implementation via simulated HTTP stream."""

    def test_ollama_streaming_successful_chunks(self) -> None:
        """Verify Ollama streaming parses newline-delimited JSON chunks correctly."""
        raw_lines = [
            json.dumps(
                {
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "The "},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "capital "},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "is Paris."},
                    "done": False,
                }
            ),
            json.dumps(
                {
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": ""},
                    "done": True,
                    "done_reason": "stop",
                    "prompt_eval_count": 15,
                    "eval_count": 8,
                }
            ),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = raw_lines

        mock_http_client = MagicMock(spec=httpx.Client)
        mock_http_client.stream.return_value.__enter__.return_value = mock_response

        client = OllamaLLMClient(
            base_url="http://localhost:11434", model="llama3.2", http_client=mock_http_client
        )
        chunks = list(
            client.generate_stream([LLMMessage(role="user", content="Capital of France?")])
        )

        assert len(chunks) == 4
        reconstructed = "".join(c.delta for c in chunks)
        assert reconstructed == "The capital is Paris."

        last_chunk = chunks[-1]
        assert last_chunk.done is True
        assert last_chunk.usage == TokenUsage(
            prompt_tokens=15, completion_tokens=8, total_tokens=23
        )
        assert last_chunk.finish_reason == "stop"

    def test_ollama_streaming_malformed_json_line(self) -> None:
        """Verify malformed non-JSON chunk raises LLMInvalidRequestError."""
        raw_lines = [
            json.dumps(
                {
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "Valid"},
                    "done": False,
                }
            ),
            "{not valid json",
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = raw_lines

        mock_http_client = MagicMock(spec=httpx.Client)
        mock_http_client.stream.return_value.__enter__.return_value = mock_response

        client = OllamaLLMClient(http_client=mock_http_client)
        gen = client.generate_stream("Prompt")

        first = next(gen)
        assert first.delta == "Valid"

        with pytest.raises(LLMInvalidRequestError, match="malformed non-JSON payload"):
            next(gen)

    def test_ollama_streaming_provider_error_payload(self) -> None:
        """Verify Ollama mid-stream error object raises LLMProviderUnavailableError."""
        raw_lines = [
            json.dumps(
                {
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "Ok"},
                    "done": False,
                }
            ),
            json.dumps({"error": "CUDA out of memory"}),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.iter_lines.return_value = raw_lines

        mock_http_client = MagicMock(spec=httpx.Client)
        mock_http_client.stream.return_value.__enter__.return_value = mock_response

        client = OllamaLLMClient(http_client=mock_http_client)
        gen = client.generate_stream("Prompt")

        first = next(gen)
        assert first.delta == "Ok"

        with pytest.raises(LLMProviderUnavailableError, match="CUDA out of memory"):
            next(gen)

    def test_ollama_streaming_model_not_found_404(self) -> None:
        """Verify 404 with missing model raises LLMInvalidRequestError."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = json.dumps({"error": "model 'nonexistent' not found, try pulling it"})
        mock_response.json.return_value = {"error": "model 'nonexistent' not found, try pulling it"}

        mock_http_client = MagicMock(spec=httpx.Client)
        mock_http_client.stream.return_value.__enter__.return_value = mock_response

        client = OllamaLLMClient(model="nonexistent", http_client=mock_http_client)
        with pytest.raises(
            LLMInvalidRequestError, match="Download the model by running 'ollama pull"
        ):
            list(client.generate_stream("Prompt"))

    def test_ollama_streaming_server_error_500(self) -> None:
        """Verify 500 error raises LLMProviderUnavailableError."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal Server Error"

        mock_http_client = MagicMock(spec=httpx.Client)
        mock_http_client.stream.return_value.__enter__.return_value = mock_response

        client = OllamaLLMClient(http_client=mock_http_client)
        with pytest.raises(LLMProviderUnavailableError, match="Ollama server error"):
            list(client.generate_stream("Prompt"))
