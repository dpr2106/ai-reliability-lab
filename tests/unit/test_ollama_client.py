"""Unit tests for the Ollama local LLM inference client and factory integration."""

from __future__ import annotations

import json
from collections.abc import Sequence

import httpx
import pytest

from reliability_lab.core.interfaces import BaseRetriever
from reliability_lab.core.models import Chunk, Query, RetrievalResult
from reliability_lab.llm import (
    GeminiLLMClient,
    LLMGenerationConfig,
    LLMInvalidRequestError,
    LLMMessage,
    LLMProviderUnavailableError,
    MockLLMClient,
    OllamaLLMClient,
    get_llm_client,
)
from reliability_lab.pipeline.models import RAGPipelineConfig
from reliability_lab.pipeline.runner import RAGEvaluationRunner


class TestOllamaInitialization:
    """Tests verifying Ollama client configuration, properties, and defaults."""

    def test_default_initialization(self) -> None:
        client = OllamaLLMClient()
        assert client.model_name == "llama3.2"
        assert client.base_url == "http://localhost:11434"
        assert client.is_mock is False
        assert client.provider == "ollama"
        client.close()

    def test_custom_parameters(self) -> None:
        client = OllamaLLMClient(
            base_url="http://192.168.1.100:11434/",
            model="mistral:7b",
            timeout=30.0,
        )
        assert client.model_name == "mistral:7b"
        assert client.base_url == "http://192.168.1.100:11434"  # Trailing slash stripped
        assert client.is_mock is False
        assert client.provider == "ollama"
        client.close()

    def test_environment_variable_overrides(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://ollama-host:11434/")
        monkeypatch.setenv("OLLAMA_MODEL", "qwen2.5:7b")

        client = OllamaLLMClient()
        assert client.base_url == "http://ollama-host:11434"
        assert client.model_name == "qwen2.5:7b"
        client.close()

    def test_context_manager_support(self) -> None:
        with OllamaLLMClient(model="llama3.2") as client:
            assert client.model_name == "llama3.2"


class TestOllamaMessageAndOptionsFormatting:
    """Tests message translation and generation option mappings."""

    def test_string_prompt_formatting(self) -> None:
        client = OllamaLLMClient()
        messages = client._format_messages("Simple query string")
        assert len(messages) == 1
        assert messages[0] == {"role": "user", "content": "Simple query string"}
        client.close()

    def test_llm_message_sequence_formatting(self) -> None:
        client = OllamaLLMClient()
        input_messages = [
            LLMMessage(role="system", content="You are a factual assistant."),
            LLMMessage(role="user", content="What is BM25?"),
            LLMMessage(role="assistant", content="BM25 is a ranking function."),
            LLMMessage(role="user", content="How does it score terms?"),
        ]
        formatted = client._format_messages(input_messages)
        assert formatted == [
            {"role": "system", "content": "You are a factual assistant."},
            {"role": "user", "content": "What is BM25?"},
            {"role": "assistant", "content": "BM25 is a ranking function."},
            {"role": "user", "content": "How does it score terms?"},
        ]
        client.close()

    def test_dict_message_sequence_formatting(self) -> None:
        client = OllamaLLMClient()
        input_dicts = [
            {"role": "system", "content": "System directive"},
            {"role": "user", "content": "User question"},
        ]
        formatted = client._format_messages(input_dicts)
        assert formatted == [
            {"role": "system", "content": "System directive"},
            {"role": "user", "content": "User question"},
        ]
        client.close()

    def test_empty_message_sequence_fallback(self) -> None:
        client = OllamaLLMClient()
        formatted = client._format_messages([])
        assert formatted == [{"role": "user", "content": ""}]
        client.close()

    def test_invalid_role_rejected(self) -> None:
        client = OllamaLLMClient()
        with pytest.raises(ValueError, match="Invalid message role 'admin'"):
            client._format_messages([{"role": "admin", "content": "Invalid role"}])
        client.close()

    def test_invalid_type_rejected(self) -> None:
        client = OllamaLLMClient()
        with pytest.raises(TypeError, match="Unsupported message type"):
            client._format_messages([123])  # type: ignore[list-item]
        client.close()

    def test_options_mapping(self) -> None:
        client = OllamaLLMClient()
        config = LLMGenerationConfig(
            temperature=0.7,
            max_output_tokens=512,
            top_p=0.95,
            top_k=40,
            stop_sequences=["\n\n", "User:"],
        )
        options = client._build_options(config)
        assert options["temperature"] == 0.7
        assert options["num_predict"] == 512
        assert options["top_p"] == 0.95
        assert options["top_k"] == 40
        assert options["stop"] == ["\n\n", "User:"]
        client.close()


class TestOllamaGeneration:
    """Hermetic tests using httpx.MockTransport for simulated Ollama interactions."""

    def test_successful_text_generation(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            assert request.url.path == "/api/chat"
            body = json.loads(request.read())
            assert body["model"] == "llama3.2"
            assert body["stream"] is False
            assert body["messages"] == [{"role": "user", "content": "Explain NDCG."}]
            assert body["options"]["temperature"] == 0.0
            assert body["options"]["num_predict"] == 1024

            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "created_at": "2026-10-31T20:00:00Z",
                    "message": {
                        "role": "assistant",
                        "content": "Normalized Discounted Cumulative Gain evaluates ranking.",
                    },
                    "done_reason": "stop",
                    "done": True,
                    "prompt_eval_count": 15,
                    "eval_count": 28,
                    "total_duration": 450000000,
                },
            )

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        resp = client.generate("Explain NDCG.")
        assert resp.text == "Normalized Discounted Cumulative Gain evaluates ranking."
        assert resp.model == "llama3.2"
        assert resp.finish_reason == "stop"
        assert resp.latency_ms >= 0.0
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 15
        assert resp.usage.completion_tokens == 28
        assert resp.usage.total_tokens == 43
        assert resp.raw_response is not None
        assert resp.raw_response["done"] is True

    def test_missing_token_usage_graceful(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {"role": "assistant", "content": "Answer without counts."},
                    "done": True,
                },
            )

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        resp = client.generate("Hello")
        assert resp.text == "Answer without counts."
        assert resp.usage is None

    def test_generate_json_structured_output(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            assert body.get("format") == "json"

            json_content = json.dumps({"verdict": "SUPPORTED", "confidence": 0.95})
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {
                        "role": "assistant",
                        "content": f"```json\n{json_content}\n```",
                    },
                    "done": True,
                },
            )

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        result = client.generate_json("Evaluate claim validity.")
        assert isinstance(result, dict)
        assert result["verdict"] == "SUPPORTED"
        assert result["confidence"] == 0.95


class TestOllamaErrorHandling:
    """Tests verifying actionable error handling across network and runtime failures."""

    def test_connection_refused_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("Connection refused to 127.0.0.1:11434")

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(
            LLMProviderUnavailableError,
            match=r"Ensure Ollama is installed and running locally",
        ):
            client.generate("Test prompt")

    def test_request_timeout_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ReadTimeout("Read timed out after 60 seconds")

        client = OllamaLLMClient(
            timeout=10.0,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(
            LLMProviderUnavailableError,
            match=r"timed out after 10\.0s",
        ):
            client.generate("Test prompt")

    def test_model_not_found_404_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                json={"error": "model 'nonexistent-model' not found, try pulling it first"},
            )

        client = OllamaLLMClient(
            model="nonexistent-model",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(
            LLMInvalidRequestError,
            match=r"ollama pull nonexistent-model",
        ) as exc_info:
            client.generate("Test prompt")

        assert exc_info.value.status_code == 404
        assert exc_info.value.provider == "ollama"

    def test_endpoint_not_found_404_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, text="404 page not found")

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(
            LLMProviderUnavailableError,
            match=r"Ollama endpoint '.*' not found",
        ):
            client.generate("Test prompt")

    def test_server_error_500(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="Internal server error in llama.cpp engine")

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMProviderUnavailableError, match="Ollama server error"):
            client.generate("Test prompt")

    def test_bad_request_400(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(400, text="Invalid context window parameter")

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMInvalidRequestError, match="Ollama invalid request"):
            client.generate("Test prompt")

    def test_malformed_json_response(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="<HTML>Not JSON</HTML>")

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMInvalidRequestError, match="malformed non-JSON payload"):
            client.generate("Test prompt")

    def test_missing_message_field(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"model": "llama3.2", "done": True})

        client = OllamaLLMClient(
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )
        with pytest.raises(LLMInvalidRequestError, match="missing valid 'message' object"):
            client.generate("Test prompt")


class TestFactoryAndCLIIntegration:
    """Tests verifying provider resolution in get_llm_client and CLI invocation."""

    def test_factory_ollama_provider_resolution(self) -> None:
        client = get_llm_client(provider="ollama", model="gemma2:9b")
        assert isinstance(client, OllamaLLMClient)
        assert client.model_name == "gemma2:9b"
        assert client.provider == "ollama"
        assert client.is_mock is False
        client.close()

    def test_factory_ollama_default_model(self) -> None:
        client = get_llm_client(provider="ollama")
        assert isinstance(client, OllamaLLMClient)
        assert client.model_name == "llama3.2"
        assert client.provider == "ollama"
        client.close()

    def test_factory_preserves_mock_and_gemini(self) -> None:
        mock_client = get_llm_client(provider="mock")
        assert isinstance(mock_client, MockLLMClient)
        assert mock_client.provider == "mock"
        assert mock_client.is_mock is True

        gemini_client = get_llm_client(provider="gemini", api_key="fake-key")
        assert isinstance(gemini_client, GeminiLLMClient)
        assert gemini_client.provider == "gemini"
        assert gemini_client.is_mock is False

    def test_factory_auto_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        auto_mock = get_llm_client(provider="auto")
        assert isinstance(auto_mock, MockLLMClient)
        assert auto_mock.is_mock is True

        monkeypatch.setenv("GEMINI_API_KEY", "test-key")
        auto_gemini = get_llm_client(provider="auto")
        assert isinstance(auto_gemini, GeminiLLMClient)
        assert auto_gemini.is_mock is False

    def test_factory_invalid_provider_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider 'openai'"):
            get_llm_client(provider="openai")

    def test_pipeline_provenance_integration(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "model": "llama3.2",
                    "message": {
                        "role": "assistant",
                        "content": "BM25 uses k1 and b parameters [CHK-001].",
                    },
                    "done": True,
                    "prompt_eval_count": 20,
                    "eval_count": 10,
                },
            )

        mock_http = httpx.Client(transport=httpx.MockTransport(handler))
        ollama_client = OllamaLLMClient(model="llama3.2", http_client=mock_http)

        class DummyRetriever(BaseRetriever):
            @property
            def name(self) -> str:
                return "DummyRetriever"

            def index(self, corpus: Sequence[Chunk]) -> None:
                pass

            def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
                return [
                    RetrievalResult(
                        chunk=Chunk(chunk_id="CHK-001", doc_id="d1", content="k1=1.5 and b=0.75"),
                        score=1.0,
                        rank=1,
                        retriever_name="DummyRetriever",
                    )
                ]

        runner = RAGEvaluationRunner(
            retriever=DummyRetriever(),
            generator=ollama_client,
            config=RAGPipelineConfig(
                top_k=1,
                enable_nli=False,
                enable_judge=False,
                enable_groundedness=False,
                enable_citations=False,
            ),
        )

        query = Query(query_id="Q-01", text="What are BM25 parameters?")
        result = runner.evaluate_query(query)

        assert result.generation_provenance.provider == "ollama"
        assert result.generation_provenance.model == "llama3.2"
        assert result.generation_provenance.is_mock is False
        assert result.generation_provenance.token_usage["prompt_tokens"] == 20
        assert result.generation_provenance.token_usage["completion_tokens"] == 10
