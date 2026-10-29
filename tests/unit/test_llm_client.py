"""Unit tests for abstract LLM client, mock provider, and Gemini adapter."""

from __future__ import annotations

import json
from unittest.mock import patch

import httpx
import pytest

from reliability_lab.llm import (
    GeminiLLMClient,
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMGenerationConfig,
    LLMInvalidRequestError,
    LLMMessage,
    LLMRateLimitError,
    LLMResponse,
    MockLLMClient,
    RateLimiter,
    TokenUsage,
    get_llm_client,
)


class TestModelsAndValidation:
    """Tests for LLM data contracts and parameter constraints."""

    def test_llm_message_creation_and_validation(self) -> None:
        msg = LLMMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"

        with pytest.raises(ValueError, match="Invalid message role"):
            LLMMessage(role="invalid_role", content="Hello")  # type: ignore[arg-type]

        with pytest.raises(TypeError, match="content must be a string"):
            LLMMessage(role="user", content=123)  # type: ignore[arg-type]

    def test_generation_config_validation(self) -> None:
        cfg = LLMGenerationConfig(temperature=0.7, max_output_tokens=500)
        assert cfg.temperature == 0.7
        assert cfg.max_output_tokens == 500

        with pytest.raises(ValueError, match="Temperature must be in"):
            LLMGenerationConfig(temperature=-0.1)

        with pytest.raises(ValueError, match="Temperature must be in"):
            LLMGenerationConfig(temperature=2.5)

        with pytest.raises(ValueError, match="max_output_tokens must be positive"):
            LLMGenerationConfig(max_output_tokens=0)


class TestMockLLMClient:
    """Tests for the deterministic zero-network MockLLMClient."""

    def test_mock_default_generation(self) -> None:
        client = MockLLMClient(model_name="test-mock", default_response="Default answer")
        assert client.model_name == "test-mock"
        assert client.call_count == 0

        resp = client.generate("What is IR?")
        assert resp.text == "Default answer"
        assert resp.model == "test-mock"
        assert resp.usage is not None
        assert resp.usage.total_tokens > 0
        assert client.call_count == 1
        assert client.history[0]["combined_text"] == "user: What is IR?"

    def test_mock_canned_pattern_matching(self) -> None:
        client = MockLLMClient(
            canned_responses={
                "recall": "Recall measures retrieved relevant items.",
                "precision": "Precision measures relevant retrieved items.",
            }
        )

        resp1 = client.generate("Tell me about recall in benchmarks")
        assert resp1.text == "Recall measures retrieved relevant items."

        resp2 = client.generate("Explain precision score")
        assert resp2.text == "Precision measures relevant retrieved items."

        resp3 = client.generate("Something unrelated")
        assert resp3.text == "Mock generation response"

    def test_mock_response_queue(self) -> None:
        client = MockLLMClient(
            response_queue=[
                "First answer",
                LLMResponse(text="Custom object", model="test-mock", usage=TokenUsage(1, 1, 2)),
                LLMRateLimitError("Simulated limit"),
            ]
        )

        assert client.generate("Q1").text == "First answer"
        assert client.generate("Q2").text == "Custom object"
        with pytest.raises(LLMRateLimitError, match="Simulated limit"):
            client.generate("Q3")

    def test_mock_direct_error_raising(self) -> None:
        client = MockLLMClient(error_to_raise=LLMAuthenticationError("Simulated auth failure"))
        with pytest.raises(LLMAuthenticationError, match="Simulated auth failure"):
            client.generate("Test prompt")

    def test_mock_history_and_reset(self) -> None:
        client = MockLLMClient()
        client.generate("Query 1")
        client.generate("Query 2")
        assert client.call_count == 2

        client.reset()
        assert client.call_count == 0
        assert len(client.history) == 0

    def test_mock_generate_json(self) -> None:
        client = MockLLMClient(
            canned_responses={
                "structured": '```json\n{"status": "PASS", "score": 0.95}\n```',
                "plain": '{"status": "OK"}',
            }
        )

        data1 = client.generate_json("Request structured data")
        assert data1 == {"status": "PASS", "score": 0.95}

        data2 = client.generate_json("Request plain data")
        assert data2 == {"status": "OK"}

    def test_mock_generate_json_invalid(self) -> None:
        client = MockLLMClient(default_response="Definitely not json {{{")
        # If response_mime_type == 'application/json', non-json string is wrapped in a dict
        data = client.generate_json("Non json query")
        assert "response" in data


class TestRateLimiter:
    """Tests for the sliding window rate limiter."""

    def test_rate_limiter_instant_acquisition(self) -> None:
        limiter = RateLimiter(requests_per_minute=10)
        assert limiter.acquire(timeout=1.0) is True

    def test_rate_limiter_invalid_rpm(self) -> None:
        with pytest.raises(ValueError, match="requests_per_minute must be strictly positive"):
            RateLimiter(requests_per_minute=0)


class TestGeminiLLMClient:
    """Tests for the Gemini Flash client with mocked HTTP transport."""

    def test_missing_api_key_raises_auth_error(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(LLMAuthenticationError, match="GEMINI_API_KEY must be provided"):
                GeminiLLMClient(api_key=None, mock_fallback=False)

    def test_missing_api_key_with_mock_fallback(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = GeminiLLMClient(api_key=None, mock_fallback=True)
            assert client.model_name == "gemini-1.5-flash"
            resp = client.generate("Hello fallback")
            assert resp.text == "Mock generation response"

    def test_successful_gemini_generation(self) -> None:
        gemini_response_payload = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Generated answer from Gemini."}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                }
            ],
            "usageMetadata": {
                "promptTokenCount": 12,
                "candidatesTokenCount": 6,
                "totalTokenCount": 18,
            },
        }

        def handler(request: httpx.Request) -> httpx.Response:
            assert "key=fake-key" in str(request.url)
            body = json.loads(request.read())
            assert "contents" in body
            assert body["generationConfig"]["temperature"] == 0.2
            return httpx.Response(200, json=gemini_response_payload)

        transport = httpx.MockTransport(handler)
        mock_http = httpx.Client(transport=transport)

        client = GeminiLLMClient(
            api_key="fake-key",
            model="gemini-1.5-flash",
            http_client=mock_http,
        )

        resp = client.generate(
            prompt="What is NDCG?",
            config=LLMGenerationConfig(temperature=0.2),
        )

        assert resp.text == "Generated answer from Gemini."
        assert resp.finish_reason == "STOP"
        assert resp.usage is not None
        assert resp.usage.prompt_tokens == 12
        assert resp.usage.completion_tokens == 6
        assert resp.usage.total_tokens == 18
        assert resp.latency_ms >= 0.0

    def test_system_prompt_and_conversation_turns(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            body = json.loads(request.read())
            assert "systemInstruction" in body
            assert body["systemInstruction"]["parts"][0]["text"] == "You are a concise evaluator."
            contents = body["contents"]
            assert len(contents) == 2
            assert contents[0]["role"] == "user"
            assert contents[0]["parts"][0]["text"] == "Turn 1"
            assert contents[1]["role"] == "model"  # mapped assistant -> model
            assert contents[1]["parts"][0]["text"] == "Response 1"

            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {"parts": [{"text": "Turn 2 response"}]},
                            "finishReason": "STOP",
                        }
                    ]
                },
            )

        client = GeminiLLMClient(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        messages = [
            LLMMessage(role="system", content="You are a concise evaluator."),
            LLMMessage(role="user", content="Turn 1"),
            LLMMessage(role="assistant", content="Response 1"),
        ]

        resp = client.generate(messages)
        assert resp.text == "Turn 2 response"

    def test_authentication_error_handling(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                401,
                json={"error": {"message": "API key not valid", "status": "UNAUTHENTICATED"}},
            )

        client = GeminiLLMClient(
            api_key="invalid-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMAuthenticationError, match="API key not valid"):
            client.generate("Test prompt")

    def test_context_length_exceeded_error(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                400,
                json={"error": {"message": "Request contains too many tokens for context length"}},
            )

        client = GeminiLLMClient(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMContextLengthExceededError, match="context length"):
            client.generate("Huge prompt")

    def test_retry_on_rate_limit_then_success(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls == 1:
                return httpx.Response(
                    429,
                    json={"error": {"message": "Resource has been exhausted (rate limit)"}},
                )
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Success after retry"}]}}]},
            )

        client = GeminiLLMClient(
            api_key="test-key",
            base_backoff_seconds=0.01,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        resp = client.generate("Retry test")
        assert calls == 2
        assert resp.text == "Success after retry"

    def test_rate_limit_exhaustion_raises(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                429,
                json={"error": {"message": "Resource has been exhausted"}},
            )

        client = GeminiLLMClient(
            api_key="test-key",
            max_retries=1,
            base_backoff_seconds=0.01,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMRateLimitError, match="rate limit exceeded"):
            client.generate("Fail prompt")

    def test_service_unavailable_retry(self) -> None:
        calls = 0

        def handler(request: httpx.Request) -> httpx.Response:
            nonlocal calls
            calls += 1
            if calls < 2:
                return httpx.Response(503, text="Service Unavailable")
            return httpx.Response(
                200,
                json={"candidates": [{"content": {"parts": [{"text": "Recovered"}]}}]},
            )

        client = GeminiLLMClient(
            api_key="test-key",
            base_backoff_seconds=0.01,
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        resp = client.generate("503 test")
        assert resp.text == "Recovered"
        assert calls == 2

    def test_empty_candidates_blocked_prompt(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={"candidates": [], "promptFeedback": {"blockReason": "SAFETY"}},
            )

        client = GeminiLLMClient(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        with pytest.raises(LLMInvalidRequestError, match="blocked: SAFETY"):
            client.generate("Toxic prompt")

    def test_generate_json_parsing(self) -> None:
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                json={
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {"text": '```json\n{"score": 0.88, "verdict": "PASS"}\n```'}
                                ]
                            }
                        }
                    ]
                },
            )

        client = GeminiLLMClient(
            api_key="test-key",
            http_client=httpx.Client(transport=httpx.MockTransport(handler)),
        )

        data = client.generate_json("Evaluate this")
        assert data == {"score": 0.88, "verdict": "PASS"}


class TestFactoryFunction:
    """Tests for the get_llm_client factory."""

    def test_factory_mock_provider(self) -> None:
        client = get_llm_client(provider="mock", model_name="custom-mock")
        assert isinstance(client, MockLLMClient)
        assert client.model_name == "custom-mock"

    def test_factory_gemini_provider(self) -> None:
        client = get_llm_client(provider="gemini", api_key="secret-key")
        assert isinstance(client, GeminiLLMClient)
        assert client.model_name == "gemini-1.5-flash"

    def test_factory_auto_with_env_key(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": "env-secret"}):
            client = get_llm_client(provider="auto")
            assert isinstance(client, GeminiLLMClient)

    def test_factory_auto_without_env_key_falls_back_to_mock(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            client = get_llm_client(provider="auto")
            assert isinstance(client, MockLLMClient)

    def test_factory_unknown_provider(self) -> None:
        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_client(provider="unsupported-provider")
