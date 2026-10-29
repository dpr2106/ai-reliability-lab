"""Provider-independent LLM client interface, Gemini Flash adapter, and deterministic mock."""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any

import httpx

from reliability_lab.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
)
from reliability_lab.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMResponse,
    TokenUsage,
)
from reliability_lab.llm.rate_limiter import RateLimiter

logger = logging.getLogger(__name__)


def _normalize_prompt(
    prompt: str | Sequence[LLMMessage | dict[str, str]],
) -> tuple[str | None, list[LLMMessage]]:
    """Normalize input into an optional system prompt and conversation messages."""
    if isinstance(prompt, str):
        return None, [LLMMessage(role="user", content=prompt)]

    system_prompt: str | None = None
    messages: list[LLMMessage] = []

    for item in prompt:
        if isinstance(item, dict):
            role = item.get("role", "user")
            content = str(item.get("content", ""))
            msg = LLMMessage(role=role, content=content)  # type: ignore[arg-type]
        elif isinstance(item, LLMMessage):
            msg = item
        else:
            raise TypeError(f"Unsupported message type: {type(item)}")

        if msg.role == "system":
            system_prompt = f"{system_prompt}\n{msg.content}" if system_prompt else msg.content
        else:
            messages.append(msg)

    if not messages:
        messages.append(LLMMessage(role="user", content=""))

    return system_prompt, messages


def _clean_json_text(text: str) -> str:
    """Extract and unescape JSON text if wrapped inside markdown code fences."""
    stripped = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", stripped)
    return match.group(1).strip() if match else stripped


class BaseLLMClient(ABC):
    """Abstract provider-independent base interface for LLM completions."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Name or identifier of the underlying generative model."""
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        """Generate a response text for the provided prompt."""
        ...

    def generate_json(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> dict[str, Any]:
        """Generate and parse structured JSON output.

        Configures response_mime_type to 'application/json' by default and strips
        markdown code fences if emitted by the model.
        """
        cfg = config or LLMGenerationConfig()
        if cfg.response_mime_type != "application/json":
            cfg = LLMGenerationConfig(
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_output_tokens,
                top_p=cfg.top_p,
                top_k=cfg.top_k,
                stop_sequences=cfg.stop_sequences,
                response_mime_type="application/json",
            )

        resp = self.generate(prompt=prompt, config=cfg)
        cleaned_text = _clean_json_text(resp.text)

        try:
            parsed = json.loads(cleaned_text)
            if not isinstance(parsed, dict):
                return {"data": parsed}
            return parsed
        except json.JSONDecodeError as exc:
            raise LLMInvalidRequestError(
                f"Model response was not valid JSON: {exc}. Response text: {resp.text[:200]}"
            ) from exc


class MockLLMClient(BaseLLMClient):
    """Deterministic, zero-network mock LLM client for testing and offline verification."""

    def __init__(
        self,
        model_name: str = "mock-model",
        default_response: str = "Mock generation response",
        canned_responses: dict[str, str] | None = None,
        response_queue: list[str | LLMResponse | Exception] | None = None,
        error_to_raise: Exception | None = None,
        latency_ms: float = 1.0,
    ):
        self._model_name = model_name
        self.default_response = default_response
        self.canned_responses = dict(canned_responses) if canned_responses else {}
        self.response_queue = list(response_queue) if response_queue else []
        self.error_to_raise = error_to_raise
        self.latency_ms = latency_ms
        self.history: list[dict[str, Any]] = []

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def call_count(self) -> int:
        return len(self.history)

    def reset(self) -> None:
        """Clear call history and queued responses."""
        self.history.clear()
        self.response_queue.clear()

    def generate(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        cfg = config or LLMGenerationConfig()
        sys_prompt, messages = _normalize_prompt(prompt)

        # Full serialized prompt text for matching and recording
        combined_text = "\n".join(
            ([f"system: {sys_prompt}"] if sys_prompt else [])
            + [f"{m.role}: {m.content}" for m in messages]
        )

        call_record: dict[str, Any] = {
            "prompt": prompt,
            "system_prompt": sys_prompt,
            "messages": messages,
            "combined_text": combined_text,
            "config": cfg,
            "timestamp": time.time(),
        }

        if self.error_to_raise is not None:
            self.history.append({**call_record, "error": str(self.error_to_raise)})
            raise self.error_to_raise

        if self.response_queue:
            item = self.response_queue.pop(0)
            if isinstance(item, Exception):
                self.history.append({**call_record, "error": str(item)})
                raise item
            if isinstance(item, LLMResponse):
                self.history.append({**call_record, "response": item})
                return item
            response_text = str(item)
        else:
            # Check canned responses by substring match
            response_text = self.default_response
            for pattern, canned in self.canned_responses.items():
                if pattern in combined_text or (sys_prompt and pattern in sys_prompt):
                    response_text = canned
                    break

        # If JSON was explicitly requested and response is not valid JSON, serialize as dict
        if cfg.response_mime_type == "application/json":
            try:
                json.loads(_clean_json_text(response_text))
            except (json.JSONDecodeError, TypeError):
                response_text = json.dumps({"response": response_text})

        prompt_tokens = max(1, len(combined_text.split()))
        completion_tokens = max(1, len(response_text.split()))

        resp = LLMResponse(
            text=response_text,
            model=self._model_name,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
            finish_reason="STOP",
            latency_ms=self.latency_ms,
            raw_response={"mock": True},
        )
        self.history.append({**call_record, "response": resp})
        return resp


class GeminiLLMClient(BaseLLMClient):
    """Production client for Google Gemini models via official Generative Language REST API.

    Uses resilient HTTP transport via httpx with exponential backoff retries on transient
    errors, rate limiting, and structured JSON configuration.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "gemini-1.5-flash",
        timeout: float = 30.0,
        max_retries: int = 3,
        base_backoff_seconds: float = 0.5,
        max_backoff_seconds: float = 8.0,
        rate_limiter: RateLimiter | None = None,
        mock_fallback: bool = False,
        http_client: httpx.Client | None = None,
    ):
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        self._model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self.base_backoff = base_backoff_seconds
        self.max_backoff = max_backoff_seconds
        self.rate_limiter = rate_limiter or RateLimiter(requests_per_minute=60)
        self.mock_fallback = mock_fallback
        self._mock_delegate: MockLLMClient | None = None

        if not resolved_key:
            if mock_fallback:
                logger.warning("GEMINI_API_KEY is not configured; falling back to MockLLMClient.")
                self._mock_delegate = MockLLMClient(model_name=f"mock-{model}")
                self.api_key = ""
            else:
                raise LLMAuthenticationError(
                    "GEMINI_API_KEY must be provided via constructor or environment variable.",
                    status_code=401,
                    provider="gemini",
                )
        else:
            self.api_key = resolved_key

        self._client = http_client or httpx.Client(timeout=self.timeout)

    @property
    def model_name(self) -> str:
        return self._model

    def _format_payload(
        self,
        system_prompt: str | None,
        messages: list[LLMMessage],
        config: LLMGenerationConfig,
    ) -> dict[str, Any]:
        """Construct the Gemini API generateContent request body."""
        contents: list[dict[str, Any]] = []

        for msg in messages:
            # Gemini roles: 'user' or 'model'
            gemini_role = "model" if msg.role == "assistant" else "user"
            contents.append(
                {
                    "role": gemini_role,
                    "parts": [{"text": msg.content}],
                }
            )

        payload: dict[str, Any] = {"contents": contents}

        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}

        generation_config: dict[str, Any] = {
            "temperature": config.temperature,
            "maxOutputTokens": config.max_output_tokens,
        }

        if config.top_p is not None:
            generation_config["topP"] = config.top_p
        if config.top_k is not None:
            generation_config["topK"] = config.top_k
        if config.stop_sequences:
            generation_config["stopSequences"] = config.stop_sequences
        if config.response_mime_type:
            generation_config["responseMimeType"] = config.response_mime_type

        payload["generationConfig"] = generation_config
        return payload

    def _execute_http_request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Execute HTTP POST with bounded exponential backoff on transient errors."""
        url = f"{self.BASE_URL}/models/{self._model}:generateContent"
        params = {"key": self.api_key}

        attempt = 0
        last_exception: Exception | None = None

        while attempt <= self.max_retries:
            # Respect rate limiter
            self.rate_limiter.acquire()
            start_time = time.perf_counter()

            try:
                response = self._client.post(url, params=params, json=payload)
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0

                if response.status_code == 200:
                    return response.json(), elapsed_ms

                # Handle HTTP errors
                error_body: dict[str, Any] = {}
                try:
                    error_body = response.json().get("error", {})
                except Exception:
                    pass

                error_message = error_body.get("message") or response.text or response.reason_phrase
                status = response.status_code

                if status in (401, 403):
                    raise LLMAuthenticationError(
                        f"Gemini authentication failed ({status}): {error_message}",
                        status_code=status,
                        provider="gemini",
                    )

                if status == 400:
                    if (
                        "token" in error_message.lower()
                        or "context length" in error_message.lower()
                    ):
                        raise LLMContextLengthExceededError(
                            f"Gemini context length exceeded: {error_message}",
                            status_code=400,
                            provider="gemini",
                        )
                    raise LLMInvalidRequestError(
                        f"Gemini bad request (400): {error_message}",
                        status_code=400,
                        provider="gemini",
                    )

                # Transient errors: 429 (RateLimit) or 503 (Unavailable) or 500
                if status == 429:
                    last_exception = LLMRateLimitError(
                        f"Gemini rate limit exceeded (429): {error_message}",
                        status_code=429,
                        provider="gemini",
                    )
                elif status >= 500:
                    last_exception = LLMProviderUnavailableError(
                        f"Gemini service unavailable ({status}): {error_message}",
                        status_code=status,
                        provider="gemini",
                    )
                else:
                    raise LLMError(
                        f"Gemini request failed ({status}): {error_message}",
                        status_code=status,
                        provider="gemini",
                    )

            except (httpx.TimeoutException, httpx.ConnectError) as exc:
                elapsed_ms = (time.perf_counter() - start_time) * 1000.0
                last_exception = LLMProviderUnavailableError(
                    f"Gemini network connection error: {exc}",
                    provider="gemini",
                )
            except (LLMAuthenticationError, LLMInvalidRequestError):
                raise

            attempt += 1
            if attempt <= self.max_retries:
                backoff = min(
                    self.max_backoff,
                    self.base_backoff * (2 ** (attempt - 1)) + random.uniform(0.05, 0.2),
                )
                logger.warning(
                    "Retrying Gemini call after transient error (%s). Attempt %d/%d, sleep %.2fs",
                    last_exception,
                    attempt,
                    self.max_retries,
                    backoff,
                )
                time.sleep(backoff)

        if last_exception:
            raise last_exception
        raise LLMProviderUnavailableError(
            "Gemini call failed with unknown error.", provider="gemini"
        )

    def generate(
        self,
        prompt: str | Sequence[LLMMessage | dict[str, str]],
        config: LLMGenerationConfig | None = None,
    ) -> LLMResponse:
        if self._mock_delegate is not None:
            return self._mock_delegate.generate(prompt=prompt, config=config)

        cfg = config or LLMGenerationConfig()
        sys_prompt, messages = _normalize_prompt(prompt)
        payload = self._format_payload(sys_prompt, messages, cfg)

        data, latency_ms = self._execute_http_request(payload)

        # Parse candidates
        candidates = data.get("candidates", [])
        if not candidates:
            # Handle block or empty response
            prompt_feedback = data.get("promptFeedback", {})
            block_reason = prompt_feedback.get("blockReason", "Unknown")
            raise LLMInvalidRequestError(
                f"Gemini generation returned no candidates (blocked: {block_reason}).",
                provider="gemini",
            )

        cand = candidates[0]
        content = cand.get("content", {})
        parts = content.get("parts", [])
        text = "".join(p.get("text", "") for p in parts)
        finish_reason = cand.get("finishReason")

        # Parse usage metadata
        usage_data = data.get("usageMetadata", {})
        prompt_tokens = usage_data.get("promptTokenCount", 0)
        completion_tokens = usage_data.get("candidatesTokenCount", 0)
        total_tokens = usage_data.get("totalTokenCount", prompt_tokens + completion_tokens)

        return LLMResponse(
            text=text,
            model=self._model,
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=data,
        )


def get_llm_client(
    provider: str = "auto",
    api_key: str | None = None,
    model: str = "gemini-1.5-flash",
    **kwargs: Any,
) -> BaseLLMClient:
    """Factory creating an LLM client instance based on configuration and available credentials.

    Args:
        provider: Provider identifier ('auto', 'gemini', 'mock').
        api_key: Optional explicit API key.
        model: Target model name.
        **kwargs: Provider-specific options (e.g. rate_limiter, mock_fallback).
    """
    provider_normalized = provider.lower().strip()

    if provider_normalized == "mock":
        mock_kwargs = dict(kwargs)
        resolved_model = mock_kwargs.pop("model_name", model)
        return MockLLMClient(model_name=resolved_model, **mock_kwargs)

    if provider_normalized == "gemini":
        return GeminiLLMClient(api_key=api_key, model=model, **kwargs)

    if provider_normalized == "auto":
        resolved_key = api_key or os.environ.get("GEMINI_API_KEY")
        if resolved_key:
            return GeminiLLMClient(api_key=resolved_key, model=model, **kwargs)
        logger.info(
            "No GEMINI_API_KEY detected; get_llm_client(provider='auto') returning MockLLMClient."
        )
        return MockLLMClient(model_name=f"mock-{model}", **kwargs)

    raise ValueError(f"Unknown LLM provider '{provider}'. Must be 'auto', 'gemini', or 'mock'.")
