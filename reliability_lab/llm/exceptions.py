"""Exception hierarchy for LLM clients and provider adapters."""

from __future__ import annotations


class LLMError(Exception):
    """Base exception for all LLM client and provider errors."""

    def __init__(self, message: str, status_code: int | None = None, provider: str = "unknown"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.provider = provider


class LLMAuthenticationError(LLMError):
    """Raised when authentication fails or API key is missing/invalid (HTTP 401/403)."""


class LLMRateLimitError(LLMError):
    """Raised when upstream rate limits or quota thresholds are exhausted (HTTP 429)."""


class LLMProviderUnavailableError(LLMError):
    """Raised when the LLM service is temporarily unreachable, timed out, or returning 5xx."""


class LLMInvalidRequestError(LLMError):
    """Raised when the generation request payload is malformed or invalid (HTTP 400)."""


class LLMContextLengthExceededError(LLMInvalidRequestError):
    """Raised when prompt tokens exceed the provider's supported context window."""

