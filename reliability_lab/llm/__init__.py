"""LLM client integration, abstract interfaces, Gemini adapter, and deterministic mocks."""

from __future__ import annotations

from reliability_lab.llm.client import (
    BaseLLMClient,
    GeminiLLMClient,
    MockLLMClient,
    OllamaLLMClient,
    get_llm_client,
)
from reliability_lab.llm.exceptions import (
    LLMAuthenticationError,
    LLMContextLengthExceededError,
    LLMError,
    LLMInvalidRequestError,
    LLMProviderUnavailableError,
    LLMRateLimitError,
)
from reliability_lab.llm.judge import (
    ClaimJudgeResult,
    CorrelationReport,
    JudgeConfig,
    JudgeResult,
    JudgeVerdict,
    LLMJudge,
    calculate_llm_nli_correlation,
)
from reliability_lab.llm.models import (
    LLMGenerationConfig,
    LLMMessage,
    LLMResponse,
    TokenUsage,
)
from reliability_lab.llm.rate_limiter import RateLimiter

__all__ = [
    "BaseLLMClient",
    "ClaimJudgeResult",
    "CorrelationReport",
    "GeminiLLMClient",
    "JudgeConfig",
    "JudgeResult",
    "JudgeVerdict",
    "LLMAuthenticationError",
    "LLMContextLengthExceededError",
    "LLMError",
    "LLMGenerationConfig",
    "LLMInvalidRequestError",
    "LLMJudge",
    "LLMMessage",
    "LLMProviderUnavailableError",
    "LLMRateLimitError",
    "LLMResponse",
    "MockLLMClient",
    "OllamaLLMClient",
    "RateLimiter",
    "TokenUsage",
    "calculate_llm_nli_correlation",
    "get_llm_client",
]
