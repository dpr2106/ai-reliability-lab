"""Domain models and data structures for provider-independent LLM interactions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

MessageRole = Literal["system", "user", "assistant"]


@dataclass(frozen=True)
class LLMMessage:
    """A single turn in an LLM conversation or generation prompt."""

    role: MessageRole
    content: str

    def __post_init__(self) -> None:
        if self.role not in {"system", "user", "assistant"}:
            raise ValueError(
                f"Invalid message role '{self.role}'. Must be 'system', 'user', or 'assistant'."
            )
        if not isinstance(self.content, str):
            raise TypeError("Message content must be a string.")


@dataclass(frozen=True)
class TokenUsage:
    """Token consumption statistics for an LLM generation call."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMGenerationConfig:
    """Tunable generation hyperparameters and constraints."""

    temperature: float = 0.0
    max_output_tokens: int = 1024
    top_p: float | None = None
    top_k: int | None = None
    stop_sequences: list[str] | None = None
    response_mime_type: str = "text/plain"

    def __post_init__(self) -> None:
        if self.temperature < 0.0 or self.temperature > 2.0:
            raise ValueError(f"Temperature must be in [0.0, 2.0], got {self.temperature}")
        if self.max_output_tokens <= 0:
            raise ValueError(f"max_output_tokens must be positive, got {self.max_output_tokens}")


@dataclass
class LLMResponse:
    """Standardized response container for generation results."""

    text: str
    model: str
    usage: TokenUsage | None = None
    finish_reason: str | None = None
    latency_ms: float = 0.0
    raw_response: dict[str, Any] | None = field(default=None, repr=False)
