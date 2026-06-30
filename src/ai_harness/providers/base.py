"""Provider interface and shared value objects."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class LLMMessage:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str


@dataclass(slots=True)
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    model: str

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


class LLMProvider(ABC):
    """Synchronous chat-completion backend."""

    name: str = "base"

    @abstractmethod
    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult:
        """Return a completion for `messages`."""
        raise NotImplementedError
