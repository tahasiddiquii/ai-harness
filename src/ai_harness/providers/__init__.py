"""LLM provider abstraction.

A tiny, synchronous interface over chat-completion backends. The `mock` provider
makes the whole system runnable offline and deterministic for tests/CI; real
providers (OpenAI, Anthropic) are imported lazily only when selected.
"""

from __future__ import annotations

from ai_harness.providers.base import LLMMessage, LLMProvider, LLMResult
from ai_harness.providers.registry import build_provider, get_provider

__all__ = ["LLMMessage", "LLMProvider", "LLMResult", "build_provider", "get_provider"]
