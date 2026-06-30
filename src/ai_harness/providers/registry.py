"""Provider registry — build and cache providers by name."""

from __future__ import annotations

from ai_harness.config import Settings
from ai_harness.providers.base import LLMProvider
from ai_harness.providers.mock import MockProvider

_CACHE: dict[str, LLMProvider] = {}


def build_provider(name: str, settings: Settings) -> LLMProvider:
    name = (name or settings.default_provider).lower()
    if name == "mock":
        return MockProvider()  # type: ignore[return-value]
    if name == "openai":
        if not settings.openai_api_key:
            raise RuntimeError("OPENAI_API_KEY is not set")
        from ai_harness.providers.openai_provider import OpenAIProvider

        return OpenAIProvider(settings.openai_api_key)  # type: ignore[return-value]
    if name == "anthropic":
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set")
        from ai_harness.providers.anthropic_provider import AnthropicProvider

        return AnthropicProvider(settings.anthropic_api_key)  # type: ignore[return-value]
    raise ValueError(f"Unknown provider: {name!r}")


def get_provider(name: str, settings: Settings) -> LLMProvider:
    key = (name or settings.default_provider).lower()
    if key not in _CACHE:
        _CACHE[key] = build_provider(key, settings)
    return _CACHE[key]


def reset_cache() -> None:
    _CACHE.clear()
