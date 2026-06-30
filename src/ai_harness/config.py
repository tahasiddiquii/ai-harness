"""Typed configuration loaded from environment / `.env`.

Defaults are chosen so the whole harness runs offline with the `mock` provider
and no API keys — important for tests, CI, and first-run experience.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", case_sensitive=False, extra="ignore"
    )

    # Providers
    default_provider: str = "mock"  # mock | openai | anthropic
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # Adaptive router
    router_policy: str = "balanced"  # cost | latency | quality | balanced

    # Guardrails
    enable_pii_redaction: bool = True
    block_on_injection: bool = True
    injection_block_threshold: float = 0.5

    # Agent
    max_agent_steps: int = 6

    # Observability (Langfuse)
    langfuse_public_key: str | None = None
    langfuse_secret_key: str | None = None
    langfuse_host: str = "https://cloud.langfuse.com"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(self.langfuse_public_key and self.langfuse_secret_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
