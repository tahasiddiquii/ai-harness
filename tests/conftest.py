"""Test fixtures.

Clears provider/observability env vars so the suite is hermetic regardless of the
developer's shell (e.g. a real OPENAI_API_KEY in the environment must not leak in
and turn the deterministic mock runs into live API calls).
"""

from __future__ import annotations

import pytest

_VARS = (
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEFAULT_PROVIDER",
    "ROUTER_POLICY",
    "LANGFUSE_PUBLIC_KEY",
    "LANGFUSE_SECRET_KEY",
    "LANGFUSE_HOST",
    "ENABLE_PII_REDACTION",
    "BLOCK_ON_INJECTION",
    "MAX_AGENT_STEPS",
)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in _VARS:
        monkeypatch.delenv(var, raising=False)
