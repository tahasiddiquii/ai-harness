"""ai-harness — a production AI harness (the decision layer around an LLM).

Agent = Model + Harness. This package is the harness: intent detection, adaptive
model routing, context retrieval, tool/agent orchestration, guardrails, structured
output + validation, and end-to-end observability with Langfuse.
"""

from __future__ import annotations

__version__ = "0.1.0"

from ai_harness.config import Settings, get_settings
from ai_harness.pipeline import Harness

__all__ = ["Harness", "Settings", "get_settings", "__version__"]
