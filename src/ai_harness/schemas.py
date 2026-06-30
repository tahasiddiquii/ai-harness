"""Pydantic models — the public API surface and the internal harness context.

`HarnessContext` is the typed object that flows through every stage of the
pipeline; each stage reads from it and writes its results back. That single
object *is* the harness state.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Intent(str, Enum):
    QA = "qa"
    CODE = "code"
    TOOL_USE = "tool_use"
    CHITCHAT = "chitchat"
    UNSAFE = "unsafe"


class Complexity(str, Enum):
    SIMPLE = "simple"
    MODERATE = "moderate"
    COMPLEX = "complex"


# --------------------------------------------------------------------------- #
# Public request / response
# --------------------------------------------------------------------------- #
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=8000)
    session_id: str = "default"
    user_id: str | None = None
    policy: str | None = Field(
        default=None, description="Override the router policy for this call."
    )


class Classification(BaseModel):
    intent: Intent
    complexity: Complexity
    needs_tools: bool
    needs_retrieval: bool
    confidence: float


class RouteDecision(BaseModel):
    provider: str
    model: str
    tier: str
    reason: str
    est_cost_per_1k_usd: float
    est_latency_ms: int


class GuardrailReport(BaseModel):
    pii_detected: bool = False
    pii_types: list[str] = Field(default_factory=list)
    injection_detected: bool = False
    injection_score: float = 0.0
    blocked: bool = False
    block_reason: str | None = None
    output_pii_leaked: bool = False


class Citation(BaseModel):
    source_id: str
    snippet: str
    score: float


class StageTiming(BaseModel):
    name: str
    duration_ms: float


class ChatResponse(BaseModel):
    answer: str
    session_id: str
    classification: Classification
    route: RouteDecision
    guardrails: GuardrailReport
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(default_factory=dict)
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    trace_id: str | None = None
    stage_timings: list[StageTiming] = Field(default_factory=list)


class FeedbackRequest(BaseModel):
    trace_id: str
    score: float = Field(
        ..., ge=-1.0, le=1.0, description="Thumbs: -1, 0, or 1 (or any value in range)."
    )
    comment: str | None = None
    correction: str | None = None


# --------------------------------------------------------------------------- #
# Internal context
# --------------------------------------------------------------------------- #
class RetrievedChunk(BaseModel):
    source_id: str
    text: str
    score: float
    retriever: str = "hybrid"


class HarnessContext(BaseModel):
    """Mutable state passed through every pipeline stage."""

    request: ChatRequest
    query: str  # working query (may be PII-redacted)
    classification: Classification | None = None
    route: RouteDecision | None = None
    guardrails: GuardrailReport = Field(default_factory=GuardrailReport)
    chunks: list[RetrievedChunk] = Field(default_factory=list)
    answer: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    confidence: float = 0.0
    tools_used: list[str] = Field(default_factory=list)
    usage: dict[str, int] = Field(
        default_factory=lambda: {"prompt_tokens": 0, "completion_tokens": 0}
    )
    cost_usd: float = 0.0
    stage_timings: list[StageTiming] = Field(default_factory=list)
    trace_id: str | None = None

    def add_usage(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.usage["prompt_tokens"] = self.usage.get("prompt_tokens", 0) + prompt_tokens
        self.usage["completion_tokens"] = self.usage.get("completion_tokens", 0) + completion_tokens
