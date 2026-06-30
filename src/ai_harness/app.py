"""FastAPI surface for the harness."""

from __future__ import annotations

from fastapi import FastAPI

from ai_harness import __version__
from ai_harness.config import get_settings
from ai_harness.pipeline import Harness
from ai_harness.schemas import ChatRequest, ChatResponse, FeedbackRequest

app = FastAPI(
    title="ai-harness",
    version=__version__,
    description="A production AI harness: routing, retrieval, agents, guardrails, and observability.",
)

_harness: Harness | None = None


def get_harness() -> Harness:
    global _harness
    if _harness is None:
        _harness = Harness(get_settings())
    return _harness


@app.get("/")
def root() -> dict[str, object]:
    settings = get_settings()
    return {
        "name": "ai-harness",
        "version": __version__,
        "default_provider": settings.default_provider,
        "router_policy": settings.router_policy,
        "langfuse": get_harness().tracer.langfuse_active,
        "endpoints": ["GET /healthz", "POST /v1/chat", "POST /v1/feedback", "GET /docs"],
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@app.post("/v1/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return get_harness().run(request)


@app.post("/v1/feedback")
def feedback(request: FeedbackRequest) -> dict[str, str]:
    get_harness().record_feedback(request)
    return {"status": "recorded", "trace_id": request.trace_id}
