"""The harness pipeline — the decision layer that wraps the model.

A single request flows through explicit, individually-traced stages:

    guardrails(in) -> classify -> route -> retrieve -> agent(+tools) -> validate

Each stage reads and writes the typed `HarnessContext`. Every stage is a Langfuse
span with cost/latency metering. This module *is* the harness.
"""

from __future__ import annotations

from time import perf_counter

from ai_harness.config import Settings, get_settings
from ai_harness.observability import Tracer
from ai_harness.providers import get_provider
from ai_harness.schemas import (
    ChatRequest,
    ChatResponse,
    Classification,
    Complexity,
    FeedbackRequest,
    HarnessContext,
    Intent,
    RetrievedChunk,
    RouteDecision,
)
from ai_harness.stages import (
    ConversationMemory,
    HybridRetriever,
    LongTermMemory,
    build_citations,
    classify,
    confidence_score,
    route,
    scan_input,
    scan_output,
    validate_output,
)
from ai_harness.stages.agent import run_agent
from ai_harness.stages.router import cost_for
from ai_harness.tools.builtins import build_default_registry

REFUSAL = "I can't help with that — the request appears to attempt to bypass safety policy."


def _context_block(chunks: list[RetrievedChunk]) -> str:
    if not chunks:
        return ""
    lines = "\n".join(f"[{c.source_id}] {c.text}" for c in chunks)
    return f"Context:\n{lines}"


def _system_prompt(tools: str, chunks: list[RetrievedChunk]) -> str:
    return (
        "You are the execution agent inside an AI harness. ACTION_PROTOCOL\n"
        "Answer the user's request, optionally using tools.\n\n"
        f"Available tools:\n{tools}\n\n"
        "Respond with exactly ONE JSON object and nothing else:\n"
        '  {"action":"tool","tool":"<name>","input":"<argument>"}   to call a tool\n'
        '  {"action":"final","answer":"<your answer>"}               to finish\n'
        f"\n{_context_block(chunks)}"
    )


class Harness:
    """Builds the stages once and runs requests through the pipeline."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.tracer = Tracer(self.settings)
        self.retriever = HybridRetriever(self.settings)
        self.registry = build_default_registry(self.retriever)
        self.memory = ConversationMemory()
        self.long_term = LongTermMemory()

    def run(self, request: ChatRequest) -> ChatResponse:
        started = perf_counter()
        ctx = HarnessContext(request=request, query=request.query)

        with self.tracer.trace(
            "chat", user_id=request.user_id, session_id=request.session_id, input=request.query
        ) as trace:
            ctx.trace_id = trace.id
            timings = ctx.stage_timings

            # 1. Entry & Protection — input guardrails
            with trace.span("guardrails.input", input=request.query, timings=timings) as span:
                report, working = scan_input(request.query, self.settings)
                ctx.guardrails = report
                ctx.query = working
                span.update(output=report.model_dump())

            if report.blocked:
                ctx.answer = REFUSAL
                ctx.classification = _unsafe_classification()
                ctx.route = _blocked_route()
                trace.score("confidence", 0.0)
                return self._finalize(ctx, started)

            # 2. Entry & Protection — intent detection & classification
            with trace.span("classify", input=ctx.query, timings=timings) as span:
                ctx.classification = classify(ctx.query)
                span.update(output=ctx.classification.model_dump())

            # 3. Entry & Protection — adaptive model routing
            with trace.span(
                "route", input=ctx.classification.model_dump(), timings=timings
            ) as span:
                ctx.route = route(ctx.classification, self.settings, request.policy)
                span.update(output=ctx.route.model_dump())

            # 4. Context Orchestration — hybrid retrieval
            if ctx.classification.needs_retrieval:
                with trace.span("retrieve", input=ctx.query, timings=timings) as span:
                    ctx.chunks = self.retriever.retrieve(ctx.query, k=3)
                    span.update(output=[c.model_dump() for c in ctx.chunks])

            # 5. Tool & Agent Orchestration — ReAct loop
            provider = get_provider(ctx.route.provider, self.settings)
            system_prompt = _system_prompt(self.registry.specs(), ctx.chunks)
            with trace.span("agent", input=ctx.query, timings=timings) as span:
                result = run_agent(
                    provider,
                    ctx.route.model,
                    self.registry,
                    system_prompt,
                    ctx.query,
                    self.memory.history(request.session_id),
                    self.settings.max_agent_steps,
                )
                ctx.answer = result["answer"]
                ctx.tools_used = result["tools_used"]
                ctx.add_usage(
                    result["usage"]["prompt_tokens"], result["usage"]["completion_tokens"]
                )
                span.update(output={"answer": ctx.answer, "tools_used": ctx.tools_used})

            # 6. Response & Quality — validation, output guardrails, citation, confidence
            with trace.span("validate", input=ctx.answer, timings=timings) as span:
                ok, _reason = validate_output(ctx.answer)
                if not ok:
                    ctx.answer = "I wasn't able to produce a valid answer."
                if scan_output(ctx.answer or ""):
                    ctx.guardrails.output_pii_leaked = True
                    _out_report, redacted = scan_input(ctx.answer or "", self.settings)
                    ctx.answer = redacted
                ctx.citations = build_citations(ctx.answer or "", ctx.chunks)
                ctx.confidence = confidence_score(
                    ctx.classification, ctx.citations, ctx.chunks, ctx.guardrails
                )
                span.update(
                    output={
                        "valid": ok,
                        "confidence": ctx.confidence,
                        "citations": [c.model_dump() for c in ctx.citations],
                    }
                )

            ctx.cost_usd = cost_for(
                ctx.route.model, ctx.usage["prompt_tokens"] + ctx.usage["completion_tokens"]
            )
            self.memory.append(request.session_id, "user", request.query)
            self.memory.append(request.session_id, "assistant", ctx.answer or "")
            self.long_term.remember(request.query, ctx.answer or "")
            trace.score("confidence", ctx.confidence)

        return self._finalize(ctx, started)

    def record_feedback(self, feedback: FeedbackRequest) -> None:
        self.tracer.score_trace(
            feedback.trace_id, "user_feedback", feedback.score, feedback.comment
        )

    # ------------------------------------------------------------------ #
    def _finalize(self, ctx: HarnessContext, started: float) -> ChatResponse:
        latency_ms = round((perf_counter() - started) * 1000, 2)
        assert ctx.classification is not None and ctx.route is not None
        return ChatResponse(
            answer=ctx.answer or "",
            session_id=ctx.request.session_id,
            classification=ctx.classification,
            route=ctx.route,
            guardrails=ctx.guardrails,
            citations=ctx.citations,
            confidence=ctx.confidence,
            tools_used=ctx.tools_used,
            usage=ctx.usage,
            cost_usd=ctx.cost_usd,
            latency_ms=latency_ms,
            trace_id=ctx.trace_id,
            stage_timings=ctx.stage_timings,
        )


def _unsafe_classification() -> Classification:
    return Classification(
        intent=Intent.UNSAFE,
        complexity=Complexity.SIMPLE,
        needs_tools=False,
        needs_retrieval=False,
        confidence=0.95,
    )


def _blocked_route() -> RouteDecision:
    return RouteDecision(
        provider="none",
        model="blocked",
        tier="none",
        reason="blocked by input guardrails before routing",
        est_cost_per_1k_usd=0.0,
        est_latency_ms=0,
    )
