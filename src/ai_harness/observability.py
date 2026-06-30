"""Observability (cross-cutting).

A tracer that always records local stage timings + a trace id, and *best-effort*
mirrors every span into Langfuse when keys are configured. The Langfuse calls are
fully guarded and feature-detected, so:

* offline / no keys  -> pure local timings, zero network, deterministic tests;
* keys configured     -> the same spans show up in the Langfuse UI.

Observability is wired into the pipeline itself, not bolted on afterwards.
"""

from __future__ import annotations

import uuid
from collections.abc import Iterator
from contextlib import contextmanager, suppress
from time import perf_counter
from typing import Any

from ai_harness.config import Settings
from ai_harness.schemas import StageTiming


def _init_langfuse(settings: Settings) -> Any | None:
    try:
        from langfuse import Langfuse

        return Langfuse(
            public_key=settings.langfuse_public_key,
            secret_key=settings.langfuse_secret_key,
            host=settings.langfuse_host,
        )
    except Exception:
        return None


def _safe(obj: Any, method: str, /, **kwargs: Any) -> Any:
    fn = getattr(obj, method, None)
    if fn is None:
        return None
    try:
        return fn(**kwargs)
    except Exception:
        return None


class StageSpan:
    def __init__(self, name: str, lf_span: Any | None, timings: list[StageTiming] | None) -> None:
        self.name = name
        self._lf = lf_span
        self._timings = timings
        self._t0 = perf_counter()
        self.duration_ms = 0.0

    def update(self, **kwargs: Any) -> None:
        if self._lf is not None:
            _safe(self._lf, "update", **kwargs)

    def end(self) -> None:
        self.duration_ms = round((perf_counter() - self._t0) * 1000, 2)
        if self._timings is not None:
            self._timings.append(StageTiming(name=self.name, duration_ms=self.duration_ms))


class TraceHandle:
    def __init__(self, trace_id: str, client: Any | None) -> None:
        self.id = trace_id
        self._client = client

    @contextmanager
    def span(
        self, name: str, *, input: Any = None, timings: list[StageTiming] | None = None
    ) -> Iterator[StageSpan]:
        lf_span = None
        cm = None
        if self._client is not None:
            cm = _safe(self._client, "start_as_current_span", name=name, input=input)
            if cm is not None:
                try:
                    lf_span = cm.__enter__()
                except Exception:
                    cm = None
                    lf_span = None
        span = StageSpan(name, lf_span, timings)
        try:
            yield span
        finally:
            span.end()
            if cm is not None:
                with suppress(Exception):
                    cm.__exit__(None, None, None)

    def score(self, name: str, value: float, comment: str | None = None) -> None:
        if self._client is not None:
            _safe(self._client, "score_current_trace", name=name, value=value, comment=comment)


class Tracer:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = _init_langfuse(settings) if settings.langfuse_enabled else None
        self.feedback_log: list[dict[str, Any]] = []

    @property
    def langfuse_active(self) -> bool:
        return self._client is not None

    @contextmanager
    def trace(
        self,
        name: str,
        *,
        user_id: str | None = None,
        session_id: str | None = None,
        input: Any = None,
    ) -> Iterator[TraceHandle]:
        trace_id = uuid.uuid4().hex
        cm = None
        if self._client is not None:
            cm = _safe(self._client, "start_as_current_span", name=name, input=input)
            if cm is not None:
                try:
                    root = cm.__enter__()
                    _safe(root, "update_trace", user_id=user_id, session_id=session_id, input=input)
                    real_id = _safe(self._client, "get_current_trace_id")
                    if real_id:
                        trace_id = real_id
                except Exception:
                    cm = None
        try:
            yield TraceHandle(trace_id, self._client)
        finally:
            if cm is not None:
                with suppress(Exception):
                    cm.__exit__(None, None, None)
            _safe(self._client, "flush")

    def score_trace(
        self, trace_id: str, name: str, value: float, comment: str | None = None
    ) -> None:
        self.feedback_log.append(
            {"trace_id": trace_id, "name": name, "value": value, "comment": comment}
        )
        if self._client is not None:
            _safe(
                self._client,
                "create_score",
                trace_id=trace_id,
                name=name,
                value=value,
                comment=comment,
            )
            _safe(self._client, "flush")
