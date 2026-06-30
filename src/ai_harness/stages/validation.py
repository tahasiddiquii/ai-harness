"""Response validation, citation & confidence scoring (Response & Quality layer)."""

from __future__ import annotations

import re

from ai_harness.schemas import Citation, Classification, GuardrailReport, RetrievedChunk

_STOP = {
    "the",
    "a",
    "an",
    "is",
    "are",
    "of",
    "to",
    "and",
    "in",
    "on",
    "for",
    "with",
    "that",
    "this",
    "it",
    "as",
    "by",
    "be",
    "based",
    "knowledge",
    "result",
    "tool",
}


def _content_tokens(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def _snippet(text: str, limit: int = 160) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0] + "…"


def build_citations(
    answer: str, chunks: list[RetrievedChunk], max_citations: int = 3
) -> list[Citation]:
    answer_tokens = _content_tokens(answer)
    if not answer_tokens:
        return []
    scored: list[tuple[float, RetrievedChunk]] = []
    for chunk in chunks:
        chunk_tokens = _content_tokens(chunk.text)
        if not chunk_tokens:
            continue
        overlap = len(answer_tokens & chunk_tokens) / len(chunk_tokens)
        if overlap > 0:
            scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [
        Citation(source_id=chunk.source_id, snippet=_snippet(chunk.text), score=round(score, 3))
        for score, chunk in scored[:max_citations]
    ]


def confidence_score(
    classification: Classification,
    citations: list[Citation],
    chunks: list[RetrievedChunk],
    guardrails: GuardrailReport,
) -> float:
    if guardrails.blocked:
        return 0.0
    grounded = min(1.0, sum(c.score for c in citations)) if citations else (0.3 if chunks else 0.55)
    confidence = 0.5 * classification.confidence + 0.5 * grounded
    return round(min(1.0, confidence), 3)


def validate_output(answer: str | None, max_chars: int = 6000) -> tuple[bool, str]:
    if not answer or not answer.strip():
        return False, "empty_answer"
    if len(answer) > max_chars:
        return False, "answer_too_long"
    return True, "ok"
