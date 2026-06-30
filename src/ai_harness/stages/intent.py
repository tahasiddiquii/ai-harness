"""Intent detection & query classification (Entry & Protection layer).

A cheap, fast, deterministic classifier. In production this is exactly where you
want a *small* model or heuristic: it runs on every request and drives routing,
retrieval, and tool selection, so it must be near-free. The output decides which
(more expensive) model the router will reach for.
"""

from __future__ import annotations

import re

from ai_harness.schemas import Classification, Complexity, Intent

_CODE = (
    "```",
    "def ",
    "function",
    "python",
    "javascript",
    "typescript",
    "sql",
    "write code",
    "regex",
    "stack trace",
    "traceback",
    "compile",
    "bug in",
    "exception",
    "refactor",
)
_TOOL = (
    "calculate",
    "what time",
    "current time",
    "current date",
    "today's date",
    "how much is",
    "what day",
)
_GREET = ("hi", "hello", "hey", "thanks", "thank you", "good morning", "how are you")
_REASON = (
    "explain",
    "compare",
    "why",
    "step by step",
    "trade-off",
    "tradeoff",
    "design",
    "architecture",
    "analyze",
    "pros and cons",
    "evaluate",
    "difference between",
)
_MATH = re.compile(r"-?\d+(?:\.\d+)?\s*[-+*/%]\s*-?\d+")


def classify(query: str) -> Classification:
    q = query.strip()
    low = q.lower()
    words = low.split()
    n = len(words)

    intent = Intent.QA
    if _MATH.search(q) or any(t in low for t in _TOOL):
        intent = Intent.TOOL_USE
    elif any(c in low for c in _CODE):
        intent = Intent.CODE
    elif n <= 4 and any(low.startswith(g) for g in _GREET):
        intent = Intent.CHITCHAT

    reasons = sum(1 for r in _REASON if r in low)
    questions = low.count("?")
    if intent == Intent.CHITCHAT:
        complexity = Complexity.SIMPLE
    elif n > 40 or reasons >= 2 or questions >= 2:
        complexity = Complexity.COMPLEX
    elif n > 14 or reasons >= 1:
        complexity = Complexity.MODERATE
    else:
        complexity = Complexity.SIMPLE

    needs_tools = intent == Intent.TOOL_USE
    needs_retrieval = intent in (Intent.QA, Intent.CODE)
    confidence = 0.9 if intent in (Intent.TOOL_USE, Intent.CHITCHAT) else 0.72

    return Classification(
        intent=intent,
        complexity=complexity,
        needs_tools=needs_tools,
        needs_retrieval=needs_retrieval,
        confidence=confidence,
    )
