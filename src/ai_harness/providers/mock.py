"""Deterministic, offline mock provider.

This is what makes ai-harness runnable with zero keys and what gives the test
suite and eval harness *reproducible* numbers (no network, no spend). It honours
two lightweight protocol markers the harness puts in the system prompt:

* ``ACTION_PROTOCOL`` — the agent loop expects a JSON tool action or final answer.
* ``JUDGE_PROTOCOL``  — the eval harness expects an LLM-as-a-judge JSON score.

Real models follow the same natural-language instructions; the markers are inert
to them. Swapping ``DEFAULT_PROVIDER=openai`` changes nothing about the harness.
"""

from __future__ import annotations

import json
import re

from ai_harness.providers.base import LLMMessage, LLMResult

_MATH_DETECT = re.compile(r"\d\s*[-+*/%]\s*\(?\s*-?\d")
_DATE_WORDS = (
    "what time",
    "current time",
    "today's date",
    "what day",
    "what is the date",
    "current date",
)


def _extract_math(text: str) -> str | None:
    """Pull the longest arithmetic sub-expression (parentheses included)."""
    candidates = re.findall(r"[-+*/%()\d.\s]+", text)
    exprs = [c.strip() for c in candidates if re.search(r"\d", c) and re.search(r"[-+*/%]", c)]
    return max(exprs, key=len).strip() if exprs else None


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
    "what",
    "how",
    "why",
    "do",
    "does",
    "this",
    "that",
    "it",
    "as",
    "by",
    "be",
}


class MockProvider:
    name = "mock"

    def complete(
        self,
        messages: list[LLMMessage],
        *,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 1024,
    ) -> LLMResult:
        system = " ".join(m.content for m in messages if m.role == "system")
        full = "\n".join(m.content for m in messages)
        user = next((m.content for m in reversed(messages) if m.role == "user"), "")

        if "JUDGE_PROTOCOL" in system:
            text = self._judge(full)
        elif "ACTION_PROTOCOL" in system:
            text = self._action(messages, user, full)
        else:
            text = self._plain_answer(user, full)

        prompt_tokens = max(1, len(full.split()))
        completion_tokens = max(1, len(text.split()))
        return LLMResult(
            text=text, prompt_tokens=prompt_tokens, completion_tokens=completion_tokens, model=model
        )

    # -- agent loop ------------------------------------------------------- #
    def _action(self, messages: list[LLMMessage], user: str, full: str) -> str:
        observation = self._last_observation(messages)
        if observation is not None:
            return json.dumps(
                {"action": "final", "answer": f"Based on the tool result, {observation}."}
            )

        if _MATH_DETECT.search(user):
            expr = _extract_math(user)
            if expr:
                return json.dumps({"action": "tool", "tool": "calculator", "input": expr})

        low = user.lower()
        if any(w in low for w in _DATE_WORDS):
            return json.dumps({"action": "tool", "tool": "datetime", "input": ""})

        snippet = self._top_context(full)
        if snippet:
            grounded = " ".join(snippet.split()[:18])
            return json.dumps(
                {"action": "final", "answer": f"Based on the knowledge base, {grounded}"}
            )

        return json.dumps({"action": "final", "answer": self._plain_answer(user, full)})

    # -- helpers ---------------------------------------------------------- #
    @staticmethod
    def _last_observation(messages: list[LLMMessage]) -> str | None:
        for m in reversed(messages):
            if m.role == "tool":
                _, _, rest = m.content.partition(":")
                return (rest or m.content).strip()
        return None

    @staticmethod
    def _top_context(full: str) -> str | None:
        m = re.search(r"\[[^\]]+\]\s*(.+)", full)
        return m.group(1).strip() if m else None

    @staticmethod
    def _plain_answer(user: str, full: str) -> str:
        snippet = MockProvider._top_context(full)
        if snippet:
            return f"Based on the knowledge base, {' '.join(snippet.split()[:18])}"
        return f"Here is a concise answer to: {user.strip()[:120]}"

    # -- llm-as-a-judge --------------------------------------------------- #
    def _judge(self, full: str) -> str:
        answer = self._section(full, "ANSWER:")
        reference = self._section(full, "REFERENCE:")
        score = self._overlap(answer, reference)
        return json.dumps(
            {"score": round(score, 2), "reasoning": f"lexical overlap with reference = {score:.2f}"}
        )

    @staticmethod
    def _section(text: str, label: str) -> str:
        idx = text.find(label)
        if idx == -1:
            return ""
        rest = text[idx + len(label) :]
        nxt = re.search(r"\n[A-Z]{3,}:", rest)
        return rest[: nxt.start()].strip() if nxt else rest.strip()

    @staticmethod
    def _overlap(answer: str, reference: str) -> float:
        a = {w for w in re.findall(r"[a-z0-9]+", answer.lower()) if w not in _STOP}
        r = {w for w in re.findall(r"[a-z0-9]+", reference.lower()) if w not in _STOP}
        if not r:
            return 1.0 if a else 0.0
        return min(1.0, len(a & r) / len(r))
