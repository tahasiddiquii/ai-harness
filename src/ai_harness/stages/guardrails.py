"""Guardrails (Safety & Guardrails — Entry & Protection layer).

Input scanning: PII detection + redaction and prompt-injection / jailbreak
heuristics. Output scanning: catch PII that leaked into a response. This is a
lightweight, dependency-free implementation; the dedicated `llm-guardrails-redteam`
repo goes deeper (Presidio, classifier-based injection detection, a red-team suite).
"""

from __future__ import annotations

import re

from ai_harness.config import Settings
from ai_harness.schemas import GuardrailReport

_PII_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b"),
    "phone": re.compile(r"(?<!\d)(?:\+?\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]\d{3}[\s-]\d{4}(?!\d)"),
    "ip": re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
}

_INJECTION: list[tuple[re.Pattern[str], float]] = [
    (
        re.compile(r"ignore (all |the |your )?(previous|prior|above) (instructions|prompt)", re.I),
        0.7,
    ),
    (re.compile(r"disregard (the |your |all )?(system|previous|above|prior)", re.I), 0.6),
    (re.compile(r"reveal (your |the )?(system prompt|instructions|prompt)", re.I), 0.7),
    (re.compile(r"print (your|the) (system )?(prompt|instructions)", re.I), 0.7),
    (re.compile(r"you are now (a |an )?(dan|developer mode|unrestricted|jailbroken)", re.I), 0.8),
    (re.compile(r"\b(jailbreak|do anything now)\b", re.I), 0.6),
    (
        re.compile(
            r"(exfiltrate|leak|dump) (all )?(the )?(data|secrets|keys|api ?keys|credentials)", re.I
        ),
        0.7,
    ),
]


def scan_input(query: str, settings: Settings) -> tuple[GuardrailReport, str]:
    """Return a guardrail report and the (optionally PII-redacted) working query."""
    report = GuardrailReport()
    redacted = query
    types: list[str] = []

    for name, pattern in _PII_PATTERNS.items():
        if pattern.search(redacted):
            types.append(name)
            if settings.enable_pii_redaction:
                redacted = pattern.sub(f"[REDACTED_{name.upper()}]", redacted)

    if types:
        report.pii_detected = True
        report.pii_types = types

    score = 0.0
    for pattern, weight in _INJECTION:
        if pattern.search(query):
            score = max(score, weight)
    report.injection_score = round(score, 2)
    report.injection_detected = score > 0.0

    if settings.block_on_injection and score >= settings.injection_block_threshold:
        report.blocked = True
        report.block_reason = "prompt_injection_detected"

    working = redacted if settings.enable_pii_redaction else query
    return report, working


def scan_output(answer: str) -> bool:
    """True if any PII pattern is present in the model's answer (a leak)."""
    return any(pattern.search(answer) for pattern in _PII_PATTERNS.values())
