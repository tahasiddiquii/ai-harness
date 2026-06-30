# AGENTS.md

Operating rules for AI agents (and humans) contributing to **ai-harness**. Keep
the bar high and the surface small.

## What this repo is
A production-shaped **AI harness**: the decision layer around an LLM — input
guardrails, intent classification, adaptive model routing, hybrid retrieval, a
tool-using agent loop, output validation, and end-to-end observability. The
harness *is* `src/ai_harness/pipeline.py`; every other module is a stage it calls.

## Golden rules
1. **It must run with zero API keys.** The `mock` provider keeps tests and evals
   deterministic and offline. Never introduce a hard dependency on a network
   service in the default path. Cloud SDKs and Docker services are opt-in extras.
2. **Every new stage is a traced span.** If you add a step to the pipeline, wrap
   it in `trace.span(...)` and record its timing. Observability is not optional.
3. **Guardrails fail closed.** Anything that could let an adversarial prompt or
   PII through must be covered by a test in `tests/test_guardrails.py` and an
   example in `evals/dataset.jsonl`.
4. **No `eval`/`exec` on model or user input.** The calculator uses an AST
   allow-list; keep that pattern for any new tool.
5. **Numbers must be real.** Anything quoted in the README or docs must be
   reproducible by `make eval`. Do not hand-write metrics.

## Definition of done
- `make lint` clean (ruff).
- `make test` green (offline).
- `make eval` passes the quality gate (`evals/run_evals.py` exits 0).
- New behaviour has both a unit test and, if user-visible, an eval example.
- Public types live in `schemas.py`; configuration goes through `config.py`
  (env-driven, never hard-coded).

## Layout
- `src/ai_harness/stages/` — one file per pipeline stage.
- `src/ai_harness/providers/` — LLM provider abstraction (`mock`/`openai`/`anthropic`).
- `src/ai_harness/tools/` — the tool registry and built-ins.
- `evals/` — golden dataset + harness + committed example report.
- `docs/` — architecture and trace-capture notes.

## Conventions
- Python 3.11+, type-hinted, `from __future__ import annotations`.
- Small, well-named functions. Comments explain *why*, not *what*.
- Prefer extending an existing stage over adding a new dependency.
