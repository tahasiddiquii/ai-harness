# Trace screenshots

This folder holds Langfuse trace screenshots that prove the harness is genuinely
observable end-to-end. They are referenced from the top-level [README](../../README.md).

## Capture them yourself (≈2 minutes)

1. Create a free project at [cloud.langfuse.com](https://cloud.langfuse.com) and copy
   the public + secret keys.
2. Export the keys (or put them in `.env`):

   ```bash
   export LANGFUSE_PUBLIC_KEY=pk-lf-...
   export LANGFUSE_SECRET_KEY=sk-lf-...
   export LANGFUSE_HOST=https://cloud.langfuse.com
   # optional: a real model instead of the mock
   export DEFAULT_PROVIDER=openai
   export OPENAI_API_KEY=sk-...
   ```

3. Generate traces:

   ```bash
   make demo        # five representative requests
   make eval        # the full golden dataset + LLM-as-a-judge scores
   ```

4. Open Langfuse and screenshot the views below.

## What to capture

| File | View | Why it matters |
| --- | --- | --- |
| `trace-overview.png` | A single `chat` trace expanded | Shows the stage spans: guardrails → classify → route → retrieve → agent → validate |
| `trace-waterfall.png` | The span waterfall with latencies | Per-stage latency + total cost in one place |
| `routing.png` | Two traces side by side (simple vs complex) | Proves adaptive routing picks different tiers |
| `guardrail-block.png` | A blocked injection trace | Refusal happens before any model call |
| `eval-scores.png` | The scores view after `make eval` | LLM-as-a-judge + confidence scores attached to traces |

Suggested: keep images ≤ 1600px wide and reference them in the README like
`![Langfuse trace](docs/traces/trace-overview.png)`.

> No keys? Everything still runs — you just get local stage timings instead of
> remote traces. The screenshots are proof for reviewers, not a runtime dependency.
