# Architecture

`ai-harness` is the **decision layer** that wraps a raw LLM and turns it into a
reliable, observable agent. A model alone is a token predictor; the harness is
everything around it that makes it safe, grounded, cheap, and debuggable.

> An agent = a model + a harness. This repo is the harness.

## Layered view

```mermaid
flowchart TB
    subgraph entry["1 · Entry & Protection"]
        G1["Input guardrails<br/>PII redaction · injection block"]
        IC["Intent & complexity<br/>classifier"]
        RT["Adaptive model router<br/>cost / latency / quality"]
    end
    subgraph context["2 · Context Orchestration"]
        HR["Hybrid retrieval<br/>BM25 + dense + RRF"]
        MEM["Memory<br/>conversation + long-term"]
    end
    subgraph exec["3 · Tool & Agent Orchestration"]
        AG["ReAct agent loop<br/>LangGraph state machine"]
        TR["Tool registry<br/>AST-safe calculator, KB search…"]
    end
    subgraph quality["4 · Response & Quality"]
        VO["Output validation"]
        G2["Output guardrails<br/>PII leak scan"]
        CT["Citations + confidence"]
    end
    OBS["Observability · Langfuse spans + cost/latency + scores"]

    REQ([request]) --> entry --> context --> exec --> quality --> RESP([response])
    OBS -.traces every stage.- entry
    OBS -.-> context
    OBS -.-> exec
    OBS -.-> quality
```

## Request lifecycle

```mermaid
sequenceDiagram
    autonumber
    participant U as Client
    participant H as Harness
    participant Gd as Guardrails
    participant R as Router
    participant Re as Retriever
    participant A as Agent+Tools
    participant V as Validator
    participant Lf as Langfuse

    U->>H: POST /v1/chat {query}
    H->>Lf: start trace
    H->>Gd: scan_input (PII, injection)
    alt blocked
        Gd-->>H: blocked=true
        H-->>U: refusal (no model call)
    else allowed
        H->>R: classify + route (policy, complexity)
        R-->>H: provider/model + tier
        opt needs retrieval
            H->>Re: hybrid retrieve (BM25+dense, RRF)
            Re-->>H: top-k chunks
        end
        H->>A: ReAct loop with tools + context
        A-->>H: answer + tools_used + usage
        H->>V: validate + output guardrails + citations + confidence
        V-->>H: final answer
    end
    H->>Lf: end trace (cost, latency, confidence score)
    H-->>U: ChatResponse (answer, route, citations, trace_id…)
```

## Stage → code map

| Layer | Stage | Module |
| --- | --- | --- |
| Entry & Protection | Input/output guardrails | [src/ai_harness/stages/guardrails.py](../src/ai_harness/stages/guardrails.py) |
| Entry & Protection | Intent & complexity | [src/ai_harness/stages/intent.py](../src/ai_harness/stages/intent.py) |
| Entry & Protection | Adaptive routing | [src/ai_harness/stages/router.py](../src/ai_harness/stages/router.py) |
| Context Orchestration | Hybrid retrieval | [src/ai_harness/stages/retrieval.py](../src/ai_harness/stages/retrieval.py) |
| Context Orchestration | Memory | [src/ai_harness/stages/memory.py](../src/ai_harness/stages/memory.py) |
| Tool & Agent Orchestration | ReAct loop | [src/ai_harness/stages/agent.py](../src/ai_harness/stages/agent.py) |
| Tool & Agent Orchestration | Tools | [src/ai_harness/tools/](../src/ai_harness/tools/) |
| Response & Quality | Validation, citations, confidence | [src/ai_harness/stages/validation.py](../src/ai_harness/stages/validation.py) |
| Cross-cutting | Tracing + cost/latency + scores | [src/ai_harness/observability.py](../src/ai_harness/observability.py) |
| Orchestrator | The pipeline itself | [src/ai_harness/pipeline.py](../src/ai_harness/pipeline.py) |

## Design decisions

- **Provider abstraction with a deterministic mock.** Every provider implements one
  `complete()` method. The `mock` provider honours the same prompt protocols as a
  real model, so the entire system runs — and is *tested and evaluated* — with zero
  API keys and zero network. Switching to OpenAI/Anthropic changes one env var.
- **Typed state object.** `HarnessContext` flows through every stage; each stage
  reads and writes it. State is explicit, not hidden in closures.
- **Routing is the highest-ROI knob.** Most requests are simple and go to a cheap,
  fast model; only genuinely complex requests earn the expensive tier. Policy
  (`cost` / `balanced` / `quality` / `latency`) is per-request overridable.
- **Hybrid retrieval over single-method.** Sparse BM25 catches exact terms, dense
  vectors catch paraphrase; Reciprocal Rank Fusion merges them without tuning weights.
- **Guardrails fail closed.** Injection over threshold refuses *before* any model
  call (cheapest, safest place). Output is re-scanned for PII leaks.
- **Tools can't execute code.** The calculator evaluates an AST allow-list, never
  `eval`. A tricked model still can't run arbitrary code (OWASP A03).
- **Observability is wired in, not bolted on.** Each stage is a span with cost and
  latency; user feedback attaches as a Langfuse score on the trace.

## Extending it

Add a stage by writing a function in `stages/`, calling it inside
`Harness.run` wrapped in a `trace.span(...)`, and adding a unit test plus an
`evals/dataset.jsonl` example. See [AGENTS.md](../AGENTS.md) for the contribution bar.
