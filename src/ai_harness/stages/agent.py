"""Agent loop (Tool & Agent Orchestration layer), built on LangGraph.

A ReAct-style state machine: the agent node asks the routed model for either a
tool call or a final answer; the tools node executes the call and feeds the
observation back. The loop ends on a final answer or when the step budget is hit.
"""

from __future__ import annotations

import json
import re
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from ai_harness.providers.base import LLMMessage, LLMProvider
from ai_harness.tools.registry import ToolRegistry

try:
    from langgraph.errors import GraphRecursionError
except Exception:  # pragma: no cover - depends on langgraph version

    class GraphRecursionError(Exception): ...


_JSON_RE = re.compile(r"\{.*\}", re.S)


class AgentState(TypedDict, total=False):
    messages: list[dict[str, str]]
    steps: int
    tools_used: list[str]
    answer: str
    done: bool
    pending_tool: str
    pending_input: str


def _parse_action(text: str) -> dict[str, str]:
    match = _JSON_RE.search(text)
    if not match:
        return {"action": "final", "answer": text.strip()}
    try:
        data = json.loads(match.group(0))
    except Exception:
        return {"action": "final", "answer": text.strip()}
    if data.get("action") == "tool" and data.get("tool"):
        return {"action": "tool", "tool": str(data["tool"]), "input": str(data.get("input", ""))}
    return {"action": "final", "answer": str(data.get("answer", text.strip()))}


def run_agent(
    provider: LLMProvider,
    model: str,
    registry: ToolRegistry,
    system_prompt: str,
    query: str,
    history: list[tuple[str, str]],
    max_steps: int,
) -> dict[str, Any]:
    usage = {"prompt_tokens": 0, "completion_tokens": 0}

    def agent_node(state: AgentState) -> dict[str, Any]:
        messages = [LLMMessage(**m) for m in state["messages"]]
        result = provider.complete(messages, model=model, temperature=0.1, max_tokens=800)
        usage["prompt_tokens"] += result.prompt_tokens
        usage["completion_tokens"] += result.completion_tokens

        steps = state.get("steps", 0) + 1
        action = _parse_action(result.text)
        new_messages = [*state["messages"], {"role": "assistant", "content": result.text}]

        if action["action"] == "final" or steps >= max_steps:
            answer = action.get("answer") or "I could not complete the request."
            return {"messages": new_messages, "steps": steps, "done": True, "answer": answer}
        return {
            "messages": new_messages,
            "steps": steps,
            "done": False,
            "pending_tool": action["tool"],
            "pending_input": action["input"],
        }

    def tools_node(state: AgentState) -> dict[str, Any]:
        tool = state.get("pending_tool", "")
        arg = state.get("pending_input", "")
        observation = registry.run(tool, arg)
        return {
            "messages": [
                *state["messages"],
                {"role": "tool", "content": f"Observation from {tool}: {observation}"},
            ],
            "tools_used": [*state.get("tools_used", []), tool],
        }

    def should_continue(state: AgentState) -> str:
        return "end" if state.get("done") else "tools"

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", tools_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    graph.add_edge("tools", "agent")
    app = graph.compile()

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for role, content in history:
        messages.append({"role": role, "content": content})
    messages.append({"role": "user", "content": query})

    initial: AgentState = {"messages": messages, "steps": 0, "tools_used": [], "done": False}
    try:
        final = app.invoke(initial, config={"recursion_limit": max_steps * 2 + 4})
    except GraphRecursionError:
        final = {"answer": "I stopped after reaching the step limit.", "tools_used": []}

    return {
        "answer": final.get("answer", ""),
        "tools_used": final.get("tools_used", []),
        "usage": usage,
    }
