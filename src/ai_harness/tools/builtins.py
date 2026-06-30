"""Built-in tools.

The calculator uses an AST allow-list — NOT ``eval`` — so a tool argument can
never execute arbitrary code (OWASP A03: Injection). This is exactly the kind of
constraint a harness enforces so the model can't hurt you even if it's tricked.
"""

from __future__ import annotations

import ast
import operator as op
from datetime import UTC, datetime

from ai_harness.tools.registry import Tool, ToolRegistry

_ALLOWED_OPS: dict[type, object] = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.FloorDiv: op.floordiv,
    ast.Mod: op.mod,
    ast.Pow: op.pow,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))  # type: ignore[operator]
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))  # type: ignore[operator]
    raise ValueError("unsupported or unsafe expression")


def calculator(expr: str) -> str:
    expr = expr.strip()
    if not expr:
        return "Error: empty expression"
    tree = ast.parse(expr, mode="eval")
    result = _eval_node(tree.body)
    if isinstance(result, float) and result.is_integer():
        result = int(result)
    return str(result)


def datetime_now(_: str = "") -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")


def word_count(text: str) -> str:
    return str(len(text.split()))


def build_default_registry(retriever: object | None = None) -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(
        Tool("calculator", "Evaluate an arithmetic expression, e.g. '12 * (3 + 4)'.", calculator)
    )
    registry.register(Tool("datetime", "Get the current UTC date and time.", datetime_now))
    registry.register(Tool("word_count", "Count the words in the provided text.", word_count))

    if retriever is not None:

        def knowledge_search(query: str) -> str:
            chunks = retriever.retrieve(query, k=2)  # type: ignore[attr-defined]
            if not chunks:
                return "No relevant documents found."
            return " | ".join(f"[{c.source_id}] {c.text[:160]}" for c in chunks)

        registry.register(
            Tool(
                "knowledge_search",
                "Search the internal knowledge base for relevant context.",
                knowledge_search,
            )
        )
    return registry
