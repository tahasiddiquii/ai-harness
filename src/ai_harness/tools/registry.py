"""A minimal, well-described tool registry.

Ten focused tools beat fifty overlapping ones: each tool's name + description is
stamped into the prompt every turn, so the menu must stay small and legible.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    func: Callable[[str], str]


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> str:
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def run(self, name: str, arg: str) -> str:
        tool = self._tools.get(name)
        if tool is None:
            return f"Error: unknown tool '{name}'."
        try:
            return tool.func(arg)
        except Exception as exc:  # tools must never crash the agent loop
            return f"Error running {name}: {exc}"
