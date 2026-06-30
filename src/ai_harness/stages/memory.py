"""Memory (Tool & Agent Orchestration layer).

Short-term conversation memory (per session, in-process — swap for Redis in
production) and a simple long-term memory with keyword recall (swap for a vector
store). Kept intentionally small; the point is the *interface* the harness uses.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


class ConversationMemory:
    """Per-session rolling window of recent turns."""

    def __init__(self, max_turns: int = 6) -> None:
        self._store: dict[str, list[tuple[str, str]]] = {}
        self._max_messages = max_turns * 2

    def history(self, session_id: str) -> list[tuple[str, str]]:
        return list(self._store.get(session_id, []))

    def append(self, session_id: str, role: str, content: str) -> None:
        buffer = self._store.setdefault(session_id, [])
        buffer.append((role, content))
        if len(buffer) > self._max_messages:
            del buffer[: len(buffer) - self._max_messages]

    def clear(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class LongTermMemory:
    """Append-only long-term store with naive keyword recall."""

    def __init__(self, path: str | Path | None = None) -> None:
        self._path = Path(path) if path else None
        self._items: list[dict[str, str]] = []

    def remember(self, query: str, answer: str) -> None:
        item = {"query": query, "answer": answer}
        self._items.append(item)
        if self._path is not None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(item) + "\n")

    def recall(self, query: str, k: int = 2) -> list[dict[str, str]]:
        terms = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for item in self._items:
            words = set(re.findall(r"[a-z0-9]+", item["query"].lower()))
            overlap = len(terms & words)
            if overlap:
                scored.append((overlap, item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:k]]
