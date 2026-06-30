"""Hybrid retrieval (Context Orchestration layer).

Sparse BM25 retrieval over a small built-in knowledge base, with reciprocal-rank
fusion (RRF) wired so a dense retriever can be fused in when an embeddings backend
is configured. Offline this runs BM25-only and fully deterministically; the
dedicated `hybrid-graph-rag` repo implements true dense + sparse + graph retrieval.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ai_harness.config import Settings
from ai_harness.schemas import RetrievedChunk

DEFAULT_KB: list[dict[str, str]] = [
    {
        "id": "harness-101",
        "text": "An agent equals a model plus a harness. The harness is every piece of "
        "code and configuration around the model: prompts, tools, context policies, "
        "guardrails, memory, and observability. A decent model with a great harness "
        "beats a great model with a bad harness.",
    },
    {
        "id": "routing",
        "text": "Adaptive model routing sends each request to the cheapest model that can "
        "handle it. Simple queries go to a small fast model; complex reasoning goes to a "
        "larger model. Routing on cost, latency, and quality is the highest-ROI lever in a harness.",
    },
    {
        "id": "rag",
        "text": "Retrieval augmented generation grounds answers in retrieved context. Hybrid "
        "RAG fuses sparse lexical search such as BM25 with dense vector search. Graph RAG "
        "adds a knowledge graph for multi-hop questions across related entities.",
    },
    {
        "id": "guardrails",
        "text": "Guardrails enforce safety and policy. Input guardrails detect and redact PII "
        "and catch prompt injection or jailbreak attempts. Output guardrails validate structure "
        "and block unsafe or ungrounded responses before they reach the user.",
    },
    {
        "id": "observability",
        "text": "Observability means tracing every stage of a request with cost and latency "
        "metering. Tools like Langfuse and OpenTelemetry capture spans so you can debug, "
        "measure quality, and run continuous evaluation on real production traffic.",
    },
    {
        "id": "evaluation",
        "text": "LLM-as-a-judge evaluation scores responses for correctness, faithfulness, and "
        "safety. Run evals offline on a golden dataset in CI to catch regressions, and online "
        "by sampling production traces. Separate the generator from the evaluator for honest scores.",
    },
]

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _rrf(rankings: list[list[int]], n: int, k: int = 60) -> list[int]:
    scores = [0.0] * n
    for ranking in rankings:
        for rank, idx in enumerate(ranking):
            scores[idx] += 1.0 / (k + rank + 1)
    return sorted(range(n), key=lambda i: scores[i], reverse=True)


class HybridRetriever:
    def __init__(self, settings: Settings, docs: list[dict[str, str]] | None = None) -> None:
        self._settings = settings
        self._docs = docs or DEFAULT_KB
        self._bm25 = BM25Okapi([_tokenize(d["text"]) for d in self._docs])

    def _dense_ranking(self, query: str) -> list[int] | None:
        # Hook for a dense retriever (embeddings). Returns None offline.
        return None

    def retrieve(self, query: str, k: int = 3) -> list[RetrievedChunk]:
        bm25_scores = self._bm25.get_scores(_tokenize(query))
        n = len(self._docs)
        bm25_ranking = sorted(range(n), key=lambda i: bm25_scores[i], reverse=True)

        rankings = [bm25_ranking]
        dense = self._dense_ranking(query)
        hybrid = dense is not None
        if dense is not None:
            rankings.append(dense)

        fused = _rrf(rankings, n) if hybrid else bm25_ranking
        out: list[RetrievedChunk] = []
        for idx in fused[:k]:
            if bm25_scores[idx] <= 0.0 and not hybrid:
                continue
            doc = self._docs[idx]
            out.append(
                RetrievedChunk(
                    source_id=doc["id"],
                    text=doc["text"],
                    score=round(float(bm25_scores[idx]), 3),
                    retriever="hybrid" if hybrid else "bm25",
                )
            )
        return out
