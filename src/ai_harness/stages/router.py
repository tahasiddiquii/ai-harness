"""Adaptive model router (Entry & Protection layer).

Picks a model per request by trading off **cost, latency, and quality** against
the classified complexity and the active policy. This is the single highest-ROI
piece of a harness: most requests are simple and should go to a cheap, fast model;
only the hard ones earn the expensive tier.

Catalog prices/latencies are illustrative blended figures for routing decisions,
not a billing source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass

from ai_harness.config import Settings
from ai_harness.schemas import Classification, Complexity, RouteDecision


@dataclass(frozen=True)
class ModelCard:
    provider: str
    model: str
    tier: str  # "small" | "large"
    cost_per_1k_usd: float
    latency_ms: int
    quality: float  # 0..1 relative


CATALOG: list[ModelCard] = [
    ModelCard("mock", "mock-small", "small", 0.0, 40, 0.55),
    ModelCard("mock", "mock-large", "large", 0.0, 90, 0.80),
    ModelCard("openai", "gpt-4o-mini", "small", 0.0006, 700, 0.78),
    ModelCard("openai", "gpt-4o", "large", 0.0050, 1300, 0.93),
    ModelCard("anthropic", "claude-3-5-haiku-latest", "small", 0.0008, 650, 0.80),
    ModelCard("anthropic", "claude-3-5-sonnet-latest", "large", 0.0060, 1400, 0.95),
]


def _available_providers(settings: Settings) -> set[str]:
    available = {"mock"}
    if settings.openai_api_key:
        available.add("openai")
    if settings.anthropic_api_key:
        available.add("anthropic")
    return available


def _tier_for(policy: str, complexity: Complexity) -> str:
    if policy == "quality":
        return "large"
    if policy in ("cost", "latency"):
        return "small"
    # balanced: spend on the large model only for genuinely complex work
    return "large" if complexity == Complexity.COMPLEX else "small"


def _pick(provider: str, tier: str) -> ModelCard:
    for card in CATALOG:
        if card.provider == provider and card.tier == tier:
            return card
    for card in CATALOG:
        if card.provider == provider:
            return card
    return CATALOG[1]  # mock-large fallback


def route(
    classification: Classification,
    settings: Settings,
    policy_override: str | None = None,
) -> RouteDecision:
    policy = (policy_override or settings.router_policy or "balanced").lower()
    available = _available_providers(settings)

    provider = settings.default_provider.lower()
    if provider not in available:
        provider = "openai" if "openai" in available else "mock"

    tier = _tier_for(policy, classification.complexity)
    card = _pick(provider, tier)
    reason = (
        f"policy={policy}, intent={classification.intent.value}, "
        f"complexity={classification.complexity.value} -> {card.tier} tier on {card.provider}"
    )
    return RouteDecision(
        provider=card.provider,
        model=card.model,
        tier=card.tier,
        reason=reason,
        est_cost_per_1k_usd=card.cost_per_1k_usd,
        est_latency_ms=card.latency_ms,
    )


def cost_for(model: str, total_tokens: int) -> float:
    for card in CATALOG:
        if card.model == model:
            return round((total_tokens / 1000.0) * card.cost_per_1k_usd, 6)
    return 0.0
