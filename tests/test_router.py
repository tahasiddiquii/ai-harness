from ai_harness.config import Settings
from ai_harness.schemas import Classification, Complexity, Intent
from ai_harness.stages.router import cost_for, route


def _cls(complexity: Complexity) -> Classification:
    return Classification(
        intent=Intent.QA,
        complexity=complexity,
        needs_tools=False,
        needs_retrieval=True,
        confidence=0.7,
    )


def test_quality_policy_always_picks_large():
    s = Settings(default_provider="mock")
    assert route(_cls(Complexity.SIMPLE), s, policy_override="quality").tier == "large"


def test_cost_policy_always_picks_small():
    s = Settings(default_provider="mock")
    assert route(_cls(Complexity.COMPLEX), s, policy_override="cost").tier == "small"


def test_balanced_policy_scales_with_complexity():
    s = Settings(default_provider="mock", router_policy="balanced")
    assert route(_cls(Complexity.SIMPLE), s).tier == "small"
    assert route(_cls(Complexity.COMPLEX), s).tier == "large"


def test_unavailable_provider_falls_back_to_mock():
    s = Settings(default_provider="openai", openai_api_key=None, anthropic_api_key=None)
    assert route(_cls(Complexity.SIMPLE), s).provider == "mock"


def test_cost_is_zero_for_mock_models():
    assert cost_for("mock-small", 10_000) == 0.0
    assert cost_for("gpt-4o-mini", 1000) > 0.0
