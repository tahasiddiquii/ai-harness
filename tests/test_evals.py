"""Validates the eval harness itself: the quality gate must pass deterministically
on the mock provider, so CI can trust it as a merge gate."""

from evals.run_evals import THRESHOLDS, run

from ai_harness.config import Settings


def test_quality_gate_passes_on_mock_provider():
    metrics, details, _ = run(Settings(default_provider="mock", router_policy="balanced"))
    assert details, "eval produced no per-example results"
    for metric, threshold in THRESHOLDS.items():
        assert metrics[metric] >= threshold, (
            f"{metric}={metrics[metric]:.2f} below gate {threshold}"
        )


def test_adversarial_prompts_are_all_blocked():
    metrics, _, _ = run(Settings(default_provider="mock"))
    assert metrics["block_accuracy"] == 1.0
