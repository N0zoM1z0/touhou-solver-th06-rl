from types import SimpleNamespace

import pytest

from th06_rl.policies.uniform_shield_exploration import (
    STATE_SCHEMA,
    UniformShieldExplorationPolicy,
)


def _policy(probability: float = 0.2) -> UniformShieldExplorationPolicy:
    policy = UniformShieldExplorationPolicy()
    policy.import_state({
        "schema": STATE_SCHEMA,
        "policy_seed": 7,
        "exploration_probability": probability,
    })
    return policy


def test_uniform_shield_policy_reports_complete_propensity() -> None:
    policy = _policy()
    context = SimpleNamespace(
        baseline_action="stay",
        locally_admissible_actions=("left", "stay"),
    )

    decision = policy.decide(context)

    probabilities = dict(decision.behavior_probabilities)
    assert probabilities == {"left": 0.1, "stay": 0.9}
    assert decision.behavior_probability == probabilities[decision.action]


def test_zero_exploration_is_deterministic_baseline() -> None:
    decision = _policy(0.0).decide(SimpleNamespace(
        baseline_action="stay",
        locally_admissible_actions=("left", "stay"),
    ))

    assert decision.action == "stay"
    assert dict(decision.behavior_probabilities) == {"left": 0.0, "stay": 1.0}


def test_policy_rejects_empty_or_outside_baseline_set() -> None:
    policy = _policy()
    with pytest.raises(ValueError, match="empty shield"):
        policy.decide(SimpleNamespace(
            baseline_action="stay", locally_admissible_actions=()
        ))
    with pytest.raises(ValueError, match="outside the shield"):
        policy.decide(SimpleNamespace(
            baseline_action="stay", locally_admissible_actions=("left",)
        ))
