from types import SimpleNamespace

from th06_rl.policies.fixed_shield_action_exposure import (
    OVERRIDE_REASON,
    STATE_SCHEMA,
    FixedShieldActionExposurePolicy,
)


def _policy(seed: int = 7) -> FixedShieldActionExposurePolicy:
    policy = FixedShieldActionExposurePolicy()
    policy.import_state({
        "schema": STATE_SCHEMA,
        "policy_seed": seed,
        "exposure_roots": 4,
    })
    return policy


def _context(*actions: str, baseline: str = "stay") -> SimpleNamespace:
    return SimpleNamespace(
        baseline_action=baseline,
        locally_admissible_actions=actions,
    )


def test_four_root_intention_has_uniform_assignment_then_conditional_actions() -> None:
    policy = _policy()
    decisions = [policy.decide(_context("left", "stay")) for _ in range(4)]
    exposure = [decision.action_exposure for decision in decisions]

    assert all(item is not None for item in exposure)
    assert [item.step for item in exposure if item is not None] == [0, 1, 2, 3]
    assert {item.group_id for item in exposure if item is not None} == {0}
    intended = exposure[0].intended_action  # type: ignore[union-attr]
    assert all(item.intended_action == intended for item in exposure if item is not None)
    assert dict(decisions[0].behavior_probabilities) == {"left": 0.5, "stay": 0.5}
    assert all(decision.action == intended for decision in decisions)
    assert all(
        dict(decision.behavior_probabilities)[intended] == 1.0
        for decision in decisions[1:]
    )

    next_decision = policy.decide(_context("left", "stay"))
    assert next_decision.action_exposure is not None
    assert next_decision.action_exposure.group_id == 1
    assert next_decision.action_exposure.step == 0


def test_inadmissible_intention_uses_declared_baseline_override() -> None:
    policy = _policy()
    first = policy.decide(_context("left", "stay"))
    assert first.action_exposure is not None
    intended = first.action_exposure.intended_action
    fallback = "right" if intended != "right" else "left"

    decision = policy.decide(_context(fallback, baseline=fallback))

    assert decision.action == fallback
    assert decision.behavior_probability == 1.0
    assert dict(decision.behavior_probabilities) == {fallback: 1.0}
    assert decision.action_exposure is not None
    assert decision.action_exposure.intended_action == intended
    assert decision.action_exposure.override_reason == OVERRIDE_REASON


def test_lifecycle_interrupt_starts_a_fresh_assignment_group() -> None:
    policy = _policy()
    first = policy.decide(_context("left", "stay"))
    policy.interrupt("physical-hit")
    second = policy.decide(_context("left", "stay"))

    assert first.action_exposure is not None
    assert second.action_exposure is not None
    assert second.action_exposure.group_id == first.action_exposure.group_id + 1
    assert second.action_exposure.step == 0
    assert policy.metrics()["interruptions"] == {"physical-hit": 1}
