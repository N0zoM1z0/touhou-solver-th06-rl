from __future__ import annotations

from th06_rl.policy_api import PolicyContext
from th06_rl.policies.autonomous_iql_actor import AutonomousIqlActorPolicy


class _ZeroRandom:
    def random(self) -> float:
        return 0.0


class _ForbiddenRandom:
    def random(self) -> float:
        raise AssertionError("shadow policy must not sample an intervention")


def _context(frame: int, legal=("stay", "left")) -> PolicyContext:
    return PolicyContext(
        frame=frame,
        scope=(6, 1, 1, 1),
        source_context="generic-test",
        baseline_action="stay",
        locally_admissible_actions=legal,
        player_x=192.0,
        player_y=384.0,
        power=128,
        bullet_count=0,
        laser_count=0,
        hard_action_count=len(legal),
        exploration_rate=0.0,
    )


def _wired_policy(mode: str) -> AutonomousIqlActorPolicy:
    policy = AutonomousIqlActorPolicy()
    policy.loaded = True
    policy.mode = mode
    policy.name = f"test-{mode}"
    policy.intervention_budget = 1 if mode == "active" else None
    policy._proposal = lambda _context, _legal, _baseline: "left"
    return policy


def test_shadow_is_deterministic_incumbent_with_probability_one() -> None:
    policy = _wired_policy("shadow")
    policy.random = _ForbiddenRandom()
    decision = policy.decide(_context(100))
    assert decision.action == "stay"
    assert decision.behavior_probability == 1.0
    assert decision.option is not None
    assert decision.option.boundary_probability == 1.0
    assert policy.proposals == 1
    assert policy.interventions == 0


def test_active_target_is_bounded_and_budget_fails_to_incumbent() -> None:
    policy = _wired_policy("active")
    policy.random = _ZeroRandom()
    first = policy.decide(_context(100))
    assert first.action == "left"
    assert first.behavior_probability == 0.1
    assert first.option is not None and first.option.boundary
    for frame in range(101, 108):
        continuation = policy.decide(_context(frame))
        assert continuation.action == "left"
        assert continuation.behavior_probability == 1.0
    after_budget = policy.decide(_context(108))
    assert after_budget.action == "stay"
    assert after_budget.behavior_probability == 1.0
    assert policy.interventions == 1
    assert policy.budget_abstentions == 1


def test_option_ends_when_native_safe_set_removes_intent() -> None:
    policy = _wired_policy("active")
    policy.random = _ZeroRandom()
    assert policy.decide(_context(100)).action == "left"
    replacement = policy.decide(_context(101, legal=("stay",)))
    assert replacement.action == "stay"
    assert replacement.option is not None
    assert replacement.option.preceding_termination_reason == "source-unsafe-intent"
    assert policy.terminations["source-unsafe-intent"] == 1
