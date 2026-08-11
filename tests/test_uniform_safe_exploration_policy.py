from __future__ import annotations

from dataclasses import replace

import pytest

from th06_rl.policies.uniform_safe_exploration import (
    POLICY_NAME,
    STATE_SCHEMA,
    UniformSafeExplorationPolicy,
)
from th06_rl.policy_api import PolicyContext


def _context(**changes) -> PolicyContext:
    base = PolicyContext(
        frame=10,
        scope=(3, 0, 0, 6),
        source_context="adapter-owned-and-ignored",
        baseline_action="stay",
        locally_admissible_actions=("stay", "left", "right"),
        player_x=192.0,
        player_y=400.0,
        power=64,
        bullet_count=300,
        laser_count=0,
        hard_action_count=3,
        exploration_rate=0.0,
    )
    return replace(base, **changes)


def _policy(*, seed: int = 7, probability: float = 0.3):
    policy = UniformSafeExplorationPolicy()
    policy.import_state({
        "schema": STATE_SCHEMA,
        "policy_seed": seed,
        "exploration_probability": probability,
    })
    return policy


def test_mixture_probability_is_exact_for_every_sampled_action() -> None:
    policy = _policy()
    observed = set()
    for _ in range(2_000):
        decision = policy.decide(_context())
        observed.add(decision.action)
        expected = 0.8 if decision.action == "stay" else 0.1
        assert decision.policy_id == POLICY_NAME
        assert decision.behavior_probability == pytest.approx(expected)
    assert observed == {"left", "right", "stay"}


def test_sampling_has_no_gameplay_location_or_hazard_gate() -> None:
    first = _policy(seed=91, probability=1.0)
    second = _policy(seed=91, probability=1.0)
    easy = _context(frame=1, player_x=10.0, player_y=20.0, bullet_count=0)
    dense = _context(
        frame=99_999,
        player_x=370.0,
        player_y=430.0,
        bullet_count=640,
        laser_count=8,
        source_context="different-game-adapter-context",
    )
    assert [first.decide(easy).action for _ in range(64)] == [
        second.decide(dense).action for _ in range(64)
    ]


def test_policy_never_escapes_safe_set_and_zero_exploration_is_baseline() -> None:
    policy = _policy(probability=0.0)
    for _ in range(32):
        decision = policy.decide(_context())
        assert decision.action == "stay"
        assert decision.behavior_probability == 1.0

    singleton = _policy(probability=1.0)
    decision = singleton.decide(
        _context(
            baseline_action="left",
            locally_admissible_actions=("left",),
        )
    )
    assert decision.action == "left"
    assert decision.behavior_probability == 1.0


@pytest.mark.parametrize(
    "state",
    [
        {},
        {"schema": STATE_SCHEMA, "policy_seed": -1, "exploration_probability": 0.1},
        {"schema": STATE_SCHEMA, "policy_seed": 0, "exploration_probability": -0.1},
        {"schema": STATE_SCHEMA, "policy_seed": 0, "exploration_probability": 1.1},
    ],
)
def test_invalid_generation_state_fails_closed(state: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        UniformSafeExplorationPolicy().import_state(state)


def test_adapter_baseline_must_be_in_safe_set() -> None:
    with pytest.raises(ValueError, match="baseline"):
        _policy().decide(
            _context(
                baseline_action="up",
                locally_admissible_actions=("left", "right"),
            )
        )
