from __future__ import annotations

from types import SimpleNamespace

import pytest

from th06_rl.policies.safe_option_exploration import (
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA,
    SafeOptionExplorationPolicy,
)
from th06_rl.policy_api import PolicyDecision, PolicyOptionTrace


def _policy(*, seed: int = 7, exploration: float = 1.0):
    policy = SafeOptionExplorationPolicy()
    policy.import_state({
        "schema": STATE_SCHEMA,
        "policy_seed": seed,
        "exploration_probability": exploration,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
    })
    return policy


def _context(
    frame: int,
    *,
    legal: tuple[str, ...] = ("left", "right"),
    baseline: str = "left",
    scope: tuple[int, int, int, int] = (3, 0, 0, 6),
):
    return SimpleNamespace(
        frame=frame,
        scope=scope,
        locally_admissible_actions=legal,
        baseline_action=baseline,
    )


def test_option_assignment_persists_but_is_recensored_each_frame() -> None:
    policy = _policy()
    first = policy.decide(_context(100))
    second = policy.decide(_context(101, baseline="right"))

    assert second.action == first.action
    assert first.option is not None and first.option.boundary is True
    assert second.option is not None and second.option.boundary is False
    assert second.option.option_id == first.option.option_id
    assert second.option.elapsed_frames == 2
    assert second.behavior_probability == 1.0

    remaining = "right" if first.action == "left" else "left"
    replacement = policy.decide(_context(
        102,
        legal=(remaining,),
        baseline=remaining,
    ))
    assert replacement.action == remaining
    assert replacement.option is not None and replacement.option.boundary is True
    assert replacement.option.option_id != first.option.option_id
    assert replacement.option.preceding_termination_reason == "source-unsafe-intent"


def test_option_has_a_fixed_eight_physical_frame_horizon() -> None:
    policy = _policy(exploration=0.0)
    decisions = [policy.decide(_context(frame)) for frame in range(1, 10)]

    first_id = decisions[0].option.option_id
    assert all(row.option.option_id == first_id for row in decisions[:8])
    assert decisions[7].option.termination_reason == "horizon"
    assert decisions[8].option.boundary is True
    assert decisions[8].option.option_id != first_id
    assert policy.metrics()["terminations"] == {"horizon": 1}


def test_observation_gap_starts_a_new_assignment() -> None:
    policy = _policy(exploration=0.0)
    first = policy.decide(_context(10))
    after_gap = policy.decide(_context(12))

    assert after_gap.option.option_id != first.option.option_id
    assert after_gap.option.preceding_termination_reason == "observation-gap"


def test_option_trace_binds_conditional_propensity() -> None:
    trace = PolicyOptionTrace("option-1", "left", False, 0.05, 2)
    with pytest.raises(ValueError, match="probability disagrees"):
        PolicyDecision("left", "test", 0.05, trace)
