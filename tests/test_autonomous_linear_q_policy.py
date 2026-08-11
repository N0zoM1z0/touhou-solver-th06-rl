from __future__ import annotations

import hashlib

import pytest

from th06_rl.autonomous_learning import fit_grouped_ridge
from th06_rl.policies.autonomous_linear_q import AutonomousLinearQPolicy
from th06_rl.policy_api import PolicyContext
from tests.test_autonomous_learning import (
    ACTION_NAMES,
    OBSERVATION_NAMES,
    _episode,
)


def _state():
    train = [
        sample
        for name, action in (
            ("train-a", "left"),
            ("train-b", "stay"),
            ("train-c", "left"),
        )
        for sample in _episode(name, action)
    ]
    validation = [
        sample
        for name, action in (
            ("validation-a", "stay"),
            ("validation-b", "left"),
        )
        for sample in _episode(name, action)
    ]
    return fit_grouped_ridge(
        train,
        validation,
        observation_names=OBSERVATION_NAMES,
        action_names=ACTION_NAMES,
        alpha=1.0,
        propensity_clip=20.0,
        minimum_train_groups=3,
        minimum_validation_groups=2,
        minimum_train_rows=1,
        minimum_non_baseline_rows=1,
        minimum_action_samples=1,
        minimum_action_ess=1.0,
        required_rmse_ratio=10.0,
        margin_rmse_fraction=0.0,
    )


def _context():
    return PolicyContext(
        frame=99,
        scope=(3, 0, 0, 6),
        source_context="ignored",
        baseline_action="stay",
        locally_admissible_actions=("left", "stay"),
        player_x=0.0,
        player_y=0.0,
        power=0,
        bullet_count=0,
        laser_count=0,
        hard_action_count=2,
        exploration_rate=0.0,
        current_action="stay",
        observation_features=(("position", 0.5),),
        action_features=(
            ("left", (("direction", -1.0),)),
            ("stay", (("direction", 0.0),)),
        ),
    )


def test_shadow_policy_never_publishes_its_learned_proposal() -> None:
    policy = AutonomousLinearQPolicy()
    policy.import_state(_state())
    decision = policy.decide(_context())
    assert decision.action == "stay"
    assert decision.behavior_probability == 1.0
    assert policy.metrics()["active_overrides"] == 0


def test_active_policy_requires_bound_shadow_authorization() -> None:
    state = _state()
    state["mode"] = "active"
    policy = AutonomousLinearQPolicy()
    with pytest.raises(ValueError, match="authorization"):
        policy.import_state(state)

    state["authorization"]["active_canary"] = {
        "shadow_audit_sha256": hashlib.sha256(b"shadow").hexdigest()
    }
    policy.import_state(state)
    assert policy.decide(_context()).action in {"left", "stay"}
