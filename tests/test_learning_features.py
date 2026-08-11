from __future__ import annotations

from types import SimpleNamespace

import pytest

from th06_rl.core.model import movement_actions
from th06_rl.learning_features import candidate_vector, feature_names
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
    project_learning_features,
)


def _projection():
    actions = {action.name: action for action in movement_actions()}
    snapshot = SimpleNamespace(
        x=192.0,
        y=400.0,
        live_bullet_count=320,
        laser_count=1,
        current_power=64,
    )
    evaluations = (
        SimpleNamespace(
            action=actions["stay"],
            min_clearance=10.0,
            final_x=192.0,
            final_y=400.0,
        ),
        SimpleNamespace(
            action=actions["left"],
            min_clearance=float("inf"),
            final_x=184.0,
            final_y=400.0,
        ),
    )
    checkpoints = (1, 2, 3, 4, 6, 8, 10, 12)
    profiles = (
        SimpleNamespace(
            action=actions["stay"],
            checkpoints=checkpoints,
            min_clearances=(12.0, 11.0, 10.0, 9.0, 0.1, 5.0, 3.0, -1.0),
        ),
        SimpleNamespace(
            action=actions["left"],
            checkpoints=checkpoints,
            min_clearances=(float("inf"),) * len(checkpoints),
        ),
    )
    return project_learning_features(
        snapshot, evaluations, ("stay", "left"), 4, profiles
    )


def test_th06_adapter_emits_versioned_finite_named_features() -> None:
    observation, actions = _projection()
    assert tuple(name for name, _ in observation) == OBSERVATION_FEATURE_NAMES
    assert all(value == pytest.approx(value) for _, value in observation)
    assert tuple(name for name, _ in actions[0][1]) == ACTION_FEATURE_NAMES
    left = dict(actions)["left"]
    assert dict(left)["clearance_unbounded"] == 1.0
    assert dict(left)["clearance_log"] == 0.0
    assert dict(left)["profile_unbounded_h12"] == 1.0
    assert dict(left)["profile_rank_h12"] == 1.0
    assert dict(dict(actions)["stay"])["profile_viable_h12"] == 0.0
    assert dict(dict(actions)["stay"])["profile_viable_h6"] == 0.0


def test_generic_vector_is_action_relative_and_has_declared_order() -> None:
    observation, actions = _projection()
    names = feature_names(OBSERVATION_FEATURE_NAMES, ACTION_FEATURE_NAMES)
    baseline = candidate_vector(
        observation_features=observation,
        action_features=actions,
        action="stay",
        baseline_action="stay",
        current_action="stay",
        observation_names=OBSERVATION_FEATURE_NAMES,
        action_names=ACTION_FEATURE_NAMES,
    )
    alternative = candidate_vector(
        observation_features=observation,
        action_features=actions,
        action="left",
        baseline_action="stay",
        current_action="stay",
        observation_names=OBSERVATION_FEATURE_NAMES,
        action_names=ACTION_FEATURE_NAMES,
    )
    assert len(baseline) == len(alternative) == len(names)
    delta_start = len(OBSERVATION_FEATURE_NAMES) + len(ACTION_FEATURE_NAMES)
    assert set(baseline[delta_start:delta_start + len(ACTION_FEATURE_NAMES)]) == {0.0}
    assert alternative != baseline


def test_generic_vector_rejects_adapter_schema_drift() -> None:
    observation, actions = _projection()
    with pytest.raises(ValueError, match="observation feature schema"):
        candidate_vector(
            observation_features=tuple(reversed(observation)),
            action_features=actions,
            action="stay",
            baseline_action="stay",
            current_action="stay",
            observation_names=OBSERVATION_FEATURE_NAMES,
            action_names=ACTION_FEATURE_NAMES,
        )
