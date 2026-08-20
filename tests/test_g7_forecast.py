from __future__ import annotations

import copy

from th06_rl.g7_forecast import (
    build_forecast_artifact,
    forecast_accepted_actions,
)
from th06_rl.g7_learner import ACTOR_FEATURE_NAMES
from th06_rl.g7_policy_math import POLICY_DISTRIBUTION_SCHEMA
from th06_rl.feature_contract import FEATURE_AVAILABILITY_SCHEMA
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.learning_features import CAUSAL_TREE_FEATURE_SCHEMA
from th06_rl.offline_options import ActorState
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


def _named(names, overrides=None):
    overrides = overrides or {}
    return tuple((name, float(overrides.get(name, 0.0))) for name in names)


def _state() -> ActorState:
    return ActorState(
        _named(OBSERVATION_FEATURE_NAMES),
        (
            ("left", _named(ACTION_FEATURE_NAMES, {"direction_x": -1.0})),
            ("stay", _named(ACTION_FEATURE_NAMES, {"stationary": 1.0})),
        ),
        _named(HISTORY_FEATURE_NAMES),
        ("left", "stay"),
        "stay",
        "stay",
    )


def _actor(left_weight: float) -> dict[str, object]:
    weights = [0.0] * len(ACTOR_FEATURE_NAMES)
    weights[ACTOR_FEATURE_NAMES.index("action:direction_x")] = -left_weight
    return {
        "schema": "th06-rl-g7-linear-awr-actor-v1",
        "feature_schema": CAUSAL_TREE_FEATURE_SCHEMA,
        "feature_availability_schema": FEATURE_AVAILABILITY_SCHEMA,
        "policy_distribution_schema": POLICY_DISTRIBUTION_SCHEMA,
        "feature_names": list(ACTOR_FEATURE_NAMES),
        "mean": [0.0] * len(ACTOR_FEATURE_NAMES),
        "scale": [1.0] * len(ACTOR_FEATURE_NAMES),
        "weights": weights,
        "reference_epsilon": 1.0,
    }


def test_unanimous_sign_consensus_accepts_only_supported_improvement() -> None:
    artifact = build_forecast_artifact([_actor(1.0), _actor(2.0), _actor(3.0)])

    assert forecast_accepted_actions(
        artifact,
        _state(),
        supported_actions=("left", "stay"),
    ) == ("left", "stay")
    assert forecast_accepted_actions(
        artifact,
        _state(),
        supported_actions=("stay",),
    ) == ("stay",)


def test_one_disagreeing_member_forces_abstention_at_unanimous_default() -> None:
    artifact = build_forecast_artifact([_actor(1.0), _actor(2.0), _actor(-1.0)])

    assert forecast_accepted_actions(
        artifact,
        _state(),
        supported_actions=("left", "stay"),
    ) == ("stay",)

    malformed = copy.deepcopy(artifact)
    malformed["actors"][0]["weights"][0] = float("nan")
    try:
        forecast_accepted_actions(
            malformed,
            _state(),
            supported_actions=("left", "stay"),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("non-finite ensemble member did not fail closed")
