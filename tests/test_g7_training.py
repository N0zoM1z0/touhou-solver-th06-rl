from __future__ import annotations

import pytest

pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("xgboost")

from th06_rl.g7_forecast import forecast_accepted_actions
from th06_rl.g7_learner import linear_actor_scores
from th06_rl.g7_support import locally_supported_actions
from th06_rl.g7_training import CANDIDATE_SCHEMA, fit_g7_candidate
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.offline_options import ActorState, OfflineOptionTransition
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


def _option(index: int) -> OfflineOptionTransition:
    action = "left" if index % 2 == 0 else "stay"
    cost = int(action == "stay")
    state = _state()
    return OfflineOptionTransition(
        "th06-rl-causal-options-v2",
        f"episode-{index:03d}",
        "complete-route",
        "safe-option-exploration-v2",
        f"{index:016x}:00000001",
        0,
        0,
        1,
        "fixture",
        action,
        0.5,
        (("left", 0.5), ("stay", 0.5)),
        state,
        None,
        cost,
        cost,
        0,
        1,
        0,
        True,
        True,
        (),
    )


def test_complete_fit_uses_only_supplied_episode_partition() -> None:
    episodes = tuple((_option(index),) for index in range(64))
    candidate = fit_g7_candidate(
        episodes,
        seed=19,
        reference_epsilon=1.0,
        awr_temperature=0.25,
        crossfit_folds=3,
        critic_estimators=40,
        n_jobs=2,
        maximum_importance_ratio=2.0,
        support_prototypes_per_action=2,
        support_minimum_samples=8,
        support_minimum_ess=8.0,
        ensemble_members=3,
        ensemble_episode_fraction=0.75,
    )
    state = _state()
    support = locally_supported_actions(candidate["local_support"], state)
    forecast = forecast_accepted_actions(
        candidate["forecast"], state, supported_actions=support
    )
    scores = dict(linear_actor_scores(candidate["actor"], state))

    assert candidate["schema"] == CANDIDATE_SCHEMA
    assert candidate["authorization"] == "offline-research-only"
    assert candidate["fit"]["episodes"] == 64
    assert scores["left"] > scores["stay"]
    assert support == ("left", "stay")
    assert forecast == ("left", "stay")
    assert all(len(row["episode_ids"]) == 48 for row in candidate["fit"]["ensemble"])
