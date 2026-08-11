from __future__ import annotations

import copy

import pytest

from th06_rl.autonomous_learning import (
    BEHAVIOR_POLICY,
    fit_grouped_ridge,
    label_episode,
)


OBSERVATION_NAMES = ("position",)
ACTION_NAMES = ("direction",)


def _row(sequence: int, *, action: str, hit: bool = False):
    legal = ["left", "stay"]
    baseline = "stay"
    probability = 0.05 if action == "left" else 0.95
    direction = -1.0 if action == "left" else 0.0
    return {
        "sequence": sequence,
        "schema_version": "th06-rl-transition-v6",
        "legal_actions": legal,
        "baseline_action": baseline,
        "proposed_action": action,
        "published_action": action,
        "behavior_probability": probability,
        "policy_id": BEHAVIOR_POLICY,
        "learning_eligible": True,
        "policy_context": {
            "current_action": "stay",
            "observation_features": [["position", sequence / 20.0]],
            "action_features": [
                ["left", [["direction", -1.0]]],
                ["stay", [["direction", 0.0]]],
            ],
        },
        "outcome_terms": {
            "elapsed_frames": 1,
            "life_lost": hit,
            "control_dead_end": False,
            "authority_lost": False,
            "bomb_used": False,
        },
    }


def _episode(name: str, action: str):
    rows = [_row(index, action=action, hit=index == 19) for index in range(20)]
    samples, excluded = label_episode(
        rows,
        episode_id=name,
        exploration_probability=0.10,
        return_horizon=5,
        gamma=0.99,
        observation_names=OBSERVATION_NAMES,
        action_names=ACTION_NAMES,
    )
    assert not excluded
    return samples


def test_factual_labels_use_only_published_propensity_bound_actions() -> None:
    rows = [_row(index, action="left", hit=index == 4) for index in range(5)]
    samples, excluded = label_episode(
        rows,
        episode_id="episode-a",
        exploration_probability=0.10,
        return_horizon=5,
        gamma=1.0,
        observation_names=OBSERVATION_NAMES,
        action_names=ACTION_NAMES,
    )
    assert len(samples) == 5
    assert samples[0].target_return == pytest.approx(-95.0)
    assert samples[-1].target_return == pytest.approx(-99.0)
    assert not excluded

    corrupted = copy.deepcopy(rows)
    corrupted[0]["behavior_probability"] = 0.5
    with pytest.raises(ValueError, match="propensity"):
        label_episode(
            corrupted,
            episode_id="bad",
            exploration_probability=0.10,
            return_horizon=5,
            gamma=1.0,
            observation_names=OBSERVATION_NAMES,
            action_names=ACTION_NAMES,
        )


def test_grouped_fit_never_mixes_episode_holdout() -> None:
    train = [
        sample
        for episode in (
            _episode("train-a", "left"),
            _episode("train-b", "stay"),
            _episode("train-c", "left"),
        )
        for sample in episode
    ]
    validation = [
        sample
        for episode in (
            _episode("validation-a", "stay"),
            _episode("validation-b", "left"),
        )
        for sample in episode
    ]
    state = fit_grouped_ridge(
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
        margin_rmse_fraction=0.1,
    )
    report = state["fit_report"]
    assert report["train_groups"] == ["train-a", "train-b", "train-c"]
    assert report["validation_groups"] == ["validation-a", "validation-b"]
    assert state["authorization"]["fit_eligible"] is True
    assert len(state["model"]["committee"]) == 3
