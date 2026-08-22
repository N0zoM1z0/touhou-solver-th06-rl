from __future__ import annotations

import numpy as np
import xgboost

from th06_rl.actions import ACTION_NAMES
from th06_rl.action_intention_dataset import ActionIntentionDataset, DATASET_SCHEMA
from th06_rl.action_intention_hazard import (
    evaluate_action_intention_hazard_models,
    fit_action_intention_hazard_models,
    intention_hazard_predictions,
)
from th06_rl.core.model import movement_actions
from th06_rl.factual_probes import PROBE_FEATURE_NAMES


_MOVEMENTS = {action.name: action for action in movement_actions()}


def _dataset(name: str, *, episodes: int = 6) -> ActionIntentionDataset:
    features = []
    labels = []
    actions = []
    episode_indices = []
    for episode in range(episodes):
        for repeat in range(4):
            for action_name in ACTION_NAMES:
                action = _MOVEMENTS[action_name]
                row = np.zeros(len(PROBE_FEATURE_NAMES), dtype=np.float64)
                row[:6] = (192.0, 400.0, 64.0, 4.0, 0.0, 18.0)
                row[6] = 0.0
                row[7] = 20.0 + repeat
                row[8] = 192.0 + action.dx
                row[9] = 400.0 + action.dy
                row[10] = 20.0
                row[11] = 0.0
                row[12] = action.dx
                row[13] = action.dy
                row[14] = float(action.focused)
                features.append(row)
                labels.append(action.dx > 0)
                actions.append(action_name)
                episode_indices.append(episode)
    label_array = np.asarray(labels, dtype=np.bool_)
    inventory = tuple({
        "episode_id": f"{name}-{episode}",
        "run_sha256": str(episode),
        "manifest_sha256": str(episode),
        "group_starts": len(ACTION_NAMES) * 4,
        "accepted_rows": len(ACTION_NAMES) * 4,
        "positives": int(np.sum(label_array[
            np.asarray(episode_indices) == episode
        ])),
        "negatives": int(np.sum(~label_array[
            np.asarray(episode_indices) == episode
        ])),
        "positive_after_control_dead_end": 0,
    } for episode in range(episodes))
    return ActionIntentionDataset(
        DATASET_SCHEMA,
        12,
        tuple(row["episode_id"] for row in inventory),
        inventory,
        np.asarray(episode_indices, dtype=np.int64),
        np.asarray(features, dtype=np.float64),
        label_array,
        tuple(actions),
        np.full(len(labels), 1.0 / len(ACTION_NAMES), dtype=np.float64),
        np.zeros(len(labels), dtype=np.bool_),
        np.zeros(len(labels), dtype=np.bool_),
        tuple(None for _ in labels),
    )


def _fit(dataset: ActionIntentionDataset) -> dict[str, object]:
    return fit_action_intention_hazard_models(
        dataset,
        boosted_rounds=16,
        maximum_depth=3,
        learning_rate=0.1,
        minimum_child_weight=1.0,
        l2_leaf_regularization=1.0,
        maximum_histogram_bins=32,
        seed=17,
        expected_xgboost_version=xgboost.__version__,
    )


def test_intention_hazard_is_deterministic_and_selects_randomized_action_signal() -> None:
    train = _dataset("train")
    validation = _dataset("validation")
    first = _fit(train)
    second = _fit(train)

    assert first == second
    predictions, bounds = intention_hazard_predictions(
        first, validation.features, model_name="full_group_start_action"
    )
    assert np.all((0.0 <= predictions) & (predictions <= 1.0))
    assert bounds["clipped_rows"] == 0

    result = evaluate_action_intention_hazard_models(
        first,
        validation,
        bootstrap_samples=100,
        bootstrap_seed=23,
        calibration_bins=10,
        minimum_validation_episodes=6,
        minimum_validation_positives=1,
        minimum_validation_negatives=1,
        minimum_positive_episodes=6,
        minimum_episodes_favoring_full=6,
        maximum_calibration_in_the_large_absolute=1.0,
        maximum_expected_calibration_error=1.0,
        maximum_full_ece_over_state_only=1.0,
        maximum_raw_clipped_fraction=1.0,
    )

    assert result["gates"]["incremental_randomized_action_signal"] is True
    assert result["summary"]["decision"] == (
        "select-h12-intention-hazard-for-export-and-wine-canary-preregistration"
    )
    assert result["summary"]["online_policy_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
