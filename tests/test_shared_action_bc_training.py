from __future__ import annotations

import numpy as np
import pytest

from th06_rl.actions import ACTION_NAMES
from th06_rl.bc_features import _features
from th06_rl.bc_target_diagnostics import PropensityDataset
from th06_rl.shared_action_bc_training import (
    _action_feature_tensor,
    _bootstrap_soft_delta_interval,
    _shared_probabilities,
)
from th06_rl.shared_action_features import ACTION_FEATURE_NAMES, action_feature_rows


def _dataset() -> PropensityDataset:
    legal = ACTION_NAMES
    evaluations = tuple(
        (name, 5.0 + index, 100.0 + index, 200.0 - index)
        for index, name in enumerate(ACTION_NAMES)
    )
    rows = tuple(
        _features(
            player_x=100.0 + row,
            player_y=200.0,
            power=16,
            bullet_count=2,
            laser_count=0,
            current_action="stay",
            legal_actions=legal,
            evaluations=evaluations,
        )
        for row in range(2)
    )
    targets = np.asarray([0, 1], dtype=np.int64)
    behavior = np.full((2, len(ACTION_NAMES)), 0.2 / len(ACTION_NAMES))
    behavior[np.arange(2), targets] += 0.8
    return PropensityDataset(
        ("episode-a", "episode-b"),
        np.asarray(rows, dtype=np.float64),
        targets,
        np.ones((2, len(ACTION_NAMES)), dtype=np.bool_),
        targets.copy(),
        behavior,
        np.asarray([0, 1], dtype=np.int64),
        (),
        (("uniform-shield-exploration-v1", 2),),
        ((len(ACTION_NAMES), 2),),
        0.0,
    )


def test_vectorized_action_features_match_online_projection() -> None:
    dataset = _dataset()
    tensor = _action_feature_tensor(dataset)
    expected = action_feature_rows(
        tuple(dataset.features[0]), ACTION_NAMES
    )
    assert tensor.shape == (2, len(ACTION_NAMES), len(ACTION_FEATURE_NAMES))
    np.testing.assert_allclose(tensor[0], np.asarray(expected))


def test_shared_zero_weights_are_uniform_over_mask() -> None:
    dataset = _dataset()
    tensor = _action_feature_tensor(dataset)
    probabilities = _shared_probabilities(
        tensor,
        dataset.legal_masks,
        np.zeros(len(ACTION_FEATURE_NAMES)),
    )
    assert probabilities == pytest.approx(
        np.full_like(probabilities, 1.0 / len(ACTION_NAMES))
    )


def test_soft_episode_bootstrap_compares_exact_distribution_loss() -> None:
    dataset = _dataset()
    comparator = dataset.behavior_targets.copy()
    candidate = np.full_like(comparator, 1.0 / len(ACTION_NAMES))
    low, high = _bootstrap_soft_delta_interval(
        candidate, comparator, dataset, seed=0, samples=100
    )
    assert low > 0.0
    assert high > 0.0
