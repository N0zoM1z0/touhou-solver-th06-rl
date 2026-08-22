from __future__ import annotations

import numpy as np

from th06_rl.factual_primitive_dataset import (
    PRIMITIVE_FEATURE_SCHEMA,
    TOKEN_FEATURE_NAMES,
    load_primitive_probe_dataset,
)
from th06_rl.factual_probes import PROBE_FEATURE_NAMES

from tests.test_factual_history_dataset import _history_episode


def test_primitive_dataset_is_bounded_factual_and_action_independent(tmp_path) -> None:
    dataset = load_primitive_probe_dataset(
        (_history_episode(tmp_path / "episode"),),
        horizons=(1,),
        token_cap=8,
    )
    view = dataset.horizons[0]

    assert dataset.feature_schema == PRIMITIVE_FEATURE_SCHEMA
    assert dataset.scalar_feature_names == PROBE_FEATURE_NAMES
    assert len(TOKEN_FEATURE_NAMES) == 13
    assert view.rows == 3
    assert view.current_features.shape == (3, 15)
    assert view.scalar_features.shape == (3, 15)
    assert view.primitive_tokens.shape == (3, 8, 13)
    assert view.primitive_masks.shape == (3, 8)
    assert tuple(view.token_counts) == (3, 3, 3)
    assert not np.any(view.truncated_token_counts)
    assert np.all(view.primitive_tokens[~view.primitive_masks] == 0.0)
    assert tuple(view.hit_labels) == (False, False, True)
    assert not any(
        forbidden in name
        for name in TOKEN_FEATURE_NAMES
        for forbidden in (
            "stage",
            "boss",
            "spell",
            "ecl",
            "rng",
            "slot",
            "source",
            "future",
            "outcome",
            "hit",
        )
    )


def test_primitive_cap_uses_frozen_physical_order_and_counts_truncation(
    tmp_path,
) -> None:
    dataset = load_primitive_probe_dataset(
        (_history_episode(tmp_path / "episode"),),
        horizons=(1,),
        token_cap=2,
    )
    view = dataset.horizons[0]

    assert tuple(view.token_counts) == (3, 3, 3)
    assert tuple(view.truncated_token_counts) == (1, 1, 1)
    assert np.all(view.primitive_masks)
    # Earliest AABBs precede the later laser under the fixed ordering.
    assert np.all(view.primitive_tokens[:, :, 0] == 1.0)
    assert np.all(view.primitive_tokens[:, :, 1] == 0.0)


def test_primitive_parallel_loader_exactly_matches_serial(tmp_path) -> None:
    paths = (
        _history_episode(tmp_path / "episode-a"),
        _history_episode(tmp_path / "episode-b", mirrored=True),
    )
    serial = load_primitive_probe_dataset(
        paths, horizons=(1,), token_cap=8, workers=1
    )
    parallel = load_primitive_probe_dataset(
        paths, horizons=(1,), token_cap=8, workers=2
    )

    assert parallel.episode_ids == serial.episode_ids
    assert parallel.inventory == serial.inventory
    assert parallel.feature_schema == serial.feature_schema
    assert parallel.scalar_feature_names == serial.scalar_feature_names
    assert parallel.token_feature_names == serial.token_feature_names
    serial_view = serial.horizons[0]
    parallel_view = parallel.horizons[0]
    assert parallel_view.horizon == serial_view.horizon
    assert parallel_view.token_cap == serial_view.token_cap
    assert parallel_view.episode_ids == serial_view.episode_ids
    assert parallel_view.published_actions == serial_view.published_actions
    assert parallel_view.baseline_actions == serial_view.baseline_actions
    assert parallel_view.lifecycle_strata == serial_view.lifecycle_strata
    for name in (
        "episode_indices",
        "current_features",
        "scalar_features",
        "primitive_tokens",
        "primitive_masks",
        "token_counts",
        "truncated_token_counts",
        "hit_labels",
        "behavior_probabilities",
    ):
        assert np.array_equal(
            getattr(parallel_view, name), getattr(serial_view, name)
        )
