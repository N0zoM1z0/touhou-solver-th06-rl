from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from th06_rl.corpus import CorpusRecorder, RunMetadata
from th06_rl.factual_history_dataset import (
    HISTORY_FEATURE_SCHEMA,
    history_feature_names,
    load_history_probe_dataset,
)
from th06_rl.factual_probes import PROBE_FEATURE_NAMES

from tests.test_episode_dataset import _decision, _snapshot
from tests.test_factual_probes import _probe_episode


def _with_primitives(decision):
    return replace(
        decision,
        shield_contract="observed-hazard-kinematics-v1",
        shield_horizon=4,
        shield_aabb_frames=(
            ((96.0, 390.0, 100.0, 394.0),),
            ((97.0, 391.0, 101.0, 395.0),),
            (),
            (),
        ),
        shield_laser_frames=(
            (),
            (),
            ((192.0, 120.0, 1.5, 250.0, 20.0, 12.0),),
            (),
        ),
    )


def _history_episode(root, *, mirrored: bool = False):
    recorder = CorpusRecorder(
        root,
        RunMetadata(root.name, "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    sign = -1.0 if mirrored else 1.0
    rows = (
        (
            _snapshot(0, x=100.0),
            _with_primitives(_decision(
                "ok", current="stay", published="left", legal=("left", "right")
            )),
        ),
        (
            _snapshot(1, x=100.0 + 4.0 * sign),
            _with_primitives(_decision(
                "ok", current="left", published="stay", legal=("stay", "left")
            )),
        ),
        (
            _snapshot(2, x=100.0 + 4.0 * sign),
            _with_primitives(_decision(
                "ok", current="stay", published="left", legal=("left", "stay")
            )),
        ),
        (
            _snapshot(3, player_state=2, lives=1, x=100.0),
            _decision("physical-hit", current="left"),
        ),
        (
            _snapshot(4, player_state=2, lives=1, x=100.0),
            _decision("player-not-active", current="stay"),
        ),
        (
            _snapshot(5, lives=1, x=100.0, in_menu=True),
            _decision("passive", current="stay"),
        ),
    )
    for snapshot, decision in rows:
        recorder.record(snapshot, decision)
    return recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 1,
    })


def test_fixed_history_is_causal_unpadded_and_reproduces_current_rows(tmp_path) -> None:
    run = _history_episode(tmp_path / "episode")

    dataset = load_history_probe_dataset(
        (run,), horizons=(1,), history_length=1
    )
    view = dataset.horizons[0]

    assert dataset.feature_schema == HISTORY_FEATURE_SCHEMA
    assert dataset.feature_names == history_feature_names(1)
    assert view.all_current_rows == 3
    assert view.rows == 2
    assert view.features.shape == (2, 2 * len(PROBE_FEATURE_NAMES))
    assert np.array_equal(
        view.features[:, -len(PROBE_FEATURE_NAMES):], view.current_features
    )
    assert np.array_equal(
        view.current_features, view.all_current_features[view.all_current_row_indices]
    )
    assert np.array_equal(
        view.hit_labels, view.all_current_hit_labels[view.all_current_row_indices]
    )
    assert tuple(view.all_current_row_indices) == (1, 2)
    assert tuple(view.history_elapsed_game_frames) == (1, 1)


def test_fixed_history_resets_at_non_unit_policy_interval(tmp_path) -> None:
    run = _probe_episode(tmp_path / "episode")

    with pytest.raises(ValueError, match="empty view"):
        load_history_probe_dataset((run,), horizons=(1,), history_length=1)


def test_history_feature_names_have_fixed_oldest_to_current_order() -> None:
    names = history_feature_names(2)

    width = len(PROBE_FEATURE_NAMES)
    assert names[0] == f"history_minus_2:{PROBE_FEATURE_NAMES[0]}"
    assert names[width] == f"history_minus_1:{PROBE_FEATURE_NAMES[0]}"
    assert names[-width] == f"current:{PROBE_FEATURE_NAMES[0]}"
    assert not any(
        forbidden in name
        for name in names
        for forbidden in (
            "outcome",
            "hit",
            "lifecycle",
            "propensity",
            "baseline",
            "stage",
            "frame",
            "ecl",
            "rng",
            "future",
        )
    )
