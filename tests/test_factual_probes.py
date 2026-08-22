from __future__ import annotations

import numpy as np

from th06_rl.corpus import CorpusRecorder, RunMetadata
from th06_rl.factual_probes import (
    PROBE_FEATURE_NAMES,
    action_conditioned_probe_features,
    evaluate_factual_probe_models,
    fit_factual_probe_models,
    load_factual_probe_dataset,
)

from tests.test_episode_dataset import _decision, _snapshot


def _probe_episode(root, *, mirrored: bool = False):
    recorder = CorpusRecorder(
        root,
        RunMetadata(root.name, "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    sign = -1.0 if mirrored else 1.0
    rows = (
        (
            _snapshot(0, x=100.0),
            _decision("ok", current="stay", published="left", legal=("left", "right")),
        ),
        (
            _snapshot(1, x=100.0 + 4.0 * sign),
            _decision("input-lease", current="right", published="left", legal=("left",)),
        ),
        (
            _snapshot(2, x=100.0 + 2.0 * sign),
            _decision("ok", current="left", published="stay", legal=("stay", "left")),
        ),
        (
            _snapshot(3, player_state=2, lives=1, x=100.0 + 2.0 * sign),
            _decision("physical-hit", current="stay"),
        ),
        (
            _snapshot(4, player_state=2, lives=1, x=100.0 + 2.0 * sign),
            _decision("player-not-active", current="stay"),
        ),
        (
            _snapshot(5, lives=1, x=100.0 + 2.0 * sign),
            _decision("ok", current="stay", published="stay", legal=("stay",)),
        ),
        (
            _snapshot(6, lives=1, x=100.0 + 2.0 * sign, in_menu=True),
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


def test_probe_uses_executed_one_frame_action_and_fixed_game_horizons(tmp_path) -> None:
    run = _probe_episode(tmp_path / "episode")

    dataset = load_factual_probe_dataset((run,), horizons=(1, 2))

    assert dataset.dynamics.rows == 3
    assert dataset.dynamics.published_actions[0] == "left"
    assert dataset.dynamics.executed_actions[0] == "right"
    assert tuple(dataset.dynamics.deltas[0]) == (4.0, 0.0)
    one = dataset.horizons[0]
    assert one.horizon == 1
    assert tuple(one.hit_labels) == (False, True, False)
    assert tuple(one.shield_collapse_labels) == (True, False, False)
    two = dataset.horizons[1]
    assert tuple(two.hit_labels) == (False, True)
    assert tuple(two.shield_collapse_labels) == (True, False)


def test_probe_features_are_current_root_and_action_conditioned_only(tmp_path) -> None:
    dataset = load_factual_probe_dataset(
        (_probe_episode(tmp_path / "episode"),), horizons=(1,)
    )
    features = dataset.horizons[0].features[0]

    assert features.shape == (len(PROBE_FEATURE_NAMES),)
    assert np.all(np.isfinite(features))
    assert not any(
        forbidden in name
        for name in PROBE_FEATURE_NAMES
        for forbidden in ("stage", "boss", "spell", "ecl", "rng", "run", "future")
    )


def test_probe_rejects_an_action_outside_the_observed_shield(tmp_path) -> None:
    run = _probe_episode(tmp_path / "episode")
    # Construct through the public feature function using the already validated
    # portable root shape retained by the loader's fixture contract.
    from th06_rl.episode_dataset import iter_decision_epochs

    root = next(iter_decision_epochs(run)).observation
    try:
        action_conditioned_probe_features(root, "up")
    except ValueError as error:
        assert "outside" in str(error)
    else:
        raise AssertionError("probe admitted an unobserved action")


def test_ridge_probe_and_episode_bootstrap_are_deterministic(tmp_path) -> None:
    train = load_factual_probe_dataset(
        (
            _probe_episode(tmp_path / "train-a"),
            _probe_episode(tmp_path / "train-b", mirrored=True),
        ),
        horizons=(1,),
    )
    validation = load_factual_probe_dataset(
        (
            _probe_episode(tmp_path / "validation-a"),
            _probe_episode(tmp_path / "validation-b", mirrored=True),
        ),
        horizons=(1,),
    )
    state = fit_factual_probe_models(train, ridge_l2=0.01)
    kwargs = {
        "dynamics_mse_ratio_max": 2.0,
        "execution_match_rate_min": 0.0,
        "mismatch_rows_min": 1,
        "mismatch_mse_ratio_max": 2.0,
        "minimum_train_positives": 1,
        "minimum_validation_positives": 1,
        "minimum_validation_negatives": 1,
        "bootstrap_samples": 100,
        "bootstrap_seed": 7,
    }

    first = evaluate_factual_probe_models(state, validation, **kwargs)
    second = evaluate_factual_probe_models(state, validation, **kwargs)

    assert first == second
    assert first["dynamics"]["published_executed_mismatch_rows"] == 2
    assert first["horizons"]["1"]["targets"]["hit"]["candidate"]["positives"] == 2
    assert first["summary"]["history_admitted"] is False
