from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from th06_rl.actions import ACTION_NAMES
from th06_rl.bc_features import features_from_portable_root
from th06_rl.bc_residual_diagnostics import (
    _reactive_structure,
    scaled_probabilities,
    select_ablation,
    train_optimal_logit_scale,
)
from th06_rl.bc_training import BehaviorDataset


def _dataset(
    features: np.ndarray,
    targets: np.ndarray,
    legal_masks: np.ndarray,
    baseline_targets: np.ndarray,
) -> BehaviorDataset:
    rows = int(targets.shape[0])
    return BehaviorDataset(
        episode_ids=("episode",),
        features=features,
        targets=targets,
        legal_masks=legal_masks,
        baseline_targets=baseline_targets,
        episode_indices=np.zeros(rows, dtype=np.int64),
        inventory=({},),
    )


def test_positive_logit_scale_is_train_only_nll_optimum() -> None:
    logits = np.tile(np.asarray([[0.2, 0.0]]), (8, 1))
    targets = np.asarray([0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int64)
    dataset = _dataset(
        np.zeros((8, 1)),
        targets,
        np.ones((8, 2), dtype=np.bool_),
        np.zeros(8, dtype=np.int64),
    )

    fit = train_optimal_logit_scale(logits, dataset)

    assert fit["root_bracketed"] is True
    assert np.isclose(fit["logit_scale"], np.log(3.0) / 0.2)
    assert abs(float(fit["derivative"])) < 1e-12


def test_scaling_preserves_mask_and_argmax() -> None:
    logits = np.asarray([[2.0, 1.0, 100.0], [-1.0, 3.0, 2.0]])
    masks = np.asarray([[True, True, False], [True, True, True]])

    original = scaled_probabilities(logits, masks, 1.0)
    scaled = scaled_probabilities(logits, masks, 2.5)

    assert np.array_equal(np.argmax(original, axis=1), np.argmax(scaled, axis=1))
    assert scaled[0, 2] == 0.0
    assert np.allclose(np.sum(scaled, axis=1), 1.0)


def test_feature_only_reactive_reconstruction_uses_nonlinear_ranking() -> None:
    root = SimpleNamespace(
        player_x=192.0,
        player_y=224.0,
        power=64,
        bullet_count=10,
        laser_count=0,
        current_action="stay",
        locally_admissible_actions=("left", "right"),
        shield_action_evaluations=(
            ("left", 5.0, 50.0, 224.0),
            ("right", 5.0, 100.0, 224.0),
        ),
    )
    features = np.asarray([features_from_portable_root(root)])
    left = ACTION_NAMES.index("left")
    right = ACTION_NAMES.index("right")
    legal = np.zeros((1, len(ACTION_NAMES)), dtype=np.bool_)
    legal[0, [left, right]] = True
    probabilities = np.zeros((1, len(ACTION_NAMES)), dtype=np.float64)
    probabilities[0, right] = 1.0
    dataset = _dataset(
        features,
        np.asarray([right]),
        legal,
        np.asarray([right]),
    )

    observed = _reactive_structure(probabilities, dataset)

    assert observed["feature_only_reconstruction_accuracy"] == 1.0
    assert observed["decision_stage"]["boundary_reserve"]["rows"] == 1


def test_selection_uses_scale_only_when_train_calibration_is_repaired() -> None:
    selected, _reason = select_ablation(
        current_feature_reconstruction_exact=True,
        scaled_train_nll=1.0,
        unscaled_train_nll=1.1,
        scaled_train_ece=0.01,
        calibration_limit=0.03,
    )
    assert selected == "train-only-scalar-calibration"

    selected, _reason = select_ablation(
        current_feature_reconstruction_exact=True,
        scaled_train_nll=1.0,
        unscaled_train_nll=1.1,
        scaled_train_ece=0.04,
        calibration_limit=0.03,
    )
    assert selected == "small-current-observation-mlp"

    selected, _reason = select_ablation(
        current_feature_reconstruction_exact=False,
        scaled_train_nll=1.0,
        unscaled_train_nll=1.1,
        scaled_train_ece=0.01,
        calibration_limit=0.03,
    )
    assert selected == "stop-diagnosis-current-feature-invariant-failed"
