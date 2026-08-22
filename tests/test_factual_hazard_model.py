from __future__ import annotations

import copy

import numpy as np
import pytest
import xgboost

from th06_rl.factual_hazard_model import (
    evaluate_action_conditioned_hazard_models,
    fit_action_conditioned_hazard_models,
    hazard_predictions,
)
from th06_rl.factual_probe_boundary_diagnostics import load_boundary_probe_dataset
from th06_rl.factual_probe_diagnostics import fit_state_only_probe_models
from th06_rl.factual_probes import fit_factual_probe_models

from tests.test_factual_probes import _probe_episode


def _dataset(tmp_path, name):
    return load_boundary_probe_dataset(
        (
            _probe_episode(tmp_path / f"{name}-a"),
            _probe_episode(tmp_path / f"{name}-b", mirrored=True),
        ),
        horizons=(1,),
    )


def _fit(dataset):
    return fit_action_conditioned_hazard_models(
        dataset.factual,
        horizon=1,
        boosted_rounds=4,
        maximum_depth=2,
        learning_rate=0.1,
        minimum_child_weight=1.0,
        l2_leaf_regularization=1.0,
        maximum_histogram_bins=16,
        seed=7,
        expected_xgboost_version=xgboost.__version__,
    )


def test_direct_hazard_fit_is_deterministic_bounded_and_action_conditioned(
    tmp_path,
) -> None:
    dataset = _dataset(tmp_path, "train")
    first = _fit(dataset)
    second = _fit(dataset)

    assert first == second
    assert first["training_proper_score"] == "mean-unweighted-row-brier"
    assert first["models"]["full_current_root_action"]["model_sha256"] == (
        second["models"]["full_current_root_action"]["model_sha256"]
    )
    predictions, raw = hazard_predictions(
        first,
        dataset.horizons[0].features,
        model_name="full_current_root_action",
    )
    assert np.all((0.0 <= predictions) & (predictions <= 1.0))
    assert raw["clipped_rows"] == 0
    assert (
        first["train"]["metrics"]["full_current_root_action"]["brier"]
        < first["train"]["metrics"]["constant_prevalence"]["brier"]
    )


def test_direct_hazard_model_rejects_tampered_identity(tmp_path) -> None:
    dataset = _dataset(tmp_path, "train")
    state = _fit(dataset)
    tampered = copy.deepcopy(state)
    tampered["models"]["full_current_root_action"]["model_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="identity hash"):
        hazard_predictions(
            tampered,
            dataset.horizons[0].features,
            model_name="full_current_root_action",
        )


def test_direct_hazard_evaluation_never_admits_policy_or_value(tmp_path) -> None:
    train = _dataset(tmp_path, "train")
    validation = _dataset(tmp_path, "validation")
    state = _fit(train)
    frozen_full = fit_factual_probe_models(train.factual, ridge_l2=0.01)
    frozen_state = fit_state_only_probe_models(train.factual, ridge_l2=0.01)

    result = evaluate_action_conditioned_hazard_models(
        state,
        frozen_full,
        frozen_state,
        validation,
        bootstrap_samples=20,
        bootstrap_seed=9,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_overall_episodes_favoring_full=1,
        minimum_nonbaseline_episodes_favoring_full=1,
        minimum_prefirst_episodes_favoring_full=1,
        calibration_in_the_large_absolute_max=1.0,
        full_ece_over_state_only_max=1.0,
        maximum_raw_clipped_fraction=1.0,
    )

    assert result["horizon_game_frames"] == 1
    assert result["summary"]["independent_confirmation"] is False
    assert result["summary"]["counterfactual_successors"] is False
    assert result["summary"]["causal_action_value_claimed"] is False
    assert result["summary"]["history_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
