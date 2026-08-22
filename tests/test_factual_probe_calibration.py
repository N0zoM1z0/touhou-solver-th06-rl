from __future__ import annotations

import numpy as np

from th06_rl.factual_probe_boundary_diagnostics import load_boundary_probe_dataset
from th06_rl.factual_probe_calibration import (
    _raw_full_scores,
    calibrated_predictions,
    evaluate_train_only_platt_calibrator,
    fit_train_only_platt_calibrator,
)
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


def test_train_only_platt_surface_is_monotone_and_deterministic(tmp_path) -> None:
    train = _dataset(tmp_path, "train")
    full = fit_factual_probe_models(train.factual, ridge_l2=0.01)
    settings = {
        "horizon": 1,
        "maximum_updates": 50,
        "minimum_updates": 1,
        "gradient_inf_tolerance": 1e-10,
        "maximum_line_search_steps": 20,
    }

    first = fit_train_only_platt_calibrator(full, train.factual, **settings)
    second = fit_train_only_platt_calibrator(full, train.factual, **settings)
    predictions = calibrated_predictions(
        full, first, train.horizons[0].features
    )
    raw = _raw_full_scores(full, 1, train.horizons[0].features)

    assert first == second
    assert first["optimization"]["converged"] is True
    assert first["raw_score_coefficients"]["slope"] > 0.0
    assert np.all((0.0 < predictions) & (predictions < 1.0))
    order = np.argsort(raw, kind="stable")
    assert np.all(np.diff(predictions[order]) >= 0.0)


def test_calibration_evaluation_never_admits_value_or_policy(tmp_path) -> None:
    train = _dataset(tmp_path, "train")
    validation = _dataset(tmp_path, "validation")
    full = fit_factual_probe_models(train.factual, ridge_l2=0.01)
    state_only = fit_state_only_probe_models(train.factual, ridge_l2=0.01)
    calibrator = fit_train_only_platt_calibrator(
        full,
        train.factual,
        horizon=1,
        maximum_updates=50,
        minimum_updates=1,
        gradient_inf_tolerance=1e-10,
        maximum_line_search_steps=20,
    )

    result = evaluate_train_only_platt_calibrator(
        full,
        state_only,
        calibrator,
        validation,
        bootstrap_samples=20,
        bootstrap_seed=5,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_episodes_favoring_calibrated=1,
        calibration_in_the_large_absolute_max=1.0,
        calibrated_ece_over_state_only_max=1.0,
    )

    assert result["horizon_game_frames"] == 1
    assert result["summary"]["independent_confirmation"] is False
    assert result["summary"]["history_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
