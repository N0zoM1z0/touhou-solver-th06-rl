from __future__ import annotations

import numpy as np
import xgboost

from th06_rl.factual_hazard_model import fit_action_conditioned_hazard_models
from th06_rl.factual_history_dataset import load_history_probe_dataset
from th06_rl.factual_history_hazard_model import (
    evaluate_history_hazard_models,
    fit_history_hazard_models,
    history_hazard_predictions,
)

from tests.test_factual_history_dataset import _history_episode


def _dataset(tmp_path, name):
    return load_history_probe_dataset(
        (
            _history_episode(tmp_path / f"{name}-a"),
            _history_episode(tmp_path / f"{name}-b", mirrored=True),
        ),
        horizons=(1,),
        history_length=1,
    )


def _fit(dataset):
    return fit_history_hazard_models(
        dataset,
        horizon=1,
        history_length=1,
        boosted_rounds=4,
        maximum_depth=2,
        learning_rate=0.1,
        minimum_child_weight=1.0,
        l2_leaf_regularization=1.0,
        maximum_histogram_bins=16,
        seed=17,
        threads=2,
        expected_xgboost_version=xgboost.__version__,
    )


def test_history_hazard_is_deterministic_and_has_three_exact_comparators(
    tmp_path,
) -> None:
    dataset = _dataset(tmp_path, "train")
    first = _fit(dataset)
    second = _fit(dataset)

    assert first == second
    assert set(first["models"]) == {
        "history_full",
        "current_only",
        "history_current_action_ablated",
    }
    assert len(first["models"]["history_full"]["feature_names"]) == 30
    assert len(first["models"]["current_only"]["feature_names"]) == 15
    ablated = first["models"]["history_current_action_ablated"]["feature_names"]
    assert len(ablated) == 21
    assert not any(
        name.startswith("current:action_") for name in ablated
    )
    predictions, bounds = history_hazard_predictions(
        first,
        dataset.horizons[0].features,
        dataset.horizons[0].current_features,
        model_name="history_full",
    )
    assert np.all((0.0 <= predictions) & (predictions <= 1.0))
    assert bounds["clipped_rows"] == 0


def test_history_hazard_evaluation_never_admits_policy_or_value(tmp_path) -> None:
    train = _dataset(tmp_path, "train")
    evaluation = _dataset(tmp_path, "evaluation")
    state = _fit(train)
    frozen = fit_action_conditioned_hazard_models(
        type("Factual", (), {"horizons": (type("View", (), {
            "horizon": 1,
            "features": train.horizons[0].all_current_features,
            "hit_labels": train.horizons[0].all_current_hit_labels,
            "rows": train.horizons[0].all_current_rows,
        })(),)})(),
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
    result = evaluate_history_hazard_models(
        state,
        frozen,
        evaluation,
        bootstrap_samples=20,
        bootstrap_seed=19,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_low_propensity_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_temporal_gain_episodes=1,
        minimum_overall_episodes_favoring_full=1,
        minimum_nonbaseline_episodes_favoring_full=1,
        minimum_low_propensity_episodes_favoring_full=1,
        minimum_prefirst_episodes_favoring_full=1,
        calibration_in_the_large_absolute_max=1.0,
        full_ece_over_action_ablated_max=1.0,
        maximum_raw_clipped_fraction=1.0,
    )

    assert result["summary"]["independent_confirmation"] is False
    assert result["summary"]["counterfactual_successors"] is False
    assert result["summary"]["history_admitted"] is True
    assert result["summary"]["object_set_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
