from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import xgboost

from th06_rl.factual_hazard_model import fit_action_conditioned_hazard_models
from th06_rl.factual_probe_boundary_diagnostics import (
    BoundaryProbeDataset,
    load_boundary_probe_dataset,
)
from th06_rl.weighted_factual_hazard_model import (
    evaluate_weighted_action_conditioned_hazard_models,
    fit_weighted_action_conditioned_hazard_models,
    uniform_shield_importance_weights,
    weighted_hazard_predictions,
)

from tests.test_factual_probes import _probe_episode


def _collector_dataset(tmp_path, name: str) -> BoundaryProbeDataset:
    original = load_boundary_probe_dataset(
        (
            _probe_episode(tmp_path / f"{name}-a"),
            _probe_episode(tmp_path / f"{name}-b", mirrored=True),
        ),
        horizons=(1,),
    )
    views = []
    for view in original.horizons:
        probabilities = np.asarray([
            (
                0.8 + 0.2 / count
                if published == baseline
                else 0.2 / count
            )
            for published, baseline, count in zip(
                view.published_actions,
                view.baseline_actions,
                view.shield_action_counts,
                strict=True,
            )
        ], dtype=np.float64)
        views.append(replace(view, behavior_probabilities=probabilities))
    return BoundaryProbeDataset(original.factual, tuple(views))


def _fit(dataset: BoundaryProbeDataset):
    return fit_weighted_action_conditioned_hazard_models(
        dataset,
        horizon=1,
        uniform_mixture_probability=0.2,
        maximum_importance_weight=5.0,
        probability_tolerance=1e-12,
        boosted_rounds=4,
        maximum_depth=2,
        learning_rate=0.1,
        minimum_child_weight=1.0,
        l2_leaf_regularization=1.0,
        maximum_histogram_bins=16,
        seed=11,
        expected_xgboost_version=xgboost.__version__,
    )


def test_uniform_shield_weights_are_exact_and_bounded() -> None:
    view = SimpleNamespace(
        rows=3,
        shield_action_counts=np.asarray([2, 2, 1], dtype=np.int64),
        behavior_probabilities=np.asarray([0.9, 0.1, 1.0]),
        published_actions=("left", "right", "stay"),
        baseline_actions=("left", "left", "stay"),
    )

    weights, summary = uniform_shield_importance_weights(
        view,
        uniform_mixture_probability=0.2,
        maximum_weight=5.0,
        probability_tolerance=1e-12,
    )

    assert np.allclose(weights, (5.0 / 9.0, 5.0, 1.0))
    assert summary["maximum"] == 5.0
    assert summary["nonbaseline_rows"] == 1
    assert summary["maximum_collector_probability_absolute_error"] == 0.0


def test_uniform_shield_weights_reject_a_false_propensity() -> None:
    view = SimpleNamespace(
        rows=1,
        shield_action_counts=np.asarray([2], dtype=np.int64),
        behavior_probabilities=np.asarray([0.2]),
        published_actions=("right",),
        baseline_actions=("left",),
    )

    with pytest.raises(ValueError, match="collector formula"):
        uniform_shield_importance_weights(
            view,
            uniform_mixture_probability=0.2,
            maximum_weight=5.0,
            probability_tolerance=1e-12,
        )


def test_weighted_hazard_fit_is_deterministic_and_propensity_is_not_a_feature(
    tmp_path,
) -> None:
    dataset = _collector_dataset(tmp_path, "train")
    first = _fit(dataset)
    second = _fit(dataset)

    assert first == second
    assert first["training_proper_score"] == (
        "uniform-observed-shield-target-importance-weighted-row-brier"
    )
    assert "propensity" not in " ".join(
        first["models"]["full_current_root_action"]["feature_names"]
    )
    assert "shield_action_count" in first["models"][
        "full_current_root_action"
    ]["feature_names"]
    probabilities, bounds = weighted_hazard_predictions(
        first,
        dataset.horizons[0].features,
        model_name="full_current_root_action",
    )
    assert np.all((0.0 <= probabilities) & (probabilities <= 1.0))
    assert bounds["clipped_rows"] == 0


def test_weighted_hazard_evaluation_never_admits_policy_or_value(tmp_path) -> None:
    train = _collector_dataset(tmp_path, "train")
    evaluation = _collector_dataset(tmp_path, "evaluation")
    state = _fit(train)
    frozen = fit_action_conditioned_hazard_models(
        train.factual,
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

    result = evaluate_weighted_action_conditioned_hazard_models(
        state,
        frozen,
        evaluation,
        uniform_mixture_probability=0.2,
        maximum_importance_weight=5.0,
        probability_tolerance=1e-12,
        bootstrap_samples=20,
        bootstrap_seed=13,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_low_propensity_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_target_gain_episodes=1,
        minimum_overall_episodes_favoring_full=1,
        minimum_nonbaseline_episodes_favoring_full=1,
        minimum_low_propensity_episodes_favoring_full=1,
        minimum_prefirst_episodes_favoring_full=1,
        weighted_calibration_in_the_large_absolute_max=1.0,
        weighted_full_ece_over_state_only_max=1.0,
        logged_calibration_in_the_large_absolute_max=1.0,
        maximum_raw_clipped_fraction=1.0,
    )

    assert result["summary"]["independent_confirmation"] is False
    assert result["summary"]["counterfactual_successors"] is False
    assert result["summary"]["propensity_is_actor_input"] is False
    assert result["summary"]["history_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
