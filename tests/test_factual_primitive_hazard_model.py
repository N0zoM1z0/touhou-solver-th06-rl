from __future__ import annotations

import numpy as np
import torch
import xgboost

from th06_rl.factual_hazard_model import fit_action_conditioned_hazard_models
from th06_rl.factual_primitive_dataset import load_primitive_probe_dataset
from th06_rl.factual_primitive_hazard_model import (
    benchmark_primitive_hazard,
    evaluate_primitive_hazard_models,
    fit_primitive_hazard_models,
    primitive_hazard_predictions,
)

from tests.test_factual_history_dataset import _history_episode


def _dataset(tmp_path, name):
    return load_primitive_probe_dataset(
        (
            _history_episode(tmp_path / f"{name}-a"),
            _history_episode(tmp_path / f"{name}-b", mirrored=True),
        ),
        horizons=(1,),
        token_cap=8,
    )


def _fit(dataset):
    return fit_primitive_hazard_models(
        dataset,
        horizon=1,
        token_cap=8,
        token_hidden=4,
        head_hidden_1=6,
        head_hidden_2=4,
        epochs=2,
        batch_size=2,
        learning_rate=0.01,
        weight_decay=0.0001,
        gradient_clip_norm=5.0,
        seed=23,
        threads=2,
        expected_torch_version=torch.__version__,
    )


def test_primitive_hazard_is_deterministic_and_permutation_invariant(
    tmp_path,
) -> None:
    dataset = _dataset(tmp_path, "train")
    first = _fit(dataset)
    second = _fit(dataset)

    assert first == second
    assert set(first["models"]) == {
        "object_full",
        "scalar_only",
        "object_current_action_ablated",
    }
    assert first["shared_initialization"] is True
    assert first["shared_minibatch_order"] is True
    assert first["parameter_count_per_model"] == 275
    view = dataset.horizons[0]
    original, surface = primitive_hazard_predictions(
        first,
        view.scalar_features,
        view.primitive_tokens,
        view.primitive_masks,
        model_name="object_full",
        batch_size=4,
    )
    permutation = np.asarray([2, 0, 1, 3, 4, 5, 6, 7])
    shuffled, _ = primitive_hazard_predictions(
        first,
        view.scalar_features,
        view.primitive_tokens[:, permutation],
        view.primitive_masks[:, permutation],
        model_name="object_full",
        batch_size=4,
    )
    assert np.allclose(original, shuffled, atol=1e-8, rtol=0.0)
    assert np.all((0.0 <= original) & (original <= 1.0))
    assert surface["saturated_rows"] == 0


def test_primitive_evaluation_never_admits_policy_or_value(tmp_path) -> None:
    train = _dataset(tmp_path, "train")
    evaluation = _dataset(tmp_path, "evaluation")
    state = _fit(train)
    state["train"]["inference_benchmark"] = benchmark_primitive_hazard(
        state,
        train,
        batch_rows=4,
        warmup_repetitions=2,
        measured_repetitions=5,
        threads=1,
    )
    view = train.horizons[0]
    frozen = fit_action_conditioned_hazard_models(
        type("Factual", (), {"horizons": (type("View", (), {
            "horizon": 1,
            "features": view.current_features,
            "hit_labels": view.hit_labels,
            "rows": view.rows,
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
    result = evaluate_primitive_hazard_models(
        state,
        frozen,
        evaluation,
        prediction_batch_size=4,
        bootstrap_samples=20,
        bootstrap_seed=29,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_low_propensity_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_object_gain_episodes=1,
        minimum_overall_episodes_favoring_full=1,
        minimum_nonbaseline_episodes_favoring_full=1,
        minimum_low_propensity_episodes_favoring_full=1,
        minimum_prefirst_episodes_favoring_full=1,
        calibration_in_the_large_absolute_max=1.0,
        full_ece_over_action_ablated_max=1.0,
        maximum_saturated_fraction=1.0,
        maximum_parameter_count=100_000,
        maximum_batch18_p99_ms=1000.0,
    )

    assert result["summary"]["independent_confirmation"] is False
    assert result["summary"]["counterfactual_successors"] is False
    assert result["summary"]["object_set_tested"] is True
    assert result["summary"]["history_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
