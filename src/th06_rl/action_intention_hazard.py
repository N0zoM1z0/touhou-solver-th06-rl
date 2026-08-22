"""Direct-Brier models for randomized fixed-horizon action intentions."""

from __future__ import annotations

from collections import Counter

import numpy as np
import xgboost as xgb

from .actions import ACTION_NAMES
from .action_intention_dataset import ActionIntentionDataset, DATASET_SCHEMA
from .factual_hazard_model import _model_state, _raw_predictions, _train_model
from .factual_probe_boundary_diagnostics import _calibration_summary, _stratum_result
from .factual_probe_diagnostics import (
    STATE_ONLY_FEATURE_INDICES,
    STATE_ONLY_FEATURE_NAMES,
)
from .factual_probes import (
    PROBE_FEATURE_NAMES,
    PROBE_FEATURE_SCHEMA,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


FIT_SCHEMA = "th06-rl-h12-action-intention-hazard-fit-v1"
EVALUATION_SCHEMA = "th06-rl-h12-action-intention-hazard-evaluation-v1"
MODEL_KIND = "paired-depth3-gradient-boosted-direct-brier-regressor"
TARGET = "physical-hit-during-randomized-h12-intention-before-next-assignment"


def intention_hazard_predictions(
    state: dict[str, object],
    features: np.ndarray,
    *,
    model_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    if state.get("schema") != FIT_SCHEMA:
        raise ValueError("action-intention hazard fit schema mismatch")
    if state.get("xgboost_version") != xgb.__version__:
        raise ValueError("action-intention hazard XGBoost version mismatch")
    if model_name == "full_group_start_action":
        selected = features
        names = PROBE_FEATURE_NAMES
    elif model_name == "state_only":
        selected = features[:, STATE_ONLY_FEATURE_INDICES]
        names = STATE_ONLY_FEATURE_NAMES
    else:
        raise ValueError(f"unknown action-intention hazard model: {model_name}")
    raw = _raw_predictions(
        state["models"][model_name],
        selected,
        expected_feature_names=names,
    )
    probabilities = np.clip(raw, 0.0, 1.0)
    clipped = raw != probabilities
    return probabilities, {
        "raw_minimum": float(np.min(raw)),
        "raw_maximum": float(np.max(raw)),
        "clipped_rows": int(np.sum(clipped)),
        "clipped_fraction": float(np.mean(clipped)),
    }


def fit_action_intention_hazard_models(
    dataset: ActionIntentionDataset,
    *,
    boosted_rounds: int,
    maximum_depth: int,
    learning_rate: float,
    minimum_child_weight: float,
    l2_leaf_regularization: float,
    maximum_histogram_bins: int,
    seed: int,
    expected_xgboost_version: str,
) -> dict[str, object]:
    """Fit paired full/state-only models once on complete train episodes."""
    if dataset.schema != DATASET_SCHEMA or dataset.exposure_roots != 12:
        raise ValueError("action-intention train dataset identity changed")
    if xgb.__version__ != expected_xgboost_version:
        raise ValueError("preregistered XGBoost version is unavailable")
    if (
        boosted_rounds <= 0
        or maximum_depth <= 0
        or not 0.0 < learning_rate <= 1.0
        or minimum_child_weight <= 0.0
        or l2_leaf_regularization < 0.0
        or maximum_histogram_bins < 2
    ):
        raise ValueError("action-intention fit settings are invalid")
    labels = dataset.labels
    positives = int(np.sum(labels))
    if positives == 0 or positives == dataset.rows:
        raise ValueError("action-intention train target requires both classes")
    prevalence = float(np.mean(labels))
    parameters: dict[str, object] = {
        "booster": "gbtree",
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "grow_policy": "depthwise",
        "max_depth": maximum_depth,
        "eta": learning_rate,
        "min_child_weight": minimum_child_weight,
        "lambda": l2_leaf_regularization,
        "alpha": 0.0,
        "gamma": 0.0,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "max_bin": maximum_histogram_bins,
        "base_score": prevalence,
        "seed": seed,
        "nthread": 1,
        "device": "cpu",
        "verbosity": 0,
    }
    full = _train_model(
        dataset.features,
        labels,
        feature_names=PROBE_FEATURE_NAMES,
        parameters=parameters,
        boosted_rounds=boosted_rounds,
    )
    state_features = dataset.features[:, STATE_ONLY_FEATURE_INDICES]
    state_only = _train_model(
        state_features,
        labels,
        feature_names=STATE_ONLY_FEATURE_NAMES,
        parameters=parameters,
        boosted_rounds=boosted_rounds,
    )
    fitted: dict[str, object] = {
        "schema": FIT_SCHEMA,
        "model": MODEL_KIND,
        "dataset_schema": DATASET_SCHEMA,
        "feature_schema": PROBE_FEATURE_SCHEMA,
        "target": TARGET,
        "exposure_roots": dataset.exposure_roots,
        "training_proper_score": "mean-unweighted-group-brier",
        "xgboost_version": xgb.__version__,
        "parameters": parameters,
        "boosted_rounds": boosted_rounds,
        "models": {
            "full_group_start_action": _model_state(
                full, feature_names=PROBE_FEATURE_NAMES
            ),
            "state_only": _model_state(
                state_only, feature_names=STATE_ONLY_FEATURE_NAMES
            ),
        },
        "train": {
            "episodes": len(dataset.episode_ids),
            "rows": dataset.rows,
            "positives": positives,
            "negatives": dataset.rows - positives,
            "prevalence": prevalence,
        },
    }
    full_probabilities, full_bounds = intention_hazard_predictions(
        fitted, dataset.features, model_name="full_group_start_action"
    )
    state_probabilities, state_bounds = intention_hazard_predictions(
        fitted, dataset.features, model_name="state_only"
    )
    constant = np.full(dataset.rows, prevalence, dtype=np.float64)
    fitted["train"]["metrics"] = {
        "full_group_start_action": _binary_metrics(full_probabilities, labels),
        "state_only": _binary_metrics(state_probabilities, labels),
        "constant_train_prevalence": _binary_metrics(constant, labels),
    }
    fitted["train"]["raw_probability_bounds"] = {
        "full_group_start_action": full_bounds,
        "state_only": state_bounds,
    }
    return fitted


def _action_support(dataset: ActionIntentionDataset) -> list[dict[str, object]]:
    actions = np.asarray(dataset.intended_actions, dtype=object)
    rows = []
    for action in ACTION_NAMES:
        mask = actions == action
        rows.append({
            "action": action,
            "rows": int(np.sum(mask)),
            "positives": int(np.sum(dataset.labels[mask])),
            "episode_count": len(set(int(value) for value in dataset.episode_indices[mask])),
            "minimum_assignment_probability": (
                None if not np.any(mask)
                else float(np.min(dataset.assignment_probabilities[mask]))
            ),
            "maximum_assignment_probability": (
                None if not np.any(mask)
                else float(np.max(dataset.assignment_probabilities[mask]))
            ),
        })
    return rows


def evaluate_action_intention_hazard_models(
    state: dict[str, object],
    dataset: ActionIntentionDataset,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_validation_episodes: int,
    minimum_validation_positives: int,
    minimum_validation_negatives: int,
    minimum_positive_episodes: int,
    minimum_episodes_favoring_full: int,
    maximum_calibration_in_the_large_absolute: float,
    maximum_expected_calibration_error: float,
    maximum_full_ece_over_state_only: float,
    maximum_raw_clipped_fraction: float,
) -> dict[str, object]:
    """Evaluate the frozen fit once on untouched complete validation episodes."""
    if (
        state.get("target") != TARGET
        or dataset.schema != DATASET_SCHEMA
        or dataset.exposure_roots != state.get("exposure_roots")
    ):
        raise ValueError("action-intention evaluation identity changed")
    labels = dataset.labels
    full, full_bounds = intention_hazard_predictions(
        state, dataset.features, model_name="full_group_start_action"
    )
    state_only, state_bounds = intention_hazard_predictions(
        state, dataset.features, model_name="state_only"
    )
    constant = np.full(
        dataset.rows,
        float(state["train"]["prevalence"]),
        dtype=np.float64,
    )
    all_rows = np.ones(dataset.rows, dtype=np.bool_)
    overall = _stratum_result(
        all_rows,
        full=full,
        state_only=state_only,
        labels=labels,
        episode_indices=dataset.episode_indices,
        episode_ids=dataset.episode_ids,
        calibration_bins=calibration_bins,
    )
    full_minus_state = _episode_bootstrap_brier_delta(
        full,
        state_only,
        labels,
        dataset.episode_indices,
        episode_count=len(dataset.episode_ids),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    full_minus_constant = _episode_bootstrap_brier_delta(
        full,
        constant,
        labels,
        dataset.episode_indices,
        episode_count=len(dataset.episode_ids),
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    full_calibration = _calibration_summary(full, labels, bins=calibration_bins)
    state_calibration = _calibration_summary(
        state_only, labels, bins=calibration_bins
    )
    positives_by_episode = np.bincount(
        dataset.episode_indices,
        weights=labels.astype(np.int64),
        minlength=len(dataset.episode_ids),
    )
    support = _action_support(dataset)
    support_sufficient = (
        len(dataset.episode_ids) >= minimum_validation_episodes
        and int(np.sum(labels)) >= minimum_validation_positives
        and dataset.rows - int(np.sum(labels)) >= minimum_validation_negatives
        and int(np.sum(positives_by_episode > 0)) >= minimum_positive_episodes
        and all(int(row["rows"]) > 0 for row in support)
    )
    action_signal = (
        float(full_minus_state["upper_95"]) < 0.0
        and int(overall["episodes_favoring_full"])
        >= minimum_episodes_favoring_full
    )
    proper_score = float(full_minus_constant["upper_95"]) < 0.0
    calibration_ready = (
        abs(float(full_calibration["calibration_in_the_large"]))
        <= maximum_calibration_in_the_large_absolute
        and float(full_calibration["expected_calibration_error"])
        <= maximum_expected_calibration_error
        and float(full_calibration["expected_calibration_error"])
        <= float(state_calibration["expected_calibration_error"])
        + maximum_full_ece_over_state_only
    )
    bounded = (
        float(full_bounds["clipped_fraction"]) <= maximum_raw_clipped_fraction
        and float(state_bounds["clipped_fraction"]) <= maximum_raw_clipped_fraction
    )
    gates = {
        "validation_support_sufficient": support_sufficient,
        "full_improves_train_prevalence_constant": proper_score,
        "incremental_randomized_action_signal": action_signal,
        "calibration_readiness": calibration_ready,
        "bounded_probability_surface": bounded,
    }
    selected = all(gates.values())
    return {
        "schema": EVALUATION_SCHEMA,
        "model": MODEL_KIND,
        "target": TARGET,
        "inventory": list(dataset.inventory),
        "metrics": {
            "full_group_start_action": _binary_metrics(full, labels),
            "state_only": _binary_metrics(state_only, labels),
            "constant_train_prevalence": _binary_metrics(constant, labels),
        },
        "calibration": {
            "full_group_start_action": full_calibration,
            "state_only": state_calibration,
        },
        "raw_probability_bounds": {
            "full_group_start_action": full_bounds,
            "state_only": state_bounds,
        },
        "bootstrap": {
            "full_minus_state_only_brier": full_minus_state,
            "full_minus_constant_brier": full_minus_constant,
        },
        "overall_action_ablation": overall,
        "action_support": support,
        "lifecycle": {
            "positive_after_control_dead_end": int(np.sum(
                labels & dataset.control_dead_ends
            )),
            "accepted_negative_with_override": int(np.sum(
                (~labels) & dataset.any_overrides
            )),
            "hit_offsets": dict(sorted(Counter(
                offset for offset in dataset.hit_offsets if offset is not None
            ).items())),
        },
        "gates": gates,
        "summary": {
            "decision": (
                "select-h12-intention-hazard-for-export-and-wine-canary-preregistration"
                if selected
                else "reject-h12-action-intention-hazard"
            ),
            "independent_heldout_episodes": True,
            "counterfactual_successors": False,
            "causal_estimand": "randomized-intention-to-treat-with-declared-mediators",
            "online_policy_admitted": False,
            "history_admitted": False,
            "value_learning_admitted": False,
        },
    }
