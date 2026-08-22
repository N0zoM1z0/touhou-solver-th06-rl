"""Direct Brier-trained action-conditioned factual HIT hazard models.

The model consumes only current-root portable facts and the action Wine
actually executed.  It predicts a fixed-horizon factual HIT under behavior
continuation; it neither constructs an unexecuted successor nor estimates a
causal or long-horizon action value.
"""

from __future__ import annotations

import base64
import hashlib

import numpy as np
import xgboost as xgb

from .factual_probe_boundary_diagnostics import (
    BoundaryProbeDataset,
    PROPENSITY_STRATA,
    _calibration_summary,
    _propensity_stratum,
    _stratum_result,
)
from .factual_probe_diagnostics import (
    STATE_ONLY_FEATURE_INDICES,
    STATE_ONLY_FEATURE_NAMES,
    _full_predictions,
    _state_only_predictions,
)
from .factual_probes import (
    FactualProbeDataset,
    PROBE_FEATURE_NAMES,
    PROBE_FEATURE_SCHEMA,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


HAZARD_FIT_SCHEMA = "th06-rl-l2f-action-conditioned-hazard-fit-v1"
HAZARD_EVALUATION_SCHEMA = "th06-rl-l2f-action-conditioned-hazard-evaluation-v1"
MODEL_KIND = "shared-depth3-gradient-boosted-brier-regressor"


def _view(dataset: FactualProbeDataset, horizon: int):
    try:
        return next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("hazard fit horizon is absent") from error


def _model_bytes(booster: xgb.Booster) -> bytes:
    raw = bytes(booster.save_raw(raw_format="json"))
    if not raw or not raw.startswith(b"{"):
        raise ValueError("hazard booster did not serialize as JSON")
    return raw


def _model_state(
    booster: xgb.Booster,
    *,
    feature_names: tuple[str, ...],
) -> dict[str, object]:
    raw = _model_bytes(booster)
    return {
        "feature_names": list(feature_names),
        "model_json_base64": base64.b64encode(raw).decode("ascii"),
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "model_bytes": len(raw),
        "boosted_rounds": booster.num_boosted_rounds(),
    }


def _load_booster(model: dict[str, object]) -> xgb.Booster:
    try:
        raw = base64.b64decode(str(model["model_json_base64"]), validate=True)
    except (KeyError, ValueError) as error:
        raise ValueError("hazard model encoding is malformed") from error
    if hashlib.sha256(raw).hexdigest() != model.get("model_sha256"):
        raise ValueError("hazard model identity hash mismatch")
    booster = xgb.Booster()
    booster.load_model(bytearray(raw))
    return booster


def _matrix(features: np.ndarray, feature_names: tuple[str, ...]) -> xgb.DMatrix:
    if (
        features.ndim != 2
        or features.shape[1] != len(feature_names)
        or not np.all(np.isfinite(features))
    ):
        raise ValueError("hazard features are malformed or non-finite")
    return xgb.DMatrix(features, feature_names=list(feature_names))


def _train_model(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    parameters: dict[str, object],
    boosted_rounds: int,
) -> xgb.Booster:
    matrix = _matrix(features, feature_names)
    matrix.set_label(labels.astype(np.float64))
    return xgb.train(
        dict(parameters),
        matrix,
        num_boost_round=boosted_rounds,
        verbose_eval=False,
    )


def _raw_predictions(
    model: dict[str, object],
    features: np.ndarray,
    *,
    expected_feature_names: tuple[str, ...],
) -> np.ndarray:
    if tuple(model.get("feature_names", ())) != expected_feature_names:
        raise ValueError("hazard model feature names changed")
    predictions = np.asarray(
        _load_booster(model).predict(_matrix(features, expected_feature_names)),
        dtype=np.float64,
    )
    if predictions.shape != (features.shape[0],) or not np.all(np.isfinite(predictions)):
        raise ValueError("hazard model produced malformed predictions")
    return predictions


def hazard_predictions(
    state: dict[str, object],
    features: np.ndarray,
    *,
    model_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return bounded probabilities and disclose every raw out-of-range row."""
    if state.get("schema") != HAZARD_FIT_SCHEMA:
        raise ValueError("hazard fit schema mismatch")
    if state.get("xgboost_version") != xgb.__version__:
        raise ValueError("hazard fit XGBoost version mismatch")
    if model_name == "full_current_root_action":
        selected = features
        feature_names = PROBE_FEATURE_NAMES
    elif model_name == "state_only":
        selected = features[:, STATE_ONLY_FEATURE_INDICES]
        feature_names = STATE_ONLY_FEATURE_NAMES
    else:
        raise ValueError(f"unknown hazard model: {model_name}")
    model = state["models"][model_name]
    raw = _raw_predictions(model, selected, expected_feature_names=feature_names)
    probabilities = np.clip(raw, 0.0, 1.0)
    clipped = raw != probabilities
    return probabilities, {
        "raw_minimum": float(np.min(raw)),
        "raw_maximum": float(np.max(raw)),
        "clipped_rows": int(np.sum(clipped)),
        "clipped_fraction": float(np.mean(clipped)),
    }


def fit_action_conditioned_hazard_models(
    dataset: FactualProbeDataset,
    *,
    horizon: int,
    boosted_rounds: int,
    maximum_depth: int,
    learning_rate: float,
    minimum_child_weight: float,
    l2_leaf_regularization: float,
    maximum_histogram_bins: int,
    seed: int,
    expected_xgboost_version: str,
) -> dict[str, object]:
    """Fit paired full and state-only Brier regressors using train episodes only."""
    if xgb.__version__ != expected_xgboost_version:
        raise ValueError("preregistered XGBoost version is unavailable")
    if (
        horizon <= 0
        or boosted_rounds <= 0
        or maximum_depth <= 0
        or not 0.0 < learning_rate <= 1.0
        or minimum_child_weight <= 0.0
        or l2_leaf_regularization < 0.0
        or maximum_histogram_bins < 2
    ):
        raise ValueError("hazard fit settings are invalid")
    view = _view(dataset, horizon)
    labels = view.hit_labels
    positives = int(np.sum(labels))
    if not positives or positives == view.rows:
        raise ValueError("hazard train target requires both classes")
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
    full_booster = _train_model(
        view.features,
        labels,
        feature_names=PROBE_FEATURE_NAMES,
        parameters=parameters,
        boosted_rounds=boosted_rounds,
    )
    state_features = view.features[:, STATE_ONLY_FEATURE_INDICES]
    state_booster = _train_model(
        state_features,
        labels,
        feature_names=STATE_ONLY_FEATURE_NAMES,
        parameters=parameters,
        boosted_rounds=boosted_rounds,
    )
    state: dict[str, object] = {
        "schema": HAZARD_FIT_SCHEMA,
        "model": MODEL_KIND,
        "feature_schema": PROBE_FEATURE_SCHEMA,
        "horizon_game_frames": horizon,
        "target": "physical-hit-within-fixed-horizon-under-behavior-continuation",
        "training_proper_score": "mean-unweighted-row-brier",
        "xgboost_version": xgb.__version__,
        "parameters": parameters,
        "boosted_rounds": boosted_rounds,
        "models": {
            "full_current_root_action": _model_state(
                full_booster, feature_names=PROBE_FEATURE_NAMES
            ),
            "state_only": _model_state(
                state_booster, feature_names=STATE_ONLY_FEATURE_NAMES
            ),
        },
        "train": {
            "rows": view.rows,
            "positives": positives,
            "negatives": view.rows - positives,
            "prevalence": prevalence,
        },
    }
    full, full_raw = hazard_predictions(
        state, view.features, model_name="full_current_root_action"
    )
    state_only, state_raw = hazard_predictions(
        state, view.features, model_name="state_only"
    )
    constant = np.full(view.rows, prevalence, dtype=np.float64)
    state["train"]["metrics"] = {
        "full_current_root_action": _binary_metrics(full, labels),
        "state_only": _binary_metrics(state_only, labels),
        "constant_prevalence": _binary_metrics(constant, labels),
    }
    state["train"]["raw_probability_bounds"] = {
        "full_current_root_action": full_raw,
        "state_only": state_raw,
    }
    return state


def _masked_bootstrap(
    mask: np.ndarray,
    *,
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    episode_count: int,
    samples: int,
    seed: int,
) -> dict[str, object] | None:
    if not np.any(mask):
        return None
    return _episode_bootstrap_brier_delta(
        candidate[mask],
        baseline[mask],
        labels[mask],
        episode_indices[mask],
        episode_count=episode_count,
        samples=samples,
        seed=seed,
    )


def _action_support(view) -> list[dict[str, object]]:
    rows = []
    for action in sorted(set(view.published_actions)):
        mask = np.asarray(
            [published == action for published in view.published_actions],
            dtype=np.bool_,
        )
        probabilities = view.behavior_probabilities[mask]
        rows.append({
            "action": action,
            "rows": int(np.sum(mask)),
            "positives": int(np.sum(view.hit_labels[mask])),
            "episode_count": len(set(int(value) for value in view.episode_indices[mask])),
            "minimum_behavior_probability": float(np.min(probabilities)),
            "maximum_behavior_probability": float(np.max(probabilities)),
        })
    return rows


def evaluate_action_conditioned_hazard_models(
    state: dict[str, object],
    frozen_full_state: dict[str, object],
    frozen_state_only_state: dict[str, object],
    dataset: BoundaryProbeDataset,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_overall_positives: int,
    minimum_overall_negatives: int,
    minimum_nonbaseline_positives: int,
    minimum_prefirst_hit_positives: int,
    minimum_overall_episodes_favoring_full: int,
    minimum_nonbaseline_episodes_favoring_full: int,
    minimum_prefirst_episodes_favoring_full: int,
    calibration_in_the_large_absolute_max: float,
    full_ece_over_state_only_max: float,
    maximum_raw_clipped_fraction: float,
) -> dict[str, object]:
    """Evaluate one frozen direct hazard fit on reused complete L2d episodes."""
    horizon = int(state["horizon_game_frames"])
    try:
        view = next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("hazard evaluation horizon is absent") from error
    labels = view.hit_labels
    full, full_raw = hazard_predictions(
        state, view.features, model_name="full_current_root_action"
    )
    state_only, state_raw = hazard_predictions(
        state, view.features, model_name="state_only"
    )
    frozen_full = _full_predictions(
        frozen_full_state, horizon, "hit", view.features
    )
    frozen_state = _state_only_predictions(
        frozen_state_only_state, horizon, "hit", view.features
    )
    all_rows = np.ones(view.rows, dtype=np.bool_)
    baseline_equal = np.asarray([
        published == baseline
        for published, baseline in zip(
            view.published_actions, view.baseline_actions, strict=True
        )
    ], dtype=np.bool_)
    lifecycle = np.asarray(view.lifecycle_strata, dtype=object)
    nonbaseline_mask = ~baseline_equal
    prefirst_mask = lifecycle == "pre-first-hit"

    def stratum(mask: np.ndarray) -> dict[str, object]:
        return _stratum_result(
            mask,
            full=full,
            state_only=state_only,
            labels=labels,
            episode_indices=view.episode_indices,
            episode_ids=view.episode_ids,
            calibration_bins=calibration_bins,
        )

    overall = stratum(all_rows)
    baseline = stratum(baseline_equal)
    nonbaseline = stratum(nonbaseline_mask)
    prefirst = stratum(prefirst_mask)
    episode_count = len(view.episode_ids)
    full_vs_state = _episode_bootstrap_brier_delta(
        full,
        state_only,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    full_vs_frozen = _episode_bootstrap_brier_delta(
        full,
        frozen_full,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    state_vs_frozen = _episode_bootstrap_brier_delta(
        state_only,
        frozen_state,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
    )
    nonbaseline_bootstrap = _masked_bootstrap(
        nonbaseline_mask,
        candidate=full,
        baseline=state_only,
        labels=labels,
        episode_indices=view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 3,
    )
    prefirst_bootstrap = _masked_bootstrap(
        prefirst_mask,
        candidate=full,
        baseline=state_only,
        labels=labels,
        episode_indices=view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 4,
    )
    metrics = {
        "full_current_root_action": _binary_metrics(full, labels),
        "state_only": _binary_metrics(state_only, labels),
        "frozen_l2_full": _binary_metrics(frozen_full, labels),
        "frozen_l2b_state_only": _binary_metrics(frozen_state, labels),
    }
    calibration = {
        "full_current_root_action": _calibration_summary(
            full, labels, bins=calibration_bins
        ),
        "state_only": _calibration_summary(
            state_only, labels, bins=calibration_bins
        ),
        "frozen_l2_full": _calibration_summary(
            frozen_full, labels, bins=calibration_bins
        ),
        "frozen_l2b_state_only": _calibration_summary(
            frozen_state, labels, bins=calibration_bins
        ),
    }
    positives = int(np.sum(labels))
    sufficient = (
        positives >= minimum_overall_positives
        and view.rows - positives >= minimum_overall_negatives
    )
    bounded = (
        float(full_raw["clipped_fraction"]) <= maximum_raw_clipped_fraction
        and float(state_raw["clipped_fraction"]) <= maximum_raw_clipped_fraction
    )
    direct_model_gain = float(full_vs_frozen["upper_95"]) < 0.0
    incremental_action = (
        sufficient
        and float(full_vs_state["upper_95"]) < 0.0
        and int(overall["episodes_favoring_full"])
        >= minimum_overall_episodes_favoring_full
    )
    nonbaseline_signal = (
        int(nonbaseline["positives"]) >= minimum_nonbaseline_positives
        and int(nonbaseline["episode_count"]) == episode_count
        and nonbaseline_bootstrap is not None
        and float(nonbaseline_bootstrap["upper_95"]) < 0.0
        and int(nonbaseline["episodes_favoring_full"])
        >= minimum_nonbaseline_episodes_favoring_full
    )
    lifecycle_signal = (
        int(prefirst["positives"]) >= minimum_prefirst_hit_positives
        and int(prefirst["episode_count"]) == episode_count
        and prefirst_bootstrap is not None
        and float(prefirst_bootstrap["upper_95"]) < 0.0
        and int(prefirst["episodes_favoring_full"])
        >= minimum_prefirst_episodes_favoring_full
    )
    full_calibration = calibration["full_current_root_action"]
    state_calibration = calibration["state_only"]
    calibration_ready = (
        abs(float(full_calibration["calibration_in_the_large"]))
        <= calibration_in_the_large_absolute_max
        and float(full_calibration["expected_calibration_error"])
        - float(state_calibration["expected_calibration_error"])
        <= full_ece_over_state_only_max
    )
    selected = all((
        sufficient,
        bounded,
        direct_model_gain,
        incremental_action,
        nonbaseline_signal,
        lifecycle_signal,
        calibration_ready,
    ))
    propensity_results = {}
    for name in PROPENSITY_STRATA:
        mask = np.asarray([
            _propensity_stratum(float(value)) == name
            for value in view.behavior_probabilities
        ], dtype=np.bool_)
        propensity_results[name] = stratum(mask)
    return {
        "schema": HAZARD_EVALUATION_SCHEMA,
        "horizon_game_frames": horizon,
        "metrics": metrics,
        "calibration": calibration,
        "raw_probability_bounds": {
            "full_current_root_action": full_raw,
            "state_only": state_raw,
        },
        "whole_episode_bootstrap": {
            "full_minus_state_only_brier": full_vs_state,
            "full_minus_frozen_l2_full_brier": full_vs_frozen,
            "state_only_minus_frozen_l2b_state_only_brier": state_vs_frozen,
            "nonbaseline_full_minus_state_only_brier": nonbaseline_bootstrap,
            "prefirst_hit_full_minus_state_only_brier": prefirst_bootstrap,
        },
        "strata": {
            "published_equals_baseline": baseline,
            "published_differs_from_baseline": nonbaseline,
            "pre_first_hit": prefirst,
            "behavior_propensity": propensity_results,
        },
        "action_support": _action_support(view),
        "gates": {
            "overall_support_sufficient": sufficient,
            "bounded_probability_surface": bounded,
            "direct_model_brier_improved_over_frozen_l2": direct_model_gain,
            "incremental_action_signal": incremental_action,
            "nonbaseline_action_signal": nonbaseline_signal,
            "prefirst_hit_lifecycle_signal": lifecycle_signal,
            "calibration_readiness_passed": calibration_ready,
            "selected_for_fresh_confirmation": selected,
        },
        "summary": {
            "decision": (
                "select-action-conditioned-h16-hazard-for-fresh-confirmation"
                if selected
                else "reject-action-conditioned-h16-hazard"
            ),
            "independent_confirmation": False,
            "counterfactual_successors": False,
            "causal_action_value_claimed": False,
            "history_admitted": False,
            "value_learning_admitted": False,
            "online_policy_admitted": False,
        },
    }
