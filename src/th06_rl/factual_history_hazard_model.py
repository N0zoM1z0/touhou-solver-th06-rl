"""Direct Brier hazard models with one fixed causal portable-root history."""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .factual_hazard_model import (
    _matrix,
    _model_state,
    _raw_predictions,
    hazard_predictions,
)
from .factual_history_dataset import (
    HISTORY_FEATURE_SCHEMA,
    HistoryProbeDataset,
    history_feature_names,
)
from .factual_probe_boundary_diagnostics import (
    _calibration_summary,
    _stratum_result,
)
from .factual_probe_diagnostics import STATE_ONLY_FEATURE_INDICES
from .factual_probes import (
    PROBE_FEATURE_NAMES,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


HISTORY_HAZARD_FIT_SCHEMA = "th06-rl-l2h-fixed-history-hazard-fit-v1"
HISTORY_HAZARD_EVALUATION_SCHEMA = "th06-rl-l2h-fixed-history-hazard-evaluation-v1"
MODEL_KIND = "shared-depth3-gradient-boosted-brier-regressor"


def _view(dataset: HistoryProbeDataset, horizon: int):
    try:
        return next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("history hazard horizon is absent") from error


def _action_ablated_indices(history_length: int) -> np.ndarray:
    history_width = history_length * len(PROBE_FEATURE_NAMES)
    return np.asarray(
        list(range(history_width))
        + [history_width + index for index in STATE_ONLY_FEATURE_INDICES],
        dtype=np.int64,
    )


def _action_ablated_names(history_length: int) -> tuple[str, ...]:
    names = history_feature_names(history_length)
    indices = _action_ablated_indices(history_length)
    return tuple(names[int(index)] for index in indices)


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
        dict(parameters), matrix, num_boost_round=boosted_rounds, verbose_eval=False
    )


def _selected_features(
    state: dict[str, object],
    features: np.ndarray,
    current_features: np.ndarray,
    model_name: str,
) -> tuple[np.ndarray, tuple[str, ...]]:
    history_length = int(state["history_length_decision_roots"])
    if model_name == "history_full":
        return features, history_feature_names(history_length)
    if model_name == "current_only":
        return current_features, tuple(
            f"current:{name}" for name in PROBE_FEATURE_NAMES
        )
    if model_name == "history_current_action_ablated":
        indices = _action_ablated_indices(history_length)
        return features[:, indices], _action_ablated_names(history_length)
    raise ValueError(f"unknown history hazard model: {model_name}")


def history_hazard_predictions(
    state: dict[str, object],
    features: np.ndarray,
    current_features: np.ndarray,
    *,
    model_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return bounded probabilities and disclose the raw regressor surface."""
    if state.get("schema") != HISTORY_HAZARD_FIT_SCHEMA:
        raise ValueError("history hazard fit schema mismatch")
    if state.get("xgboost_version") != xgb.__version__:
        raise ValueError("history hazard XGBoost version mismatch")
    selected, names = _selected_features(
        state, features, current_features, model_name
    )
    raw = _raw_predictions(
        state["models"][model_name], selected, expected_feature_names=names
    )
    probabilities = np.clip(raw, 0.0, 1.0)
    clipped = raw != probabilities
    return probabilities, {
        "raw_minimum": float(np.min(raw)),
        "raw_maximum": float(np.max(raw)),
        "clipped_rows": int(np.sum(clipped)),
        "clipped_fraction": float(np.mean(clipped)),
    }


def fit_history_hazard_models(
    dataset: HistoryProbeDataset,
    *,
    horizon: int,
    history_length: int,
    boosted_rounds: int,
    maximum_depth: int,
    learning_rate: float,
    minimum_child_weight: float,
    l2_leaf_regularization: float,
    maximum_histogram_bins: int,
    seed: int,
    threads: int,
    expected_xgboost_version: str,
) -> dict[str, object]:
    """Fit history, same-row current-only, and current-action ablation models."""
    if xgb.__version__ != expected_xgboost_version:
        raise ValueError("preregistered XGBoost version is unavailable")
    if (
        history_length <= 0
        or horizon <= 0
        or boosted_rounds <= 0
        or maximum_depth <= 0
        or not 0.0 < learning_rate <= 1.0
        or minimum_child_weight <= 0.0
        or l2_leaf_regularization < 0.0
        or maximum_histogram_bins < 2
        or threads <= 0
    ):
        raise ValueError("history hazard fit settings are invalid")
    view = _view(dataset, horizon)
    if view.history_length != history_length:
        raise ValueError("history hazard length differs from dataset")
    labels = view.hit_labels
    positives = int(np.sum(labels))
    if not positives or positives == view.rows:
        raise ValueError("history hazard train target requires both classes")
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
        "nthread": threads,
        "device": "cpu",
        "verbosity": 0,
    }
    names = {
        "history_full": history_feature_names(history_length),
        "current_only": tuple(f"current:{name}" for name in PROBE_FEATURE_NAMES),
        "history_current_action_ablated": _action_ablated_names(history_length),
    }
    matrices = {
        "history_full": view.features,
        "current_only": view.current_features,
        "history_current_action_ablated": view.features[
            :, _action_ablated_indices(history_length)
        ],
    }
    boosters = {
        name: _train_model(
            matrices[name],
            labels,
            feature_names=names[name],
            parameters=parameters,
            boosted_rounds=boosted_rounds,
        )
        for name in names
    }
    state: dict[str, object] = {
        "schema": HISTORY_HAZARD_FIT_SCHEMA,
        "model": MODEL_KIND,
        "feature_schema": HISTORY_FEATURE_SCHEMA,
        "history_length_decision_roots": history_length,
        "history_order": "oldest-to-newest-then-current",
        "history_padding": "none-drop-row-without-complete-prefix",
        "horizon_game_frames": horizon,
        "target": "physical-hit-within-fixed-horizon-under-behavior-continuation",
        "training_proper_score": "mean-unweighted-row-brier",
        "xgboost_version": xgb.__version__,
        "parameters": parameters,
        "boosted_rounds": boosted_rounds,
        "models": {
            name: _model_state(boosters[name], feature_names=names[name])
            for name in names
        },
        "train": {
            "all_current_rows": view.all_current_rows,
            "history_ready_rows": view.rows,
            "dropped_without_complete_history": view.all_current_rows - view.rows,
            "positives": positives,
            "negatives": view.rows - positives,
            "prevalence": prevalence,
        },
    }
    predictions = {}
    bounds = {}
    for name in names:
        predictions[name], bounds[name] = history_hazard_predictions(
            state,
            view.features,
            view.current_features,
            model_name=name,
        )
    constant = np.full(view.rows, prevalence, dtype=np.float64)
    state["train"]["metrics"] = {
        **{
            name: _binary_metrics(predictions[name], labels)
            for name in names
        },
        "constant_prevalence": _binary_metrics(constant, labels),
    }
    state["train"]["raw_probability_bounds"] = bounds
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
    if len(set(int(value) for value in episode_indices[mask])) != episode_count:
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


def _episodes_favoring(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    episode_ids: tuple[str, ...],
) -> tuple[int, list[dict[str, object]]]:
    targets = labels.astype(np.float64)
    rows = []
    for index, episode_id in enumerate(episode_ids):
        members = episode_indices == index
        delta = (
            (candidate[members] - targets[members]) ** 2
            - (baseline[members] - targets[members]) ** 2
        )
        rows.append({
            "episode_id": episode_id,
            "rows": int(np.sum(members)),
            "candidate_minus_baseline_brier": float(np.mean(delta)),
        })
    return sum(row["candidate_minus_baseline_brier"] < 0.0 for row in rows), rows


def evaluate_history_hazard_models(
    state: dict[str, object],
    frozen_l2f_state: dict[str, object],
    dataset: HistoryProbeDataset,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_overall_positives: int,
    minimum_overall_negatives: int,
    minimum_nonbaseline_positives: int,
    minimum_low_propensity_positives: int,
    minimum_prefirst_hit_positives: int,
    minimum_temporal_gain_episodes: int,
    minimum_overall_episodes_favoring_full: int,
    minimum_nonbaseline_episodes_favoring_full: int,
    minimum_low_propensity_episodes_favoring_full: int,
    minimum_prefirst_episodes_favoring_full: int,
    calibration_in_the_large_absolute_max: float,
    full_ece_over_action_ablated_max: float,
    maximum_raw_clipped_fraction: float,
) -> dict[str, object]:
    """Evaluate one frozen L2h fit on reused complete L2d episodes."""
    horizon = int(state["horizon_game_frames"])
    view = _view(dataset, horizon)
    predictions = {}
    raw_bounds = {}
    for name in (
        "history_full",
        "current_only",
        "history_current_action_ablated",
    ):
        predictions[name], raw_bounds[name] = history_hazard_predictions(
            state,
            view.features,
            view.current_features,
            model_name=name,
        )
    full = predictions["history_full"]
    current_only = predictions["current_only"]
    action_ablated = predictions["history_current_action_ablated"]
    frozen_subset, frozen_subset_raw = hazard_predictions(
        frozen_l2f_state,
        view.current_features,
        model_name="full_current_root_action",
    )
    frozen_all, frozen_all_raw = hazard_predictions(
        frozen_l2f_state,
        view.all_current_features,
        model_name="full_current_root_action",
    )
    labels = view.hit_labels
    all_rows = np.ones(view.rows, dtype=np.bool_)
    baseline_equal = np.asarray([
        published == baseline
        for published, baseline in zip(
            view.published_actions, view.baseline_actions, strict=True
        )
    ], dtype=np.bool_)
    nonbaseline = ~baseline_equal
    low_propensity = view.behavior_probabilities < 0.025
    lifecycle = np.asarray(view.lifecycle_strata, dtype=object)
    prefirst = lifecycle == "pre-first-hit"
    episode_count = len(view.episode_ids)

    def action_stratum(mask: np.ndarray) -> dict[str, object]:
        return _stratum_result(
            mask,
            full=full,
            state_only=action_ablated,
            labels=labels,
            episode_indices=view.episode_indices,
            episode_ids=view.episode_ids,
            calibration_bins=calibration_bins,
        )

    overall = action_stratum(all_rows)
    baseline = action_stratum(baseline_equal)
    nonbaseline_result = action_stratum(nonbaseline)
    low_propensity_result = action_stratum(low_propensity)
    prefirst_result = action_stratum(prefirst)
    temporal_bootstrap = _episode_bootstrap_brier_delta(
        full,
        current_only,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    action_bootstrap = _episode_bootstrap_brier_delta(
        full,
        action_ablated,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    frozen_bootstrap = _episode_bootstrap_brier_delta(
        full,
        frozen_subset,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
    )
    nonbaseline_bootstrap = _masked_bootstrap(
        nonbaseline,
        candidate=full,
        baseline=action_ablated,
        labels=labels,
        episode_indices=view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 3,
    )
    low_propensity_bootstrap = _masked_bootstrap(
        low_propensity,
        candidate=full,
        baseline=action_ablated,
        labels=labels,
        episode_indices=view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 4,
    )
    prefirst_bootstrap = _masked_bootstrap(
        prefirst,
        candidate=full,
        baseline=action_ablated,
        labels=labels,
        episode_indices=view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 5,
    )
    temporal_favorable, temporal_per_episode = _episodes_favoring(
        full,
        current_only,
        labels,
        view.episode_indices,
        view.episode_ids,
    )
    calibration = {
        name: _calibration_summary(
            predictions[name], labels, bins=calibration_bins
        )
        for name in predictions
    }
    positives = int(np.sum(labels))
    support = (
        positives >= minimum_overall_positives
        and view.rows - positives >= minimum_overall_negatives
        and int(np.min(np.bincount(
            view.episode_indices, minlength=episode_count
        ))) > 0
    )
    temporal_gain = (
        float(temporal_bootstrap["upper_95"]) < 0.0
        and temporal_favorable >= minimum_temporal_gain_episodes
    )
    action_signal = (
        float(action_bootstrap["upper_95"]) < 0.0
        and int(overall["episodes_favoring_full"])
        >= minimum_overall_episodes_favoring_full
    )

    def boundary_gate(
        result: dict[str, object],
        interval: dict[str, object] | None,
        *,
        minimum_positives: int,
        minimum_favorable: int,
    ) -> bool:
        return (
            int(result["positives"]) >= minimum_positives
            and int(result["episode_count"]) == episode_count
            and interval is not None
            and float(interval["upper_95"]) < 0.0
            and int(result["episodes_favoring_full"]) >= minimum_favorable
        )

    nonbaseline_signal = boundary_gate(
        nonbaseline_result,
        nonbaseline_bootstrap,
        minimum_positives=minimum_nonbaseline_positives,
        minimum_favorable=minimum_nonbaseline_episodes_favoring_full,
    )
    low_propensity_signal = boundary_gate(
        low_propensity_result,
        low_propensity_bootstrap,
        minimum_positives=minimum_low_propensity_positives,
        minimum_favorable=minimum_low_propensity_episodes_favoring_full,
    )
    lifecycle_signal = boundary_gate(
        prefirst_result,
        prefirst_bootstrap,
        minimum_positives=minimum_prefirst_hit_positives,
        minimum_favorable=minimum_prefirst_episodes_favoring_full,
    )
    full_calibration = calibration["history_full"]
    ablated_calibration = calibration["history_current_action_ablated"]
    calibration_ready = (
        abs(float(full_calibration["calibration_in_the_large"]))
        <= calibration_in_the_large_absolute_max
        and float(full_calibration["expected_calibration_error"])
        - float(ablated_calibration["expected_calibration_error"])
        <= full_ece_over_action_ablated_max
    )
    candidate_bounded = (
        float(raw_bounds["history_full"]["clipped_fraction"])
        <= maximum_raw_clipped_fraction
    )
    selected = all((
        support,
        candidate_bounded,
        temporal_gain,
        action_signal,
        nonbaseline_signal,
        low_propensity_signal,
        lifecycle_signal,
        calibration_ready,
    ))
    return {
        "schema": HISTORY_HAZARD_EVALUATION_SCHEMA,
        "horizon_game_frames": horizon,
        "history_length_decision_roots": view.history_length,
        "rows": {
            "all_current": view.all_current_rows,
            "history_ready": view.rows,
            "dropped_without_complete_history": view.all_current_rows - view.rows,
            "positives": positives,
            "negatives": view.rows - positives,
        },
        "history_elapsed_game_frames": {
            "minimum": int(np.min(view.history_elapsed_game_frames)),
            "median": float(np.median(view.history_elapsed_game_frames)),
            "maximum": int(np.max(view.history_elapsed_game_frames)),
        },
        "metrics": {
            **{
                name: _binary_metrics(predictions[name], labels)
                for name in predictions
            },
            "frozen_l2f_full_same_rows": _binary_metrics(frozen_subset, labels),
            "frozen_l2f_full_all_current_rows": _binary_metrics(
                frozen_all, view.all_current_hit_labels
            ),
        },
        "calibration": calibration,
        "raw_probability_bounds": {
            **raw_bounds,
            "frozen_l2f_full_same_rows": frozen_subset_raw,
            "frozen_l2f_full_all_current_rows": frozen_all_raw,
        },
        "whole_episode_bootstrap": {
            "history_full_minus_current_only_brier": {
                **temporal_bootstrap,
                "episodes_favoring_history": temporal_favorable,
                "per_episode": temporal_per_episode,
            },
            "history_full_minus_current_action_ablated_brier": action_bootstrap,
            "history_full_minus_frozen_l2f_full_same_rows_brier": frozen_bootstrap,
            "nonbaseline_full_minus_action_ablated_brier": nonbaseline_bootstrap,
            "low_propensity_full_minus_action_ablated_brier": (
                low_propensity_bootstrap
            ),
            "prefirst_hit_full_minus_action_ablated_brier": prefirst_bootstrap,
        },
        "strata": {
            "overall": overall,
            "published_equals_baseline": baseline,
            "published_differs_from_baseline": nonbaseline_result,
            "behavior_propensity_below_0.025": low_propensity_result,
            "pre_first_hit": prefirst_result,
        },
        "gates": {
            "overall_support_sufficient": support,
            "candidate_probability_surface_bounded": candidate_bounded,
            "history_temporal_gain": temporal_gain,
            "current_action_signal": action_signal,
            "nonbaseline_action_signal": nonbaseline_signal,
            "low_propensity_action_signal": low_propensity_signal,
            "prefirst_hit_lifecycle_signal": lifecycle_signal,
            "calibration_readiness_passed": calibration_ready,
            "selected_for_fresh_confirmation": selected,
        },
        "summary": {
            "decision": (
                "select-fixed-history-h16-hazard-for-fresh-confirmation"
                if selected
                else "reject-fixed-history-h16-hazard"
            ),
            "independent_confirmation": False,
            "counterfactual_successors": False,
            "causal_action_value_claimed": False,
            "history_admitted": True,
            "object_set_admitted": False,
            "value_learning_admitted": False,
            "online_policy_admitted": False,
        },
    }
