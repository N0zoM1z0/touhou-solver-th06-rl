"""Bounded action-measure correction for the factual h16 HIT hazard.

The model still consumes only portable current-root/action facts for the action
Wine actually executed.  Exact recorder propensities and observed-shield set
sizes define offline fit/evaluation weights; they are never model inputs.  The
weight changes a one-step action distribution, not the factual successor or
the behavior-continuation HIT target.
"""

from __future__ import annotations

import numpy as np
import xgboost as xgb

from .actions import ACTION_NAMES
from .factual_hazard_model import (
    _matrix,
    _model_state,
    _raw_predictions,
    hazard_predictions,
)
from .factual_probe_boundary_diagnostics import (
    BoundaryHorizonDataset,
    BoundaryProbeDataset,
    _calibration_summary,
)
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


WEIGHTED_HAZARD_FIT_SCHEMA = "th06-rl-l2g-weighted-hazard-fit-v1"
WEIGHTED_HAZARD_EVALUATION_SCHEMA = "th06-rl-l2g-weighted-hazard-evaluation-v1"
MODEL_KIND = "uniform-shield-weighted-shared-depth3-brier-regressor"


def _view(dataset: BoundaryProbeDataset, horizon: int) -> BoundaryHorizonDataset:
    try:
        return next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("weighted hazard fit horizon is absent") from error


def uniform_shield_importance_weights(
    view: BoundaryHorizonDataset,
    *,
    uniform_mixture_probability: float,
    maximum_weight: float,
    probability_tolerance: float,
) -> tuple[np.ndarray, dict[str, object]]:
    """Audit the declared collector and return q(a|s)/mu(a|s).

    The target q is uniform over the observed-shield-admissible action set.
    The known behavior mu is the frozen reactive-baseline/uniform mixture.
    """
    if (
        not 0.0 < uniform_mixture_probability < 1.0
        or maximum_weight <= 0.0
        or probability_tolerance < 0.0
    ):
        raise ValueError("importance-weight settings are invalid")
    counts = np.asarray(view.shield_action_counts, dtype=np.int64)
    behavior = np.asarray(view.behavior_probabilities, dtype=np.float64)
    if (
        counts.shape != (view.rows,)
        or behavior.shape != (view.rows,)
        or np.any(counts <= 0)
        or np.any(counts > len(ACTION_NAMES))
        or not np.all(np.isfinite(behavior))
        or np.any(behavior <= 0.0)
        or np.any(behavior > 1.0)
    ):
        raise ValueError("importance-weight recorder facts are malformed")
    baseline_equal = np.asarray([
        published == baseline
        for published, baseline in zip(
            view.published_actions, view.baseline_actions, strict=True
        )
    ], dtype=np.bool_)
    target = 1.0 / counts.astype(np.float64)
    expected_behavior = uniform_mixture_probability * target
    expected_behavior[baseline_equal] += 1.0 - uniform_mixture_probability
    probability_error = np.abs(behavior - expected_behavior)
    if float(np.max(probability_error)) > probability_tolerance:
        raise ValueError("recorded behavior probability violates collector formula")
    weights = target / behavior
    if (
        not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
        or float(np.max(weights)) > maximum_weight + probability_tolerance
    ):
        raise ValueError("uniform-shield importance weight is outside its bound")
    nonbaseline = ~baseline_equal
    nonbaseline_error = 0.0
    if np.any(nonbaseline):
        expected_nonbaseline = 1.0 / uniform_mixture_probability
        nonbaseline_error = float(np.max(np.abs(
            weights[nonbaseline] - expected_nonbaseline
        )))
        if nonbaseline_error > probability_tolerance:
            raise ValueError("nonbaseline importance weight is not the frozen ratio")
    weight_sum = float(np.sum(weights))
    return weights, {
        "formula": "(1 / observed_shield_action_count) / behavior_probability",
        "target_action_measure": "uniform-over-observed-shield-actions",
        "uniform_mixture_probability": uniform_mixture_probability,
        "rows": view.rows,
        "nonbaseline_rows": int(np.sum(nonbaseline)),
        "minimum": float(np.min(weights)),
        "maximum": float(np.max(weights)),
        "mean": float(np.mean(weights)),
        "sum": weight_sum,
        "effective_sample_size": float(
            weight_sum * weight_sum / np.sum(weights * weights)
        ),
        "maximum_collector_probability_absolute_error": float(
            np.max(probability_error)
        ),
        "maximum_nonbaseline_weight_absolute_error": nonbaseline_error,
        "bounded": True,
    }


def _weighted_metrics(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
) -> dict[str, object]:
    targets = labels.astype(np.float64)
    clipped = np.clip(probabilities, 1e-9, 1.0 - 1e-9)
    weight_sum = float(np.sum(weights))
    if (
        probabilities.shape != labels.shape
        or weights.shape != labels.shape
        or weight_sum <= 0.0
        or not np.all(np.isfinite(weights))
        or np.any(weights <= 0.0)
    ):
        raise ValueError("weighted metric inputs are malformed")
    return {
        "rows": int(labels.size),
        "positives": int(np.sum(labels)),
        "negatives": int(labels.size - np.sum(labels)),
        "weight_sum": weight_sum,
        "effective_sample_size": float(
            weight_sum * weight_sum / np.sum(weights * weights)
        ),
        "weighted_prevalence": float(np.sum(weights * targets) / weight_sum),
        "brier": float(np.sum(weights * (probabilities - targets) ** 2) / weight_sum),
        "negative_log_likelihood": float(-np.sum(weights * (
            targets * np.log(clipped)
            + (1.0 - targets) * np.log(1.0 - clipped)
        )) / weight_sum),
    }


def _weighted_calibration_summary(
    probabilities: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    bins: int,
) -> dict[str, object]:
    if bins <= 1:
        raise ValueError("weighted calibration requires at least two bins")
    weight_sum = float(np.sum(weights))
    indices = np.minimum(
        np.floor(probabilities * bins).astype(np.int64),
        bins - 1,
    )
    rows = []
    ece = 0.0
    for index in range(bins):
        members = indices == index
        count = int(np.sum(members))
        member_weight = float(np.sum(weights[members]))
        if count:
            mean_prediction = float(
                np.sum(weights[members] * probabilities[members]) / member_weight
            )
            event_rate = float(
                np.sum(weights[members] * labels[members]) / member_weight
            )
            ece += member_weight / weight_sum * abs(mean_prediction - event_rate)
        else:
            mean_prediction = None
            event_rate = None
        rows.append({
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "rows": count,
            "weight_sum": member_weight,
            "mean_prediction": mean_prediction,
            "event_rate": event_rate,
        })
    mean_prediction = float(np.sum(weights * probabilities) / weight_sum)
    event_rate = float(np.sum(weights * labels) / weight_sum)
    return {
        "mean_prediction": mean_prediction,
        "event_rate": event_rate,
        "calibration_in_the_large": mean_prediction - event_rate,
        "expected_calibration_error": ece,
        "zero_clipped_rows": int(np.sum(probabilities == 0.0)),
        "one_clipped_rows": int(np.sum(probabilities == 1.0)),
        "equal_width_bins": rows,
    }


def _weighted_episode_bootstrap_brier_delta(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    episode_indices: np.ndarray,
    episode_ids: tuple[str, ...],
    *,
    samples: int,
    seed: int,
) -> dict[str, object] | None:
    episode_count = len(episode_ids)
    if samples <= 0 or episode_count <= 0:
        raise ValueError("weighted episode bootstrap settings must be positive")
    if not labels.size:
        return None
    targets = labels.astype(np.float64)
    weighted_delta = weights * (
        (candidate - targets) ** 2 - (baseline - targets) ** 2
    )
    sums = np.bincount(
        episode_indices, weights=weighted_delta, minlength=episode_count
    )
    weight_sums = np.bincount(
        episode_indices, weights=weights, minlength=episode_count
    )
    row_counts = np.bincount(episode_indices, minlength=episode_count)
    if np.any(row_counts == 0) or np.any(weight_sums <= 0.0):
        return None
    per_episode = [
        {
            "episode_id": episode_ids[index],
            "rows": int(row_counts[index]),
            "weight_sum": float(weight_sums[index]),
            "candidate_minus_baseline_brier": float(
                sums[index] / weight_sums[index]
            ),
        }
        for index in range(episode_count)
    ]
    random = np.random.default_rng(seed)
    draws = random.integers(0, episode_count, size=(samples, episode_count))
    values = np.sum(sums[draws], axis=1) / np.sum(weight_sums[draws], axis=1)
    return {
        "unit": "complete-physical-episode",
        "target_action_measure": "uniform-over-observed-shield-actions",
        "samples": samples,
        "seed": seed,
        "point": float(np.sum(weighted_delta) / np.sum(weights)),
        "lower_95": float(np.quantile(values, 0.025)),
        "upper_95": float(np.quantile(values, 0.975)),
        "episodes_favoring_candidate": sum(
            row["candidate_minus_baseline_brier"] < 0.0 for row in per_episode
        ),
        "per_episode": per_episode,
    }


def _train_model(
    features: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    *,
    feature_names: tuple[str, ...],
    parameters: dict[str, object],
    boosted_rounds: int,
) -> xgb.Booster:
    matrix = _matrix(features, feature_names)
    matrix.set_label(labels.astype(np.float64))
    matrix.set_weight(weights)
    return xgb.train(
        dict(parameters), matrix, num_boost_round=boosted_rounds, verbose_eval=False
    )


def weighted_hazard_predictions(
    state: dict[str, object],
    features: np.ndarray,
    *,
    model_name: str,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return clipped probabilities and disclose the unconstrained surface."""
    if state.get("schema") != WEIGHTED_HAZARD_FIT_SCHEMA:
        raise ValueError("weighted hazard fit schema mismatch")
    if state.get("xgboost_version") != xgb.__version__:
        raise ValueError("weighted hazard XGBoost version mismatch")
    if model_name == "full_current_root_action":
        selected = features
        feature_names = PROBE_FEATURE_NAMES
    elif model_name == "state_only":
        selected = features[:, STATE_ONLY_FEATURE_INDICES]
        feature_names = STATE_ONLY_FEATURE_NAMES
    else:
        raise ValueError(f"unknown weighted hazard model: {model_name}")
    raw = _raw_predictions(
        state["models"][model_name],
        selected,
        expected_feature_names=feature_names,
    )
    probabilities = np.clip(raw, 0.0, 1.0)
    clipped = raw != probabilities
    return probabilities, {
        "raw_minimum": float(np.min(raw)),
        "raw_maximum": float(np.max(raw)),
        "clipped_rows": int(np.sum(clipped)),
        "clipped_fraction": float(np.mean(clipped)),
    }


def fit_weighted_action_conditioned_hazard_models(
    dataset: BoundaryProbeDataset,
    *,
    horizon: int,
    uniform_mixture_probability: float,
    maximum_importance_weight: float,
    probability_tolerance: float,
    boosted_rounds: int,
    maximum_depth: int,
    learning_rate: float,
    minimum_child_weight: float,
    l2_leaf_regularization: float,
    maximum_histogram_bins: int,
    seed: int,
    expected_xgboost_version: str,
) -> dict[str, object]:
    """Fit paired full/state models under the uniform-shield target measure."""
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
        raise ValueError("weighted hazard fit settings are invalid")
    view = _view(dataset, horizon)
    labels = view.hit_labels
    positives = int(np.sum(labels))
    if not positives or positives == view.rows:
        raise ValueError("weighted hazard train target requires both classes")
    weights, weight_summary = uniform_shield_importance_weights(
        view,
        uniform_mixture_probability=uniform_mixture_probability,
        maximum_weight=maximum_importance_weight,
        probability_tolerance=probability_tolerance,
    )
    weighted_prevalence = float(np.sum(weights * labels) / np.sum(weights))
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
        "base_score": weighted_prevalence,
        "seed": seed,
        "nthread": 1,
        "device": "cpu",
        "verbosity": 0,
    }
    full_booster = _train_model(
        view.features,
        labels,
        weights,
        feature_names=PROBE_FEATURE_NAMES,
        parameters=parameters,
        boosted_rounds=boosted_rounds,
    )
    state_booster = _train_model(
        view.features[:, STATE_ONLY_FEATURE_INDICES],
        labels,
        weights,
        feature_names=STATE_ONLY_FEATURE_NAMES,
        parameters=parameters,
        boosted_rounds=boosted_rounds,
    )
    state: dict[str, object] = {
        "schema": WEIGHTED_HAZARD_FIT_SCHEMA,
        "model": MODEL_KIND,
        "feature_schema": PROBE_FEATURE_SCHEMA,
        "horizon_game_frames": horizon,
        "target": "physical-hit-within-fixed-horizon-under-behavior-continuation",
        "training_proper_score": (
            "uniform-observed-shield-target-importance-weighted-row-brier"
        ),
        "xgboost_version": xgb.__version__,
        "parameters": parameters,
        "boosted_rounds": boosted_rounds,
        "importance_weight": weight_summary,
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
            "logged_prevalence": float(np.mean(labels)),
            "uniform_target_prevalence": weighted_prevalence,
        },
    }
    full, full_raw = weighted_hazard_predictions(
        state, view.features, model_name="full_current_root_action"
    )
    state_only, state_raw = weighted_hazard_predictions(
        state, view.features, model_name="state_only"
    )
    constant = np.full(view.rows, weighted_prevalence, dtype=np.float64)
    state["train"]["metrics"] = {
        "uniform_shield_target": {
            "full_current_root_action": _weighted_metrics(full, labels, weights),
            "state_only": _weighted_metrics(state_only, labels, weights),
            "constant_prevalence": _weighted_metrics(constant, labels, weights),
        },
        "logged_measure": {
            "full_current_root_action": _binary_metrics(full, labels),
            "state_only": _binary_metrics(state_only, labels),
        },
    }
    state["train"]["raw_probability_bounds"] = {
        "full_current_root_action": full_raw,
        "state_only": state_raw,
    }
    return state


def _stratum(
    mask: np.ndarray,
    *,
    full: np.ndarray,
    state_only: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    episode_indices: np.ndarray,
    episode_ids: tuple[str, ...],
    calibration_bins: int,
) -> dict[str, object]:
    if not np.any(mask):
        return {
            "rows": 0,
            "positives": 0,
            "episode_count": 0,
            "episodes_favoring_full": 0,
            "full_minus_state_only_brier": None,
            "full": None,
            "state_only": None,
        }
    targets = labels[mask]
    selected_weights = weights[mask]
    selected_episodes = sorted(set(int(value) for value in episode_indices[mask]))
    per_episode = []
    for episode_index in selected_episodes:
        members = mask & (episode_indices == episode_index)
        member_targets = labels[members].astype(np.float64)
        delta = weights[members] * (
            (full[members] - member_targets) ** 2
            - (state_only[members] - member_targets) ** 2
        )
        per_episode.append({
            "episode_id": episode_ids[episode_index],
            "rows": int(np.sum(members)),
            "positives": int(np.sum(labels[members])),
            "weight_sum": float(np.sum(weights[members])),
            "full_minus_state_only_brier": float(
                np.sum(delta) / np.sum(weights[members])
            ),
        })
    full_metrics = _weighted_metrics(full[mask], targets, selected_weights)
    state_metrics = _weighted_metrics(state_only[mask], targets, selected_weights)
    return {
        "rows": int(np.sum(mask)),
        "positives": int(np.sum(targets)),
        "episode_count": len(selected_episodes),
        "episodes_favoring_full": sum(
            row["full_minus_state_only_brier"] < 0.0 for row in per_episode
        ),
        "full_minus_state_only_brier": (
            float(full_metrics["brier"]) - float(state_metrics["brier"])
        ),
        "full": full_metrics,
        "state_only": state_metrics,
        "full_logged_metrics": _binary_metrics(full[mask], targets),
        "state_only_logged_metrics": _binary_metrics(state_only[mask], targets),
        "full_calibration": _weighted_calibration_summary(
            full[mask], targets, selected_weights, bins=calibration_bins
        ),
        "state_only_calibration": _weighted_calibration_summary(
            state_only[mask], targets, selected_weights, bins=calibration_bins
        ),
        "per_episode": per_episode,
    }


def evaluate_weighted_action_conditioned_hazard_models(
    state: dict[str, object],
    frozen_unweighted_state: dict[str, object],
    dataset: BoundaryProbeDataset,
    *,
    uniform_mixture_probability: float,
    maximum_importance_weight: float,
    probability_tolerance: float,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_overall_positives: int,
    minimum_overall_negatives: int,
    minimum_nonbaseline_positives: int,
    minimum_low_propensity_positives: int,
    minimum_prefirst_hit_positives: int,
    minimum_target_gain_episodes: int,
    minimum_overall_episodes_favoring_full: int,
    minimum_nonbaseline_episodes_favoring_full: int,
    minimum_low_propensity_episodes_favoring_full: int,
    minimum_prefirst_episodes_favoring_full: int,
    weighted_calibration_in_the_large_absolute_max: float,
    weighted_full_ece_over_state_only_max: float,
    logged_calibration_in_the_large_absolute_max: float,
    maximum_raw_clipped_fraction: float,
) -> dict[str, object]:
    """Evaluate L2g once on the already-inspected complete L2d episodes."""
    horizon = int(state["horizon_game_frames"])
    view = _view(dataset, horizon)
    labels = view.hit_labels
    weights, weight_summary = uniform_shield_importance_weights(
        view,
        uniform_mixture_probability=uniform_mixture_probability,
        maximum_weight=maximum_importance_weight,
        probability_tolerance=probability_tolerance,
    )
    full, full_raw = weighted_hazard_predictions(
        state, view.features, model_name="full_current_root_action"
    )
    state_only, state_raw = weighted_hazard_predictions(
        state, view.features, model_name="state_only"
    )
    unweighted_full, unweighted_full_raw = hazard_predictions(
        frozen_unweighted_state,
        view.features,
        model_name="full_current_root_action",
    )
    all_rows = np.ones(view.rows, dtype=np.bool_)
    baseline_equal = np.asarray([
        published == baseline
        for published, baseline in zip(
            view.published_actions, view.baseline_actions, strict=True
        )
    ], dtype=np.bool_)
    lifecycle = np.asarray(view.lifecycle_strata, dtype=object)
    nonbaseline = ~baseline_equal
    low_propensity = view.behavior_probabilities < 0.025
    prefirst = lifecycle == "pre-first-hit"

    def result(mask: np.ndarray) -> dict[str, object]:
        return _stratum(
            mask,
            full=full,
            state_only=state_only,
            labels=labels,
            weights=weights,
            episode_indices=view.episode_indices,
            episode_ids=view.episode_ids,
            calibration_bins=calibration_bins,
        )

    overall = result(all_rows)
    baseline = result(baseline_equal)
    nonbaseline_result = result(nonbaseline)
    low_propensity_result = result(low_propensity)
    prefirst_result = result(prefirst)

    def bootstrap(
        mask: np.ndarray,
        candidate: np.ndarray,
        comparator: np.ndarray,
        seed_offset: int,
    ) -> dict[str, object] | None:
        return _weighted_episode_bootstrap_brier_delta(
            candidate[mask],
            comparator[mask],
            labels[mask],
            weights[mask],
            view.episode_indices[mask],
            view.episode_ids,
            samples=bootstrap_samples,
            seed=bootstrap_seed + seed_offset,
        )

    full_vs_unweighted = bootstrap(all_rows, full, unweighted_full, 0)
    full_vs_state = bootstrap(all_rows, full, state_only, 1)
    nonbaseline_bootstrap = bootstrap(nonbaseline, full, state_only, 2)
    low_propensity_bootstrap = bootstrap(low_propensity, full, state_only, 3)
    prefirst_bootstrap = bootstrap(prefirst, full, state_only, 4)
    assert full_vs_unweighted is not None and full_vs_state is not None

    target_calibration = {
        "full_current_root_action": _weighted_calibration_summary(
            full, labels, weights, bins=calibration_bins
        ),
        "state_only": _weighted_calibration_summary(
            state_only, labels, weights, bins=calibration_bins
        ),
        "frozen_unweighted_l2f_full": _weighted_calibration_summary(
            unweighted_full, labels, weights, bins=calibration_bins
        ),
    }
    logged_calibration = {
        "full_current_root_action": _calibration_summary(
            full, labels, bins=calibration_bins
        ),
        "state_only": _calibration_summary(
            state_only, labels, bins=calibration_bins
        ),
        "frozen_unweighted_l2f_full": _calibration_summary(
            unweighted_full, labels, bins=calibration_bins
        ),
    }
    positives = int(np.sum(labels))
    episode_count = len(view.episode_ids)
    support = (
        positives >= minimum_overall_positives
        and view.rows - positives >= minimum_overall_negatives
    )
    target_gain = (
        float(full_vs_unweighted["upper_95"]) < 0.0
        and int(full_vs_unweighted["episodes_favoring_candidate"])
        >= minimum_target_gain_episodes
    )
    action_signal = (
        float(full_vs_state["upper_95"]) < 0.0
        and int(full_vs_state["episodes_favoring_candidate"])
        >= minimum_overall_episodes_favoring_full
    )

    def boundary_gate(
        boundary: dict[str, object],
        interval: dict[str, object] | None,
        *,
        minimum_positives: int,
        minimum_favorable: int,
    ) -> bool:
        return (
            int(boundary["positives"]) >= minimum_positives
            and int(boundary["episode_count"]) == episode_count
            and interval is not None
            and float(interval["upper_95"]) < 0.0
            and int(boundary["episodes_favoring_full"]) >= minimum_favorable
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
    target_full_calibration = target_calibration["full_current_root_action"]
    target_state_calibration = target_calibration["state_only"]
    target_calibration_ready = (
        abs(float(target_full_calibration["calibration_in_the_large"]))
        <= weighted_calibration_in_the_large_absolute_max
        and float(target_full_calibration["expected_calibration_error"])
        - float(target_state_calibration["expected_calibration_error"])
        <= weighted_full_ece_over_state_only_max
    )
    logged_calibration_ready = (
        abs(float(logged_calibration["full_current_root_action"][
            "calibration_in_the_large"
        ]))
        <= logged_calibration_in_the_large_absolute_max
    )
    candidate_bounded = (
        float(full_raw["clipped_fraction"]) <= maximum_raw_clipped_fraction
    )
    selected = all((
        support,
        bool(weight_summary["bounded"]),
        candidate_bounded,
        target_gain,
        action_signal,
        nonbaseline_signal,
        low_propensity_signal,
        lifecycle_signal,
        target_calibration_ready,
        logged_calibration_ready,
    ))
    return {
        "schema": WEIGHTED_HAZARD_EVALUATION_SCHEMA,
        "horizon_game_frames": horizon,
        "importance_weight": weight_summary,
        "uniform_shield_target_metrics": {
            "full_current_root_action": _weighted_metrics(full, labels, weights),
            "state_only": _weighted_metrics(state_only, labels, weights),
            "frozen_unweighted_l2f_full": _weighted_metrics(
                unweighted_full, labels, weights
            ),
        },
        "logged_measure_metrics": {
            "full_current_root_action": _binary_metrics(full, labels),
            "state_only": _binary_metrics(state_only, labels),
            "frozen_unweighted_l2f_full": _binary_metrics(unweighted_full, labels),
        },
        "calibration": {
            "uniform_shield_target": target_calibration,
            "logged_measure": logged_calibration,
        },
        "raw_probability_bounds": {
            "full_current_root_action": full_raw,
            "state_only": state_raw,
            "frozen_unweighted_l2f_full": unweighted_full_raw,
        },
        "whole_episode_bootstrap": {
            "weighted_full_minus_unweighted_l2f_full_brier": full_vs_unweighted,
            "weighted_full_minus_weighted_state_only_brier": full_vs_state,
            "nonbaseline_full_minus_state_only_brier": nonbaseline_bootstrap,
            "low_propensity_full_minus_state_only_brier": low_propensity_bootstrap,
            "prefirst_hit_full_minus_state_only_brier": prefirst_bootstrap,
            "logged_full_minus_unweighted_l2f_full_brier": (
                _episode_bootstrap_brier_delta(
                    full,
                    unweighted_full,
                    labels,
                    view.episode_indices,
                    episode_count=episode_count,
                    samples=bootstrap_samples,
                    seed=bootstrap_seed + 5,
                )
            ),
        },
        "strata": {
            "published_equals_baseline": baseline,
            "published_differs_from_baseline": nonbaseline_result,
            "behavior_propensity_below_0.025": low_propensity_result,
            "pre_first_hit": prefirst_result,
            "overall": overall,
        },
        "gates": {
            "overall_support_sufficient": support,
            "importance_weight_contract_passed": bool(weight_summary["bounded"]),
            "candidate_probability_surface_bounded": candidate_bounded,
            "uniform_target_brier_improved_over_unweighted_l2f": target_gain,
            "incremental_action_signal": action_signal,
            "nonbaseline_action_signal": nonbaseline_signal,
            "low_propensity_action_signal": low_propensity_signal,
            "prefirst_hit_lifecycle_signal": lifecycle_signal,
            "uniform_target_calibration_readiness_passed": target_calibration_ready,
            "logged_calibration_guardrail_passed": logged_calibration_ready,
            "selected_for_fresh_confirmation": selected,
        },
        "summary": {
            "decision": (
                "select-weighted-h16-hazard-for-fresh-confirmation"
                if selected
                else "reject-weighted-h16-hazard"
            ),
            "independent_confirmation": False,
            "counterfactual_successors": False,
            "causal_action_value_claimed": False,
            "propensity_is_actor_input": False,
            "history_admitted": False,
            "value_learning_admitted": False,
            "online_policy_admitted": False,
        },
    }
