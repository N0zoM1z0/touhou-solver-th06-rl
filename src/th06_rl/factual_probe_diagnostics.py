"""Single-variable attribution for the frozen L2 factual-risk probe."""

from __future__ import annotations

import math

import numpy as np

from .factual_probes import (
    FactualProbeDataset,
    PROBE_FEATURE_NAMES,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


DIAGNOSIS_SCHEMA = "th06-rl-l2b-incremental-action-diagnosis-v1"
STATE_ONLY_FEATURE_NAMES = PROBE_FEATURE_NAMES[:6]
STATE_ONLY_FEATURE_INDICES = tuple(range(len(STATE_ONLY_FEATURE_NAMES)))


def _fit_ridge(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    ridge_l2: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(features, axis=0)
    scale = np.std(features, axis=0)
    scale[scale < 1e-9] = 1.0
    normalized = (features - mean) / scale
    design = np.column_stack((np.ones(features.shape[0]), normalized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_l2
    penalty[0, 0] = 0.0
    gram = design.T @ design / features.shape[0] + penalty
    coefficients = np.linalg.solve(
        gram,
        design.T @ labels.astype(np.float64) / features.shape[0],
    )
    if any(np.any(~np.isfinite(value)) for value in (mean, scale, coefficients)):
        raise ValueError("state-only attribution fit is non-finite")
    return mean, scale, coefficients


def fit_state_only_probe_models(
    dataset: FactualProbeDataset,
    *,
    ridge_l2: float,
) -> dict[str, object]:
    """Fit the frozen ridge model after removing every action-relative feature."""
    if not math.isfinite(ridge_l2) or ridge_l2 <= 0.0:
        raise ValueError("state-only attribution ridge L2 must be positive")
    horizons = {}
    for view in dataset.horizons:
        features = view.features[:, STATE_ONLY_FEATURE_INDICES]
        target_states = {}
        for name, labels in (
            ("hit", view.hit_labels),
            ("shield_collapse", view.shield_collapse_labels),
        ):
            mean, scale, coefficients = _fit_ridge(
                features,
                labels,
                ridge_l2=ridge_l2,
            )
            target_states[name] = {
                "normalization": {
                    "mean": mean.tolist(),
                    "scale": scale.tolist(),
                },
                "coefficients": coefficients.tolist(),
            }
        horizons[str(view.horizon)] = {
            "rows": view.rows,
            "targets": target_states,
        }
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "model": "standardized-ridge-linear-probability",
        "ridge_l2": ridge_l2,
        "feature_names": list(STATE_ONLY_FEATURE_NAMES),
        "removed_feature_names": list(PROBE_FEATURE_NAMES[len(
            STATE_ONLY_FEATURE_NAMES
        ):]),
        "horizons": horizons,
    }


def _full_predictions(
    full_state: dict[str, object],
    horizon: int,
    target: str,
    features: np.ndarray,
) -> np.ndarray:
    fitted = full_state["horizons"][str(horizon)]
    normalization = fitted["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    coefficients = np.asarray(fitted["coefficients"][target], dtype=np.float64)
    design = np.column_stack((np.ones(features.shape[0]), (features - mean) / scale))
    return np.clip(design @ coefficients, 0.0, 1.0)


def _state_only_predictions(
    state: dict[str, object],
    horizon: int,
    target: str,
    features: np.ndarray,
) -> np.ndarray:
    fitted = state["horizons"][str(horizon)]["targets"][target]
    normalization = fitted["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    coefficients = np.asarray(fitted["coefficients"], dtype=np.float64)
    selected = features[:, STATE_ONLY_FEATURE_INDICES]
    design = np.column_stack((np.ones(selected.shape[0]), (selected - mean) / scale))
    return np.clip(design @ coefficients, 0.0, 1.0)


def _calibration_bins(
    probabilities: np.ndarray,
    labels: np.ndarray,
    *,
    bins: int,
) -> dict[str, object]:
    if bins <= 1:
        raise ValueError("calibration diagnosis requires at least two bins")
    indices = np.minimum(
        np.floor(probabilities * bins).astype(np.int64),
        bins - 1,
    )
    rows = []
    ece = 0.0
    for index in range(bins):
        members = indices == index
        count = int(np.sum(members))
        if count:
            mean_prediction = float(np.mean(probabilities[members]))
            event_rate = float(np.mean(labels[members]))
            ece += count / labels.size * abs(mean_prediction - event_rate)
        else:
            mean_prediction = None
            event_rate = None
        rows.append({
            "bin": index,
            "lower": index / bins,
            "upper": (index + 1) / bins,
            "rows": count,
            "mean_prediction": mean_prediction,
            "event_rate": event_rate,
        })
    return {"bins": rows, "expected_calibration_error": ece}


def _per_episode_brier_delta(
    full: np.ndarray,
    state_only: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    episode_ids: tuple[str, ...],
) -> list[dict[str, object]]:
    targets = labels.astype(np.float64)
    row_delta = (full - targets) ** 2 - (state_only - targets) ** 2
    return [
        {
            "episode_id": episode_id,
            "rows": int(np.sum(episode_indices == index)),
            "full_minus_state_only_brier": float(np.mean(
                row_delta[episode_indices == index]
            )),
        }
        for index, episode_id in enumerate(episode_ids)
    ]


def diagnose_incremental_action_signal(
    full_state: dict[str, object],
    state_only_state: dict[str, object],
    validation: FactualProbeDataset,
    source_evaluation: dict[str, object],
    *,
    supported_hit_horizons: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    reproduction_tolerance: float,
) -> dict[str, object]:
    """Compare frozen full and state-only probes on reused held-out episodes."""
    if not supported_hit_horizons:
        raise ValueError("incremental-action diagnosis needs supported HIT horizons")
    results = {}
    hit_passes = 0
    shield_passes = 0
    maximum_reproduction_error = 0.0
    for horizon_index, view in enumerate(validation.horizons):
        target_results = {}
        for target_index, (name, labels) in enumerate((
            ("hit", view.hit_labels),
            ("shield_collapse", view.shield_collapse_labels),
        )):
            full = _full_predictions(full_state, view.horizon, name, view.features)
            state_only = _state_only_predictions(
                state_only_state,
                view.horizon,
                name,
                view.features,
            )
            full_metrics = _binary_metrics(full, labels)
            state_metrics = _binary_metrics(state_only, labels)
            recorded_brier = float(
                source_evaluation["horizons"][str(view.horizon)]["targets"][name]
                ["candidate"]["brier"]
            )
            error = abs(float(full_metrics["brier"]) - recorded_brier)
            maximum_reproduction_error = max(maximum_reproduction_error, error)
            bootstrap = _episode_bootstrap_brier_delta(
                full,
                state_only,
                labels,
                view.episode_indices,
                episode_count=len(view.episode_ids),
                samples=bootstrap_samples,
                seed=bootstrap_seed + horizon_index * 2 + target_index,
            )
            eligible = (
                view.horizon in supported_hit_horizons
                if name == "hit" else True
            )
            passed = eligible and float(bootstrap["upper_95"]) < 0.0
            if name == "hit":
                hit_passes += int(passed)
            else:
                shield_passes += int(passed)
            target_results[name] = {
                "eligible_for_gate": eligible,
                "incremental_action_gate_passed": passed,
                "full_current_root_action": full_metrics,
                "state_only": state_metrics,
                "whole_episode_bootstrap_full_minus_state_only_brier": bootstrap,
                "full_calibration_10_bin": _calibration_bins(
                    full, labels, bins=calibration_bins
                ),
                "state_only_calibration_10_bin": _calibration_bins(
                    state_only, labels, bins=calibration_bins
                ),
                "per_episode": _per_episode_brier_delta(
                    full,
                    state_only,
                    labels,
                    view.episode_indices,
                    view.episode_ids,
                ),
                "source_full_brier_reproduction_error": error,
            }
        results[str(view.horizon)] = {"targets": target_results}
    reproduced = maximum_reproduction_error <= reproduction_tolerance
    if not reproduced:
        decision = "stop-source-probe-reproduction-failed"
    elif hit_passes:
        decision = "proceed-action-relative-current-root-signal"
    else:
        decision = "retain-state-only-risk-representation"
    return {
        "schema": DIAGNOSIS_SCHEMA,
        "horizons": results,
        "summary": {
            "source_probe_reproduced": reproduced,
            "maximum_source_brier_reproduction_error": maximum_reproduction_error,
            "supported_hit_horizons": list(supported_hit_horizons),
            "hit_horizons_with_incremental_action_signal": hit_passes,
            "shield_horizons_with_incremental_action_signal": shield_passes,
            "decision": decision,
            "independent_confirmation": False,
            "history_admitted": False,
            "value_learning_admitted": False,
        },
    }
