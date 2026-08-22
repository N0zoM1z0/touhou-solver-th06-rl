"""Train-only scalar calibration for the frozen factual HIT-risk probe.

The calibrator is a two-parameter monotone probability surface over the
unclipped scalar score of the already-fitted L2 ridge probe.  It neither
changes the feature representation nor constructs a counterfactual successor.
"""

from __future__ import annotations

import math

import numpy as np

from .factual_probe_boundary_diagnostics import (
    BoundaryProbeDataset,
    _calibration_summary,
    _stratum_result,
)
from .factual_probe_diagnostics import _full_predictions, _state_only_predictions
from .factual_probes import (
    FactualProbeDataset,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


CALIBRATOR_SCHEMA = "th06-rl-l2e-train-only-platt-calibrator-v1"
CALIBRATION_EVALUATION_SCHEMA = "th06-rl-l2e-calibration-evaluation-v1"


def _raw_full_scores(
    full_state: dict[str, object],
    horizon: int,
    features: np.ndarray,
) -> np.ndarray:
    fitted = full_state["horizons"][str(horizon)]
    normalization = fitted["normalization"]
    mean = np.asarray(normalization["mean"], dtype=np.float64)
    scale = np.asarray(normalization["scale"], dtype=np.float64)
    coefficients = np.asarray(fitted["coefficients"]["hit"], dtype=np.float64)
    design = np.column_stack((np.ones(features.shape[0]), (features - mean) / scale))
    scores = design @ coefficients
    if np.any(~np.isfinite(scores)):
        raise ValueError("frozen full probe produced a non-finite raw score")
    return scores


def _sigmoid(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=np.float64)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponent = np.exp(values[~positive])
    result[~positive] = exponent / (1.0 + exponent)
    return result


def _objective(design: np.ndarray, labels: np.ndarray, coefficients: np.ndarray) -> float:
    logits = design @ coefficients
    return float(np.mean(np.logaddexp(0.0, logits) - labels * logits))


def fit_train_only_platt_calibrator(
    full_state: dict[str, object],
    dataset: FactualProbeDataset,
    *,
    horizon: int,
    maximum_updates: int,
    minimum_updates: int,
    gradient_inf_tolerance: float,
    maximum_line_search_steps: int,
) -> dict[str, object]:
    """Fit one affine-logistic map over a frozen raw L2 HIT score."""
    if (
        maximum_updates <= 0
        or not 0 <= minimum_updates <= maximum_updates
        or not math.isfinite(gradient_inf_tolerance)
        or gradient_inf_tolerance <= 0.0
        or maximum_line_search_steps <= 0
    ):
        raise ValueError("Platt optimizer settings are invalid")
    try:
        view = next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("Platt horizon is absent from the train dataset") from error
    targets = view.hit_labels.astype(np.float64)
    positives = int(np.sum(view.hit_labels))
    if positives == 0 or positives == view.rows:
        raise ValueError("Platt fitting requires both HIT outcomes")
    raw_scores = _raw_full_scores(full_state, horizon, view.features)
    score_mean = float(np.mean(raw_scores))
    score_scale = float(np.std(raw_scores))
    if not math.isfinite(score_scale) or score_scale < 1e-12:
        raise ValueError("frozen full score has no calibration variation")
    normalized = (raw_scores - score_mean) / score_scale
    design = np.column_stack((np.ones(view.rows), normalized))
    prevalence = positives / view.rows
    coefficients = np.asarray([
        math.log(prevalence / (1.0 - prevalence)),
        0.0,
    ], dtype=np.float64)
    initial_objective = _objective(design, targets, coefficients)
    converged = False
    accepted_steps = 0
    final_gradient_inf = math.inf
    final_objective = initial_objective
    for update in range(maximum_updates + 1):
        logits = design @ coefficients
        probabilities = _sigmoid(logits)
        gradient = design.T @ (probabilities - targets) / view.rows
        final_gradient_inf = float(np.max(np.abs(gradient)))
        final_objective = _objective(design, targets, coefficients)
        if update >= minimum_updates and final_gradient_inf <= gradient_inf_tolerance:
            converged = True
            break
        if update == maximum_updates:
            break
        weights = probabilities * (1.0 - probabilities)
        hessian = (design.T * weights) @ design / view.rows
        hessian += np.eye(2, dtype=np.float64) * 1e-12
        direction = np.linalg.solve(hessian, gradient)
        directional_derivative = float(gradient @ direction)
        accepted = False
        step = 1.0
        for _line_search in range(maximum_line_search_steps):
            proposal = coefficients - step * direction
            proposal_objective = _objective(design, targets, proposal)
            if proposal_objective <= final_objective - 1e-4 * step * directional_derivative:
                coefficients = proposal
                accepted = True
                accepted_steps += 1
                break
            step *= 0.5
        if not accepted:
            break
    if np.any(~np.isfinite(coefficients)) or not math.isfinite(final_objective):
        raise ValueError("Platt fit produced non-finite state")
    raw_slope = float(coefficients[1] / score_scale)
    raw_intercept = float(coefficients[0] - coefficients[1] * score_mean / score_scale)
    return {
        "schema": CALIBRATOR_SCHEMA,
        "model": "affine-logistic-over-frozen-unclipped-l2-score",
        "horizon_game_frames": horizon,
        "target": "physical-hit-within-fixed-horizon",
        "source_feature_names": list(full_state["feature_names"]),
        "score_normalization": {"mean": score_mean, "scale": score_scale},
        "normalized_coefficients": {
            "intercept": float(coefficients[0]),
            "slope": float(coefficients[1]),
        },
        "raw_score_coefficients": {
            "intercept": raw_intercept,
            "slope": raw_slope,
        },
        "train": {
            "rows": view.rows,
            "positives": positives,
            "prevalence": prevalence,
        },
        "optimization": {
            "method": "damped-newton-with-backtracking",
            "maximum_updates": maximum_updates,
            "minimum_updates": minimum_updates,
            "gradient_inf_tolerance": gradient_inf_tolerance,
            "maximum_line_search_steps": maximum_line_search_steps,
            "accepted_steps": accepted_steps,
            "converged": converged,
            "final_gradient_inf": final_gradient_inf,
            "initial_mean_nll": initial_objective,
            "final_mean_nll": final_objective,
        },
    }


def calibrated_predictions(
    full_state: dict[str, object],
    calibrator: dict[str, object],
    features: np.ndarray,
) -> np.ndarray:
    if calibrator.get("schema") != CALIBRATOR_SCHEMA:
        raise ValueError("Platt calibrator schema mismatch")
    horizon = int(calibrator["horizon_game_frames"])
    raw_scores = _raw_full_scores(full_state, horizon, features)
    normalization = calibrator["score_normalization"]
    coefficients = calibrator["normalized_coefficients"]
    normalized = (
        raw_scores - float(normalization["mean"])
    ) / float(normalization["scale"])
    logits = float(coefficients["intercept"]) + float(coefficients["slope"]) * normalized
    probabilities = _sigmoid(logits)
    if np.any(~np.isfinite(probabilities)):
        raise ValueError("Platt prediction is non-finite")
    return probabilities


def evaluate_train_only_platt_calibrator(
    full_state: dict[str, object],
    state_only_state: dict[str, object],
    calibrator: dict[str, object],
    dataset: BoundaryProbeDataset,
    *,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_overall_positives: int,
    minimum_overall_negatives: int,
    minimum_nonbaseline_positives: int,
    minimum_prefirst_hit_positives: int,
    minimum_episodes_favoring_calibrated: int,
    calibration_in_the_large_absolute_max: float,
    calibrated_ece_over_state_only_max: float,
) -> dict[str, object]:
    """Evaluate the frozen calibrator once on complete reused L2d episodes."""
    horizon = int(calibrator["horizon_game_frames"])
    try:
        view = next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("calibration evaluation horizon is absent") from error
    labels = view.hit_labels
    calibrated = calibrated_predictions(full_state, calibrator, view.features)
    uncalibrated = _full_predictions(full_state, horizon, "hit", view.features)
    state_only = _state_only_predictions(
        state_only_state, horizon, "hit", view.features
    )
    all_rows = np.ones(view.rows, dtype=np.bool_)
    baseline_equal = np.asarray([
        published == baseline
        for published, baseline in zip(
            view.published_actions, view.baseline_actions, strict=True
        )
    ], dtype=np.bool_)
    lifecycle = np.asarray(view.lifecycle_strata, dtype=object)

    def stratum(mask: np.ndarray) -> dict[str, object]:
        return _stratum_result(
            mask,
            full=calibrated,
            state_only=state_only,
            labels=labels,
            episode_indices=view.episode_indices,
            episode_ids=view.episode_ids,
            calibration_bins=calibration_bins,
        )

    overall = stratum(all_rows)
    nonbaseline = stratum(~baseline_equal)
    baseline = stratum(baseline_equal)
    prefirst = stratum(lifecycle == "pre-first-hit")
    calibrated_vs_state = _episode_bootstrap_brier_delta(
        calibrated,
        state_only,
        labels,
        view.episode_indices,
        episode_count=len(view.episode_ids),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    calibrated_vs_uncalibrated = _episode_bootstrap_brier_delta(
        calibrated,
        uncalibrated,
        labels,
        view.episode_indices,
        episode_count=len(view.episode_ids),
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    calibrated_metrics = _binary_metrics(calibrated, labels)
    uncalibrated_metrics = _binary_metrics(uncalibrated, labels)
    state_metrics = _binary_metrics(state_only, labels)
    calibrated_calibration = _calibration_summary(
        calibrated, labels, bins=calibration_bins
    )
    uncalibrated_calibration = _calibration_summary(
        uncalibrated, labels, bins=calibration_bins
    )
    state_calibration = _calibration_summary(
        state_only, labels, bins=calibration_bins
    )
    positives = int(np.sum(labels))
    negatives = view.rows - positives
    sufficient = (
        positives >= minimum_overall_positives
        and negatives >= minimum_overall_negatives
    )
    monotone = float(calibrator["raw_score_coefficients"]["slope"]) > 0.0
    optimizer_converged = bool(calibrator["optimization"]["converged"])
    proper_score_repaired = float(calibrated_vs_uncalibrated["upper_95"]) < 0.0
    signal_preserved = (
        sufficient
        and float(calibrated_vs_state["upper_95"]) < 0.0
        and int(overall["episodes_favoring_full"])
        >= minimum_episodes_favoring_calibrated
    )
    support_preserved = (
        int(nonbaseline["positives"]) >= minimum_nonbaseline_positives
        and int(nonbaseline["episode_count"]) == len(view.episode_ids)
        and float(nonbaseline["full_minus_state_only_brier"]) < 0.0
    )
    lifecycle_preserved = (
        int(prefirst["positives"]) >= minimum_prefirst_hit_positives
        and int(prefirst["episode_count"]) == len(view.episode_ids)
        and float(prefirst["full_minus_state_only_brier"]) < 0.0
    )
    calibration_ready = (
        abs(float(calibrated_calibration["calibration_in_the_large"]))
        <= calibration_in_the_large_absolute_max
        and float(calibrated_calibration["expected_calibration_error"])
        - float(state_calibration["expected_calibration_error"])
        <= calibrated_ece_over_state_only_max
    )
    selected = all((
        optimizer_converged,
        monotone,
        proper_score_repaired,
        signal_preserved,
        support_preserved,
        lifecycle_preserved,
        calibration_ready,
    ))
    return {
        "schema": CALIBRATION_EVALUATION_SCHEMA,
        "horizon_game_frames": horizon,
        "metrics": {
            "calibrated_full": calibrated_metrics,
            "uncalibrated_full": uncalibrated_metrics,
            "state_only": state_metrics,
        },
        "calibration": {
            "calibrated_full": calibrated_calibration,
            "uncalibrated_full": uncalibrated_calibration,
            "state_only": state_calibration,
        },
        "whole_episode_bootstrap": {
            "calibrated_minus_state_only_brier": calibrated_vs_state,
            "calibrated_minus_uncalibrated_full_brier": calibrated_vs_uncalibrated,
        },
        "strata": {
            "published_equals_baseline": baseline,
            "published_differs_from_baseline": nonbaseline,
            "pre_first_hit": prefirst,
        },
        "gates": {
            "optimizer_converged": optimizer_converged,
            "positive_monotone_slope": monotone,
            "overall_support_sufficient": sufficient,
            "proper_score_repaired": proper_score_repaired,
            "signal_preserved": signal_preserved,
            "nonbaseline_support_preserved": support_preserved,
            "prefirst_hit_lifecycle_preserved": lifecycle_preserved,
            "calibration_readiness_passed": calibration_ready,
            "selected_for_fresh_confirmation": selected,
        },
        "summary": {
            "decision": (
                "select-platt-calibrated-probe-for-fresh-confirmation"
                if selected else "reject-train-only-platt-calibration"
            ),
            "independent_confirmation": False,
            "history_admitted": False,
            "value_learning_admitted": False,
            "online_policy_admitted": False,
        },
    }
