"""Independent confirmation of frozen current-root factual HIT probes."""

from __future__ import annotations

import numpy as np

from .factual_probe_boundary_diagnostics import (
    BoundaryProbeDataset,
    diagnose_probe_boundaries,
)
from .factual_probe_diagnostics import _full_predictions, _state_only_predictions
from .factual_probes import _episode_bootstrap_brier_delta


CONFIRMATION_SCHEMA = "th06-rl-l2d-frozen-probe-confirmation-v1"


def evaluate_frozen_probe_confirmation(
    full_state: dict[str, object],
    state_only_state: dict[str, object],
    dataset: BoundaryProbeDataset,
    *,
    primary_horizon: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_overall_positives: int,
    minimum_overall_negatives: int,
    minimum_nonbaseline_positives: int,
    minimum_prefirst_hit_positives: int,
    minimum_episodes_favoring_full: int,
    calibration_in_the_large_absolute_max: float,
    full_ece_over_state_only_max: float,
) -> dict[str, object]:
    """Evaluate untouched L2/L2b models on wholly new complete episodes."""
    if bootstrap_samples <= 0:
        raise ValueError("confirmation bootstrap samples must be positive")
    if not 0 <= minimum_episodes_favoring_full <= len(dataset.factual.episode_ids):
        raise ValueError("confirmation episode-direction threshold is invalid")
    boundaries = diagnose_probe_boundaries(
        full_state,
        state_only_state,
        dataset,
        calibration_bins=calibration_bins,
    )
    try:
        view = next(row for row in dataset.horizons if row.horizon == primary_horizon)
    except StopIteration as error:
        raise ValueError("primary confirmation horizon is absent") from error
    full = _full_predictions(full_state, primary_horizon, "hit", view.features)
    state_only = _state_only_predictions(
        state_only_state, primary_horizon, "hit", view.features
    )
    bootstrap = _episode_bootstrap_brier_delta(
        full,
        state_only,
        view.hit_labels,
        view.episode_indices,
        episode_count=len(view.episode_ids),
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    primary = boundaries["horizons"][str(primary_horizon)]
    overall = primary["overall"]
    nonbaseline = primary["support"]["published_differs_from_baseline"]
    prefirst = primary["lifecycle"]["pre-first-hit"]
    overall_support = (
        int(overall["positives"]) >= minimum_overall_positives
        and int(overall["rows"]) - int(overall["positives"])
        >= minimum_overall_negatives
    )
    signal_replication = (
        overall_support
        and float(bootstrap["upper_95"]) < 0.0
        and int(overall["episodes_favoring_full"])
        >= minimum_episodes_favoring_full
    )
    support_boundary = (
        int(nonbaseline["positives"]) >= minimum_nonbaseline_positives
        and int(nonbaseline["episode_count"]) == len(view.episode_ids)
        and float(nonbaseline["full_minus_state_only_brier"]) < 0.0
    )
    lifecycle_boundary = (
        int(prefirst["positives"]) >= minimum_prefirst_hit_positives
        and int(prefirst["episode_count"]) == len(view.episode_ids)
        and float(prefirst["full_minus_state_only_brier"]) < 0.0
    )
    full_calibration = overall["full_calibration"]
    state_calibration = overall["state_only_calibration"]
    calibration_ready = (
        abs(float(full_calibration["calibration_in_the_large"]))
        <= calibration_in_the_large_absolute_max
        and float(full_calibration["expected_calibration_error"])
        - float(state_calibration["expected_calibration_error"])
        <= full_ece_over_state_only_max
    )
    if not signal_replication:
        decision = "do-not-confirm-action-relative-hit-signal"
    elif not support_boundary or not lifecycle_boundary:
        decision = "inconclusive-boundary-localized-action-signal"
    elif not calibration_ready:
        decision = "confirm-predictive-signal-calibration-not-ready"
    else:
        decision = "confirm-predictive-action-signal-not-action-value"
    return {
        "schema": CONFIRMATION_SCHEMA,
        "boundaries": boundaries,
        "primary": {
            "horizon_game_frames": primary_horizon,
            "whole_episode_bootstrap_full_minus_state_only_brier": bootstrap,
            "minimum_overall_positives": minimum_overall_positives,
            "minimum_overall_negatives": minimum_overall_negatives,
            "minimum_nonbaseline_positives": minimum_nonbaseline_positives,
            "minimum_prefirst_hit_positives": minimum_prefirst_hit_positives,
            "minimum_episodes_favoring_full": minimum_episodes_favoring_full,
            "overall_support_sufficient": overall_support,
            "signal_replication_passed": signal_replication,
            "nonbaseline_support_boundary_passed": support_boundary,
            "prefirst_hit_lifecycle_boundary_passed": lifecycle_boundary,
            "calibration_readiness_passed": calibration_ready,
            "calibration_in_the_large_absolute_max": (
                calibration_in_the_large_absolute_max
            ),
            "full_ece_over_state_only_max": full_ece_over_state_only_max,
        },
        "summary": {
            "decision": decision,
            "independent_confirmation": True,
            "causal_action_effect_identified": False,
            "history_admitted": False,
            "value_learning_admitted": False,
            "online_policy_admitted": False,
        },
    }
