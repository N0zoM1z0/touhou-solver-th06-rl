from __future__ import annotations

from th06_rl.factual_probe_boundary_diagnostics import load_boundary_probe_dataset
from th06_rl.factual_probe_confirmation import evaluate_frozen_probe_confirmation
from th06_rl.factual_probe_diagnostics import fit_state_only_probe_models
from th06_rl.factual_probes import fit_factual_probe_models

from tests.test_factual_probes import _probe_episode


def test_fresh_confirmation_never_admits_value_or_online_policy(tmp_path) -> None:
    dataset = load_boundary_probe_dataset(
        (
            _probe_episode(tmp_path / "a"),
            _probe_episode(tmp_path / "b", mirrored=True),
        ),
        horizons=(1,),
    )
    full = fit_factual_probe_models(dataset.factual, ridge_l2=0.01)
    state_only = fit_state_only_probe_models(dataset.factual, ridge_l2=0.01)

    result = evaluate_frozen_probe_confirmation(
        full,
        state_only,
        dataset,
        primary_horizon=1,
        bootstrap_samples=20,
        bootstrap_seed=7,
        calibration_bins=10,
        minimum_overall_positives=1,
        minimum_overall_negatives=1,
        minimum_nonbaseline_positives=1,
        minimum_prefirst_hit_positives=1,
        minimum_episodes_favoring_full=1,
        calibration_in_the_large_absolute_max=1.0,
        full_ece_over_state_only_max=1.0,
    )

    assert result["primary"]["horizon_game_frames"] == 1
    assert result["primary"][
        "whole_episode_bootstrap_full_minus_state_only_brier"
    ]["unit"] == "complete-physical-episode"
    assert result["summary"]["independent_confirmation"] is True
    assert result["summary"]["causal_action_effect_identified"] is False
    assert result["summary"]["history_admitted"] is False
    assert result["summary"]["value_learning_admitted"] is False
    assert result["summary"]["online_policy_admitted"] is False
