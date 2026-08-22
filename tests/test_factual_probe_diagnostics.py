from __future__ import annotations

from th06_rl.factual_probe_diagnostics import (
    STATE_ONLY_FEATURE_NAMES,
    diagnose_incremental_action_signal,
    fit_state_only_probe_models,
)
from th06_rl.factual_probes import (
    evaluate_factual_probe_models,
    fit_factual_probe_models,
    load_factual_probe_dataset,
)

from tests.test_factual_probes import _probe_episode


def _dataset(tmp_path, name):
    return load_factual_probe_dataset(
        (
            _probe_episode(tmp_path / f"{name}-a"),
            _probe_episode(tmp_path / f"{name}-b", mirrored=True),
        ),
        horizons=(1,),
    )


def test_incremental_action_diagnosis_reproduces_the_frozen_full_probe(
    tmp_path,
) -> None:
    train = _dataset(tmp_path, "train")
    validation = _dataset(tmp_path, "validation")
    full = fit_factual_probe_models(train, ridge_l2=0.01)
    state_only = fit_state_only_probe_models(train, ridge_l2=0.01)
    source = evaluate_factual_probe_models(
        full,
        validation,
        dynamics_mse_ratio_max=2.0,
        execution_match_rate_min=0.0,
        mismatch_rows_min=1,
        mismatch_mse_ratio_max=2.0,
        minimum_train_positives=1,
        minimum_validation_positives=1,
        minimum_validation_negatives=1,
        bootstrap_samples=20,
        bootstrap_seed=3,
    )

    diagnosis = diagnose_incremental_action_signal(
        full,
        state_only,
        validation,
        source,
        supported_hit_horizons=(1,),
        bootstrap_samples=50,
        bootstrap_seed=9,
        calibration_bins=10,
        reproduction_tolerance=1e-12,
    )

    assert state_only["feature_names"] == list(STATE_ONLY_FEATURE_NAMES)
    assert state_only["removed_feature_names"]
    assert diagnosis["summary"]["source_probe_reproduced"] is True
    assert diagnosis["summary"]["independent_confirmation"] is False
    assert diagnosis["summary"]["history_admitted"] is False
    hit = diagnosis["horizons"]["1"]["targets"]["hit"]
    assert hit["source_full_brier_reproduction_error"] == 0.0
    assert sum(
        row["rows"] for row in hit["full_calibration_10_bin"]["bins"]
    ) == validation.horizons[0].rows
    assert len(hit["per_episode"]) == 2


def test_incremental_action_diagnosis_stops_on_source_reproduction_error(
    tmp_path,
) -> None:
    train = _dataset(tmp_path, "train")
    validation = _dataset(tmp_path, "validation")
    full = fit_factual_probe_models(train, ridge_l2=0.01)
    state_only = fit_state_only_probe_models(train, ridge_l2=0.01)
    source = evaluate_factual_probe_models(
        full,
        validation,
        dynamics_mse_ratio_max=2.0,
        execution_match_rate_min=0.0,
        mismatch_rows_min=1,
        mismatch_mse_ratio_max=2.0,
        minimum_train_positives=1,
        minimum_validation_positives=1,
        minimum_validation_negatives=1,
        bootstrap_samples=20,
        bootstrap_seed=3,
    )
    source["horizons"]["1"]["targets"]["hit"]["candidate"]["brier"] += 0.1

    diagnosis = diagnose_incremental_action_signal(
        full,
        state_only,
        validation,
        source,
        supported_hit_horizons=(1,),
        bootstrap_samples=20,
        bootstrap_seed=9,
        calibration_bins=10,
        reproduction_tolerance=1e-12,
    )

    assert diagnosis["summary"]["source_probe_reproduced"] is False
    assert diagnosis["summary"]["decision"] == (
        "stop-source-probe-reproduction-failed"
    )
