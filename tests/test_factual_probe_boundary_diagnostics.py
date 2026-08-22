from __future__ import annotations

import numpy as np

from th06_rl.factual_probe_boundary_diagnostics import (
    LIFECYCLE_STRATA,
    diagnose_probe_boundaries,
    load_boundary_probe_dataset,
)
from th06_rl.factual_probe_diagnostics import fit_state_only_probe_models
from th06_rl.factual_probes import fit_factual_probe_models

from tests.test_factual_probes import _probe_episode


def test_boundary_loader_exactly_reproduces_factual_rows_and_hit_lifecycle(
    tmp_path,
) -> None:
    dataset = load_boundary_probe_dataset(
        (_probe_episode(tmp_path / "episode"),),
        horizons=(1,),
    )

    boundary = dataset.horizons[0]
    factual = dataset.factual.horizons[0]
    assert np.array_equal(boundary.features, factual.features)
    assert np.array_equal(boundary.hit_labels, factual.hit_labels)
    assert boundary.lifecycle_strata == (
        LIFECYCLE_STRATA[0],
        LIFECYCLE_STRATA[0],
        LIFECYCLE_STRATA[1],
    )
    assert boundary.frames_since_prior_hit == (None, None, 2)
    assert tuple(boundary.behavior_probabilities) == (1.0, 1.0, 1.0)


def test_boundary_diagnosis_is_descriptive_and_handles_empty_support_strata(
    tmp_path,
) -> None:
    dataset = load_boundary_probe_dataset(
        (
            _probe_episode(tmp_path / "a"),
            _probe_episode(tmp_path / "b", mirrored=True),
        ),
        horizons=(1,),
    )
    full = fit_factual_probe_models(dataset.factual, ridge_l2=0.01)
    state_only = fit_state_only_probe_models(dataset.factual, ridge_l2=0.01)

    diagnosis = diagnose_probe_boundaries(
        full,
        state_only,
        dataset,
        calibration_bins=10,
    )

    horizon = diagnosis["horizons"]["1"]
    assert horizon["overall"]["rows"] == dataset.horizons[0].rows
    assert horizon["support"]["published_differs_from_baseline"]["rows"] == 0
    assert sum(row["rows"] for row in horizon["lifecycle"].values()) == (
        dataset.horizons[0].rows
    )
    assert diagnosis["summary"]["fresh_confirmation_required"] is True
    assert diagnosis["summary"]["history_admitted"] is False
    assert diagnosis["summary"]["value_learning_admitted"] is False
