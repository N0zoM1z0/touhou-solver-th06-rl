from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from th06_rl.iql_actor_learning import IqlActorModel, NativeIqlActorPopulation
from th06_rl.low_rank_learning import FeatureRoleLayout
from th06_rl.native_decision_conformance import (
    actor_centered_forward_reference,
    actor_forward_reference,
    certify_mean_population_decision,
    native_order_float32_centered_advantages,
    native_order_float32_scores,
    native_order_centered_portability_reference,
)


def _model(seed: int = 4813) -> IqlActorModel:
    generator = np.random.default_rng(seed)
    layout = FeatureRoleLayout(
        names=("observation:x", "action:dx", "observation:y"),
        state_indices=(0, 2),
        action_indices=(1,),
    )
    return IqlActorModel(
        layout=layout,
        state_mean=np.asarray([0.2, -0.1], dtype=np.float32),
        state_scale=np.asarray([1.3, 0.7], dtype=np.float32),
        action_mean=np.asarray([0.1], dtype=np.float32),
        action_scale=np.asarray([0.8], dtype=np.float32),
        state_hidden_weight=generator.normal(size=(2, 5)).astype(np.float32),
        state_hidden_bias=generator.normal(size=5).astype(np.float32),
        state_latent_weight=generator.normal(size=(5, 3)).astype(np.float32),
        state_latent_bias=generator.normal(size=3).astype(np.float32),
        action_hidden_weight=generator.normal(size=(1, 5)).astype(np.float32),
        action_hidden_bias=generator.normal(size=5).astype(np.float32),
        action_latent_weight=generator.normal(size=(5, 3)).astype(np.float32),
        action_latent_bias=generator.normal(size=3).astype(np.float32),
        action_score_weight=generator.normal(size=5).astype(np.float32),
        action_score_bias=float(generator.normal()),
    )


def test_scalar_reference_and_envelope_cover_frozen_native_kernel() -> None:
    library = (
        Path(__file__).resolve().parents[1]
        / "build/native/libth06_rl_ranker.so"
    )
    if not library.is_file():
        import pytest
        pytest.skip("native ranker build is absent")
    model = _model()
    rows = [(0.7, value, -0.4) for value in (-1.0, -0.5, 0.0, 0.5, 1.0)]
    native = NativeIqlActorPopulation(
        library,
        expected_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        models=[model],
    )
    actual = np.asarray(native.predict(rows)[0], dtype=np.float64)
    scalar = native_order_float32_scores(model, rows).astype(np.float64)
    reference = actor_forward_reference(model, rows)

    assert np.allclose(actual, scalar, rtol=0.0, atol=2e-6)
    assert np.all(
        np.abs(actual - reference.scores) <= reference.error_bounds
    )
    assert np.all(
        np.abs(model.predict(rows) - reference.scores)
        <= reference.error_bounds
    )
    centered = np.asarray(
        native.predict_centered(rows, baseline_index=2)[0], dtype=np.float64
    )
    centered_reference = actor_centered_forward_reference(
        model, rows, baseline_index=2
    )
    centered_scalar = native_order_float32_centered_advantages(
        model, rows, baseline_index=2
    )
    portability = native_order_centered_portability_reference(
        model, rows, baseline_index=2
    )
    assert centered[2] == 0.0
    assert np.array_equal(portability.scores, centered_scalar)
    assert np.all(
        np.abs(centered - portability.scores) <= portability.error_bounds
    )
    assert np.all(
        np.abs(centered - centered_reference.scores)
        <= centered_reference.error_bounds
    )


def test_decision_certificate_uses_baseline_centered_margin() -> None:
    certificate = certify_mean_population_decision(
        [[1000.0, 1000.25, 1000.10], [900.0, 900.20, 900.05]],
        [[0.01, 0.01, 0.01], [0.01, 0.01, 0.01]],
        ("stay", "left", "right"),
        "stay",
        (True, True, True),
    )

    assert certificate.choice == "left"
    assert certificate.certified is True
    assert certificate.decision_margin > certificate.error_envelope


def test_decision_certificate_rejects_numerically_unresolved_choice() -> None:
    certificate = certify_mean_population_decision(
        [[0.0, 0.0001], [0.0, 0.0001]],
        [[0.001, 0.001], [0.001, 0.001]],
        ("stay", "left"),
        "stay",
        (True, True),
    )

    assert certificate.choice == "left"
    assert certificate.certified is False
    assert certificate.margin_ratio < 1.0


def test_decision_certificate_honors_support_and_lexical_tie_break() -> None:
    certificate = certify_mean_population_decision(
        [[0.0, 1.0, 1.0]],
        [[0.0, 0.0, 0.0]],
        ("stay", "left", "right"),
        "stay",
        (True, False, True),
    )

    assert certificate.choice == "right"
    assert certificate.certified is True
