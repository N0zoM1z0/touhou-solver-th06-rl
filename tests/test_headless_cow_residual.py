from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_headless_cow_residual import _metrics, choose_threshold, risk_label
from scripts.collect_headless_dagger import DistilledRanker, supported_residual_ranking
from scripts.train_headless_cow_value import ValueGroup
from tests.test_headless_teacher import decision


def value_group(*, selected: str, completed_or_best: tuple[str, ...]) -> ValueGroup:
    current = replace(decision(), selected_action=selected)
    return ValueGroup(
        seed=current.seed,
        observation_sha256="digest",
        decision=current,
        actions=("stay", "left"),
        labels=(0, 1),
        best_actions=("left",),
        completed_or_best_actions=completed_or_best,
    )


def test_risk_label_marks_only_incumbent_without_robust_value_support() -> None:
    assert risk_label(value_group(selected="left", completed_or_best=("left",))) == 0
    assert risk_label(value_group(selected="stay", completed_or_best=("left",))) == 1


def test_threshold_is_selected_from_training_evidence_only() -> None:
    threshold, report = choose_threshold(
        [1, 0, 1, 0],
        np.asarray([0.98, 0.10, 0.92, 0.20]),
        np.asarray([0.05, 0.15, 0.25, 0.30]),
        min_precision=1.0,
        max_behavior_activation=0.0,
    )

    assert threshold == pytest.approx(0.92)
    assert report["precision"] == 1.0
    assert report["recall"] == 1.0
    assert report["behavior_activation_ratio"] == 0.0


def test_threshold_fails_closed_when_no_training_threshold_is_safe() -> None:
    threshold, report = choose_threshold(
        [1, 0],
        np.asarray([0.6, 0.9]),
        np.asarray([0.95]),
        min_precision=1.0,
        max_behavior_activation=0.0,
    )

    assert threshold > 1.0
    assert report["status"] == "no-safe-training-threshold"
    assert report["predicted_risk_groups"] == 0


def test_gate_metrics_reject_misaligned_inputs() -> None:
    with pytest.raises(ValueError, match="differ in length"):
        _metrics([1, 0], np.asarray([0.9]), 0.5)


def test_residual_correction_cannot_change_native_safe_membership() -> None:
    assert supported_residual_ranking(("stay", "left"), ("left", "stay")) == (
        "left",
        "stay",
    )
    with pytest.raises(ValueError, match="same nonempty native-safe set"):
        supported_residual_ranking(("stay", "left"), ("left", "right"))


@pytest.mark.parametrize(
    ("probability", "expected", "activated", "overrode"),
    [
        (0.95, ("left", "stay"), True, True),
        (0.50, ("stay", "left"), False, False),
    ],
)
def test_residual_ranker_records_gate_and_override_diagnostics(
    probability: float,
    expected: tuple[str, ...],
    activated: bool,
    overrode: bool,
) -> None:
    ranker = DistilledRanker.__new__(DistilledRanker)
    ranker.threads = 1
    ranker.last_decision_diagnostics = None
    ranker.members = []
    ranker.residual = {
        "base": "base",
        "correction": "correction",
        "gate_model": SimpleNamespace(
            booster_=SimpleNamespace(
                predict=lambda _matrix, num_threads: np.asarray([probability])
            )
        ),
        "gate_encoder": SimpleNamespace(encode=lambda rows: rows),
        "threshold": 0.9,
    }
    ranker._rank_member = lambda member, _decision: (
        ("stay", "left") if member == "base" else ("left", "stay")
    )

    assert ranker.rank_decision(decision()) == expected
    assert ranker.last_decision_diagnostics == {
        "kind": "supported-residual",
        "risk_probability": probability,
        "threshold": 0.9,
        "activated": activated,
        "base_action": "stay",
        "correction_action": "left" if activated else None,
        "overrode": overrode,
    }
