from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np

from th06_rl.bc_training import _expected_calibration_error, fit_behavior_clone
from th06_rl.bc_features import (
    features_from_policy_context,
    linear_action_scores,
    masked_softmax_probabilities,
    normalized_features,
)
from th06_rl.corpus import CorpusRecorder, RunMetadata
from th06_rl.policy_api import PolicyContext
from th06_rl.policy_loader import ImmutablePolicy

from tests.test_episode_dataset import _decision, _snapshot


REPOSITORY = Path(__file__).resolve().parents[1]
PLUGIN = REPOSITORY / "src/th06_rl/policies/linear_behavior_clone.py"


def _learning_episode(root: Path, *, x: float, action: str):
    recorder = CorpusRecorder(
        root,
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    other = "right" if action == "left" else "left"
    legal = ("left", "right")
    recorder.record(
        _snapshot(0, x=x),
        _decision("ok", current=other, published=action, legal=legal),
    )
    recorder.record(
        _snapshot(1, x=x),
        _decision("ok", current=action, published=action, legal=legal),
    )
    recorder.record(
        _snapshot(2, x=x),
        _decision("ok", current=action, published=action, legal=legal),
    )
    recorder.record(
        _snapshot(3, x=x, in_menu=True),
        _decision("passive", current=action),
    )
    return recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })


def _context(x: float, *, baseline: str = "left") -> PolicyContext:
    evaluations = (
        ("left", 10.0, 190.0, 399.0),
        ("right", 11.0, 191.0, 400.0),
    )
    return PolicyContext(
        baseline_action=baseline,
        locally_admissible_actions=("left", "right"),
        player_x=x,
        player_y=400.0,
        power=64,
        bullet_count=10,
        laser_count=0,
        shield_action_count=2,
        current_action="stay",
        shield_admissible_actions=("left", "right"),
        shield_action_evaluations=evaluations,
    )


def test_fixture_loop_fits_exports_and_reloads_immutable_policy(tmp_path) -> None:
    train = (
        _learning_episode(tmp_path / "train-left", x=96.0, action="left"),
        _learning_episode(tmp_path / "train-right", x=288.0, action="right"),
    )
    validation = (
        _learning_episode(tmp_path / "validation-left", x=104.0, action="left"),
        _learning_episode(tmp_path / "validation-right", x=280.0, action="right"),
    )

    state = fit_behavior_clone(
        train,
        validation,
        epochs=400,
        learning_rate=0.1,
        seed=7,
        bootstrap_samples=200,
        calibration_tolerance=0.1,
        policy_plugin_sha256=hashlib.sha256(PLUGIN.read_bytes()).hexdigest(),
    )
    state_path = tmp_path / "linear-bc.json"
    state_path.write_text(
        json.dumps(state, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    policy = ImmutablePolicy(PLUGIN, state_path=state_path)

    left = policy.decide(_context(100.0, baseline="right"))
    right = policy.decide(_context(284.0, baseline="left"))

    assert left.action == "left"
    assert right.action == "right"
    def exported_probabilities(context):
        features = normalized_features(
            features_from_policy_context(context),
            tuple(state["normalization"]["mean"]),
            tuple(state["normalization"]["scale"]),
        )
        scores = linear_action_scores(
            features,
            tuple(tuple(row) for row in state["model"]["weights"]),
            tuple(state["model"]["biases"]),
        )
        return masked_softmax_probabilities(
            scores,
            context.locally_admissible_actions,
        )

    assert left.behavior_probabilities == exported_probabilities(
        _context(100.0, baseline="right")
    )
    assert right.behavior_probabilities == exported_probabilities(
        _context(284.0, baseline="left")
    )
    assert dict(left.behavior_probabilities)[left.action] == left.behavior_probability
    assert dict(right.behavior_probabilities)[right.action] == right.behavior_probability
    assert state["fit"]["learnability_gate_passed"] is True
    assert state["fit"]["validation"]["negative_log_likelihood"] < state["fit"][
        "action_frequency_validation"
    ]["negative_log_likelihood"]
    assert all(
        "path" not in record
        for split in state["inventory"].values()
        for record in split
    )
    assert policy.status(include_metrics=False)["policy_failures"] == 0


def test_export_rejects_tampered_policy_identity(tmp_path) -> None:
    left = _learning_episode(tmp_path / "left", x=96.0, action="left")
    right = _learning_episode(tmp_path / "right", x=288.0, action="right")
    state = fit_behavior_clone(
        (left,),
        (right,),
        epochs=1,
        bootstrap_samples=1,
        policy_plugin_sha256=hashlib.sha256(PLUGIN.read_bytes()).hexdigest(),
    )
    state["model"]["biases"][0] += 1.0
    state_path = tmp_path / "tampered.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    try:
        ImmutablePolicy(PLUGIN, state_path=state_path)
    except ValueError as error:
        assert "identity hash mismatch" in str(error)
    else:
        raise AssertionError("tampered behavior-clone state retained its identity")


def test_fit_is_deterministic_and_rejects_episode_leakage(tmp_path) -> None:
    left = _learning_episode(tmp_path / "left", x=96.0, action="left")
    right = _learning_episode(tmp_path / "right", x=288.0, action="right")

    first = fit_behavior_clone(
        (left,),
        (right,),
        epochs=5,
        bootstrap_samples=10,
    )
    second = fit_behavior_clone(
        (left,),
        (right,),
        epochs=5,
        bootstrap_samples=10,
    )

    assert first == second

    try:
        fit_behavior_clone((left,), (left,), epochs=1, bootstrap_samples=1)
    except ValueError as error:
        assert "episode leakage" in str(error)
    else:
        raise AssertionError("train/validation episode leakage was accepted")


def test_ece_assigns_every_exact_decimal_boundary_once() -> None:
    confidence = np.asarray([index / 10 for index in range(11)], dtype=np.float64)
    correct = np.ones(11, dtype=np.bool_)

    observed = _expected_calibration_error(confidence, correct)
    expected = float(np.mean(1.0 - confidence))

    assert np.isclose(observed, expected)
    assert np.isclose(
        _expected_calibration_error(np.asarray([0.6]), np.asarray([True])),
        0.4,
    )


def test_train_only_gradient_stop_is_recorded_and_gates_fit(tmp_path) -> None:
    train = (
        _learning_episode(tmp_path / "train-left", x=96.0, action="left"),
        _learning_episode(tmp_path / "train-right", x=288.0, action="right"),
    )
    validation = (
        _learning_episode(tmp_path / "validation-left", x=104.0, action="left"),
        _learning_episode(tmp_path / "validation-right", x=280.0, action="right"),
    )

    state = fit_behavior_clone(
        train,
        validation,
        epochs=2_000,
        minimum_updates=10,
        relative_gradient_l2_tolerance=0.05,
        learning_rate=0.1,
        calibration_tolerance=0.1,
        bootstrap_samples=20,
    )

    optimization = state["fit"]["optimization"]
    assert optimization["converged"] is True
    assert 10 <= optimization["updates_completed"] < 2_000
    assert optimization["stop_reason"] == "relative-gradient-l2"
    assert optimization["final_to_initial_gradient_l2_ratio"] <= 0.05
    assert state["fit"]["optimization_gate_passed"] is True
