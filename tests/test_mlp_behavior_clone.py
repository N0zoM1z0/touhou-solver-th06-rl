from __future__ import annotations

import hashlib
import json
from pathlib import Path

from th06_rl.bc_training import fit_behavior_clone
from th06_rl.mlp_bc_training import fit_small_mlp_behavior_clone
from th06_rl.policy_loader import ImmutablePolicy

from tests.test_behavior_clone import _context, _learning_episode


REPOSITORY = Path(__file__).resolve().parents[1]
LINEAR_PLUGIN = REPOSITORY / "src/th06_rl/policies/linear_behavior_clone.py"
MLP_PLUGIN = REPOSITORY / "src/th06_rl/policies/small_mlp_behavior_clone.py"


def _fixture_fit(tmp_path: Path) -> tuple[dict[str, object], Path]:
    train = (
        _learning_episode(tmp_path / "train-left", x=96.0, action="left"),
        _learning_episode(tmp_path / "train-right", x=288.0, action="right"),
    )
    validation = (
        _learning_episode(tmp_path / "validation-left", x=104.0, action="left"),
        _learning_episode(tmp_path / "validation-right", x=280.0, action="right"),
    )
    linear = fit_behavior_clone(
        train,
        validation,
        epochs=50,
        seed=7,
        bootstrap_samples=20,
        policy_plugin_sha256=hashlib.sha256(LINEAR_PLUGIN.read_bytes()).hexdigest(),
    )
    linear_path = tmp_path / "linear.json"
    linear_path.write_text(json.dumps(linear, sort_keys=True), encoding="utf-8")
    state = fit_small_mlp_behavior_clone(
        train,
        validation,
        linear_comparator_state=linear_path,
        epochs=200,
        minimum_updates=10,
        relative_gradient_l2_tolerance=0.5,
        learning_rate=0.1,
        seed=7,
        bootstrap_samples=20,
        calibration_tolerance=0.1,
        policy_plugin_sha256=hashlib.sha256(MLP_PLUGIN.read_bytes()).hexdigest(),
    )
    return state, linear_path


def test_small_mlp_fit_exports_a_mask_bounded_immutable_policy(tmp_path: Path) -> None:
    state, _ = _fixture_fit(tmp_path)
    state_path = tmp_path / "small-mlp.json"
    state_path.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")
    policy = ImmutablePolicy(MLP_PLUGIN, state_path=state_path)

    decision = policy.decide(_context(100.0, baseline="right"))

    assert state["model"]["hidden_width"] == 32
    assert state["initialization"]["kind"] == "fixed-seed-he-normal"
    assert state["fit"]["optimization"]["converged"] is True
    assert set(dict(decision.behavior_probabilities)) == {"left", "right"}
    assert abs(sum(dict(decision.behavior_probabilities).values()) - 1.0) < 1e-12
    assert decision.action in {"left", "right"}
    assert policy.status(include_metrics=False)["policy_failures"] == 0


def test_small_mlp_fit_is_deterministic_and_binds_linear_comparator(
    tmp_path: Path,
) -> None:
    first, comparator = _fixture_fit(tmp_path / "first")

    # Reusing the exact episodes in a second fit would require reconstructing
    # the corpus; deterministic initialization itself is exposed in the state.
    second_root = tmp_path / "second"
    second, second_comparator = _fixture_fit(second_root)

    assert first["initialization"] == second["initialization"]
    assert first["model"] == second["model"]
    assert first["fit"] == second["fit"]
    assert first["provenance"]["frozen_linear_comparator_sha256"] == hashlib.sha256(
        comparator.read_bytes()
    ).hexdigest()
    assert second["provenance"]["frozen_linear_comparator_sha256"] == hashlib.sha256(
        second_comparator.read_bytes()
    ).hexdigest()


def test_small_mlp_export_rejects_tampered_identity(tmp_path: Path) -> None:
    state, _ = _fixture_fit(tmp_path)
    state["model"]["output_biases"][0] += 1.0
    state_path = tmp_path / "tampered.json"
    state_path.write_text(json.dumps(state), encoding="utf-8")

    try:
        ImmutablePolicy(MLP_PLUGIN, state_path=state_path)
    except ValueError as error:
        assert "identity hash mismatch" in str(error)
    else:
        raise AssertionError("tampered small-MLP state retained its identity")
