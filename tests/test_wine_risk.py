from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

from th06_rl.offline_learning import LabeledTransition
import pytest

from scripts.replay_wine_risk_guard import _score_sweep
from scripts.train_wine_risk_guard import (
    Encoder,
    _precision_lower_bound,
    _validate_factual_action_audit,
    _validate_residual_proposal_audit,
)
from th06_rl.wine_risk import (
    _validate_clean_run_outcome,
    _validate_failure_window,
    label_failure_risk,
)


def _row(frame: int, phase: str, action: str = "right") -> LabeledTransition:
    return LabeledTransition(
        run_id="run",
        sequence=frame,
        frame=frame,
        source_context=phase,
        action=action,
        baseline_action="left",
        legal_actions=("left", "right"),
        behavior_probability=1.0,
        features={},
        reward=1.0,
    )


def test_failure_risk_is_phase_bounded_and_factual() -> None:
    labeled = label_failure_risk(
        [_row(79, "p2"), _row(80, "p1"), _row(150, "p2"), _row(200, "p2")],
        failure_frame=200,
        failure_context="p2",
    )
    assert [row.failure_within_120 for row in labeled] == [False, False, True, True]
    assert [row.frames_to_failure for row in labeled] == [None, None, 50, 0]
    assert all(row.fallback_opportunity for row in labeled)


def test_failure_risk_marks_no_fallback_when_actions_match() -> None:
    row = _row(100, "p", action="left")
    labeled = label_failure_risk(
        [row], failure_frame=110, failure_context="p",
    )
    assert labeled[0].failure_within_120 is True
    assert labeled[0].fallback_opportunity is False


def test_failure_risk_excludes_an_earlier_reused_context_segment() -> None:
    labeled = label_failure_risk(
        [_row(100, "p"), _row(150, "other"), _row(190, "p")],
        failure_frame=200,
        failure_context="p",
        segment_start_frame=180,
    )
    assert [row.failure_within_120 for row in labeled] == [False, False, True]


def _outcome(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "capture_failures": 0,
        "infrastructure_failures": 0,
        "corpus_failures": 0,
        "trace_failures": 0,
        "background_reactivations": 0,
        "corpus_failure": None,
        "policy_transaction_failure": None,
        "policy_state_recovered": False,
        "policy_state_committed": False,
        "policy_state_rolled_back": False,
        "stage_completed": False,
        "controller_exit_code": 12,
        "physical_hits": 0,
        "termination_reason": "authority-stop:Hard safe set empty",
    }
    value.update(overrides)
    return value


def test_first_failure_outcome_rejects_capture_recovery() -> None:
    _validate_clean_run_outcome(
        {"run_outcome": _outcome()}, failure_kind="control-dead-end",
    )
    with pytest.raises(ValueError, match="capture_failures"):
        _validate_clean_run_outcome(
            {"run_outcome": _outcome(capture_failures=1)},
            failure_kind="control-dead-end",
        )


def _raw_window_row(source: int, elapsed: int) -> dict[str, object]:
    return {
        "snapshot_ref": f"run:00000000:f{source}",
        "next_snapshot_ref": f"run:00000001:f{source + elapsed}",
        "scope": {"phase_id": "terminal-phase"},
        "outcome_terms": {"elapsed_frames": elapsed},
    }


def test_first_failure_window_rejects_only_overlapping_observation_gap() -> None:
    _validate_failure_window(
        [_raw_window_row(850, 20), _raw_window_row(900, 1)],
        failure_frame=1000,
        failure_context="terminal-phase",
        segment_start_frame=880,
    )
    with pytest.raises(ValueError, match="crosses an observation gap"):
        _validate_failure_window(
            [_raw_window_row(879, 2), _raw_window_row(900, 1)],
            failure_frame=1000,
            failure_context="terminal-phase",
            segment_start_frame=879,
        )


def test_precision_gate_uses_a_one_sided_wilson_lower_bound() -> None:
    assert _precision_lower_bound(168, 280) < 0.60
    assert _precision_lower_bound(163, 250) > 0.60
    assert _precision_lower_bound(0, 0) is None


def test_replay_score_sweep_audits_all_requested_thresholds() -> None:
    scored = [
        {"run_id": "r1", "score": 0.9, "label": True},
        {"run_id": "r1", "score": 0.8, "label": False},
        {"run_id": "r2", "score": 0.7, "label": None},
    ]

    high, low = _score_sweep(
        scored, thresholds=(0.85, 0.65), policy_calls=100,
    )

    assert high["activations"] == 1
    assert high["candidate_positive"] == 1
    assert high["labeled_precision"] == 1.0
    assert low["activations"] == 3
    assert low["candidate_negative"] == 1
    assert low["candidate_unlabeled"] == 1
    assert low["activated_runs"] == ["r1", "r2"]


def test_factual_action_audit_binds_exact_corpus_and_state(tmp_path) -> None:
    state = tmp_path / "state.json"
    state.write_text("{}\n", encoding="utf-8")
    state_sha256 = hashlib.sha256(state.read_bytes()).hexdigest()
    report = {
        "schema": "th06-rl-wine-risk-consensus-replay-v1",
        "passed": True,
        "mode": "shadow",
        "expect_recorded_actions": "incumbent",
        "scope": [3, 0, 0, 6],
        "state": str(state),
        "state_sha256": state_sha256,
        "runs": [{
            "run_id": "run-a",
            "manifest_sha256": "a" * 64,
            "run_sha256": "b" * 64,
            "policy_calls": 10,
            "recorded_incumbent_mismatches": [],
            "recorded_policy_mismatches": [],
            "shadow_action_contract_violations": [],
        }],
        "totals": {
            "runs": 1,
            "policy_calls": 10,
            "recorded_incumbent_mismatches": 0,
            "recorded_policy_mismatches": 0,
            "shadow_action_contract_violations": 0,
        },
    }
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    prefixes = [SimpleNamespace(
        run_id="run-a", manifest_sha256="a" * 64, run_sha256="b" * 64,
    )]

    provenance = _validate_factual_action_audit(
        path, scope=(3, 0, 0, 6), prefixes=prefixes,
    )
    assert provenance["contract"] == "recorded-action-equals-frozen-incumbent"

    report["runs"][0]["manifest_sha256"] = "c" * 64
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="exactly cover"):
        _validate_factual_action_audit(
            path, scope=(3, 0, 0, 6), prefixes=prefixes,
        )


def test_residual_proposal_audit_binds_only_exact_factual_examples(
    tmp_path,
) -> None:
    states = {}
    for name in ("incumbent_state", "ranker_state", "risk_state"):
        state = tmp_path / f"{name}.json"
        state.write_text("{}\n", encoding="utf-8")
        states[name] = state
    examples = label_failure_risk(
        [_row(10, "phase")], failure_frame=20, failure_context="phase",
    )
    prefix = SimpleNamespace(
        run_id="run-a",
        manifest_sha256="a" * 64,
        run_sha256="b" * 64,
        examples=examples,
    )
    report = {
        "schema": "th06-rl-wine-offline-residual-replay-v1",
        "passed": True,
        "scope": [3, 0, 0, 6],
        "semantics": {
            "published_action": "none-offline-replay-only",
            "recorded_action_authority": "frozen-incumbent",
            "residual_guard_eligibility": (
                "exact-factual-risk-example-and-fallback-opportunity"
            ),
        },
        "runs": [{
            "run_id": "run-a",
            "manifest_sha256": "a" * 64,
            "run_sha256": "b" * 64,
            "policy_calls": 10,
            "recorded_incumbent_mismatch_count": 0,
            "action_contract_violation_count": 0,
            "residual_guard_eligible": 1,
            "residual_guard_eligible_sequences": [10],
        }],
        "totals": {
            "runs": 1,
            "policy_calls": 10,
            "recorded_incumbent_mismatches": 0,
            "action_contract_violations": 0,
        },
    }
    for name, state in states.items():
        report[name] = str(state)
        report[f"{name}_sha256"] = hashlib.sha256(state.read_bytes()).hexdigest()
    path = tmp_path / "residual-audit.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    provenance, eligible = _validate_residual_proposal_audit(
        path, scope=(3, 0, 0, 6), prefixes=[prefix],
    )
    assert provenance["eligible_rows"] == 1
    assert eligible == {"run-a": frozenset({10})}

    report["runs"][0]["residual_guard_eligible_sequences"] = [11]
    path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(ValueError, match="escaped factual risk eligibility"):
        _validate_residual_proposal_audit(
            path, scope=(3, 0, 0, 6), prefixes=[prefix],
        )


def test_encoder_can_condition_on_automatic_source_context() -> None:
    risk = SimpleNamespace(
        transition=_row(10, "boss:0:sub31:auto"),
        features={"player_x": 42.0},
    )
    encoder = Encoder(
        [risk],
        feature_names=("source_context", "player_x"),
        categorical_features=("source_context",),
    )

    encoded = encoder.encode([risk])
    assert encoded.tolist() == [[0.0, 42.0]]
    assert encoder.manifest()["source_context"] == ["boss:0:sub31:auto"]
