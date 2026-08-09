from __future__ import annotations

import json

from scripts.audit_headless_feasibility_oracle import (
    audit_file,
    representation_probe,
)
from scripts.label_headless_feasibility_oracle import (
    action_trace_sha256,
    checkpoint_verdict,
    exact_snapshot_features,
)


def _branch(action: str, continuation: str, *, feasible: bool):
    return {
        "continuation": continuation,
        "first_action": action,
        "termination_reason": "tick-limit" if feasible else "authority-failure",
        "authority_failure_reason": None if feasible else "native safe set is empty",
        "survival_ticks": 60 if feasible else 20,
        "actions_issued": 60 if feasible else 20,
        "minimum_native_legal_actions": 2 if feasible else 0,
        "terminal_boundary_reserve": 30.0 if feasible else 2.0,
        "physical_deaths_delta": 0,
        "bombs_used_delta": 0,
        "feasible": feasible,
    }


def _document():
    branches = [
        _branch(action, continuation, feasible=action == "left")
        for action in ("left", "right")
        for continuation in ("generic-clearance", "native-local-h12")
    ]
    return {
        "schema": "th06-rl-headless-feasibility-oracle-v1",
        "authority": "exact-checkpoint-native-first-actions-multi-continuation-v1",
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 3},
        "initial_seed": 9,
        "branch_frames": 60,
        "native_set_revision_allowed": False,
        "runtime_source": {"clean": True, "commit": "abc", "binary_sha256": "def"},
        "input_source": {"clean": True, "commit": "abc", "binary_sha256": "def"},
        "code_source": {"clean": True, "commit": "123"},
        "input_corpus": {
            "manifest_sha256": "1" * 64,
            "transitions": {"path": "transitions.jsonl.gz", "sha256": "2" * 64},
        },
        "continuations": [
            {"name": "generic-clearance", "kind": "generic-clearance", "horizon": 0},
            {"name": "native-local-h12", "kind": "native-local-plan", "horizon": 12},
        ],
        "checkpoints": [{
            "sequence": 10,
            "checkpoint_tick": 11,
            "observation_sha256": "0" * 64,
            "source_context": "timeline:1/2/3",
            "compact_state": {"player_x": 192.0, "previous_action": "stay_fast"},
            "action_candidates": [
                {"action": "left", "min_clearance": 20.0},
                {"action": "right", "min_clearance": 20.0},
            ],
            "exact_snapshot_features": {"bullet_0_dx": -10.0},
            "factual_action": "right",
            "local_teacher_action": "right",
            "native_legal_actions": ["left", "right"],
            "input_native_legal_actions": ["left", "right"],
            "native_set_revised": False,
            "branch_frames": 60,
            "continuation_count": 2,
            "feasible_actions": ["left"],
            "best_actions": ["left"],
            "factual_action_has_witness": False,
            "local_teacher_action_has_witness": False,
            "verdict": "policy-selection-witness",
            "branches": branches,
        }],
    }


def test_feasibility_audit_accepts_complete_action_continuation_product(tmp_path) -> None:
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(_document()), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is True
    assert result["branches"] == 4
    assert result["verdicts"] == {"policy-selection-witness": 1}


def test_feasibility_audit_accepts_declared_branch_extension_subset(tmp_path) -> None:
    document = _document()
    document["evaluation_mode"] = "declared-subset"
    checkpoint = document["checkpoints"][0]
    checkpoint["evaluated_first_actions"] = ["left"]
    checkpoint["branches"] = [
        branch for branch in checkpoint["branches"]
        if branch["first_action"] == "left"
    ]
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is True
    assert result["native_actions"] == 2
    assert result["evaluated_actions"] == 1
    assert result["subset_checkpoints"] == 1


def test_feasibility_audit_rechecks_reproducible_action_trace(tmp_path) -> None:
    document = _document()
    checkpoint = document["checkpoints"][0]
    branch = checkpoint["branches"][0]
    actions = [branch["first_action"]] * branch["actions_issued"]
    branch.update({
        "action_trace_rle": [{"action": branch["first_action"], "ticks": len(actions)}],
        "action_trace_sha256": action_trace_sha256(actions),
        "terminal_tick": checkpoint["checkpoint_tick"] + branch["survival_ticks"],
        "terminal_observation_sha256": "3" * 64,
    })
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document), encoding="utf-8")
    assert audit_file(path)["valid"] is True

    branch["action_trace_sha256"] = "4" * 64
    path.write_text(json.dumps(document), encoding="utf-8")
    result = audit_file(path)
    assert result["valid"] is False
    assert "trace SHA-256" in " ".join(result["errors"])


def test_feasibility_audit_rejects_missing_continuation_branch(tmp_path) -> None:
    document = _document()
    document["checkpoints"][0]["branches"].pop()
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is False
    assert "action-continuation product" in " ".join(result["errors"])


def test_feasibility_audit_requires_declared_native_set_revision(tmp_path) -> None:
    document = _document()
    checkpoint = document["checkpoints"][0]
    checkpoint["input_native_legal_actions"] = ["left"]
    checkpoint["native_set_revised"] = True
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is False
    assert "undeclared native set revision" in " ".join(result["errors"])


def test_feasibility_audit_accepts_declared_native_set_revision(tmp_path) -> None:
    document = _document()
    document["native_set_revision_allowed"] = True
    checkpoint = document["checkpoints"][0]
    checkpoint["input_native_legal_actions"] = ["left"]
    checkpoint["native_set_revised"] = True
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is True
    assert result["native_set_revisions"] == 1


def test_feasibility_audit_rejects_runtime_candidates_outside_native_set(
    tmp_path,
) -> None:
    document = _document()
    checkpoint = document["checkpoints"][0]
    checkpoint["runtime_compact_state"] = checkpoint["compact_state"]
    checkpoint["runtime_action_candidates"] = [{"action": "left"}]
    path = tmp_path / "oracle.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is False
    assert "runtime candidates" in " ".join(result["errors"])


def test_checkpoint_verdict_preserves_no_witness_epistemic_boundary() -> None:
    assert checkpoint_verdict(feasible_actions=(), factual_action="left") == "oracle-no-witness"
    assert checkpoint_verdict(
        feasible_actions=("left",), factual_action="right"
    ) == "policy-selection-witness"


def test_exact_snapshot_probe_is_fixed_width_and_uses_source_motion() -> None:
    observation = {
        "player": {"x": 100.0, "y": 200.0},
        "bullets": [{
            "x": 103.0,
            "y": 196.0,
            "vx": 1.5,
            "vy": -2.0,
            "state": 1,
            "ex_flags": 16,
        }],
        "lasers": [],
        "enemies": [],
    }

    features = exact_snapshot_features(observation)

    assert features["bullet_0_present"] == 1.0
    assert features["bullet_0_dx"] == 3.0
    assert features["bullet_0_dy"] == -4.0
    assert features["bullet_0_vx"] == 1.5
    assert features["bullet_31_present"] == 0.0


def _probe_checkpoint(seed: int, index: int, mode: int):
    feasible = "left" if mode == 0 else "right"
    return {
        "observation_sha256": f"{seed:02x}{index:062x}",
        "source_context": "shared",
        "compact_state": {"player_x": 192.0, "previous_action": "stay_fast"},
        "action_candidates": [
            {"action": "left", "min_clearance": 10.0},
            {"action": "right", "min_clearance": 10.0},
        ],
        "exact_snapshot_features": {"mode": float(mode)},
        "factual_action": "right",
        "local_teacher_action": "right",
        "native_legal_actions": ["left", "right"],
        "feasible_actions": [feasible],
    }


def test_representation_probe_detects_exact_derived_separation() -> None:
    results = []
    for seed in (1, 2):
        results.append({
            "valid": True,
            "document": {
                "initial_seed": seed,
                "checkpoints": [
                    _probe_checkpoint(seed, index, index % 2)
                    for index in range(8)
                ],
            },
        })

    probe = representation_probe(results, threads=1)

    assert probe["status"] == "complete"
    assert probe["exact_derived"]["top1_feasible_rate"] == 1.0
    assert probe["exact_derived"]["top1_feasible_rate"] > probe["compact"]["top1_feasible_rate"]
