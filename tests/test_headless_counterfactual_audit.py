from __future__ import annotations

import json

from scripts.audit_headless_counterfactuals import audit_file


def document() -> dict[str, object]:
    return {
        "schema": "th06-rl-headless-cow-counterfactual-v1",
        "authority": "first-action-native-legal-and-dynamic-continuation-revalidated",
        "scope": {"stage": 3},
        "initial_seed": 9,
        "branch_frames": 180,
        "runtime_source": {"clean": True},
        "runtime_delivery_contract": "synchronous-step-v1",
        "runtime_delivery_delays": [0],
        "observation_digest_contract": "physical-observation-without-diagnostic-events-v1",
        "checkpoints": [{
            "native_legal_actions": ["up", "stay_fast"],
            "local_teacher_action": "stay_fast",
            "factual_action": "stay_fast",
            "best_actions": ["up"],
            "local_teacher_action_is_best": False,
            "factual_action_is_best": False,
            "outcomes": [
                {
                    "first_action": "up",
                    "termination_reason": "tick-limit",
                    "survival_ticks": 180,
                    "actions_issued": 180,
                    "minimum_native_legal_actions": 18,
                    "terminal_boundary_reserve": 60.0,
                },
                {
                    "first_action": "stay_fast",
                    "termination_reason": "authority-failure",
                    "survival_ticks": 100,
                    "actions_issued": 100,
                    "minimum_native_legal_actions": 2,
                    "terminal_boundary_reserve": 0.0,
                },
            ],
        }],
    }


def test_counterfactual_audit_accepts_complete_native_action_table(tmp_path) -> None:
    path = tmp_path / "label.json"
    path.write_text(json.dumps(document()), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is True
    assert result["outcomes"] == 2
    assert result["unique_best_checkpoints"] == 1
    assert result["learning_informative_checkpoints"] == 1


def test_counterfactual_audit_reports_equal_dead_end_as_uninformative(tmp_path) -> None:
    value = document()
    outcomes = value["checkpoints"][0]["outcomes"]  # type: ignore[index]
    for index, row in enumerate(outcomes):
        row.update({  # type: ignore[union-attr]
            "termination_reason": "authority-failure",
            "survival_ticks": 1,
            "actions_issued": 1,
            "minimum_native_legal_actions": 18 - index,
            "terminal_boundary_reserve": 10.0 + index,
        })
    value["checkpoints"][0].update({  # type: ignore[index]
        "best_actions": ["up"],
        "local_teacher_action_is_best": False,
        "factual_action_is_best": False,
    })
    path = tmp_path / "dead-end.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is True
    assert result["learning_informative_checkpoints"] == 0


def test_counterfactual_audit_rejects_summary_tampering(tmp_path) -> None:
    value = document()
    value["checkpoints"][0]["best_actions"] = ["stay_fast"]  # type: ignore[index]
    path = tmp_path / "tampered.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is False
    assert "best action summary mismatch" in " ".join(result["errors"])


def test_counterfactual_audit_rejects_false_synchronous_delivery(tmp_path) -> None:
    value = document()
    value["runtime_delivery_delays"] = [0, 1]
    path = tmp_path / "bad-delivery.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is False
    assert "delivery" in " ".join(result["errors"])


def test_counterfactual_audit_rejects_unknown_observation_digest(tmp_path) -> None:
    value = document()
    value["observation_digest_contract"] = "unknown"
    path = tmp_path / "bad-digest.json"
    path.write_text(json.dumps(value), encoding="utf-8")

    result = audit_file(path)

    assert result["valid"] is False
    assert "digest" in " ".join(result["errors"])
