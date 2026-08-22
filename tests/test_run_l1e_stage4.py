from __future__ import annotations

import json
from pathlib import Path

from scripts.run_l1e_stage4 import load_prereg, result_decision
from th06_rl.shared_action_features import ACTION_FEATURE_NAMES


REPOSITORY = Path(__file__).resolve().parents[1]
PREREGISTRATION = (
    REPOSITORY / "experiments/l1e-stage4-bc-shared-action-v1.json"
)


def test_l1e_preregistration_is_bound_and_single_variable() -> None:
    prereg = load_prereg(PREREGISTRATION)
    assert prereg["fit"]["action_feature_names"] == list(ACTION_FEATURE_NAMES)
    assert prereg["fit"]["target_schema"] == (
        "th06-rl-published-executed-action-target-v1"
    )
    assert prereg["model_scope"]["parameter_count"] == 7
    assert prereg["online_canary"]["run_in_this_experiment"] is False


def test_l1e_result_decision_requires_convergence_and_joint_gate() -> None:
    assert result_decision({
        "fit": {
            "optimization": {"converged": False},
            "learnability_gate_passed": False,
        }
    }) == "inconclusive-l1e-shared-action-optimization-not-converged"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": False,
        }
    }) == "stop-l1e-shared-action-current-observation"
    assert result_decision({
        "fit": {
            "optimization": {"converged": True},
            "learnability_gate_passed": True,
        }
    }) == "admit-stage4-shared-action-bc-integration-canary"


def test_l1e_preregistration_json_is_canonical_json_data() -> None:
    observed = json.loads(PREREGISTRATION.read_text(encoding="utf-8"))
    assert observed["schema"] == (
        "th06-rl-l1e-stage4-bc-shared-action-prereg-v1"
    )
