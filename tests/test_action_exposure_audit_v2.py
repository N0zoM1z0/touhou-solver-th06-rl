from th06_rl.actions import ACTION_NAMES
from th06_rl.action_exposure_audit_v2 import summarize_audits


def _episode(*, violations: int = 0) -> dict[str, object]:
    return {
        "episode_id": "fixture",
        "transitions": 20000,
        "eligible_complete_groups": 2000,
        "no_override_groups": 1600,
        "full_intended_executions_groups": 1500,
        "assignment_counts": {action: 120 for action in ACTION_NAMES},
        "control_dead_ends": 20,
        "bombs": 0,
        "infrastructure_failures": 0,
        "contract_violation_count": violations,
    }


def _target(*, positives: int = 10) -> dict[str, object]:
    return {
        "accepted_actions": {action: 100 for action in ACTION_NAMES},
        "positive_actions": {"left": positives},
        "status": {"accepted-label-0": 1900, "accepted-label-1": positives},
    }


def _summarize(*, positives: int = 10, violations: int = 0) -> dict[str, object]:
    return summarize_audits(
        [_episode(violations=violations), _episode()],
        [_target(positives=positives), _target(positives=positives)],
        exposure_roots=12,
        minimum_complete_groups_per_episode=1800,
        minimum_assignments_per_action=100,
        minimum_no_override_fraction=0.7,
        minimum_full_execution_fraction=0.7,
        maximum_control_dead_end_rate=0.005,
        hit_support_diagnostic_minimum=16,
    )


def test_long_exposure_contract_and_hit_support_select_serial_collection() -> None:
    result = _summarize()

    assert all(result["gates"].values())
    assert result["aggregate"]["hit_support_ready"] is True
    assert result["decision"] == "proceed-serial-action-exposure-training-collection"


def test_low_hit_support_does_not_reclassify_valid_contract_as_failure() -> None:
    result = _summarize(positives=3)

    assert all(result["gates"].values())
    assert result["aggregate"]["hit_support_ready"] is False
    assert result["decision"] == "pass-action-exposure-contract-insufficient-hit-support"


def test_metadata_violation_rejects_long_exposure_contract() -> None:
    result = _summarize(violations=1)

    assert result["gates"]["exact_exposure_contract"] is False
    assert result["decision"] == "reject-action-exposure-collection-contract"
