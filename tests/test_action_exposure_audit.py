from th06_rl.actions import ACTION_NAMES
from th06_rl.action_exposure_audit import summarize_action_exposure_audits


def _episode(*, positives: int = 40, violations: int = 0) -> dict[str, object]:
    return {
        "episode_id": "fixture",
        "transitions": 20000,
        "policy_invocations": 18000,
        "eligible_complete_groups": 4500,
        "no_override_groups": 4200,
        "four_intended_executions_groups": 4100,
        "assignment_counts": {action: 120 for action in ACTION_NAMES},
        "h16_group_starts": {
            "positive": positives,
            "negative": 4000,
            "unsupported": 0,
        },
        "control_dead_ends": 2,
        "bombs": 0,
        "infrastructure_failures": 0,
        "contract_violation_count": violations,
    }


def _summarize(episodes: list[dict[str, object]]) -> dict[str, object]:
    return summarize_action_exposure_audits(
        episodes,
        exposure_roots=4,
        minimum_complete_groups_per_episode=4000,
        minimum_assignments_per_action=100,
        minimum_no_override_fraction=0.8,
        minimum_four_execution_fraction=0.8,
        maximum_control_dead_end_rate=0.005,
        h16_support_diagnostic_minimum=64,
    )


def test_contract_and_h16_support_select_scaled_collection() -> None:
    result = _summarize([_episode(), _episode()])

    assert all(result["gates"].values())
    assert result["aggregate"]["h16_positive_group_starts"] == 80
    assert result["decision"] == "proceed-action-exposure-training-collection"


def test_low_hit_support_does_not_reclassify_a_valid_contract_as_failure() -> None:
    result = _summarize([_episode(positives=10), _episode(positives=10)])

    assert all(result["gates"].values())
    assert result["decision"] == "pass-action-exposure-contract-insufficient-h16-support"


def test_metadata_violation_rejects_collection_contract() -> None:
    result = _summarize([_episode(violations=1), _episode()])

    assert result["gates"]["exact_exposure_contract"] is False
    assert result["decision"] == "reject-action-exposure-collection-contract"
