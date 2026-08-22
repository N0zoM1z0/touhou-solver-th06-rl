from pathlib import Path

from scripts.run_l2m_stage4_intention_hazard import (
    _split_support,
    load_prereg,
)
from th06_rl.actions import ACTION_NAMES


REPOSITORY = Path(__file__).resolve().parents[1]


def _target(*, rows_per_action: int, positives: int) -> dict[str, object]:
    return {
        "status": {
            "accepted-label-0": rows_per_action * len(ACTION_NAMES) - positives,
            "accepted-label-1": positives,
        },
        "accepted_actions": {action: rows_per_action for action in ACTION_NAMES},
        "positive_actions": {"left": positives},
    }


def test_l2m_preregistration_freezes_whole_episode_split_and_fit() -> None:
    prereg = load_prereg(
        REPOSITORY / "experiments/l2m-stage4-intention-hazard-v1.json"
    )

    assert [row["split"] for row in prereg["collection"]["episodes"]] == (
        ["train"] * 10 + ["validation"] * 6
    )
    assert prereg["data"]["train_episode_count"] == 12
    assert prereg["data"]["validation_episode_count"] == 6
    assert prereg["fit"]["single_fit_no_sweep"] is True
    assert prereg["gate"]["online_learned_policy_admitted"] is False


def test_l2m_split_support_includes_pilot_only_in_train() -> None:
    prereg = load_prereg(
        REPOSITORY / "experiments/l2m-stage4-intention-hazard-v1.json"
    )
    pilot = {
        "action_exposure_audit": {
            "target_episodes": [
                _target(rows_per_action=100, positives=16),
                _target(rows_per_action=100, positives=16),
            ]
        }
    }
    audited = [
        ({}, _target(rows_per_action=80 if index < 10 else 100, positives=12))
        for index in range(16)
    ]

    support = _split_support(prereg, audited, pilot)

    assert support["train"]["episodes"] == 12
    assert support["validation"]["episodes"] == 6
    assert support["train"]["positives"] == 152
    assert support["validation"]["positives"] == 72
    assert all(support["gates"].values())
