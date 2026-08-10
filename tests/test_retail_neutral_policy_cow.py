from __future__ import annotations

from types import SimpleNamespace

from scripts.audit_retail_neutral_policy_cow import unanimous_neutral_candidate
from scripts.select_retail_neutral_anchors import select_anchor


def _example(*, sequence: int, lag: int, **overrides: object) -> SimpleNamespace:
    features: dict[str, object] = {
        "action": "down_right",
        "baseline_action": "down_left",
        "edge_reserve": 100.0,
        "bullet_count": 80.0,
        "laser_count": 48.0,
        "hard_action_count": 6.0,
        "hard_down_right": 1.0,
        "hard_stay": 1.0,
        "hard_stay_fast": 1.0,
    }
    features.update(overrides)
    transition = SimpleNamespace(
        sequence=sequence,
        frame=1000 + sequence,
        source_context="boss:0:sub31:life_cb31:timer_cb19:spell",
        legal_actions=("down_right", "stay", "stay_fast"),
    )
    return SimpleNamespace(
        features=features,
        transition=transition,
        failure_within_120=True,
        frames_to_failure=lag,
    )


def test_anchor_selection_uses_closest_eligible_row() -> None:
    selected, count = select_anchor(
        (
            _example(sequence=4, lag=12),
            _example(sequence=7, lag=8),
            _example(sequence=9, lag=4, hard_stay=0.0),
        )
    )

    assert count == 2
    assert selected is not None
    assert selected["sequence"] == 7
    assert selected["frames_to_failure"] == 8


def test_neutral_gate_requires_same_unique_winner_twice() -> None:
    assert (
        unanimous_neutral_candidate(
            (
                {"robust_winners": ["stay"]},
                {"robust_winners": ["stay"]},
            )
        )
        == "stay"
    )
    assert (
        unanimous_neutral_candidate(
            (
                {"robust_winners": ["stay", "stay_fast"]},
                {"robust_winners": ["stay"]},
            )
        )
        is None
    )
    assert (
        unanimous_neutral_candidate(
            (
                {"robust_winners": ["stay"]},
                {"robust_winners": ["stay_fast"]},
            )
        )
        is None
    )
    assert (
        unanimous_neutral_candidate(
            (
                {"robust_winners": ["down_right"]},
                {"robust_winners": ["down_right"]},
            )
        )
        is None
    )
