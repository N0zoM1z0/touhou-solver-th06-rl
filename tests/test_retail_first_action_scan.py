from __future__ import annotations

from scripts.audit_retail_first_action_scan import select_discovery_candidate


def _outcome(
    action: str,
    *,
    width: int,
    reserve: float = 32.0,
    survival: int = 600,
) -> dict[str, object]:
    return {
        "first_action": action,
        "termination_reason": "tick-limit",
        "physical_deaths_delta": 0,
        "survival_ticks": survival,
        "minimum_native_legal_actions": width,
        "terminal_boundary_reserve": reserve,
    }


def test_scan_requires_one_unique_robust_nonincumbent_winner() -> None:
    result = select_discovery_candidate(
        {"outcomes": [
            _outcome("down_right", width=3),
            _outcome("left", width=5),
            _outcome("right", width=3),
        ]},
        incumbent_action="down_right",
        excluded_actions=("down_left", "down_fast"),
    )

    assert result["robust_winners"] == ["left"]
    assert result["candidate"] == "left"
    assert result["conclusion"] == "confirmation-required"


def test_scan_rejects_a_robust_tie_without_pixel_level_selection() -> None:
    result = select_discovery_candidate(
        {"outcomes": [
            _outcome("down_right", width=5, reserve=65.0),
            _outcome("left", width=6, reserve=100.0),
        ]},
        incumbent_action="down_right",
        excluded_actions=(),
    )

    assert result["robust_winners"] == ["down_right", "left"]
    assert result["candidate"] is None
    assert result["conclusion"] == "discovery-robust-tie-rejected"


def test_scan_does_not_reopen_a_previously_rejected_action() -> None:
    result = select_discovery_candidate(
        {"outcomes": [
            _outcome("down_right", width=3),
            _outcome("down_fast", width=5),
        ]},
        incumbent_action="down_right",
        excluded_actions=("down_fast",),
    )

    assert result["robust_winners"] == ["down_fast"]
    assert result["candidate"] is None
    assert result["conclusion"] == "discovery-previously-rejected-action-wins"
