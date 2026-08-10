from __future__ import annotations

from scripts.label_headless_cow_counterfactuals import outcome_rank


def outcome(
    terminal: str,
    survival: int,
    legal: int,
    reserve: float,
) -> dict[str, object]:
    return {
        "termination_reason": terminal,
        "survival_ticks": survival,
        "minimum_native_legal_actions": legal,
        "terminal_boundary_reserve": reserve,
    }


def test_counterfactual_rank_prefers_completed_horizon_over_pretty_dead_end() -> None:
    completed = outcome("tick-limit", 240, 2, 8.0)
    dead_end = outcome("authority-failure", 239, 18, 100.0)

    assert outcome_rank(completed) > outcome_rank(dead_end)


def test_counterfactual_rank_prefers_maneuverability_after_equal_survival() -> None:
    flexible = outcome("tick-limit", 240, 8, 20.0)
    cornered = outcome("tick-limit", 240, 3, 100.0)

    assert outcome_rank(flexible) > outcome_rank(cornered)


def test_counterfactual_rank_treats_stage_clear_as_completed() -> None:
    cleared = outcome("stage-clear-success", 180, 4, 20.0)
    dead_end = outcome("authority-failure", 180, 18, 100.0)

    assert outcome_rank(cleared) > outcome_rank(dead_end)
