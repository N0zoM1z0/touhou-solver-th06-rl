from __future__ import annotations

from scripts.audit_targeted_headless_cow import (
    compare_actions,
    robust_outcome_rank,
    targeted_family,
)


def _row(**state_overrides: object) -> dict[str, object]:
    state: dict[str, object] = {
        "boundary_reserve": 8.0,
        "bullet_count": 420,
        "laser_count": 0,
    }
    state.update(state_overrides)
    return {
        "benchmark_forced_action": False,
        "source_context": "boss:0/10",
        "legal_actions": ["stay", "left", "right", "left_fast", "right_fast"],
        "state": state,
    }


def _outcome(action: str, *, survival: int, terminal: str) -> dict[str, object]:
    return {
        "first_action": action,
        "termination_reason": terminal,
        "survival_ticks": survival,
        "minimum_native_legal_actions": 5,
        "terminal_boundary_reserve": 16.0,
        "physical_deaths_delta": 0,
    }


def test_targeted_families_use_only_automatic_context_and_generic_bins() -> None:
    assert targeted_family(_row()) == "sub10-dense-boundary-broad"
    assert targeted_family(_row(bullet_count=383)) is None

    sub31 = _row(boundary_reserve=24.0, bullet_count=40, laser_count=48)
    sub31["source_context"] = "boss:0/31"
    assert targeted_family(sub31) == "sub31-interior-lasers-broad"

    sub18 = _row(bullet_count=220)
    sub18["source_context"] = "boss:0/18"
    assert targeted_family(sub18) == "sub18-medium-boundary-broad"


def test_forced_or_narrow_rows_never_enter_targeted_cow() -> None:
    forced = _row()
    forced["benchmark_forced_action"] = True
    assert targeted_family(forced) is None
    narrow = _row()
    narrow["legal_actions"] = ["stay", "left", "right", "left_fast"]
    assert targeted_family(narrow) is None


def test_robust_comparison_prefers_survival_before_reserve() -> None:
    completed = _outcome("left_fast", survival=600, terminal="tick-limit")
    failed = _outcome("right_fast", survival=599, terminal="authority-failure")
    failed["terminal_boundary_reserve"] = 100.0
    checkpoint = {"outcomes": [completed, failed]}

    assert robust_outcome_rank(completed) > robust_outcome_rank(failed)
    assert compare_actions(checkpoint, "left_fast", "right_fast") == "left-better"
    assert compare_actions(checkpoint, "up", "right_fast") is None
