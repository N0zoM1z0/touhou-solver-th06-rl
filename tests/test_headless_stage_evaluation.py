from __future__ import annotations

import pytest

from scripts.evaluate_headless_stage import classify_stage_runs


def run(*, seed: int, stage: int = 6, rows: int = 2999) -> dict[str, object]:
    return {
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": stage},
        "initial_seed": seed,
        "valid": True,
        "rows": rows,
        "termination_reason": "tick-limit",
        "physical_hit": False,
        "authority_failure": None,
        "nmnb_stage_clear": False,
    }


def test_two_audited_seeds_qualify_only_as_bounded_survival() -> None:
    result = classify_stage_runs([run(seed=7), run(seed=12)], required_seeds=2, minimum_ticks=3000)

    assert result["bounded_survival_qualified"] is True
    assert result["headless_nmnb_stage_clear_qualified"] is False
    assert result["headless_status"] == "bounded-headless-survival-candidate"
    assert result["windows_promotion_allowed"] is False


def test_authority_failure_rejects_the_candidate() -> None:
    failed = {
        **run(seed=12, rows=1612),
        "termination_reason": "authority-failure",
        "authority_failure": "native safe set is empty",
    }

    result = classify_stage_runs([run(seed=7), failed], required_seeds=2, minimum_ticks=3000)

    assert result["headless_status"] == "rejected"


def test_stage_evaluation_refuses_scope_mixing() -> None:
    with pytest.raises(ValueError, match="mix scopes"):
        classify_stage_runs([run(seed=7, stage=5), run(seed=12)], required_seeds=2, minimum_ticks=3000)


def test_two_audited_nmnb_stage_clears_qualify() -> None:
    cleared = {
        **run(seed=7, rows=18000),
        "termination_reason": "stage-clear-success",
        "nmnb_stage_clear": True,
    }
    second = {**cleared, "initial_seed": 12}

    result = classify_stage_runs([cleared, second], required_seeds=2, minimum_ticks=3000)

    assert result["headless_nmnb_stage_clear_qualified"] is True
    assert result["headless_status"] == "headless-nmnb-stage-clear-candidate"
