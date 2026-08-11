from __future__ import annotations

import pytest

from scripts.audit_full_stage_hit_windows import audit_hit_windows


def _row(
    frame: int,
    reason: str,
    hard_count: int,
    action: str | None = "left",
    effort_horizon: int | None = None,
):
    return {
        "frame": frame,
        "time": float(frame),
        "reason": reason,
        "hard_count": hard_count,
        "effort_horizon": (12 if hard_count else 0)
        if effort_horizon is None
        else effort_horizon,
        "action": action,
        "source_context": "boss:0:sub10:auto",
        "x": 8.0,
        "y": 400.0,
        "bullets": 400,
        "lasers": 0,
    }


def test_audit_keeps_hits_separate_and_finds_final_collapse() -> None:
    rows = [
        _row(1, "ok", 18),
        _row(2, "ok", 3, effort_horizon=4),
        _row(3, "stale-retry", 3, None),
        _row(4, "control-dead-end:Hard safe set empty", 0, None),
        _row(5, "control-dead-end:Hard safe set empty", 0, None),
        _row(6, "physical-hit", 0, None),
        _row(7, "ok", 8, effort_horizon=4),
        _row(8, "control-dead-end:Hard safe set empty", 0, None),
        _row(9, "physical-hit", 0, None),
    ]

    result = audit_hit_windows(rows, window_frames=120)

    assert result["totals"]["physical_hits"] == 2
    assert result["totals"]["hits_with_narrow_ok_row"] == 1
    assert result["totals"]["hits_with_final_dead_end"] == 2
    first, second = result["hits"]
    assert first["final_dead_end"] == {"rows": 2, "start_frame": 4, "lead_frames": 2}
    assert first["last_ok_before_final_dead_end"]["frame"] == 2
    assert first["last_narrow_ok_before_final_dead_end"]["hard_count"] == 3
    assert first["first_degraded_ok_before_final_dead_end"]["frame"] == 2
    assert first["final_degraded_ok_run"] == {
        "rows": 0,
        "start_frame": None,
        "lead_frames": None,
    }
    assert second["window_start_frame"] == 7
    assert second["rows_in_window"] == 2
    assert second["last_narrow_ok_before_final_dead_end"] is None
    assert second["final_degraded_ok_run"] == {
        "rows": 1,
        "start_frame": 7,
        "lead_frames": 2,
    }
    assert result["totals"]["hits_with_degraded_horizon_ok_row"] == 2
    assert result["totals"]["degraded_horizon_ok_runs"] == 2
    assert result["collection_trigger_diagnostics"]["1"]["distinct_hits_covered"] == 2
    assert result["collection_trigger_diagnostics"]["2"]["activations"] == 0


def test_audit_validates_parameters() -> None:
    with pytest.raises(ValueError):
        audit_hit_windows([], window_frames=0)
    with pytest.raises(ValueError):
        audit_hit_windows([], narrow_max=0)
