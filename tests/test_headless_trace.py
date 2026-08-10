from __future__ import annotations

from scripts.compare_headless_traces import first_difference


def test_headless_trace_comparison_reports_first_physical_path() -> None:
    left = {"tick": 3, "player": {"x": 10.0}, "bullets": [{"x": 20.0}]}
    right = {"tick": 3, "player": {"x": 10.0}, "bullets": [{"x": 20.25}]}

    difference = first_difference(left, right, absolute_tolerance=0.01)

    assert difference == {
        "path": "$.bullets[0].x",
        "left": 20.0,
        "right": 20.25,
        "reason": "value",
    }


def test_headless_trace_comparison_accepts_configured_float_tolerance() -> None:
    assert first_difference(1.0, 1.0001, absolute_tolerance=0.001) is None
