from __future__ import annotations

import json
from pathlib import Path

from scripts.run_wine_retail import _summarize_trace, _windows_path, parse_args


def test_windows_path_uses_wines_z_drive(tmp_path: Path) -> None:
    assert _windows_path(tmp_path / "hello") == "Z:" + str(
        (tmp_path / "hello").resolve()
    ).replace("/", "\\")


def test_trace_summary_retains_hit_and_fail_close_counts(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    rows = [
        {
            "frame": 120,
            "bullets": 3,
            "event": "continuous-fail-close",
        },
        {
            "frame": 121,
            "bullets": 9,
            "policy": {
                "metrics": {"physical_hit_events": 2, "decisions": 77}
            },
        },
    ]
    trace.write_text("".join(json.dumps(row) + "\n" for row in rows))

    summary = _summarize_trace(trace)

    assert summary["rows"] == 2
    assert summary["event_counts"] == {"continuous-fail-close": 1}
    assert summary["first_frame"] == 120
    assert summary["last_frame"] == 121
    assert summary["max_bullets"] == 9
    assert summary["physical_hit_events"] == 2
    assert summary["decisions"] == 77


def test_runner_accepts_route_as_a_distinct_mode(tmp_path: Path) -> None:
    args = parse_args(["--start-route", "--artifact-dir", str(tmp_path / "run")])
    assert args.start_route
    assert args.practice_stage is None
    assert args.wine_prefix == Path("/home/c/.wine-th06-rl-retail")
    assert args.score_template.name == "full-unlock-score.dat"
    assert args.policy_plugin.name == "adaptive.py"
    assert args.policy_state.name == "lunatic_reimu_a_route.json"
    assert args.exploration_rate == 0.03


def test_runner_records_a_scope_specific_practice_policy_state(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--practice-stage",
            "4",
            "--difficulty",
            "hard",
            "--exploration-rate",
            "0",
            "--artifact-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.policy_state.name == "hard_reimu_a_stage4.json"
    assert args.exploration_rate == 0.0
