from __future__ import annotations

import json
from pathlib import Path

import pytest

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
            "reason": "physical-hit",
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
    assert summary["physical_hits_in_run"] == 1
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


def test_immutable_runner_requires_zero_exploration(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--practice-stage",
                "6",
                "--immutable-policy",
                "--artifact-dir",
                str(tmp_path / "run"),
            ]
        )

    args = parse_args(
        [
            "--practice-stage",
            "6",
            "--immutable-policy",
            "--exploration-rate",
            "0",
            "--artifact-dir",
            str(tmp_path / "run"),
        ]
    )
    assert args.immutable_policy


def test_runner_accepts_isolated_offline_scorer(tmp_path: Path) -> None:
    scorer = tmp_path / "ranker.dll"
    args = parse_args(
        [
            "--practice-stage",
            "6",
            "--policy-scorer-library",
            str(scorer),
            "--artifact-dir",
            str(tmp_path / "run"),
        ]
    )
    assert args.policy_scorer_library == scorer


def test_first_failure_corpus_requires_frozen_natural_practice(
    tmp_path: Path,
) -> None:
    common = [
        "--practice-stage",
        "6",
        "--first-failure-corpus-root",
        str(tmp_path / "corpus"),
        "--artifact-dir",
        str(tmp_path / "run"),
    ]
    with pytest.raises(SystemExit):
        parse_args(common)

    args = parse_args([
        *common,
        "--immutable-policy",
        "--exploration-rate",
        "0",
    ])
    assert args.first_failure_corpus_root == tmp_path / "corpus"


def test_trace_summary_counts_default_first_hit_stop(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({
            "frame": 800,
            "reason": "authority-stop:physical HIT",
            "bullets": 200,
        })
        + "\n",
        encoding="utf-8",
    )

    assert _summarize_trace(trace)["physical_hits_in_run"] == 1
