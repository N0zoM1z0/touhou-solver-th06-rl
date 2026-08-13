from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from scripts.run_wine_retail import (
    _bounded_priority_command,
    _summarize_controller_completion,
    _summarize_trace,
    _windows_path,
    parse_args,
)


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
            "run_id": "run-a",
        },
        {
            "frame": 121,
            "bullets": 9,
            "reason": "physical-hit",
            "hard_collision_margin": 0.0,
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
    assert summary["source_exact_hard_fallbacks"] == 1
    assert summary["decisions"] == 77
    assert summary["corpus_run_ids"] == ["run-a"]
    assert summary["last_policy_metrics"] == {
        "physical_hit_events": 2,
        "decisions": 77,
    }


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


def test_runner_requires_bounded_priority_as_a_complete_cpu_contract(
    tmp_path: Path,
) -> None:
    cpus = sorted(os.sched_getaffinity(0))[:4]
    common = [
        "--practice-stage", "6",
        "--artifact-dir", str(tmp_path / "run"),
        "--game-cpu-list", ",".join(map(str, cpus[:2])),
        "--controller-cpu-list", ",".join(map(str, cpus[2:])),
    ]
    with pytest.raises(SystemExit):
        parse_args([*common, "--game-nice", "-10"])
    with pytest.raises(SystemExit):
        parse_args([
            *common, "--game-nice", "-16", "--controller-nice", "-10"
        ])

    args = parse_args([
        *common, "--game-nice", "-10", "--controller-nice", "-10"
    ])
    assert args.game_nice == args.controller_nice == -10


def test_bounded_priority_wrapper_is_explicit_and_attested(tmp_path: Path) -> None:
    command = _bounded_priority_command(
        ["wine", "game.exe"],
        cpu_list="8-31",
        nice=-10,
        attestation=tmp_path / "priority.json",
    )

    assert command[:3] == [
        "sudo", "-n",
        "--preserve-env=DISPLAY,LANG,LC_ALL,LP_NUM_THREADS,MESA_GLTHREAD,"
        "TH06_RL_OFFLINE_SCORER_LIBRARY,WINEDEBUG,WINEDLLOVERRIDES,WINEPREFIX",
    ]
    assert command[command.index("--nice") + 1] == "-10"
    assert command[command.index("--cpu-list") + 1] == "8-31"
    assert command[-2:] == ["wine", "game.exe"]


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


def test_controller_completion_is_explicitly_parsed_for_full_stage(tmp_path: Path) -> None:
    log = tmp_path / "controller.log"
    log.write_bytes(
        b"noise\nPractice Stage 6 complete; physical_hits=10\ncleanup\n"
    )
    assert _summarize_controller_completion(log) == {
        "practice_stage_completed": True,
        "practice_stage": 6,
        "physical_hits": 10,
    }


def test_fixed_rng_is_allowed_only_for_training_corpus(tmp_path: Path) -> None:
    common = [
        "--practice-stage",
        "6",
        "--immutable-policy",
        "--exploration-rate",
        "0",
        "--diagnostic-rng-seed",
        "0x1234",
        "--artifact-dir",
        str(tmp_path / "run"),
    ]
    with pytest.raises(SystemExit):
        parse_args(common)

    args = parse_args([
        *common,
        "--first-failure-corpus-root",
        str(tmp_path / "corpus"),
    ])
    assert args.diagnostic_rng_seed == 0x1234


def test_complete_stage_training_corpus_allows_natural_or_fixed_rng_and_requires_immutable(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "complete-corpus")
    with pytest.raises(SystemExit):
        parse_args([
            "--practice-stage", "6",
            "--complete-stage-training-corpus-root", root,
        ])
    args = parse_args([
        "--practice-stage", "6",
        "--complete-stage-training-corpus-root", root,
        "--immutable-policy",
        "--exploration-rate", "0",
    ])
    assert args.complete_stage_training_corpus_root == tmp_path / "complete-corpus"
    assert args.diagnostic_rng_seed is None
    fixed = parse_args([
        "--practice-stage", "6",
        "--complete-stage-training-corpus-root", root,
        "--diagnostic-rng-seed", "123",
        "--immutable-policy",
        "--exploration-rate", "0",
    ])
    assert fixed.diagnostic_rng_seed == 123


def test_option_smoke_is_time_bounded_fixed_rng_and_non_evidence(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "smoke-corpus")
    common = [
        "--practice-stage", "6",
        "--option-smoke-corpus-root", root,
        "--immutable-policy",
        "--exploration-rate", "0",
        "--diagnostic-rng-seed", "0xd53c",
    ]
    with pytest.raises(SystemExit):
        parse_args(common)
    args = parse_args([*common, "--seconds", "45"])
    assert args.option_smoke_corpus_root == tmp_path / "smoke-corpus"
    assert args.seconds == 45.0


def test_option_smoke_is_exclusive_with_evidence_corpus_modes(
    tmp_path: Path,
) -> None:
    with pytest.raises(SystemExit):
        parse_args([
            "--practice-stage", "6",
            "--option-smoke-corpus-root", str(tmp_path / "smoke"),
            "--complete-stage-training-corpus-root", str(tmp_path / "training"),
            "--seconds", "45",
            "--immutable-policy",
            "--exploration-rate", "0",
            "--diagnostic-rng-seed", "1",
        ])
