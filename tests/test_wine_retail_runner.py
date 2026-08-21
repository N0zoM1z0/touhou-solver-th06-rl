from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.run_wine_retail import (
    _attested_process_pid,
    _bounded_priority_command,
    _repository_worktree_clean,
    _summarize_controller_completion,
    _summarize_trace,
    _windows_path,
    parse_args as _parse_args,
)


FROZEN_POLICY_ARGS = [
    "--policy-plugin", "candidate.py",
    "--policy-state", "candidate.json",
    "--immutable-policy",
]


def parse_args(args: list[str]):
    return _parse_args([*args, *FROZEN_POLICY_ARGS])


def test_windows_path_uses_wines_z_drive(tmp_path: Path) -> None:
    assert _windows_path(tmp_path / "hello") == "Z:" + str(
        (tmp_path / "hello").resolve()
    ).replace("/", "\\")


def test_repository_evidence_rejects_tracked_or_untracked_drift(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=repository,
        check=True,
    )
    tracked = repository / "tracked.txt"
    tracked.write_text("committed\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=repository, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=repository, check=True)

    assert _repository_worktree_clean(repository) is True
    tracked.write_text("dirty\n", encoding="utf-8")
    assert _repository_worktree_clean(repository) is False
    subprocess.run(["git", "restore", "tracked.txt"], cwd=repository, check=True)
    (repository / "untracked.txt").write_text("unbound\n", encoding="utf-8")
    assert _repository_worktree_clean(repository) is False


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
            "shield_collision_margin": 0.0,
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
    assert summary["zero_margin_frames"] == 1
    assert summary["invalid_shield_collision_margin_frames"] == 1
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
    assert args.wine_prefix == (
        Path(__file__).resolve().parents[1]
        / "reference/wine-prefixes/th06-retail"
    )
    assert args.score_template.name == "full-unlock-score.dat"
    assert args.policy_plugin.name == "candidate.py"
    assert args.policy_state.name == "candidate.json"


def test_runner_uses_only_the_explicit_policy_state(tmp_path: Path) -> None:
    args = parse_args(
        [
            "--practice-stage",
            "4",
            "--difficulty",
            "hard",
            "--artifact-dir",
            str(tmp_path / "run"),
        ]
    )

    assert args.policy_state.name == "candidate.json"


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
        "PYTHONDONTWRITEBYTECODE,PYTHONPATH,TH06_RL_OFFLINE_SCORER_LIBRARY,"
        "WINEARCH,WINEDEBUG,WINEDLLOVERRIDES,WINEPREFIX",
    ]
    assert command[command.index("--nice") + 1] == "-10"
    assert command[command.index("--cpu-list") + 1] == "8-31"
    assert command[-2:] == ["wine", "game.exe"]


def test_attested_child_pid_bypasses_sudo_monitor_pid(tmp_path: Path) -> None:
    process = subprocess.Popen([
        sys.executable, "-c", "import time; time.sleep(30)"
    ])
    try:
        attestation = tmp_path / "priority.json"
        attestation.write_text(json.dumps({
            "schema": "bounded-wine-process-priority-v1",
            "pid": os.getpid(),
        }), encoding="utf-8")
        assert _attested_process_pid(attestation, process) == os.getpid()
        assert process.pid != os.getpid()
    finally:
        process.terminate()
        process.wait(timeout=5)


def test_trace_summary_counts_default_first_hit_stop(tmp_path: Path) -> None:
    trace = tmp_path / "trace.jsonl"
    trace.write_text(
        json.dumps({
            "frame": 800,
            "reason": "infrastructure-stop:physical HIT",
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
        "route_completed": False,
        "practice_stage": 6,
        "physical_hits": 10,
    }


def test_controller_completion_parses_full_route_hit_total(tmp_path: Path) -> None:
    log = tmp_path / "controller.log"
    log.write_bytes(b"Full route reached Ending; physical_hits=37\n")

    assert _summarize_controller_completion(log) == {
        "practice_stage_completed": False,
        "route_completed": True,
        "practice_stage": None,
        "physical_hits": 37,
    }


def test_fixed_rng_is_allowed_only_for_training_corpus(tmp_path: Path) -> None:
    common = [
        "--practice-stage",
        "6",
        "--immutable-policy",
        "--diagnostic-rng-seed",
        "0x1234",
        "--artifact-dir",
        str(tmp_path / "run"),
    ]
    with pytest.raises(SystemExit):
        parse_args(common)

    args = parse_args([
        *common,
        "--complete-stage-training-corpus-root",
        str(tmp_path / "corpus"),
    ])
    assert args.diagnostic_rng_seed == 0x1234


def test_complete_stage_training_corpus_allows_natural_or_fixed_rng(
    tmp_path: Path,
) -> None:
    root = str(tmp_path / "complete-corpus")
    args = parse_args([
        "--practice-stage", "6",
        "--complete-stage-training-corpus-root", root,
    ])
    assert args.complete_stage_training_corpus_root == tmp_path / "complete-corpus"
    assert args.diagnostic_rng_seed is None
    fixed = parse_args([
        "--practice-stage", "6",
        "--complete-stage-training-corpus-root", root,
        "--diagnostic-rng-seed", "123",
    ])
    assert fixed.diagnostic_rng_seed == 123


def test_complete_route_corpus_requires_natural_full_route(tmp_path: Path) -> None:
    root = str(tmp_path / "route-corpus")
    args = parse_args([
        "--start-route",
        "--complete-route-corpus-root", root,
    ])
    assert args.complete_route_corpus_root == tmp_path / "route-corpus"

    with pytest.raises(SystemExit):
        parse_args([
            "--practice-stage", "1",
            "--complete-route-corpus-root", root,
        ])
    with pytest.raises(SystemExit):
        parse_args([
            "--start-route",
            "--complete-route-corpus-root", root,
            "--diagnostic-rng-seed", "1",
        ])


def test_runner_has_no_implicit_policy() -> None:
    with pytest.raises(SystemExit):
        _parse_args(["--start-route"])
