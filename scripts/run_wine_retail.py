#!/usr/bin/env python3
"""Run exact TH06 1.02h retail under Wine without a PTY and retain evidence."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
from typing import Any


RETAIL_SHA256 = "9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245"
RETAIL_EXECUTABLE = "東方紅魔郷.exe"
STARTUP_MARKER = "TH06_RL_WINE_STARTUP normalized=1"
FULL_UNLOCK_SCORE_SHA256 = "54cd436d5d8a7a904190c792a977bf270ab1cb759fd72101e51e94d26b749c71"
_PRACTICE_COMPLETE_RE = re.compile(
    rb"Practice Stage (\d+) complete; physical_hits=(\d+)"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _windows_path(path: Path) -> str:
    resolved = path.resolve()
    if not resolved.is_absolute():
        raise ValueError("Wine path must be absolute")
    return "Z:" + str(resolved).replace("/", "\\")


def _repository_commit(repository: Path) -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _prefix_processes(prefix: Path) -> list[dict[str, Any]]:
    """Return only live processes explicitly bound to this Wine prefix."""
    wanted = str(prefix.resolve())
    matches = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            environment = (entry / "environ").read_bytes().split(b"\0")
            values = {}
            for item in environment:
                if b"=" in item:
                    key, value = item.split(b"=", 1)
                    if key in (b"WINEPREFIX", b"DISPLAY"):
                        values[key.decode()] = value.decode(errors="replace")
            executable = os.path.realpath(entry / "exe")
            process_prefix = values.get("WINEPREFIX")
            if process_prefix is None and "wine" in executable.lower():
                process_prefix = str(Path.home() / ".wine")
            if os.path.realpath(process_prefix or "") != wanted:
                continue
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(
                errors="replace"
            ).strip()
            matches.append(
                {
                    "pid": int(entry.name),
                    "command": command,
                    "display": values.get("DISPLAY"),
                }
            )
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
    return sorted(matches, key=lambda item: item["pid"])


def _summarize_trace(path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "path": str(path),
        "rows": 0,
        "event_counts": {},
        "first_frame": None,
        "last_frame": None,
        "max_bullets": 0,
        "physical_hit_events": None,
        "physical_hits_in_run": 0,
        "source_exact_hard_fallbacks": 0,
        "decisions": None,
        "last_policy_metrics": None,
        "corpus_run_ids": [],
    }
    if not path.is_file():
        return summary
    events: Counter[str] = Counter()
    frames = []
    corpus_run_ids = set()
    last_policy_metrics = None
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            record = json.loads(line)
            summary["rows"] += 1
            event = record.get("event")
            if isinstance(event, str):
                events[event] += 1
            frame = record.get("frame")
            if isinstance(frame, int):
                frames.append(frame)
            run_id = record.get("run_id")
            if isinstance(run_id, str) and run_id:
                corpus_run_ids.add(run_id)
            bullets = record.get("bullets")
            if isinstance(bullets, int):
                summary["max_bullets"] = max(summary["max_bullets"], bullets)
            metrics = (record.get("policy") or {}).get("metrics")
            if isinstance(metrics, dict):
                last_policy_metrics = metrics
            if record.get("reason") in {
                "physical-hit",
                "authority-stop:physical HIT",
            }:
                summary["physical_hits_in_run"] += 1
            if record.get("hard_collision_margin") == 0.0:
                summary["source_exact_hard_fallbacks"] += 1
    summary["event_counts"] = dict(sorted(events.items()))
    summary["corpus_run_ids"] = sorted(corpus_run_ids)
    if frames:
        summary["first_frame"] = min(frames)
        summary["last_frame"] = max(frames)
    if last_policy_metrics is not None:
        summary["last_policy_metrics"] = last_policy_metrics
        summary["physical_hit_events"] = last_policy_metrics.get(
            "physical_hit_events"
        )
        summary["decisions"] = last_policy_metrics.get("decisions")
    return summary


def _summarize_controller_completion(path: Path) -> dict[str, Any]:
    summary = {
        "practice_stage_completed": False,
        "practice_stage": None,
        "physical_hits": None,
    }
    if not path.is_file():
        return summary
    matches = list(_PRACTICE_COMPLETE_RE.finditer(path.read_bytes()))
    if matches:
        stage, hits = matches[-1].groups()
        summary.update({
            "practice_stage_completed": True,
            "practice_stage": int(stage),
            "physical_hits": int(hits),
        })
    return summary


def _stop_process(process: subprocess.Popen[Any] | None, timeout: float = 5.0) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=timeout)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--game-dir",
        type=Path,
        default=repository / "reference/th06-game-original/th06",
    )
    parser.add_argument(
        "--wine-prefix",
        type=Path,
        default=Path("/home/c/.wine-th06-rl-retail"),
    )
    parser.add_argument("--wine", type=Path, default=Path("/usr/bin/wine"))
    parser.add_argument(
        "--windows-python",
        type=Path,
        default=(
            repository
            / "reference/tools/windows-python-3.11.9-embed-win32/python.exe"
        ),
    )
    parser.add_argument(
        "--native-library",
        type=Path,
        default=repository / "build/native-win32-fully-static/libth06_rl_native.dll",
    )
    parser.add_argument(
        "--score-template",
        type=Path,
        default=(
            repository
            / "reference/th06-game-original/full-unlock-score.dat"
        ),
        help="ignored canonical full-unlock score.dat restored before every trial",
    )
    parser.add_argument(
        "--policy-plugin",
        type=Path,
        default=repository / "src/th06_rl/policies/adaptive.py",
    )
    parser.add_argument("--policy-state", type=Path)
    parser.add_argument(
        "--policy-scorer-library",
        type=Path,
        help="isolated native batch scorer used by an offline policy plug-in",
    )
    parser.add_argument(
        "--immutable-policy",
        action="store_true",
        help=(
            "copy the declared policy state into the run artifact, disable "
            "learning/checkpoint writes, and assert before/after SHA equality"
        ),
    )
    parser.add_argument(
        "--first-failure-corpus-root",
        type=Path,
        help=(
            "collect one lossless physical Practice prefix and stop on the first "
            "HIT or authority failure; unlike continuation benchmarks, this "
            "mode does not patch lives"
        ),
    )
    parser.add_argument(
        "--complete-stage-training-corpus-root",
        type=Path,
        help=(
            "collect one fixed-RNG, patched-life, HIT-continuation Practice "
            "Stage for factual offline-RL training"
        ),
    )
    parser.add_argument(
        "--option-smoke-corpus-root",
        type=Path,
        help=(
            "collect a fixed-RNG, time-bounded, patched-life option corpus "
            "for non-evidence Generation-3 wiring smoke only"
        ),
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--practice-stage", type=int, choices=range(1, 7))
    mode.add_argument("--start-route", action="store_true")
    parser.add_argument(
        "--difficulty", choices=("normal", "hard", "lunatic"), default="lunatic"
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=0.0,
        help="0 lets the Practice stage reach its natural result path",
    )
    parser.add_argument(
        "--exploration-rate",
        type=float,
        default=0.03,
        help=(
            "adaptive-policy exploration probability; use 0 only for an "
            "explicit frozen-action-selection evaluation"
        ),
    )
    parser.add_argument(
        "--diagnostic-rng-seed",
        type=lambda value: int(value, 0),
        choices=range(0x10000),
        metavar="0..0xffff",
        help=(
            "fixed original-retail RNG seed for diagnostic training variance; "
            "forbidden in final HIT-continuation evaluation"
        ),
    )
    parser.add_argument("--display", default=":97")
    parser.add_argument("--artifact-dir", type=Path)
    args = parser.parse_args(argv)
    if args.seconds < 0:
        parser.error("--seconds cannot be negative")
    if not 0.0 <= args.exploration_rate <= 1.0:
        parser.error("--exploration-rate must be in [0, 1]")
    if args.immutable_policy and args.exploration_rate != 0.0:
        parser.error("--immutable-policy requires --exploration-rate 0")
    if args.first_failure_corpus_root is not None:
        if args.start_route:
            parser.error("first-failure corpus collection currently requires Practice")
        if args.seconds != 0.0:
            parser.error("first-failure corpus collection requires --seconds 0")
        if not args.immutable_policy:
            parser.error(
                "first-failure corpus collection requires --immutable-policy"
            )
    corpus_modes = sum(
        value is not None
        for value in (
            args.first_failure_corpus_root,
            args.complete_stage_training_corpus_root,
            args.option_smoke_corpus_root,
        )
    )
    if corpus_modes > 1:
        parser.error("Wine corpus modes are mutually exclusive")
    if args.complete_stage_training_corpus_root is not None:
        if args.start_route:
            parser.error("complete-Stage training corpus currently requires Practice")
        if args.seconds != 0.0:
            parser.error("complete-Stage training corpus requires --seconds 0")
        if not args.immutable_policy:
            parser.error(
                "complete-Stage training corpus requires --immutable-policy"
            )
        if args.diagnostic_rng_seed is None:
            parser.error(
                "complete-Stage training corpus requires --diagnostic-rng-seed"
            )
    if args.option_smoke_corpus_root is not None:
        if args.start_route:
            parser.error("option smoke currently requires Practice")
        if args.seconds <= 0.0:
            parser.error("option smoke requires a positive --seconds limit")
        if not args.immutable_policy:
            parser.error("option smoke requires --immutable-policy")
        if args.diagnostic_rng_seed is None:
            parser.error("option smoke requires --diagnostic-rng-seed")
    if (
        args.diagnostic_rng_seed is not None
        and args.first_failure_corpus_root is None
        and args.complete_stage_training_corpus_root is None
        and args.option_smoke_corpus_root is None
    ):
        parser.error(
            "--diagnostic-rng-seed is training-only and requires "
            "a declared corpus root"
        )
    if not args.display.startswith(":") or not args.display[1:].isdigit():
        parser.error("--display must look like :97")
    if args.artifact_dir is None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.artifact_dir = (
            repository
            / "artifacts"
            / (
                f"wine-retail-route-{timestamp}"
                if args.start_route
                else f"wine-retail-stage{args.practice_stage}-{timestamp}"
            )
        )
    if args.policy_state is None:
        label = (
            f"{args.difficulty}_reimu_a_route"
            if args.start_route
            else f"{args.difficulty}_reimu_a_stage{args.practice_stage}"
        )
        args.policy_state = repository / f"artifacts/policy/{label}.json"
    args.repository = repository
    return args


def run(args: argparse.Namespace) -> int:
    repository = args.repository.resolve()
    game_dir = args.game_dir.resolve()
    prefix = args.wine_prefix.resolve()
    executable = game_dir / RETAIL_EXECUTABLE
    score = game_dir / "score.dat"
    score_template = args.score_template.resolve()
    config = game_dir / "東方紅魔郷.cfg"
    native = args.native_library.resolve()
    windows_python = args.windows_python.resolve()
    policy_plugin = args.policy_plugin.resolve()
    policy_state = args.policy_state.resolve()
    policy_scorer_library = (
        args.policy_scorer_library.resolve()
        if args.policy_scorer_library is not None
        else None
    )
    first_failure_corpus_root = (
        args.first_failure_corpus_root.resolve()
        if args.first_failure_corpus_root is not None
        else None
    )
    complete_stage_training_corpus_root = (
        args.complete_stage_training_corpus_root.resolve()
        if args.complete_stage_training_corpus_root is not None
        else None
    )
    option_smoke_corpus_root = (
        args.option_smoke_corpus_root.resolve()
        if args.option_smoke_corpus_root is not None
        else None
    )
    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=False)
    report_path = artifact_dir / "report.json"
    trace_path = artifact_dir / "trace.jsonl"
    report: dict[str, Any] = {
        "schema": "th06-rl-wine-retail-run-v1",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "practice_stage": args.practice_stage,
        "start_route": args.start_route,
        "difficulty": args.difficulty,
        "seconds": args.seconds,
        "exploration_rate": args.exploration_rate,
        "immutable_policy": args.immutable_policy,
        "diagnostic_rng_seed": args.diagnostic_rng_seed,
        "evaluation_mode": (
            "fixed-rng-complete-stage-training"
            if complete_stage_training_corpus_root is not None
            else "fixed-rng-option-smoke-non-evidence"
            if option_smoke_corpus_root is not None
            else "fixed-rng-first-failure-training"
            if args.diagnostic_rng_seed is not None
            else "first-failure-corpus"
            if first_failure_corpus_root is not None
            else "hit-continuation-benchmark"
        ),
        "first_failure_corpus_root": (
            str(first_failure_corpus_root)
            if first_failure_corpus_root is not None
            else None
        ),
        "complete_stage_training_corpus_root": (
            str(complete_stage_training_corpus_root)
            if complete_stage_training_corpus_root is not None
            else None
        ),
        "option_smoke_corpus_root": (
            str(option_smoke_corpus_root)
            if option_smoke_corpus_root is not None
            else None
        ),
        "display": args.display,
        "wine_prefix": str(prefix),
        "retail_executable": str(executable),
        "expected_retail_sha256": RETAIL_SHA256,
        "controller_returncode": None,
        "gdb_normalized": False,
        "error": None,
    }
    game_process = None
    xvfb_process = None
    prefix_owned = False
    controller_log = (artifact_dir / "controller.log").open("wb")
    game_log = (artifact_dir / "game.log").open("wb")
    xvfb_log = (artifact_dir / "xvfb.log").open("wb")
    controller_policy_state = policy_state
    policy_state_sha256_before = None
    controller_policy_state_sha256_before = None
    try:
        for required in (
            executable,
            score_template,
            config,
            native,
            windows_python,
            policy_plugin,
        ):
            if not required.is_file():
                raise RuntimeError(f"required Wine retail input is absent: {required}")
        if policy_scorer_library is not None and not policy_scorer_library.is_file():
            raise RuntimeError(
                f"required offline scorer is absent: {policy_scorer_library}"
            )
        if args.immutable_policy:
            if not policy_state.is_file():
                raise RuntimeError(
                    "immutable Wine evaluation requires an existing policy state"
                )
            controller_policy_state = artifact_dir / "policy-state-input.json"
            shutil.copy2(policy_state, controller_policy_state)
        policy_state_sha256_before = (
            _sha256(policy_state) if policy_state.is_file() else None
        )
        controller_policy_state_sha256_before = (
            _sha256(controller_policy_state)
            if controller_policy_state.is_file()
            else None
        )
        score_template_sha = _sha256(score_template)
        if score_template_sha != FULL_UNLOCK_SCORE_SHA256:
            raise RuntimeError(
                f"full-unlock score template SHA mismatch: {score_template_sha}"
            )
        shutil.copy2(score_template, score)
        retail_sha = _sha256(executable)
        if retail_sha != RETAIL_SHA256:
            raise RuntimeError(f"retail SHA mismatch: {retail_sha}")
        report.update(
            {
                "retail_sha256": retail_sha,
                "score_sha256": _sha256(score),
                "score_template": str(score_template),
                "score_template_sha256": score_template_sha,
                "config_sha256_before": _sha256(config),
                "native_sha256": _sha256(native),
                "windows_python_sha256": _sha256(windows_python),
                "repository_commit": _repository_commit(repository),
                "policy_plugin": str(policy_plugin),
                "policy_plugin_sha256": _sha256(policy_plugin),
                "policy_state": str(policy_state),
                "policy_state_sha256_before": policy_state_sha256_before,
                "policy_state_size_before": (
                    policy_state.stat().st_size if policy_state.is_file() else None
                ),
                "controller_policy_state": str(controller_policy_state),
                "controller_policy_state_sha256_before": (
                    controller_policy_state_sha256_before
                ),
                "policy_scorer_library": (
                    str(policy_scorer_library)
                    if policy_scorer_library is not None
                    else None
                ),
                "policy_scorer_library_sha256": (
                    _sha256(policy_scorer_library)
                    if policy_scorer_library is not None
                    else None
                ),
            }
        )
        version = subprocess.run(
            [str(args.wine), "--version"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        report["wine_version"] = version

        existing = _prefix_processes(prefix)
        if existing:
            raise RuntimeError(
                "Wine prefix already has live processes; refusing shared-prefix "
                f"cleanup: {existing}"
            )
        prefix_owned = True
        socket = Path("/tmp/.X11-unix") / f"X{args.display[1:]}"
        if socket.exists():
            raise RuntimeError(f"X display socket already exists: {socket}")

        config_result = subprocess.run(
            [
                sys.executable,
                str(repository / "scripts/configure_wine_retail.py"),
                str(game_dir),
                "--report",
                str(artifact_dir / "config.json"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        (artifact_dir / "config.stdout").write_text(
            config_result.stdout, encoding="utf-8"
        )
        report["config_sha256_after"] = _sha256(config)

        environment = os.environ.copy()
        environment.update(
            {
                "WINEPREFIX": str(prefix),
                "DISPLAY": args.display,
                "LANG": "ja_JP.UTF-8",
                "LC_ALL": "ja_JP.UTF-8",
                "WINEDEBUG": "-all",
                "LP_NUM_THREADS": "1",
                "MESA_GLTHREAD": "false",
                # TH06 and the embeddable controller do not require .NET or
                # MSHTML. Disable Wine's interactive Mono/Gecko installers so
                # a new headless prefix cannot hang on an unseen dialog.
                "WINEDLLOVERRIDES": "mscoree,mshtml=",
            }
        )
        if policy_scorer_library is not None:
            environment["TH06_RL_OFFLINE_SCORER_LIBRARY"] = _windows_path(
                policy_scorer_library
            )
        xvfb_process = subprocess.Popen(
            ["Xvfb", args.display, "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
            stdin=subprocess.DEVNULL,
            stdout=xvfb_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(0.5)
        if xvfb_process.poll() is not None:
            raise RuntimeError(f"Xvfb exited early with {xvfb_process.returncode}")

        prefix_marker = prefix / ".th06-rl-retail-ready-v1"
        prefix_initialized = not prefix_marker.is_file()
        report["prefix_initialized"] = prefix_initialized
        if prefix_initialized:
            with (artifact_dir / "wineboot.log").open("wb") as wineboot_log:
                initialized = subprocess.run(
                    ["wineboot", "-u"],
                    env=environment,
                    stdin=subprocess.DEVNULL,
                    stdout=wineboot_log,
                    stderr=subprocess.STDOUT,
                    check=False,
                    timeout=180,
                )
            if initialized.returncode:
                raise RuntimeError(
                    f"dedicated Wine prefix initialization failed: {initialized.returncode}"
                )
            subprocess.run(
                ["wineserver", "-k"],
                env=environment,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            prefix_marker.write_text("wine-11.0 headless retail\n", encoding="utf-8")

        game_process = subprocess.Popen(
            [str(args.wine), f"./{RETAIL_EXECUTABLE}"],
            cwd=game_dir,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=game_log,
            stderr=subprocess.STDOUT,
        )
        report["game_host_pid"] = game_process.pid
        time.sleep(1.0)
        gdb_result = subprocess.run(
            [
                "sudo",
                "-n",
                "gdb",
                "-q",
                "-nx",
                "-batch",
                "-p",
                str(game_process.pid),
                "-x",
                str(repository / "scripts/gdb/normalize_wine_retail_startup.py"),
            ],
            check=False,
            capture_output=True,
            timeout=60,
        )
        gdb_output = gdb_result.stdout + gdb_result.stderr
        (artifact_dir / "gdb.log").write_bytes(gdb_output)
        if STARTUP_MARKER.encode() not in gdb_output:
            raise RuntimeError(
                f"Wine startup normalization failed with GDB rc {gdb_result.returncode}"
            )
        report["gdb_normalized"] = True

        controller = [
            str(args.wine),
            _windows_path(windows_python),
            _windows_path(repository / "scripts/run_th06_rl.py"),
            "--game-dir",
            _windows_path(game_dir),
            "--game-executable-name",
            RETAIL_EXECUTABLE,
            "--native-library",
            _windows_path(native),
            "--difficulty",
            args.difficulty,
            "--armed",
            "--seconds",
            str(args.seconds),
            "--exploration-rate",
            str(args.exploration_rate),
            "--no-post-run-audit",
            "--stop-game",
            "--min-commit-headroom-gib",
            "0",
            "--trace",
            _windows_path(trace_path),
        ]
        if args.diagnostic_rng_seed is not None:
            controller.extend((
                "--diagnostic-rng-seed",
                hex(args.diagnostic_rng_seed),
            ))
        if complete_stage_training_corpus_root is not None:
            controller.extend((
                "--patch-lives",
                "--continuous-stage",
                "--corpus-root",
                _windows_path(complete_stage_training_corpus_root),
            ))
        elif option_smoke_corpus_root is not None:
            controller.extend((
                "--patch-lives",
                "--continuous-stage",
                "--corpus-root",
                _windows_path(option_smoke_corpus_root),
            ))
        elif first_failure_corpus_root is None:
            controller.extend(("--patch-lives", "--continuous-stage", "--no-corpus"))
        else:
            controller.extend(
                (
                    "--corpus-root",
                    _windows_path(first_failure_corpus_root),
                )
            )
        if args.start_route:
            controller.append("--start-route")
        else:
            controller.extend(
                (
                    "--practice-stage",
                    str(args.practice_stage),
                    "--expected-stage",
                    str(args.practice_stage),
                )
            )
        controller.extend(("--policy-plugin", _windows_path(policy_plugin)))
        controller.extend(
            ("--policy-state", _windows_path(controller_policy_state))
        )
        if args.immutable_policy:
            controller.append("--immutable-policy")
        report["controller_command"] = controller
        result = subprocess.run(
            controller,
            cwd=repository,
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=controller_log,
            stderr=subprocess.STDOUT,
            check=False,
        )
        report["controller_returncode"] = result.returncode
        if args.immutable_policy:
            source_after = _sha256(policy_state) if policy_state.is_file() else None
            controller_after = (
                _sha256(controller_policy_state)
                if controller_policy_state.is_file()
                else None
            )
            if source_after != policy_state_sha256_before:
                raise RuntimeError("immutable source policy state changed during run")
            if controller_after != controller_policy_state_sha256_before:
                raise RuntimeError("immutable evaluation policy state changed during run")
        return result.returncode
    except BaseException as error:
        report["error"] = f"{type(error).__name__}: {error}"
        if isinstance(error, KeyboardInterrupt):
            return 130
        return 78
    finally:
        _stop_process(game_process)
        if prefix_owned:
            environment = os.environ.copy()
            environment["WINEPREFIX"] = str(prefix)
            subprocess.run(
                ["wineserver", "-k"],
                env=environment,
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        _stop_process(xvfb_process)
        controller_log.close()
        game_log.close()
        xvfb_log.close()
        report["trace"] = _summarize_trace(trace_path)
        report["controller_completion"] = _summarize_controller_completion(
            artifact_dir / "controller.log"
        )
        report["policy_state_sha256_after"] = (
            _sha256(policy_state) if policy_state.is_file() else None
        )
        report["policy_state_size_after"] = (
            policy_state.stat().st_size if policy_state.is_file() else None
        )
        report["controller_policy_state_sha256_after"] = (
            _sha256(controller_policy_state)
            if controller_policy_state.is_file()
            else None
        )
        report["immutable_policy_state_equal"] = (
            args.immutable_policy
            and report.get("policy_state_sha256_after")
            == policy_state_sha256_before
            and report.get("controller_policy_state_sha256_after")
            == controller_policy_state_sha256_before
        )
        if score.is_file():
            report["score_sha256_after"] = _sha256(score)
        report["leftover_prefix_processes"] = _prefix_processes(prefix)
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(report, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
