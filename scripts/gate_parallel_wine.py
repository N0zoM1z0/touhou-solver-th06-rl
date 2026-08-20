#!/usr/bin/env python3
"""Require exact fixed-seed serial/two-worker original-Wine compatibility."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.corpus import FRAME_SCHEMA  # noqa: E402
from th06_rl.corpus_digest import normalized_factual_digest  # noqa: E402
from th06_rl.wine_workers import (  # noqa: E402
    RETAIL_EXECUTABLE_SHA256,
    validate_wine_worker,
    validate_worker_specifications,
)


POOL_SCHEMA = "th06-rl-normal-speed-wine-pool-v1"
GATE_SCHEMA = "th06-rl-exact-parallel-wine-gate-v1"
_CLEAN_FIELDS = (
    "background_reactivations",
    "capture_failures",
    "corpus_failures",
    "infrastructure_failures",
    "trace_failures",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _repository_commit() -> str:
    if subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPOSITORY,
        check=True, capture_output=True, text=True,
    ).stdout:
        raise RuntimeError("parallel Wine gate requires a clean committed tree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY,
        check=True, capture_output=True, text=True,
    ).stdout.strip()


def _pool(path: Path) -> dict[str, Any]:
    pool = _object(path.resolve())
    workers = pool.get("workers")
    if pool.get("schema") != POOL_SCHEMA or not isinstance(workers, list):
        raise ValueError("unsupported Wine worker pool")
    specifications = [{
        key: row.get(key) for key in (
            "worker", "directory", "display", "game_cpu_list",
            "controller_cpu_list",
        )
    } for row in workers if isinstance(row, dict)]
    validate_worker_specifications(specifications)
    if len(workers) != 2 or len(specifications) != 2:
        raise ValueError("parallel compatibility gate requires exactly two workers")
    validated = [validate_wine_worker(row) for row in workers]
    game_dirs = {str(row["game_dir"]) for row in validated}
    prefixes = {str(row["wine_prefix"]) for row in validated}
    sources = {str(row["source_game_dir"]) for row in validated}
    if len(game_dirs) != 2 or len(prefixes) != 2 or len(sources) != 1:
        raise ValueError("Wine worker filesystem ownership overlaps or differs")
    return pool


def build_runner_command(
    *, worker: dict[str, Any], score_template: Path, policy_plugin: Path,
    policy_state: Path, stage: int, rng_seed: int | None, artifact_dir: Path,
    corpus_root: Path,
) -> list[str]:
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/run_wine_retail.py"),
        "--game-dir", str(worker["game_dir"]),
        "--wine-prefix", str(worker["wine_prefix"]),
        "--score-template", str(score_template),
        "--practice-stage", str(stage),
        "--difficulty", "lunatic",
        "--seconds", "0",
        "--complete-stage-training-corpus-root", str(corpus_root),
        "--policy-plugin", str(policy_plugin),
        "--policy-state", str(policy_state),
        "--immutable-policy",
        "--display", str(worker["display"]),
        "--game-cpu-list", str(worker["game_cpu_list"]),
        "--controller-cpu-list", str(worker["controller_cpu_list"]),
        "--artifact-dir", str(artifact_dir),
    ]
    if rng_seed is not None:
        command.extend(("--diagnostic-rng-seed", hex(rng_seed)))
    return command


def run_batch(commands: list[tuple[str, list[str], Path]]) -> dict[str, int]:
    """Run one batch; any failure interrupts and reaps every sibling."""
    processes: dict[str, subprocess.Popen[bytes]] = {}
    logs = {}
    try:
        for name, command, log_path in commands:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log = log_path.open("wb")
            logs[name] = log
            processes[name] = subprocess.Popen(
                command,
                cwd=REPOSITORY,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        while True:
            statuses = {name: process.poll() for name, process in processes.items()}
            failed = [name for name, status in statuses.items() if status not in (None, 0)]
            if failed:
                for name, process in processes.items():
                    if process.poll() is None:
                        try:
                            os.killpg(process.pid, signal.SIGINT)
                        except ProcessLookupError:
                            pass
                break
            if all(status == 0 for status in statuses.values()):
                break
            time.sleep(0.2)
        deadline = time.monotonic() + 30.0
        for process in processes.values():
            remaining = max(0.1, deadline - time.monotonic())
            try:
                process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        result = {name: int(process.returncode) for name, process in processes.items()}
        if any(result.values()):
            raise RuntimeError(f"Wine worker batch failed: {result}")
        return result
    finally:
        live = [process for process in processes.values() if process.poll() is None]
        for process in live:
            try:
                os.killpg(process.pid, signal.SIGINT)
            except ProcessLookupError:
                pass
        deadline = time.monotonic() + 30.0
        for process in live:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        for log in logs.values():
            log.close()


def _run_dir(corpus_root: Path) -> Path:
    manifests = sorted(corpus_root.glob("*/manifest.json"))
    if len(manifests) != 1:
        raise ValueError(
            f"expected one corpus run below {corpus_root}, found {len(manifests)}"
        )
    return manifests[0].parent


def validate_gate_run(
    *, report: dict[str, Any], run: dict[str, Any],
    manifest: dict[str, Any], audit: dict[str, Any], stage: int,
) -> dict[str, object]:
    completion = report.get("controller_completion")
    outcome = manifest.get("run_outcome")
    planner = (run.get("metadata") or {}).get("planner")
    schemas = run.get("schemas")
    successor = audit.get("source_successor_coverage")
    numeric_successor = audit.get("source_numeric_successor_parity")
    player_successor = audit.get("player_successor_parity")
    parity = audit.get("dense_hard_parity")
    latency = audit.get("latency")
    if not all(isinstance(value, dict) for value in (
        completion, outcome, planner, schemas, successor, numeric_successor,
        player_successor, parity, latency,
    )):
        raise ValueError("parallel gate run is missing structured evidence")
    checks = {
        "runner": report.get("controller_returncode") == 0,
        "completion": (
            completion.get("practice_stage_completed") is True
            and completion.get("practice_stage") == stage
        ),
        "retail": report.get("retail_sha256") == RETAIL_EXECUTABLE_SHA256,
        "cleanup": report.get("leftover_prefix_processes") == [],
        "manifest": (
            manifest.get("complete") is True
            and manifest.get("stage_trajectory_complete") is True
            and manifest.get("dropped_records") == 0
            and outcome.get("stage_completed") is True
            and outcome.get("controller_exit_code") == 0
            and outcome.get("corpus_failure") is None
            and all(outcome.get(field) == 0 for field in _CLEAN_FIELDS)
        ),
        "hit_conservation": (
            isinstance(outcome.get("physical_hits"), int)
            and outcome.get("physical_hits") == completion.get("physical_hits")
            and outcome.get("physical_hits") == audit.get("physical_hits")
        ),
        "source_contract": (
            schemas.get("frame") == FRAME_SCHEMA
            and planner.get("source_commitment") == "source-complete-hard-v1"
            and planner.get("publication_epoch")
            == "source-root-process-suspended-v1"
            and planner.get("algorithm")
            == "source-hard4-paused-publication-v2"
            and planner.get("hard_horizon") == 4
            and planner.get("learner_feature_horizon") == 4
            and planner.get("minimum_collision_margin") == 0.35
            and planner.get("zero_margin_fallback") is False
        ),
        "audit": (
            audit.get("integrity_errors") == []
            and audit.get("bomb_events") == 0
        ),
        "successor": (
            successor.get("method") == "retained-next-root-one-sided-coverage-v1"
            and successor.get("checked_links", 0) > 0
            and successor.get("actual_lasers_checked", 0) > 0
            and successor.get("uncovered_aabbs") == 0
            and successor.get("uncovered_lasers") == 0
            and (successor.get(
                "retained_laser_geometry_unavailable", {}
            ) or {}).get("invalid-state", 0) == 0
        ),
        "numeric_successor": (
            numeric_successor.get("method")
            == "stable-retained-bullet-center-successor-v2"
            and numeric_successor.get("arithmetic_comparison")
            == "float32-bit-exact"
            and numeric_successor.get("required_collision_margin") == 0.35
            and numeric_successor.get("transcendental_axis_error_budget", 1.0)
            < 0.35
            and 0
            < numeric_successor.get(
                "global_release_acceleration_axis_bound", 1.0
            )
            < 0.35
            and numeric_successor.get("global_mutation_semantics")
            == "source branch union"
            and (
                numeric_successor.get("linear_exact_checked", 0)
                + numeric_successor.get("acceleration_exact_checked", 0)
            ) > 0
            and numeric_successor.get("transcendental_checked", 0) > 0
            and (
                numeric_successor.get("global_stop_union_checked", 0)
                + numeric_successor.get("global_release_union_checked", 0)
                + numeric_successor.get("global_combined_union_checked", 0)
            ) > 0
            and numeric_successor.get("exact_mismatches") == 0
            and numeric_successor.get("transcendental_budget_violations") == 0
            and numeric_successor.get("nonfinite_successors") == 0
            and numeric_successor.get("global_mutation_union_violations") == 0
        ),
        "player_successor": (
            player_successor.get("method")
            == "contiguous-active-player-center-successor-v1"
            and player_successor.get("arithmetic_comparison")
            == "float32-bit-exact"
            and player_successor.get("input_semantics")
            == "next-completed-root-sampled-input"
            and player_successor.get("movement_order")
            == "Player-before-Enemy-before-Bullet"
            and player_successor.get("checked_links", 0) > 0
            and player_successor.get("mismatches") == 0
        ),
        "native_parity": (
            parity.get("checked", 0) > 0
            and parity.get("unsafe_divergences") == []
            and parity.get("conservative_divergences") == []
        ),
        "online_latency": (
            (latency.get("capture") or {}).get("p99_ms", float("inf")) <= 40.0
            and (latency.get("solve") or {}).get("p99_ms", float("inf")) <= 16.7
            and latency.get("observation_gap_rate", 1.0) <= 0.005
            and latency.get("stale_retry_rate", 1.0) == 0.0
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"parallel gate run failed: {','.join(failed)}")
    return {
        "physical_hits": outcome["physical_hits"],
        "retail_sha256": report["retail_sha256"],
        "native_sha256": report.get("native_sha256"),
        "policy_plugin_sha256": report.get("policy_plugin_sha256"),
        "policy_state_sha256": report.get("policy_state_sha256_before"),
        "latency": latency,
        "checks": checks,
    }


def _audit(run_dir: Path, artifact_dir: Path) -> dict[str, Any]:
    output = artifact_dir / "infra-audit.json"
    subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/audit_run.py"),
            str(run_dir),
            "--native-library",
            str(REPOSITORY / "build/native/libth06_rl_native.so"),
            "--output", str(output),
        ],
        cwd=REPOSITORY,
        check=True,
    )
    return _object(output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pool", type=Path,
        default=REPOSITORY / "reference/wine-workers-v2/pool.json",
    )
    parser.add_argument("--policy-plugin", type=Path, required=True)
    parser.add_argument("--policy-state", type=Path, required=True)
    parser.add_argument("--practice-stage", type=int, choices=range(1, 7), default=4)
    parser.add_argument(
        "--diagnostic-rng-seed", type=lambda value: int(value, 0),
        choices=range(0x10000), required=True,
    )
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> dict[str, object]:
    commit = _repository_commit()
    pool_path = args.pool.resolve()
    pool = _pool(pool_path)
    workers = sorted(pool["workers"], key=lambda row: int(row["worker"]))
    score_template = Path(pool["score_template"]).resolve()
    policy_plugin = args.policy_plugin.resolve()
    policy_state = args.policy_state.resolve()
    artifact_root = args.artifact_root.resolve()
    corpus_root = args.corpus_root.resolve()
    output = args.output.resolve()
    for path in (artifact_root, corpus_root, output):
        if path.exists():
            raise FileExistsError(path)

    rows = [
        ("serial-worker-0", workers[0]),
        ("concurrent-worker-0", workers[0]),
        ("concurrent-worker-1", workers[1]),
    ]
    commands = {}
    for name, worker in rows:
        commands[name] = build_runner_command(
            worker=worker,
            score_template=score_template,
            policy_plugin=policy_plugin,
            policy_state=policy_state,
            stage=args.practice_stage,
            rng_seed=args.diagnostic_rng_seed,
            artifact_dir=artifact_root / name,
            corpus_root=corpus_root / name,
        )
    run_batch([(
        "serial-worker-0", commands["serial-worker-0"],
        artifact_root / "serial-worker-0-launcher.log",
    )])
    run_batch([(
        name, commands[name], artifact_root / f"{name}-launcher.log",
    ) for name in ("concurrent-worker-0", "concurrent-worker-1")])

    evidence = {}
    for name, _worker in rows:
        run_dir = _run_dir(corpus_root / name)
        artifact_dir = artifact_root / name
        audit = _audit(run_dir, artifact_dir)
        validated = validate_gate_run(
            report=_object(artifact_dir / "report.json"),
            run=_object(run_dir / "run.json"),
            manifest=_object(run_dir / "manifest.json"),
            audit=audit,
            stage=args.practice_stage,
        )
        evidence[name] = {
            **validated,
            "run_dir": str(run_dir),
            "digest": normalized_factual_digest(run_dir),
        }
    reference = evidence["serial-worker-0"]
    for name in ("concurrent-worker-0", "concurrent-worker-1"):
        candidate = evidence[name]
        if (
            candidate["physical_hits"] != reference["physical_hits"]
            or candidate["digest"] != reference["digest"]
            or any(
                candidate[key] != reference[key]
                for key in (
                    "retail_sha256", "native_sha256", "policy_plugin_sha256",
                    "policy_state_sha256",
                )
            )
        ):
            raise RuntimeError(f"parallel Wine factual differential failed: {name}")
    result = {
        "schema": GATE_SCHEMA,
        "passed": True,
        "repository_commit": commit,
        "pool_sha256": _sha256(pool_path),
        "policy_plugin_sha256": _sha256(policy_plugin),
        "policy_state_sha256": _sha256(policy_state),
        "native_sha256": _sha256(
            REPOSITORY / "build/native-win32-fully-static/libth06_rl_native.dll"
        ),
        "practice_stage": args.practice_stage,
        "diagnostic_rng_seed": args.diagnostic_rng_seed,
        "game_clock": "original-retail-normal-speed",
        "evidence": evidence,
    }
    _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
