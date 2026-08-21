#!/usr/bin/env python3
"""Collect a predeclared natural-RNG corpus after the exact parallel gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.gate_parallel_wine import (  # noqa: E402
    GATE_SCHEMA,
    _atomic_json,
    _audit,
    _object,
    _pool,
    _repository_commit,
    _run_dir,
    _sha256,
    build_runner_command,
    run_batch,
    validate_gate_run,
)
from scripts.prepare_wine_workers import (  # noqa: E402
    GIB,
    linux_mem_available_bytes,
    require_host_capacity,
)
from th06_rl.corpus_digest import normalized_factual_digest  # noqa: E402


COLLECTION_SCHEMA = "th06-rl-observed-shield-parallel-collection-v1"
SCHEDULE_SCHEMA = "th06-rl-parallel-collection-schedule-v1"


def _stages(value: str) -> tuple[int, ...]:
    try:
        stages = tuple(int(item) for item in value.split(","))
    except ValueError as error:
        raise argparse.ArgumentTypeError("stages must be comma-separated integers") from error
    if not stages or any(stage not in range(1, 7) for stage in stages):
        raise argparse.ArgumentTypeError("stages must be in 1..6")
    return stages


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate", type=Path, required=True)
    parser.add_argument(
        "--pool", type=Path,
        default=REPOSITORY / "reference/wine-workers-v2/pool.json",
    )
    parser.add_argument("--policy-plugin", type=Path, required=True)
    parser.add_argument("--policy-state", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--stages", type=_stages, default=tuple(range(1, 7)))
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes < 1:
        parser.error("--episodes must be positive")
    return args


def _validate_gate_binding(
    *, gate_path: Path, pool_path: Path, policy_plugin: Path,
    policy_state: Path, commit: str,
) -> dict[str, Any]:
    gate = _object(gate_path)
    native = REPOSITORY / "build/native-win32-fully-static/libth06_rl_native.dll"
    checks = {
        "schema": gate.get("schema") == GATE_SCHEMA,
        "passed": gate.get("passed") is True,
        "clock": gate.get("game_clock") == "original-retail-normal-speed",
        "commit": gate.get("repository_commit") == commit,
        "pool": gate.get("pool_sha256") == _sha256(pool_path),
        "policy_plugin": gate.get("policy_plugin_sha256") == _sha256(policy_plugin),
        "policy_state": gate.get("policy_state_sha256") == _sha256(policy_state),
        "native": gate.get("native_sha256") == _sha256(native),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"parallel Wine gate binding differs: {','.join(failed)}")
    return gate


def _schedule(
    *, episodes: int, stages: tuple[int, ...], workers: list[dict[str, Any]],
    gate_sha256: str, commit: str,
) -> dict[str, object]:
    rows = []
    for index in range(episodes):
        worker = workers[index % len(workers)]
        stage = stages[index % len(stages)]
        rows.append({
            "episode": index,
            "worker": int(worker["worker"]),
            "stage": stage,
            "name": f"episode-{index:06d}-worker-{worker['worker']}-stage-{stage}",
        })
    return {
        "schema": SCHEDULE_SCHEMA,
        "repository_commit": commit,
        "gate_sha256": gate_sha256,
        "natural_rng": True,
        "game_clock": "original-retail-normal-speed",
        "episodes": rows,
    }


def _episode_evidence(
    *, row: dict[str, Any], artifact_root: Path, corpus_root: Path,
) -> dict[str, object]:
    name = str(row["name"])
    run_dir = _run_dir(corpus_root / name)
    artifact_dir = artifact_root / name
    audit = _audit(run_dir, artifact_dir)
    validated = validate_gate_run(
        report=_object(artifact_dir / "report.json"),
        run=_object(run_dir / "run.json"),
        manifest=_object(run_dir / "manifest.json"),
        audit=audit,
        stage=int(row["stage"]),
    )
    return {
        **row,
        **validated,
        "run": str(run_dir.relative_to(corpus_root)),
        "run_sha256": _sha256(run_dir / "run.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "audit_sha256": _sha256(artifact_dir / "infra-audit.json"),
        "digest": normalized_factual_digest(run_dir),
    }


def run(args: argparse.Namespace) -> dict[str, object]:
    commit = _repository_commit()
    gate_path = args.gate.resolve()
    pool_path = args.pool.resolve()
    policy_plugin = args.policy_plugin.resolve()
    policy_state = args.policy_state.resolve()
    _validate_gate_binding(
        gate_path=gate_path,
        pool_path=pool_path,
        policy_plugin=policy_plugin,
        policy_state=policy_state,
        commit=commit,
    )
    pool = _pool(pool_path)
    workers = sorted(pool["workers"], key=lambda row: int(row["worker"]))
    artifact_root = args.artifact_root.resolve()
    corpus_root = args.corpus_root.resolve()
    output = args.output.resolve()
    score_template = Path(pool["score_template"]).resolve()
    schedule = _schedule(
        episodes=args.episodes,
        stages=args.stages,
        workers=workers,
        gate_sha256=_sha256(gate_path),
        commit=commit,
    )
    schedule_path = artifact_root / "schedule.json"
    if schedule_path.is_file():
        if _object(schedule_path) != schedule:
            raise ValueError("parallel collection resume schedule differs")
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("parallel collection root lacks its immutable schedule")
        _atomic_json(schedule_path, schedule)
    if output.is_file():
        result = _object(output)
        if result.get("schedule_sha256") != _sha256(schedule_path):
            raise ValueError("completed parallel collection schedule differs")
        return result

    corpus_root.parent.mkdir(parents=True, exist_ok=True)
    rows = schedule["episodes"]
    evidence: dict[int, dict[str, object]] = {}
    for row in rows:
        name = str(row["name"])
        artifact_dir = artifact_root / name
        episode_corpus = corpus_root / name
        if artifact_dir.exists() or episode_corpus.exists():
            if not (
                (artifact_dir / "report.json").is_file()
                and len(list(episode_corpus.glob("*/manifest.json"))) == 1
            ):
                raise ValueError(f"partial scheduled episode requires triage: {name}")
            evidence[int(row["episode"])] = _episode_evidence(
                row=row, artifact_root=artifact_root, corpus_root=corpus_root,
            )

    for wave_start in range(0, len(rows), len(workers)):
        wave = [
            row for row in rows[wave_start : wave_start + len(workers)]
            if int(row["episode"]) not in evidence
        ]
        if not wave:
            continue
        resource = pool["resource_contract"]
        require_host_capacity(
            workers=len(wave),
            memory_available=linux_mem_available_bytes(),
            disk_available=shutil.disk_usage(corpus_root.parent).free,
            memory_per_worker=int(resource["minimum_memory_gib_per_worker"] * GIB),
            disk_per_worker=int(resource["minimum_disk_gib_per_worker"] * GIB),
        )
        commands = []
        for row in wave:
            worker = next(
                item for item in workers if int(item["worker"]) == int(row["worker"])
            )
            name = str(row["name"])
            command = build_runner_command(
                worker=worker,
                score_template=score_template,
                policy_plugin=policy_plugin,
                policy_state=policy_state,
                stage=int(row["stage"]),
                rng_seed=None,
                artifact_dir=artifact_root / name,
                corpus_root=corpus_root / name,
            )
            commands.append((name, command, artifact_root / f"{name}-launcher.log"))
        run_batch(commands)
        for row in wave:
            evidence[int(row["episode"])] = _episode_evidence(
                row=row, artifact_root=artifact_root, corpus_root=corpus_root,
            )

    if len(evidence) != len(rows):
        raise RuntimeError("parallel collection did not validate its full schedule")
    result = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "repository_commit": commit,
        "schedule_sha256": _sha256(schedule_path),
        "gate_sha256": _sha256(gate_path),
        "pool_sha256": _sha256(pool_path),
        "policy_plugin_sha256": _sha256(policy_plugin),
        "policy_state_sha256": _sha256(policy_state),
        "game_clock": "original-retail-normal-speed",
        "natural_rng": True,
        "episodes": [evidence[index] for index in sorted(evidence)],
    }
    _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
