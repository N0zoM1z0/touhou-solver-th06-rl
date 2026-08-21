#!/usr/bin/env python3
"""Collect complete natural-RNG routes on gated isolated Wine workers."""

from __future__ import annotations

import argparse
import hashlib
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
    _atomic_json,
    _audit,
    _object,
    _pool,
    _repository_commit,
    _run_dir,
    _sha256,
    run_batch,
)
from scripts.collect_wine_parallel import _validate_gate_binding  # noqa: E402
from scripts.prepare_wine_workers import (  # noqa: E402
    GIB,
    linux_mem_available_bytes,
    require_host_capacity,
)
from scripts.verify_baseline_route import verify as verify_route  # noqa: E402
from th06_rl.corpus_digest import normalized_factual_digest  # noqa: E402
from th06_rl.policies.uniform_shield_exploration import STATE_SCHEMA  # noqa: E402


COLLECTION_SCHEMA = "th06-rl-observed-shield-parallel-route-collection-v1"
SCHEDULE_SCHEMA = "th06-rl-parallel-route-schedule-v1"
MAX_CORPUS_GIB = 4


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY))
    except ValueError as error:
        raise ValueError(f"collection path must be inside the repository: {path}") from error


def _canonical(value: object) -> bytes:
    return (json.dumps(
        value,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ) + "\n").encode("utf-8")


def derive_episode_policy_state(
    base_state: dict[str, object],
    *,
    episode: int,
) -> dict[str, object]:
    """Derive independent, replayable exploration RNG streams per episode."""
    if base_state.get("schema") != STATE_SCHEMA:
        raise ValueError("route collection requires uniform-shield-exploration-v1")
    base_seed = int(base_state.get("policy_seed", -1))
    if not 0 <= base_seed < 2**64 or episode < 0:
        raise ValueError("base policy seed and episode index are invalid")
    digest = hashlib.sha256(
        f"th06-rl:route:{base_seed:016x}:{episode:016x}".encode("ascii")
    ).digest()
    result = dict(base_state)
    result["policy_seed"] = int.from_bytes(digest[:8], "big")
    return result


def _schedule(
    *,
    episodes: int,
    workers: list[dict[str, Any]],
    base_state: dict[str, object],
    gate_sha256: str,
    commit: str,
) -> dict[str, object]:
    rows = []
    seeds = set()
    for index in range(episodes):
        worker = workers[index % len(workers)]
        state = derive_episode_policy_state(base_state, episode=index)
        seed = int(state["policy_seed"])
        if seed in seeds:
            raise RuntimeError("derived episode policy seeds collided")
        seeds.add(seed)
        name = f"route-{index:06d}-worker-{worker['worker']}"
        rows.append({
            "episode": index,
            "worker": int(worker["worker"]),
            "name": name,
            "policy_seed": seed,
            "policy_state_path": f"policy-states/{name}.json",
            "policy_state_sha256": hashlib.sha256(_canonical(state)).hexdigest(),
        })
    return {
        "schema": SCHEDULE_SCHEMA,
        "repository_commit": commit,
        "gate_sha256": gate_sha256,
        "natural_rng": True,
        "game_clock": "original-retail-normal-speed",
        "episode_unit": "complete-route",
        "episodes": rows,
    }


def build_route_runner_command(
    *,
    worker: dict[str, Any],
    score_template: Path,
    policy_plugin: Path,
    policy_state: Path,
    artifact_dir: Path,
    corpus_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(REPOSITORY / "scripts/run_wine_retail.py"),
        "--game-dir", str(worker["game_dir"]),
        "--wine-prefix", str(worker["wine_prefix"]),
        "--score-template", str(score_template),
        "--start-route",
        "--difficulty", "lunatic",
        "--seconds", "0",
        "--complete-route-corpus-root", str(corpus_root),
        "--policy-plugin", str(policy_plugin),
        "--policy-state", str(policy_state),
        "--immutable-policy",
        "--display", str(worker["display"]),
        "--game-cpu-list", str(worker["game_cpu_list"]),
        "--controller-cpu-list", str(worker["controller_cpu_list"]),
        "--artifact-dir", str(artifact_dir),
    ]


def _episode_evidence(
    *,
    row: dict[str, Any],
    artifact_root: Path,
    corpus_root: Path,
) -> dict[str, object]:
    name = str(row["name"])
    run_dir = _run_dir(corpus_root / name)
    artifact_dir = artifact_root / name
    audit = _audit(run_dir, artifact_dir)
    report = _object(artifact_dir / "report.json")
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    verification = verify_route(report, run, manifest, audit)
    return {
        **row,
        "verification": verification,
        "run_dir": _relative(run_dir),
        "report_path": _relative(artifact_dir / "report.json"),
        "audit_path": _relative(artifact_dir / "infra-audit.json"),
        "run_sha256": _sha256(run_dir / "run.json"),
        "manifest_sha256": _sha256(run_dir / "manifest.json"),
        "report_sha256": _sha256(artifact_dir / "report.json"),
        "audit_sha256": _sha256(artifact_dir / "infra-audit.json"),
        "digest": normalized_factual_digest(run_dir),
    }


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
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.episodes < 2:
        parser.error("complete-route offline collection requires at least two episodes")
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    commit = _repository_commit()
    gate_path = args.gate.resolve()
    pool_path = args.pool.resolve()
    policy_plugin = args.policy_plugin.resolve()
    base_state_path = args.policy_state.resolve()
    base_state = _object(base_state_path)
    _validate_gate_binding(
        gate_path=gate_path,
        pool_path=pool_path,
        policy_plugin=policy_plugin,
        policy_state=base_state_path,
        commit=commit,
    )
    pool = _pool(pool_path)
    workers = sorted(pool["workers"], key=lambda row: int(row["worker"]))
    artifact_root = args.artifact_root.resolve()
    corpus_root = args.corpus_root.resolve()
    output = args.output.resolve()
    for path in (artifact_root, corpus_root, output):
        _relative(path)
    schedule = _schedule(
        episodes=args.episodes,
        workers=workers,
        base_state=base_state,
        gate_sha256=_sha256(gate_path),
        commit=commit,
    )
    schedule_path = artifact_root / "schedule.json"
    if schedule_path.is_file():
        if _object(schedule_path) != schedule:
            raise ValueError("parallel route resume schedule differs")
    else:
        if artifact_root.exists() and any(artifact_root.iterdir()):
            raise ValueError("parallel route root lacks its immutable schedule")
        _atomic_json(schedule_path, schedule)
    if output.is_file():
        result = _object(output)
        if (
            result.get("schema") != COLLECTION_SCHEMA
            or result.get("complete") is not True
            or result.get("schedule_sha256") != _sha256(schedule_path)
        ):
            raise ValueError("completed route collection schedule differs")
        return result

    for row in schedule["episodes"]:
        state = derive_episode_policy_state(base_state, episode=int(row["episode"]))
        state_path = artifact_root / str(row["policy_state_path"])
        if state_path.is_file():
            if _object(state_path) != state:
                raise ValueError(f"episode policy state differs: {state_path}")
        else:
            _atomic_json(state_path, state)
        if _sha256(state_path) != row["policy_state_sha256"]:
            raise ValueError(f"episode policy state hash differs: {state_path}")

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
                raise ValueError(f"partial scheduled route requires triage: {name}")
            evidence[int(row["episode"])] = _episode_evidence(
                row=row,
                artifact_root=artifact_root,
                corpus_root=corpus_root,
            )

    score_template = Path(pool["score_template"]).resolve()
    for wave_start in range(0, len(rows), len(workers)):
        wave = [
            row for row in rows[wave_start:wave_start + len(workers)]
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
            disk_per_worker=max(
                int(resource["minimum_disk_gib_per_worker"] * GIB),
                (MAX_CORPUS_GIB + 1) * GIB,
            ),
        )
        commands = []
        for row in wave:
            worker = next(
                item for item in workers
                if int(item["worker"]) == int(row["worker"])
            )
            name = str(row["name"])
            commands.append((
                name,
                build_route_runner_command(
                    worker=worker,
                    score_template=score_template,
                    policy_plugin=policy_plugin,
                    policy_state=artifact_root / str(row["policy_state_path"]),
                    artifact_dir=artifact_root / name,
                    corpus_root=corpus_root / name,
                ),
                artifact_root / f"{name}-launcher.log",
            ))
        run_batch(commands)
        for row in wave:
            evidence[int(row["episode"])] = _episode_evidence(
                row=row,
                artifact_root=artifact_root,
                corpus_root=corpus_root,
            )

    if len(evidence) != len(rows):
        raise RuntimeError("parallel route collection lacks scheduled evidence")
    result = {
        "schema": COLLECTION_SCHEMA,
        "complete": True,
        "repository_commit": commit,
        "schedule_path": _relative(schedule_path),
        "schedule_sha256": _sha256(schedule_path),
        "gate_sha256": _sha256(gate_path),
        "pool_sha256": _sha256(pool_path),
        "policy_plugin_sha256": _sha256(policy_plugin),
        "base_policy_state_sha256": _sha256(base_state_path),
        "game_clock": "original-retail-normal-speed",
        "natural_rng": True,
        "episode_unit": "complete-route",
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
