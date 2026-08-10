#!/usr/bin/env python3
"""Run an exact-model, exact-seed headless HIT-continuation panel.

This is benchmark orchestration only.  Every child keeps the native gate and
fresh issue check in authority, disables the artificial tick limit, forbids
Bomb through the underlying collector, and runs until the game reports a
natural Practice terminal.  The resulting continuation streams are never
training eligible.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence

try:
    from compare_headless_paired_panel import compare
except ModuleNotFoundError:  # Imported as scripts.run_headless_paired_continuation.
    from scripts.compare_headless_paired_panel import compare


LABEL_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")


@dataclass(frozen=True)
class Candidate:
    label: str
    model: Path


def parse_candidate(value: str) -> Candidate:
    label, separator, raw_model = value.partition("=")
    if not separator or not LABEL_RE.fullmatch(label) or not raw_model:
        raise argparse.ArgumentTypeError(
            "candidate must be a filesystem-safe LABEL=MODEL"
        )
    return Candidate(label=label, model=Path(raw_model))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def child_command(
    *,
    root: Path,
    candidate: Candidate,
    seed: int,
    stage: int,
    difficulty: int,
    character: int,
    shot_type: int,
    threads: int,
    anchor_stride: int,
    teacher_horizon: int,
    binary: Path,
    game_directory: Path,
    output_root: Path,
) -> list[str]:
    return [
        sys.executable,
        str(root / "scripts/collect_headless_dagger.py"),
        "--model",
        str(candidate.model),
        "--seed",
        str(seed),
        "--max-ticks",
        "0",
        "--anchor-stride",
        str(anchor_stride),
        "--teacher-horizon",
        str(teacher_horizon),
        "--threads",
        str(threads),
        "--continue-after-hit",
        "--difficulty",
        str(difficulty),
        "--character",
        str(character),
        "--shot-type",
        str(shot_type),
        "--stage",
        str(stage),
        "--binary",
        str(binary),
        "--game-directory",
        str(game_directory),
        "--output-root",
        str(output_root / candidate.label),
    ]


def _run_child(
    command: Sequence[str],
    *,
    log_path: Path,
    cwd: Path,
    env: dict[str, str],
) -> dict[str, Any]:
    with log_path.open("wb") as log:
        result = subprocess.run(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            check=False,
        )
    return {
        "command": list(command),
        "log": str(log_path),
        "returncode": result.returncode,
    }


def _unique(values: Iterable[int]) -> tuple[int, ...]:
    result = tuple(dict.fromkeys(values))
    if len(result) < 2:
        raise ValueError("paired continuation requires at least two unique seeds")
    return result


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate",
        action="append",
        required=True,
        type=parse_candidate,
        help="filesystem-safe LABEL=MODEL; repeat for every policy",
    )
    parser.add_argument("--seed", action="append", required=True, type=int)
    parser.add_argument("--stage", required=True, type=int, choices=range(1, 7))
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--anchor-stride", type=int, default=4096)
    parser.add_argument("--teacher-horizon", type=int, default=12)
    parser.add_argument(
        "--binary",
        type=Path,
        default=root / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument(
        "--game-directory",
        type=Path,
        default=root / "reference/th06-game-original/th06",
    )
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()

    candidates: list[Candidate] = args.candidate
    if len(candidates) < 2:
        parser.error("paired continuation requires at least two candidates")
    labels = [candidate.label for candidate in candidates]
    if len(labels) != len(set(labels)):
        parser.error("candidate labels must be unique")
    try:
        seeds = _unique(args.seed)
    except ValueError as error:
        parser.error(str(error))
    if not 1 <= args.workers <= 12 or not 1 <= args.threads <= 12:
        parser.error("workers and threads must each be in 1..12 on the shared VPS")
    if args.workers * args.threads > 24:
        parser.error("workers * threads must not exceed the shared-VPS ceiling of 24")
    if min(args.anchor_stride, args.teacher_horizon) <= 0:
        parser.error("anchor stride and teacher horizon must be positive")

    binary = args.binary.resolve()
    game_directory = args.game_directory.resolve()
    output_root = args.output_root.resolve()
    if not binary.is_file():
        parser.error(f"headless binary does not exist: {binary}")
    if not game_directory.is_dir():
        parser.error(f"game directory does not exist: {game_directory}")
    resolved: list[Candidate] = []
    for candidate in candidates:
        model = candidate.model.resolve()
        if not model.is_file():
            parser.error(f"candidate model does not exist: {model}")
        resolved.append(Candidate(candidate.label, model))
    candidates = resolved

    if output_root.exists() and any(output_root.iterdir()):
        parser.error("output root must be absent or empty")
    log_root = output_root / "logs"
    log_root.mkdir(parents=True, exist_ok=True)
    for candidate in candidates:
        (output_root / candidate.label).mkdir(parents=True, exist_ok=True)

    jobs = [
        (candidate, seed)
        for candidate in candidates
        for seed in seeds
    ]
    commands = {
        (candidate.label, seed): child_command(
            root=root,
            candidate=candidate,
            seed=seed,
            stage=args.stage,
            difficulty=args.difficulty,
            character=args.character,
            shot_type=args.shot_type,
            threads=args.threads,
            anchor_stride=args.anchor_stride,
            teacher_horizon=args.teacher_horizon,
            binary=binary,
            game_directory=game_directory,
            output_root=output_root,
        )
        for candidate, seed in jobs
    }
    plan = {
        "schema": "th06-rl-headless-paired-continuation-plan-v1",
        "scope": {
            "difficulty": args.difficulty,
            "character": args.character,
            "shot_type": args.shot_type,
            "stage": args.stage,
        },
        "seeds": list(seeds),
        "workers": args.workers,
        "threads_per_worker": args.threads,
        "max_ticks": 0,
        "continue_after_hit": True,
        "candidates": [
            {
                "label": candidate.label,
                "model": str(candidate.model),
                "model_sha256": _sha256(candidate.model),
            }
            for candidate in candidates
        ],
    }
    (output_root / "run-plan.json").write_text(
        json.dumps(plan, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [str(root), str(root / "src")]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _run_child,
                commands[(candidate.label, seed)],
                log_path=log_root / f"{candidate.label}-seed{seed}.log",
                cwd=root,
                env=env,
            ): (candidate.label, seed)
            for candidate, seed in jobs
        }
        for future in as_completed(futures):
            label, seed = futures[future]
            row = future.result()
            row.update({"label": label, "seed": seed})
            results.append(row)
    results.sort(key=lambda row: (row["label"], row["seed"]))
    failures = [row for row in results if row["returncode"] != 0]
    (output_root / "child-results.json").write_text(
        json.dumps(results, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if failures:
        print(json.dumps({"failures": failures}, indent=2), file=sys.stderr)
        return 1

    panel = compare(
        (candidate.label, output_root / candidate.label)
        for candidate in candidates
    )
    panel_path = output_root / "paired-panel.json"
    panel_path.write_text(
        json.dumps(panel, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "paired_panel": str(panel_path),
        "candidates": [
            {
                "label": row["label"],
                "physical_hits": row["physical_hits"],
                "benchmark_forced_rows": row["benchmark_forced_rows"],
                "ranker_sha256": row["ranker_sha256"],
            }
            for row in panel["candidates"]
        ],
        "seedwise_dominators": panel["seedwise_dominators"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
