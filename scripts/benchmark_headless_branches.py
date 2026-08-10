#!/usr/bin/env python3
"""Benchmark exact TH06 counterfactual branches from one replay checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import tempfile
import time
from typing import Iterable

from th06_rl.headless import HeadlessScope
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.native import ACTIONS


ACTION_NAMES = tuple(action.name for action in ACTIONS)


def _normalized_prefix(path: Path, required_actions: int) -> tuple[tuple[int, str], ...]:
    """Read the runtime RLE syntax and retain exactly one checkpoint prefix."""
    if required_actions <= 0:
        raise ValueError("checkpoint prefix must contain at least one action")
    retained: list[tuple[int, str]] = []
    remaining = required_actions
    for line_number, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        if len(fields) == 1:
            repeats, action = 1, fields[0]
        elif len(fields) == 2:
            try:
                repeats = int(fields[0])
            except ValueError as error:
                raise ValueError(f"invalid action repeat at {path}:{line_number}") from error
            action = fields[1]
        else:
            raise ValueError(f"invalid action line at {path}:{line_number}")
        if repeats <= 0:
            raise ValueError(f"nonpositive action repeat at {path}:{line_number}")
        if action not in ACTION_NAMES:
            raise ValueError(f"unknown or forbidden action at {path}:{line_number}: {action}")
        used = min(repeats, remaining)
        retained.append((used, action))
        remaining -= used
        if remaining == 0:
            break
    if remaining:
        raise ValueError(f"action prefix is {remaining} physical actions short")
    return tuple(retained)


def _write_actions(path: Path, rows: Iterable[tuple[int, str]]) -> None:
    path.write_text(
        "".join(f"{repeats} {action}\n" for repeats, action in rows),
        encoding="utf-8",
    )


def _low_priority(command: list[str]) -> list[str]:
    nice = shutil.which("nice")
    ionice = shutil.which("ionice")
    if nice is not None:
        command = [nice, "-n", "15", *command]
    if ionice is not None:
        command = [ionice, "-c", "2", "-n", "7", *command]
    return command


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_revision(binary: Path) -> dict[str, object]:
    root = binary.resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(root), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "commit": commit,
        "clean": not dirty,
        "binary_sha256": _sha256(binary),
    }


def _fresh_command(
    *,
    binary: Path,
    scope: HeadlessScope,
    seed: int,
    terminal_tick: int,
    actions: Path,
    trace: Path,
) -> list[str]:
    return _low_priority([
        str(binary),
        "--headless",
        "--seed",
        str(seed),
        "--max-ticks",
        str(terminal_tick),
        "--practice-stage",
        str(scope.stage),
        "--difficulty",
        str(scope.difficulty),
        "--character",
        str(scope.character),
        "--shot-type",
        str(scope.shot_type),
        "--actions",
        str(actions),
        "--trace",
        str(trace),
        "--trace-final-only",
        "--auto-shoot",
    ])


def benchmark_once(
    *,
    binary: Path,
    game_directory: Path,
    scope: HeadlessScope,
    seed: int,
    checkpoint_tick: int,
    branch_frames: int,
    prefix_rows: tuple[tuple[int, str], ...],
    workspace: Path,
) -> dict[str, object]:
    terminal_tick = checkpoint_tick + branch_frames
    prefix_path = workspace / "prefix.txt"
    _write_actions(prefix_path, prefix_rows)
    branch_paths: dict[str, Path] = {}
    combined_paths: dict[str, Path] = {}
    for action in ACTION_NAMES:
        branch_paths[action] = workspace / f"branch-{action}.txt"
        combined_paths[action] = workspace / f"combined-{action}.txt"
        _write_actions(branch_paths[action], ((branch_frames, action),))
        _write_actions(
            combined_paths[action],
            (*prefix_rows, (branch_frames, action)),
        )

    nested_outputs: dict[str, Path] = {}
    started = time.perf_counter()
    server = HeadlessForkserver(
        binary=binary,
        game_directory=game_directory,
        scope=scope,
        seed=seed,
    )
    try:
        root_tick = server.start()
        if checkpoint_tick - root_tick != sum(row[0] for row in prefix_rows):
            raise ValueError("normalized prefix length does not match live root checkpoint")
        server.enter_checkpoint(
            terminal_tick=checkpoint_tick,
            actions_path=prefix_path,
        )
        for action in ACTION_NAMES:
            output = workspace / f"nested-{action}.jsonl"
            server.run(
                terminal_tick=terminal_tick,
                actions_path=branch_paths[action],
                trace_path=output,
                summary_only=True,
            )
            nested_outputs[action] = output
        server.leave_checkpoint()
    finally:
        server.close()
    nested_seconds = time.perf_counter() - started

    fresh_outputs: dict[str, Path] = {}
    started = time.perf_counter()
    for action in ACTION_NAMES:
        output = workspace / f"fresh-{action}.jsonl"
        subprocess.run(
            _fresh_command(
                binary=binary,
                scope=scope,
                seed=seed,
                terminal_tick=terminal_tick,
                actions=combined_paths[action],
                trace=output,
            ),
            cwd=game_directory,
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        fresh_outputs[action] = output
    fresh_seconds = time.perf_counter() - started

    mismatches = tuple(
        action
        for action in ACTION_NAMES
        if nested_outputs[action].read_bytes() != fresh_outputs[action].read_bytes()
    )
    outcomes = {}
    for action, path in nested_outputs.items():
        rows = path.read_text(encoding="utf-8").splitlines()
        if len(rows) != 1:
            raise ValueError(f"terminal-only branch {action} emitted {len(rows)} rows")
        observation = json.loads(rows[0])
        outcomes[action] = {
            "tick": observation.get("tick"),
            "terminal_reason": observation.get("terminal_reason"),
            "deaths": observation.get("deaths"),
            "lives": observation.get("lives"),
        }
    return {
        "nested_seconds": nested_seconds,
        "fresh_seconds": fresh_seconds,
        "speedup": fresh_seconds / nested_seconds,
        "byte_exact_matches": len(ACTION_NAMES) - len(mismatches),
        "mismatches": list(mismatches),
        "outcomes": outcomes,
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--prefix-actions", type=Path, required=True)
    parser.add_argument("--checkpoint-tick", type=int, required=True)
    parser.add_argument("--branch-frames", type=int, default=60)
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=root / "artifacts/benchmarks/headless-branches.json",
    )
    args = parser.parse_args()
    if args.checkpoint_tick <= 1:
        parser.error("checkpoint tick must follow the stage-entry checkpoint")
    if args.branch_frames <= 0 or args.repetitions <= 0:
        parser.error("branch frames and repetitions must be positive")
    binary = args.binary.resolve()
    game_directory = args.game_directory.resolve()
    if not binary.is_file():
        parser.error(f"headless binary not found: {binary}")
    if not game_directory.is_dir():
        parser.error(f"game directory not found: {game_directory}")
    scope = HeadlessScope(
        args.difficulty,
        args.character,
        args.shot_type,
        args.stage,
    )
    prefix_rows = _normalized_prefix(
        args.prefix_actions,
        args.checkpoint_tick - 1,
    )
    runs = []
    with tempfile.TemporaryDirectory(prefix="th06-branch-benchmark-") as raw:
        workspace = Path(raw)
        for repetition in range(args.repetitions):
            run = workspace / f"run-{repetition}"
            run.mkdir()
            runs.append(benchmark_once(
                binary=binary,
                game_directory=game_directory,
                scope=scope,
                seed=args.seed,
                checkpoint_tick=args.checkpoint_tick,
                branch_frames=args.branch_frames,
                prefix_rows=prefix_rows,
                workspace=run,
            ))
    result = {
        "schema": "th06-headless-branch-benchmark-v1",
        "scope": {
            "difficulty": scope.difficulty,
            "character": scope.character,
            "shot_type": scope.shot_type,
            "stage": scope.stage,
        },
        "seed": args.seed,
        "root_checkpoint_tick": 1,
        "checkpoint_tick": args.checkpoint_tick,
        "branch_terminal_tick": args.checkpoint_tick + args.branch_frames,
        "branch_frames": args.branch_frames,
        "branches": len(ACTION_NAMES),
        "repetitions": args.repetitions,
        "median_nested_seconds": statistics.median(
            float(run["nested_seconds"]) for run in runs
        ),
        "median_fresh_seconds": statistics.median(
            float(run["fresh_seconds"]) for run in runs
        ),
        "median_speedup": statistics.median(float(run["speedup"]) for run in runs),
        "all_byte_exact": all(not run["mismatches"] for run in runs),
        "source": _source_revision(binary),
        "runs": runs,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["all_byte_exact"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
