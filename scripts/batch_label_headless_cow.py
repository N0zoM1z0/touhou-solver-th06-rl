#!/usr/bin/env python3
"""Label a resumable, worker-bounded batch of headless COW checkpoints."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import gzip
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Iterable


SCHEMA = "th06-rl-headless-cow-counterfactual-v1"
CORRECTIVE_TERMINATIONS = frozenset({"authority-failure", "physical-hit"})


def _run_directories(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if (path / "manifest.json").is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(item.parent for item in path.rglob("manifest.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def checkpoint_sequences(
    transition_count: int,
    *,
    tail_transitions: int,
    stride: int,
) -> tuple[int, ...]:
    """Select a terminal neighborhood and retain the final reconstructable row."""
    if transition_count < 2:
        return ()
    start = max(1, transition_count - tail_transitions)
    sequences = list(range(start, transition_count, stride))
    final = transition_count - 1
    if final not in sequences:
        sequences.append(final)
    return tuple(sequences)


def checkpoint_groups(
    sequences: tuple[int, ...],
    *,
    checkpoints_per_task: int,
) -> tuple[tuple[int, ...], ...]:
    """Optionally shard one replay into bounded parallel checkpoint groups."""
    if not sequences:
        return ()
    if checkpoints_per_task <= 0 or checkpoints_per_task >= len(sequences):
        return (sequences,)
    return tuple(
        sequences[offset:offset + checkpoints_per_task]
        for offset in range(0, len(sequences), checkpoints_per_task)
    )


def event_checkpoint_sequences(
    rows: list[dict[str, Any]],
    *,
    event_window: int,
    stride: int,
    termination_reason: str | None = None,
) -> tuple[int, ...]:
    """Sample legal states leading into observed or terminal failure events.

    A fail-close run ends before it can append an authority-release transition,
    so the manifest termination reason is required to retain that final failure
    neighborhood.  Continued-HIT runs still expose their individual events in
    the transition stream.
    """
    event_rows = []
    previous_forced = False
    for index, row in enumerate(rows):
        forced = row.get("benchmark_forced_action") is True
        outcome = row.get("outcome_terms")
        hit = isinstance(outcome, dict) and int(outcome.get("deaths_delta", 0)) > 0
        if hit or (forced and not previous_forced):
            target = index
            while target >= 1 and not row_is_labelable(rows[target]):
                target -= 1
            if target >= 1:
                event_rows.append(target)
        previous_forced = forced
    if termination_reason in CORRECTIVE_TERMINATIONS:
        target = len(rows) - 1
        while target >= 1 and not row_is_labelable(rows[target]):
            target -= 1
        if target >= 1:
            event_rows.append(target)
    selected = set()
    for target in event_rows:
        lower = max(1, target - event_window)
        for sequence in range(target, lower - 1, -stride):
            if row_is_labelable(rows[sequence]):
                selected.add(sequence)
        if row_is_labelable(rows[lower]):
            selected.add(lower)
    return tuple(sorted(selected))


def row_is_labelable(row: dict[str, Any]) -> bool:
    legal = row.get("legal_actions")
    return (
        isinstance(legal, list)
        and bool(legal)
        and row.get("benchmark_forced_action") is not True
    )


def _transition_rows(run: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(run / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
        rows.extend(json.loads(line) for line in stream)
    return rows


def _output_path(
    output_root: Path,
    run: Path,
    manifest: dict[str, Any],
    *,
    group_index: int = 0,
    group_count: int = 1,
) -> Path:
    scope = manifest["scope"]
    stem = (
        f"stage{int(scope['stage'])}-seed{int(manifest['initial_seed'])}-"
        f"{run.name}"
    )
    if group_count > 1:
        stem += f"-part{group_index + 1:04d}-of{group_count:04d}"
    return output_root / f"{stem}.json"


def _completed_output(path: Path, run: Path, sequences: tuple[int, ...]) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    try:
        recorded = Path(str(document.get("input_run", ""))).resolve()
    except OSError:
        return False
    recorded_sequences = tuple(
        int(checkpoint.get("sequence", -1))
        for checkpoint in document.get("checkpoints", [])
        if isinstance(checkpoint, dict)
    )
    return (
        document.get("schema") == SCHEMA
        and recorded == run.resolve()
        and recorded_sequences == sequences
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tail-transitions", type=int, default=600)
    parser.add_argument("--event-window", type=int, default=240)
    parser.add_argument("--stride", type=int, default=80)
    parser.add_argument(
        "--selection",
        choices=("tail", "events", "hybrid"),
        default="tail",
    )
    parser.add_argument("--branch-frames", type=int, default=180)
    parser.add_argument("--teacher-horizon", type=int, default=12)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument(
        "--checkpoints-per-task",
        type=int,
        default=0,
        help=(
            "split each run into replay tasks of this many checkpoints; 0 keeps "
            "one efficient sequential replay per run"
        ),
    )
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
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if min(
        args.tail_transitions,
        args.event_window,
        args.stride,
        args.branch_frames,
        args.teacher_horizon,
        args.workers,
    ) <= 0:
        parser.error("batch, branch, teacher, and worker bounds must be positive")
    if args.workers > 64:
        parser.error("workers must be at most 64 on the shared VPS")
    if args.checkpoints_per_task < 0:
        parser.error("checkpoints per task must be nonnegative")

    runs = _run_directories(args.paths)
    if not runs:
        parser.error("no compact headless corpus manifests found")
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    tasks = []
    skipped = []
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("transaction_complete") is not True:
            continue
        tail = checkpoint_sequences(
            int(manifest.get("transition_count", 0)),
            tail_transitions=args.tail_transitions,
            stride=args.stride,
        ) if args.selection in {"tail", "hybrid"} else ()
        events = event_checkpoint_sequences(
            _transition_rows(run),
            event_window=args.event_window,
            stride=args.stride,
            termination_reason=manifest.get("termination_reason"),
        ) if args.selection in {"events", "hybrid"} else ()
        sequences = tuple(sorted(set(tail).union(events)))
        if not sequences:
            continue
        groups = checkpoint_groups(
            sequences,
            checkpoints_per_task=args.checkpoints_per_task,
        )
        for group_index, group in enumerate(groups):
            output = _output_path(
                output_root,
                run,
                manifest,
                group_index=group_index,
                group_count=len(groups),
            )
            if _completed_output(output, run, group):
                skipped.append(str(output))
                continue
            command = [
                sys.executable,
                str(root / "scripts/label_headless_cow_counterfactuals.py"),
                str(run),
            ]
            for sequence in group:
                command.extend(("--checkpoint-sequence", str(sequence)))
            command.extend((
                "--branch-frames",
                str(args.branch_frames),
                "--teacher-horizon",
                str(args.teacher_horizon),
                "--binary",
                str(args.binary.resolve()),
                "--game-directory",
                str(args.game_directory.resolve()),
                "--output",
                str(output),
            ))
            tasks.append((run, output, command, len(group)))

    results = []
    failures = []

    def execute(task):
        run, output, command, checkpoints = task
        process = subprocess.run(command, capture_output=True, text=True)
        return run, output, checkpoints, process

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(execute, task): task for task in tasks}
        for future in as_completed(futures):
            run, output, checkpoints, process = future.result()
            record = {
                "run": str(run),
                "output": str(output),
                "checkpoints": checkpoints,
                "returncode": process.returncode,
            }
            results.append(record)
            if process.returncode:
                failures.append({
                    **record,
                    "stderr": process.stderr[-4000:],
                })
                print(f"FAILED {run.name}", file=sys.stderr, flush=True)
            else:
                print(f"DONE {run.name} checkpoints={checkpoints}", flush=True)

    summary = {
        "schema": "th06-rl-headless-cow-batch-v2",
        "requested_runs": len(runs),
        "requested_tasks": len(tasks) + len(skipped),
        "launched_tasks": len(tasks),
        "completed_tasks": sum(record["returncode"] == 0 for record in results),
        "skipped_completed_tasks": len(skipped),
        "failed_tasks": len(failures),
        "workers": args.workers,
        "checkpoints_per_task": args.checkpoints_per_task,
        "tail_transitions": args.tail_transitions,
        "event_window": args.event_window,
        "selection": args.selection,
        "stride": args.stride,
        "branch_frames": args.branch_frames,
        "teacher_horizon": args.teacher_horizon,
        "results": sorted(results, key=lambda record: record["run"]),
        "skipped_outputs": sorted(skipped),
        "failures": sorted(failures, key=lambda record: record["run"]),
    }
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
