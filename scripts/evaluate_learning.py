#!/usr/bin/env python3
"""Summarize complete-stage learning progress and learned UCB evidence."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gzip
import json
import math
from pathlib import Path
from statistics import median


DIFFICULTIES = {"normal": 1, "hard": 2, "lunatic": 3}


def _rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(fraction * len(ordered)) - 1)]


def _timings(values: list[float]) -> dict[str, float | None]:
    return {
        "p50_ms": median(values) if values else None,
        "p95_ms": _percentile(values, 0.95),
        "p99_ms": _percentile(values, 0.99),
        "max_ms": max(values) if values else None,
    }


def _frame_from_ref(value: object) -> int | None:
    marker = str(value).rsplit(":f", 1)
    if len(marker) != 2:
        return None
    try:
        return int(marker[1])
    except ValueError:
        return None


def _longest_survival(start: int | None, end: int | None, hits: list[int]) -> int | None:
    if start is None or end is None or end < start:
        return None
    boundaries = [start, *sorted(frame for frame in hits if start <= frame <= end), end]
    return max((right - left for left, right in zip(boundaries, boundaries[1:])), default=0)


def summarize_run(run_dir: Path) -> dict[str, object]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    stored_summary = manifest.get("summary")
    if isinstance(stored_summary, dict):
        outcome = manifest.get("run_outcome") or {}
        active_frames = int(
            stored_summary.get("learning_eligible_elapsed_frames", 0)
        )
        hit_frames = [int(value) for value in stored_summary.get("hit_frames", ())]
        hits = int(outcome.get("physical_hits", len(hit_frames)))
        frames = int(stored_summary.get("frames", 0))
        compressed = int(manifest.get("compressed_bytes", 0))
        return {
            "run_id": run_dir.name,
            "stage_trajectory_complete": bool(
                manifest.get("stage_trajectory_complete")
            ),
            "termination_reason": outcome.get("termination_reason"),
            "physical_hits": hits,
            "control_dead_ends": int(outcome.get("control_dead_ends", 0)),
            "capture_failures": int(outcome.get("capture_failures", 0)),
            "infrastructure_failures": int(
                outcome.get("infrastructure_failures", 0)
            ),
            "recorded_frames": frames,
            "active_elapsed_frames": active_frames,
            "hit_rate_per_1000_active_frames": (
                1000.0 * hits / active_frames if active_frames else None
            ),
            "first_hit_frame": min(hit_frames) if hit_frames else None,
            "longest_observed_no_hit_frames": stored_summary.get(
                "longest_observed_no_hit_frames"
            ),
            "compressed_bytes": compressed,
            "compressed_bytes_per_frame": (
                compressed / frames if frames else None
            ),
            "capture_timing": stored_summary.get("capture_timing", {}),
            "solve_timing": stored_summary.get("solve_timing", {}),
            "reason_counts": stored_summary.get("reason_counts", {}),
            "stale_retry_rate": stored_summary.get("stale_retry_rate"),
            "phases": stored_summary.get("phases", []),
        }
    phase = defaultdict(lambda: {
        "transitions": 0,
        "elapsed_frames": 0,
        "hits": 0,
        "control_dead_ends": 0,
        "published_actions": Counter(),
        "legal_opportunities": Counter(),
        "hard_sum": 0,
    })
    hit_frames: list[int] = []
    first_frame = None
    last_frame = None
    transitions = 0
    active_frames = 0
    for row in _rows(sorted(run_dir.glob("transitions-*.jsonl.gz"))):
        transitions += 1
        frame = _frame_from_ref(row.get("snapshot_ref"))
        next_frame = _frame_from_ref(row.get("next_snapshot_ref"))
        first_frame = frame if first_frame is None else min(first_frame, frame or first_frame)
        last_frame = next_frame if last_frame is None else max(last_frame, next_frame or last_frame)
        outcome = row.get("outcome_terms", {})
        elapsed = max(0, int(outcome.get("elapsed_frames", 0)))
        if row.get("learning_eligible"):
            active_frames += elapsed
        scope = row.get("scope", {})
        key = str(scope.get("key", "unknown"))
        item = phase[key]
        item["transitions"] += 1
        item["elapsed_frames"] += elapsed
        item["hard_sum"] += max(0, int(outcome.get("hard_count_before", 0)))
        action = row.get("published_action")
        if action:
            item["published_actions"][str(action)] += 1
        for legal in row.get("legal_actions", ()):
            item["legal_opportunities"][str(legal)] += 1
        if outcome.get("life_lost"):
            item["hits"] += 1
            if next_frame is not None:
                hit_frames.append(next_frame)
        if outcome.get("control_dead_end"):
            item["control_dead_ends"] += 1

    capture: list[float] = []
    solve: list[float] = []
    reasons = Counter()
    frame_rows = 0
    for row in _rows(sorted(run_dir.glob("frames-*.jsonl.gz"))):
        frame_rows += 1
        decision = row.get("decision", {})
        try:
            capture.append(float(decision["capture_ms"]))
            solve.append(float(decision["solve_ms"]))
        except (KeyError, TypeError, ValueError):
            pass
        reasons[str(decision.get("reason", "unknown"))] += 1

    phase_rows = []
    for key, item in phase.items():
        count = int(item["transitions"])
        phase_rows.append({
            "scope": key,
            "transitions": count,
            "elapsed_frames": int(item["elapsed_frames"]),
            "hits": int(item["hits"]),
            "control_dead_ends": int(item["control_dead_ends"]),
            "mean_hard_actions": item["hard_sum"] / count if count else None,
            "published_actions": dict(item["published_actions"].most_common()),
            "legal_opportunities": dict(item["legal_opportunities"].most_common()),
        })
    phase_rows.sort(key=lambda item: (-item["hits"], -item["elapsed_frames"], item["scope"]))
    outcome = manifest.get("run_outcome") or {}
    compressed = int(manifest.get("compressed_bytes", 0))
    frames = max(int(manifest.get("written_frames", 0)), frame_rows)
    return {
        "run_id": run_dir.name,
        "stage_trajectory_complete": bool(manifest.get("stage_trajectory_complete")),
        "termination_reason": outcome.get("termination_reason"),
        "physical_hits": int(outcome.get("physical_hits", len(hit_frames))),
        "control_dead_ends": int(outcome.get("control_dead_ends", 0)),
        "capture_failures": int(outcome.get("capture_failures", 0)),
        "infrastructure_failures": int(outcome.get("infrastructure_failures", 0)),
        "recorded_frames": frames,
        "active_elapsed_frames": active_frames,
        "hit_rate_per_1000_active_frames": (
            1000.0 * len(hit_frames) / active_frames if active_frames else None
        ),
        "first_hit_frame": min(hit_frames) if hit_frames else None,
        "longest_observed_no_hit_frames": _longest_survival(
            first_frame, last_frame, hit_frames
        ),
        "compressed_bytes": compressed,
        "compressed_bytes_per_frame": compressed / frames if frames else None,
        "capture_timing": _timings(capture),
        "solve_timing": _timings(solve),
        "reason_counts": dict(reasons.most_common()),
        "phases": phase_rows,
    }


def summarize_policy(path: Path) -> dict[str, object] | None:
    if not path.is_file():
        return None
    state = json.loads(path.read_text(encoding="utf-8"))
    from th06_rl.policies.adaptive import unpack_state

    state = unpack_state(state)
    trials = state.get("trials", {})
    rewards = state.get("reward_sum", {})
    opportunities = state.get("opportunities", {})
    rows = []
    if isinstance(trials, dict) and isinstance(rewards, dict):
        for key, raw_trials in trials.items():
            count = int(raw_trials)
            if count <= 0:
                continue
            reward = float(rewards.get(key, 0.0))
            rows.append({
                "context_action": str(key),
                "trials": count,
                "legal_opportunities": int(opportunities.get(key, 0)),
                "mean_observed_reward": reward / count,
            })
    by_support = sorted(rows, key=lambda row: (-row["trials"], row["context_action"]))
    by_reward = sorted(
        (row for row in rows if row["trials"] >= 3),
        key=lambda row: (-row["mean_observed_reward"], -row["trials"]),
    )
    return {
        "schema": state.get("schema"),
        "reward_version": state.get("reward_version"),
        "decisions": int(state.get("decisions", 0)),
        "exploratory_decisions": int(state.get("exploratory_decisions", 0)),
        "trained_context_actions": len(rows),
        "observed_trials": sum(row["trials"] for row in rows),
        "most_supported": by_support[:20],
        "highest_observed_reward_min_3_trials": by_reward[:20],
        "lowest_observed_reward_min_3_trials": list(reversed(by_reward[-20:])),
    }


def _trend(complete_runs: list[dict[str, object]]) -> dict[str, object]:
    values = [
        float(run["hit_rate_per_1000_active_frames"])
        for run in complete_runs
        if run["hit_rate_per_1000_active_frames"] is not None
    ]
    if len(values) < 2:
        return {"complete_runs": len(values), "hit_rate_slope_per_run": None}
    mean_x = (len(values) - 1) / 2.0
    mean_y = sum(values) / len(values)
    denominator = sum((index - mean_x) ** 2 for index in range(len(values)))
    slope = sum(
        (index - mean_x) * (value - mean_y)
        for index, value in enumerate(values)
    ) / denominator
    return {
        "complete_runs": len(values),
        "hit_rate_slope_per_run": slope,
        "first_hit_rate": values[0],
        "latest_hit_rate": values[-1],
        "best_hit_rate": min(values),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-root", type=Path, default=Path("artifacts/corpus"))
    parser.add_argument("--difficulty", choices=tuple(DIFFICULTIES), default="lunatic")
    parser.add_argument("--stage", type=int, choices=range(1, 7), default=4)
    parser.add_argument("--policy-state", type=Path, default=None)
    parser.add_argument("--recent", type=int, default=20)
    parser.add_argument(
        "--include-active",
        action="store_true",
        help="scan active/incomplete shards; expensive and unsuitable during play",
    )
    args = parser.parse_args()
    if args.recent <= 0:
        parser.error("--recent must be positive")
    if args.policy_state is None:
        args.policy_state = Path(
            f"artifacts/policy/{args.difficulty}_reimu_a_stage{args.stage}.json"
        )
    run_dirs = []
    for path in sorted(args.corpus_root.glob("*/manifest.json")):
        run_path = path.parent / "run.json"
        if not run_path.is_file():
            continue
        metadata = json.loads(run_path.read_text(encoding="utf-8")).get(
            "metadata", {}
        )
        if (
            metadata.get("difficulty") != DIFFICULTIES[args.difficulty]
            or metadata.get("character") != 0
            or metadata.get("shot_type") != 0
            or metadata.get("stage") != args.stage
        ):
            continue
        if not args.include_active:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if "closed_unix_ns" not in manifest:
                continue
        run_dirs.append(path.parent)
    run_dirs = run_dirs[-args.recent:]
    runs = [summarize_run(path) for path in run_dirs]
    complete = [run for run in runs if run["stage_trajectory_complete"]]
    print(json.dumps({
        "corpus_root": str(args.corpus_root.resolve()),
        "scope": {
            "difficulty": args.difficulty,
            "character_shot": "Reimu-A",
            "stage": args.stage,
        },
        "runs": runs,
        "complete_stage_trend": _trend(complete),
        "policy": summarize_policy(args.policy_state),
        "interpretation": (
            "Chronological physical evidence only; trend is descriptive, not "
            "a causal off-policy estimate. Incomplete stages are excluded."
        ),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
