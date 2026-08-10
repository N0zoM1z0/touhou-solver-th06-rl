#!/usr/bin/env python3
"""Check compact v5 decisions against sampled physical authority frames."""

from __future__ import annotations

import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path


BOMB_BIT = 0x02


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _rows(path: Path):
    with gzip.open(path, "rt", encoding="utf-8") as source:
        for line_number, line in enumerate(source, 1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"row is not an object: {path}:{line_number}")
            yield value


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _latency(rows: list[tuple[int, float, float]]) -> dict[str, object]:
    capture = [row[1] for row in rows]
    solve = [row[2] for row in rows]
    total = [left + right for left, right in zip(capture, solve, strict=True)]
    return {
        "frames": len(rows),
        "capture_p95_ms": _percentile(capture, 0.95),
        "solve_p95_ms": _percentile(solve, 0.95),
        "control_p50_ms": _percentile(total, 0.50),
        "control_p95_ms": _percentile(total, 0.95),
        "control_p99_ms": _percentile(total, 0.99),
        "control_max_ms": max(total) if total else None,
        "over_16_67_ms_rate": (
            sum(value > 1000.0 / 60.0 for value in total) / len(total)
            if total else None
        ),
    }


def _close(left: object, right: object, tolerance: float = 1e-6) -> bool:
    try:
        return abs(float(left) - float(right)) <= tolerance
    except (TypeError, ValueError):
        return left == right


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    run_dir = args.dataset / "runs" / args.run_id
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run.get("schemas", {}).get("transition") != "th06-rl-transition-v5":
        raise SystemExit("recorded compact-context benchmark requires transition v5")
    shard_rows = [row for row in manifest.get("shards", []) if isinstance(row, dict)]
    frames = [
        row for row in shard_rows
        if row.get("stream") == "frames" and (run_dir / str(row.get("path"))).is_file()
    ]
    if not frames:
        raise SystemExit("no local sampled frame shards found")
    transitions = [row for row in shard_rows if row.get("stream") == "transitions"]
    transition_by_range = {
        (int(row["first_sequence"]), int(row["last_sequence"])): row
        for row in transitions
    }
    mismatches: Counter[str] = Counter()
    reasons: Counter[str] = Counter()
    phases: set[str] = set()
    latencies: list[tuple[int, float, float]] = []
    planning_effort: Counter[int] = Counter()
    legal_sizes: Counter[int] = Counter()
    hard_sizes: Counter[int] = Counter()
    frame_count = 0
    paired = 0
    trailing = 0
    sampled_bytes = 0
    sampled_shards = []
    for frame_shard in sorted(frames, key=lambda row: int(row["first_sequence"])):
        frame_path = run_dir / str(frame_shard["path"])
        if _sha256(frame_path) != frame_shard.get("sha256"):
            raise ValueError(f"sampled frame digest mismatch: {frame_path}")
        sampled_bytes += frame_path.stat().st_size
        sampled_shards.append(frame_path.name)
        first = int(frame_shard["first_sequence"])
        last = int(frame_shard["last_sequence"])
        candidates = [
            row for (left, right), row in transition_by_range.items()
            if left <= first <= right or left <= last <= right or first <= left <= last
        ]
        transition_map = {}
        for transition_shard in candidates:
            path = run_dir / str(transition_shard["path"])
            if _sha256(path) != transition_shard.get("sha256"):
                raise ValueError(f"transition digest mismatch: {path}")
            for transition in _rows(path):
                transition_map[int(transition["sequence"])] = transition
        for frame in _rows(frame_path):
            frame_count += 1
            sequence = int(frame["sequence"])
            transition = transition_map.get(sequence)
            if transition is None:
                trailing += 1
                continue
            paired += 1
            decision = frame.get("decision") if isinstance(frame.get("decision"), dict) else {}
            snapshot = frame.get("snapshot") if isinstance(frame.get("snapshot"), dict) else {}
            scope = frame.get("scope") if isinstance(frame.get("scope"), dict) else {}
            transition_scope = (
                transition.get("scope")
                if isinstance(transition.get("scope"), dict)
                else {}
            )
            context = (
                transition.get("policy_context")
                if isinstance(transition.get("policy_context"), dict)
                else {}
            )
            hard = tuple(str(item[0]) for item in decision.get("hard_actions", []))
            legal = tuple(str(item) for item in decision.get("locally_admissible_actions", []))
            phases.add(str(decision.get("phase_id")))
            reasons[str(decision.get("reason"))] += 1
            hard_sizes[len(hard)] += 1
            legal_sizes[len(legal)] += 1
            planning_effort[int(decision.get("effort_horizon", 0))] += 1
            bullets = int(snapshot.get("live_bullet_count", len(snapshot.get("bullets", []))))
            latencies.append((bullets, float(decision.get("capture_ms", 0.0)), float(decision.get("solve_ms", 0.0))))

            comparisons = {
                "sequence": sequence == int(transition.get("sequence", -1)),
                "scope": scope.get("key") == transition_scope.get("key"),
                "phase": decision.get("phase_id") == transition_scope.get("phase_id"),
                "baseline_action": decision.get("baseline_action") == transition.get("baseline_action"),
                "proposed_action": decision.get("proposed_action") == transition.get("proposed_action"),
                "published_action": decision.get("published_action") == transition.get("published_action"),
                "legal_actions": legal == tuple(str(item) for item in transition.get("legal_actions", [])),
                "behavior_probability": _close(decision.get("behavior_probability"), transition.get("behavior_probability"), 1e-12),
                "current_action": decision.get("current_action") == context.get("current_action"),
                "hard_actions": hard == tuple(str(item) for item in context.get("hard_admissible_actions", [])),
                "phase_elapsed_frames": int(decision.get("phase_elapsed_frames", -1)) == int(context.get("phase_elapsed_frames", -2)),
                "player_x": _close(snapshot.get("x"), context.get("player_x")),
                "player_y": _close(snapshot.get("y"), context.get("player_y")),
                "power": int(snapshot.get("current_power", -1)) == int(context.get("power", -2)),
                "bullet_count": bullets == int(context.get("bullet_count", -1)),
                "laser_count": int(snapshot.get("laser_count", -1)) == int(context.get("laser_count", -2)),
                "hard_action_count": len(hard) == int(context.get("hard_action_count", -1)),
            }
            for name, matches in comparisons.items():
                mismatches[name] += not matches
            mismatches["legal_not_subset_of_hard"] += not set(legal).issubset(hard)
            mismatches["duplicate_hard_action"] += len(hard) != len(set(hard))
            mismatches["bomb_input"] += bool(int(snapshot.get("input_mask", 0)) & BOMB_BIT)
            mismatches["bomb_active"] += bool(snapshot.get("bomb_active"))
            x = float(snapshot.get("x", -math.inf))
            y = float(snapshot.get("y", -math.inf))
            mismatches["player_outside_runtime_bounds"] += not (8.0 <= x <= 376.0 and 16.0 <= y <= 432.0)
            for item in decision.get("hard_actions", []):
                mismatches["hard_endpoint_outside_runtime_bounds"] += not (
                    8.0 <= float(item[2]) <= 376.0
                    and 16.0 <= float(item[3]) <= 432.0
                )

    density_bins = {
        "000-099": (0, 100),
        "100-299": (100, 300),
        "300-499": (300, 500),
        "500+": (500, math.inf),
    }
    nonzero_mismatches = {name: value for name, value in sorted(mismatches.items()) if value}
    result = {
        "schema": "th06-rl-recorded-frame-benchmark-v1",
        "dataset_revision": args.revision,
        "run_id": args.run_id,
        "frame_schema": run.get("schemas", {}).get("frame"),
        "transition_schema": run.get("schemas", {}).get("transition"),
        "sample": {
            "frame_shards": sampled_shards,
            "compressed_bytes": sampled_bytes,
            "frame_rows": frame_count,
            "paired_transition_rows": paired,
            "trailing_frames_without_transition": trailing,
            "source_contexts": len(phases),
        },
        "coherence_mismatches": dict(sorted(mismatches.items())),
        "nonzero_coherence_mismatches": nonzero_mismatches,
        "decision_reasons": dict(reasons.most_common()),
        "hard_action_set_sizes": {str(key): value for key, value in sorted(hard_sizes.items())},
        "local_action_set_sizes": {str(key): value for key, value in sorted(legal_sizes.items())},
        "planning_effort_horizon": {str(key): value for key, value in sorted(planning_effort.items())},
        "physical_control_latency": {
            "overall": _latency(latencies),
            "by_live_bullet_count": {
                name: _latency([row for row in latencies if low <= row[0] < high])
                for name, (low, high) in density_bins.items()
            },
        },
        "passes": not nonzero_mismatches,
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if result["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
