#!/usr/bin/env python3
"""Replay a bounded sample of lossless TH06 learning snapshots offline."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from statistics import mean, median
import time

from th06_rl.corpus import expand_compact
from th06_rl.core.planner import LocalPlannerConfig
from th06_rl.native import NativeKernel, PackedHazards
from th06_rl.retail.barrage_lab.corpus import decode_snapshot
from th06_rl.th06.control_capture import decode_control_snapshot
from th06_rl.th06.source import (
    AuthorityUnavailable,
    core_action_from_input,
    kinematics_from_snapshot,
    lower_observed_hazards,
    lower_source_forecast,
)


def _rows(paths: list[Path]):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _load_objects(run_dir: Path) -> dict[str, object]:
    return {
        row["object_id"]: row["payload"]
        for row in _rows(sorted(run_dir.glob("objects-*.jsonl.gz")))
    }


def _hydrate(value: object, objects: dict[str, object]) -> object:
    return expand_compact(value, objects)


def _encoded_row_count(value: object) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict) and value.get("codec") == "dataclass-rows-v1":
        rows = value.get("rows", ())
        return len(rows) if isinstance(rows, list) else 0
    return 0


def _encoded_bullet_count(row: dict) -> int:
    direct = row["snapshot"].get("live_bullet_count")
    return int(direct) if direct is not None else _encoded_row_count(
        row["snapshot"]["bullets"]
    )


def _sample_indices(rows: list[dict], evenly: int, dense: int) -> tuple[int, ...]:
    if not rows:
        return ()
    even_count = min(evenly, len(rows))
    even = {
        round(index * (len(rows) - 1) / max(1, even_count - 1))
        for index in range(even_count)
    }
    densest = {
        index
        for index, _row in sorted(
            enumerate(rows),
            key=lambda pair: _encoded_bullet_count(pair[1]),
            reverse=True,
        )[:dense]
    }
    return tuple(sorted(even | densest))


def _timing_summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    percentile_95 = ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]
    return {
        "mean_ms": mean(values),
        "median_ms": median(values),
        "p95_ms": percentile_95,
        "max_ms": ordered[-1],
    }


def replay(args: argparse.Namespace) -> dict[str, object]:
    run_dir = args.run_dir.resolve()
    objects = _load_objects(run_dir)
    rows = list(_rows(sorted(run_dir.glob("frames-*.jsonl.gz"))))
    if not rows:
        raise RuntimeError(f"no snapshot shards in {run_dir}")

    kernel = NativeKernel(args.native_library)
    timings: dict[str, list[float]] = {"source": [], "hard": [], "plan": []}
    authority_stops: list[dict[str, object]] = []
    empty_hard: list[dict[str, int]] = []
    failures: list[dict[str, object]] = []
    sampled = 0
    maximum_bullets = 0
    maximum_lasers = 0
    minimum_hard = 18
    partial_forecasts = 0
    forecast_function = (
        lower_observed_hazards
        if args.forecast_mode == "observed"
        else lower_source_forecast
    )

    for index in _sample_indices(rows, args.samples, args.dense):
        row = rows[index]
        try:
            raw_snapshot = _hydrate(row["snapshot"], objects)
            if str(raw_snapshot.get("capture_tier", "")).startswith("control-v"):
                if args.forecast_mode != "observed":
                    raise RuntimeError(
                        "source-ecl replay requires authoritative anchor rows"
                    )
                snapshot = decode_control_snapshot(raw_snapshot)
            else:
                snapshot = decode_snapshot(raw_snapshot)
            if (
                snapshot.player_state not in (0, 3)
                or snapshot.in_menu
                or snapshot.time_stopped
            ):
                continue
            source_started = time.perf_counter()
            forecast = forecast_function(snapshot, args.horizon)
            hard_started = time.perf_counter()
            hard_hazards = PackedHazards(
                forecast.hazards.aabb_frames[: forecast.hard_horizon],
                forecast.hazards.laser_frames[: forecast.hard_horizon],
            )
            current = core_action_from_input(snapshot.input_mask)
            kinematics = kinematics_from_snapshot(snapshot)
            hard = kernel.certify_actions(
                x=snapshot.x,
                y=snapshot.y,
                half_width=snapshot.half_width,
                half_height=snapshot.half_height,
                kinematics=kinematics,
                current_action=current,
                hazards=hard_hazards,
            )
            plan_started = time.perf_counter()
            plan = (
                kernel.plan(
                    x=snapshot.x,
                    y=snapshot.y,
                    half_width=snapshot.half_width,
                    half_height=snapshot.half_height,
                    kinematics=kinematics,
                    current_action=current,
                    hazards=forecast.hazards,
                    hard=hard,
                    config=LocalPlannerConfig(horizon=forecast.source_coverage),
                )
                if hard
                else None
            )
            finished = time.perf_counter()
        except AuthorityUnavailable as error:
            authority_stops.append({
                "sequence": row["sequence"],
                "frame": row["snapshot"]["frame"],
                "reason": str(error),
            })
            continue
        except Exception as error:  # A replay is a diagnostic boundary.
            failures.append({
                "sequence": row.get("sequence"),
                "frame": row.get("snapshot", {}).get("frame"),
                "error": f"{type(error).__name__}: {error}",
            })
            continue

        sampled += 1
        timings["source"].append((hard_started - source_started) * 1000.0)
        timings["hard"].append((plan_started - hard_started) * 1000.0)
        timings["plan"].append((finished - plan_started) * 1000.0)
        maximum_bullets = max(maximum_bullets, len(snapshot.bullets))
        maximum_lasers = max(maximum_lasers, len(snapshot.lasers))
        minimum_hard = min(minimum_hard, len(hard))
        partial_forecasts += forecast.source_coverage < args.horizon
        if not hard:
            empty_hard.append({"sequence": row["sequence"], "frame": snapshot.frame})
        elif plan is None:
            failures.append({
                "sequence": row["sequence"],
                "frame": snapshot.frame,
                "error": "non-empty Hard set had no local continuation",
            })

    return {
        "run_id": run_dir.name,
        "forecast_mode": args.forecast_mode,
        "total_snapshots": len(rows),
        "sampled_active_snapshots": sampled,
        "source_objects": len(objects),
        "maximum_bullets": maximum_bullets,
        "maximum_lasers": maximum_lasers,
        "minimum_hard_actions": minimum_hard,
        "partial_forecasts": partial_forecasts,
        "empty_hard": empty_hard,
        "authority_stops": authority_stops,
        "failures": failures,
        "timings": {
            name: _timing_summary(values) for name, values in timings.items()
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--samples", type=int, default=120)
    parser.add_argument("--dense", type=int, default=30)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument(
        "--forecast-mode",
        choices=("observed", "source-ecl"),
        default="observed",
    )
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    if args.samples <= 0 or args.dense < 0:
        parser.error("--samples must be positive and --dense cannot be negative")
    result = replay(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if result["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
