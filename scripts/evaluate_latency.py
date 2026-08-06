#!/usr/bin/env python3
"""Measure control-loop latency, especially in high-density TH06 frames."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


FRAME_BUDGET_MS = 1000.0 / 60.0
BINS = (
    ("000-099", 0, 100),
    ("100-299", 100, 300),
    ("300-499", 300, 500),
    ("500+", 500, math.inf),
)


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(len(ordered) * fraction) - 1)]


def _summary(rows: list[dict[str, object]]) -> dict[str, object]:
    capture = [float(row["capture_ms"]) for row in rows]
    solve = [float(row["solve_ms"]) for row in rows]
    total = [left + right for left, right in zip(capture, solve)]
    return {
        "frames": len(rows),
        "capture_p50_ms": _percentile(capture, 0.50),
        "capture_p95_ms": _percentile(capture, 0.95),
        "capture_p99_ms": _percentile(capture, 0.99),
        "control_p95_ms": _percentile(total, 0.95),
        "control_p99_ms": _percentile(total, 0.99),
        "control_max_ms": max(total) if total else None,
        "over_frame_budget_rate": (
            sum(value > FRAME_BUDGET_MS for value in total) / len(total)
            if total else None
        ),
        "stale_retry_rate": (
            sum(row.get("reason") == "stale-retry" for row in rows) / len(rows)
            if rows else None
        ),
        "observation_gap_rate": (
            sum(int(row.get("observation_gap", 1)) != 1 for row in rows) / len(rows)
            if rows else None
        ),
        "max_observation_gap": max(
            (int(row.get("observation_gap", 1)) for row in rows),
            default=None,
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument(
        "--run-id",
        help="select one run; default is the last run_id in the trace",
    )
    args = parser.parse_args()
    rows = []
    with args.trace.open("r", encoding="utf-8") as source:
        for line in source:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not all(key in row for key in ("capture_ms", "solve_ms", "bullets")):
                continue
            if row.get("run_id") is not None:
                rows.append(row)
    if not rows:
        raise SystemExit("trace has no run-tagged latency rows")
    run_id = args.run_id or str(rows[-1]["run_id"])
    rows = [row for row in rows if str(row.get("run_id")) == run_id]
    if not rows:
        raise SystemExit(f"run_id not found: {run_id}")
    density = {
        label: _summary([
            row for row in rows if low <= int(row["bullets"]) < high
        ])
        for label, low, high in BINS
    }
    result = {
        "run_id": run_id,
        "frame_budget_ms": FRAME_BUDGET_MS,
        "overall": _summary(rows),
        "by_live_bullet_count": density,
        "lunatic_dense_acceptance": {
            "bin": "500+",
            "target_control_p95_ms_lte": FRAME_BUDGET_MS,
            "target_stale_retry_rate_lte": 0.02,
            "passes": bool(
                density["500+"]["frames"]
                and density["500+"]["control_p95_ms"] <= FRAME_BUDGET_MS
                and density["500+"]["stale_retry_rate"] <= 0.02
            ),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
