#!/usr/bin/env python3
"""Summarize pre-HIT warning windows in an evaluation-only Wine trace.

This tool never emits training examples or counterfactual labels.  It reduces
each physical HIT to fixed, generic native-safe-set and delivery diagnostics so
that an expensive new data-collection protocol can be scoped without mining
actions from a post-HIT continuation benchmark.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
from typing import Any, Iterable, Mapping


AUDIT_SCHEMA = "th06-rl-full-stage-hit-window-audit-v1"
DEAD_END_REASON = "control-dead-end:Hard safe set empty"
HIT_REASON = "physical-hit"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"trace line {line_number} is not an object")
            if isinstance(value.get("frame"), int):
                rows.append(value)
    return rows


def _count_events(frames: Iterable[int]) -> int:
    count = 0
    previous: int | None = None
    for frame in sorted(set(frames)):
        if previous is None or frame > previous + 1:
            count += 1
        previous = frame
    return count


def _last(rows: Iterable[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    return max(rows, key=lambda row: int(row["frame"]), default=None)


def _row_projection(row: Mapping[str, Any] | None, hit_frame: int) -> dict[str, Any] | None:
    if row is None:
        return None
    frame = int(row["frame"])
    return {
        "frame": frame,
        "frames_before_hit": hit_frame - frame,
        "hard_count": row.get("hard_count"),
        "effort_horizon": row.get("effort_horizon"),
        "action": row.get("action"),
        "reason": row.get("reason"),
        "x": row.get("x"),
        "y": row.get("y"),
        "bullets": row.get("bullets"),
        "lasers": row.get("lasers"),
        "source_context": row.get("source_context"),
    }


def audit_hit_windows(
    rows: list[dict[str, Any]], *, window_frames: int = 120, narrow_max: int = 4
) -> dict[str, Any]:
    if window_frames <= 0:
        raise ValueError("window_frames must be positive")
    if narrow_max <= 0:
        raise ValueError("narrow_max must be positive")

    ordered = sorted(rows, key=lambda row: (int(row["frame"]), float(row.get("time", 0))))
    hit_rows = [row for row in ordered if row.get("reason") == HIT_REASON]
    degraded_runs: list[list[dict[str, Any]]] = []
    current_degraded_run: list[dict[str, Any]] = []
    for row in ordered:
        if (
            row.get("reason") == "ok"
            and isinstance(row.get("effort_horizon"), (int, float))
            and int(row["effort_horizon"]) < 12
        ):
            current_degraded_run.append(row)
        elif current_degraded_run:
            degraded_runs.append(current_degraded_run)
            current_degraded_run = []
    if current_degraded_run:
        degraded_runs.append(current_degraded_run)

    trigger_diagnostics: dict[str, dict[str, Any]] = {}
    hit_frames = [int(row["frame"]) for row in hit_rows]
    for consecutive_rows in (1, 2, 3):
        activations = []
        covered_hits: set[int] = set()
        for run in degraded_runs:
            if len(run) < consecutive_rows:
                continue
            trigger_frame = int(run[consecutive_rows - 1]["frame"])
            next_hits = [frame for frame in hit_frames if frame > trigger_frame]
            next_hit = min(next_hits) if next_hits else None
            frames_to_hit = next_hit - trigger_frame if next_hit is not None else None
            predicts_hit = frames_to_hit is not None and frames_to_hit <= 15
            if predicts_hit:
                covered_hits.add(int(next_hit))
            activations.append(
                {
                    "trigger_frame": trigger_frame,
                    "run_start_frame": int(run[0]["frame"]),
                    "run_rows": len(run),
                    "next_hit_frame": next_hit,
                    "frames_to_next_hit": frames_to_hit,
                    "hit_within_15_frames": predicts_hit,
                }
            )
        trigger_diagnostics[str(consecutive_rows)] = {
            "consecutive_degraded_ok_rows": consecutive_rows,
            "activations": len(activations),
            "hit_within_15_activations": sum(
                bool(activation["hit_within_15_frames"]) for activation in activations
            ),
            "other_activations": sum(
                not bool(activation["hit_within_15_frames"]) for activation in activations
            ),
            "distinct_hits_covered": len(covered_hits),
            "physical_hits": len(hit_frames),
            "details": activations,
        }
    hits: list[dict[str, Any]] = []
    previous_hit_frame: int | None = None

    for hit_index, hit in enumerate(hit_rows, start=1):
        hit_frame = int(hit["frame"])
        segment_start = 0 if previous_hit_frame is None else previous_hit_frame + 1
        window_start = max(segment_start, hit_frame - window_frames)
        preceding = [
            row
            for row in ordered
            if window_start <= int(row["frame"]) < hit_frame
        ]
        ok_rows = [row for row in preceding if row.get("reason") == "ok"]
        hard_rows = [
            row for row in preceding if isinstance(row.get("hard_count"), (int, float))
        ]
        narrow_rows = [
            row
            for row in ok_rows
            if 1 <= int(row.get("hard_count", -1)) <= narrow_max
        ]
        medium_rows = [
            row
            for row in ok_rows
            if narrow_max < int(row.get("hard_count", -1)) <= 9
        ]
        degraded_rows = [
            row
            for row in ok_rows
            if isinstance(row.get("effort_horizon"), (int, float))
            and int(row["effort_horizon"]) < 12
        ]
        dead_end_rows = [row for row in preceding if row.get("reason") == DEAD_END_REASON]
        stale_rows = [row for row in preceding if row.get("reason") == "stale-retry"]
        lease_rows = [row for row in preceding if row.get("reason") == "input-lease"]

        final_dead_end: list[dict[str, Any]] = []
        for row in reversed(preceding):
            if row.get("reason") != DEAD_END_REASON:
                break
            final_dead_end.append(row)
        final_dead_end.reverse()
        final_start = final_dead_end[0] if final_dead_end else None
        final_start_frame = int(final_start["frame"]) if final_start else hit_frame
        before_final = [row for row in preceding if int(row["frame"]) < final_start_frame]
        last_ok_before_final = _last(row for row in before_final if row.get("reason") == "ok")
        last_narrow_before_final = _last(
            row
            for row in before_final
            if row.get("reason") == "ok"
            and 1 <= int(row.get("hard_count", -1)) <= narrow_max
        )
        first_degraded_before_final = min(
            (
                row
                for row in before_final
                if row.get("reason") == "ok"
                and isinstance(row.get("effort_horizon"), (int, float))
                and int(row["effort_horizon"]) < 12
            ),
            key=lambda row: int(row["frame"]),
            default=None,
        )
        final_degraded: list[dict[str, Any]] = []
        for row in reversed(before_final):
            if (
                row.get("reason") != "ok"
                or not isinstance(row.get("effort_horizon"), (int, float))
                or int(row["effort_horizon"]) >= 12
            ):
                break
            final_degraded.append(row)
        final_degraded.reverse()

        hard_histogram = Counter()
        for row in hard_rows:
            count = int(row["hard_count"])
            if count == 0:
                hard_histogram["zero"] += 1
            elif count <= narrow_max:
                hard_histogram[f"one_to_{narrow_max}"] += 1
            elif count <= 9:
                hard_histogram[f"{narrow_max + 1}_to_9"] += 1
            else:
                hard_histogram["ten_plus"] += 1

        hits.append(
            {
                "hit_index": hit_index,
                "hit_frame": hit_frame,
                "source_context": hit.get("source_context"),
                "position": {"x": hit.get("x"), "y": hit.get("y")},
                "window_start_frame": window_start,
                "rows_in_window": len(preceding),
                "hard_count_histogram": dict(sorted(hard_histogram.items())),
                "ok_rows": len(ok_rows),
                "narrow_ok_rows": len(narrow_rows),
                "narrow_ok_events": _count_events(int(row["frame"]) for row in narrow_rows),
                "medium_ok_rows": len(medium_rows),
                "degraded_horizon_ok_rows": len(degraded_rows),
                "degraded_horizon_ok_events": _count_events(
                    int(row["frame"]) for row in degraded_rows
                ),
                "dead_end_rows": len(dead_end_rows),
                "dead_end_events": _count_events(int(row["frame"]) for row in dead_end_rows),
                "stale_retry_rows": len(stale_rows),
                "input_lease_rows": len(lease_rows),
                "final_dead_end": {
                    "rows": len(final_dead_end),
                    "start_frame": int(final_start["frame"]) if final_start else None,
                    "lead_frames": hit_frame - int(final_start["frame"]) if final_start else None,
                },
                "last_ok_before_final_dead_end": _row_projection(
                    last_ok_before_final, hit_frame
                ),
                "last_narrow_ok_before_final_dead_end": _row_projection(
                    last_narrow_before_final, hit_frame
                ),
                "first_degraded_ok_before_final_dead_end": _row_projection(
                    first_degraded_before_final, hit_frame
                ),
                "final_degraded_ok_run": {
                    "rows": len(final_degraded),
                    "start_frame": int(final_degraded[0]["frame"])
                    if final_degraded
                    else None,
                    "lead_frames": hit_frame - int(final_degraded[0]["frame"])
                    if final_degraded
                    else None,
                },
            }
        )
        previous_hit_frame = hit_frame

    final_leads = [
        int(hit["final_dead_end"]["lead_frames"])
        for hit in hits
        if hit["final_dead_end"]["lead_frames"] is not None
    ]
    last_ok_widths = [
        int(hit["last_ok_before_final_dead_end"]["hard_count"])
        for hit in hits
        if hit["last_ok_before_final_dead_end"] is not None
        and isinstance(hit["last_ok_before_final_dead_end"]["hard_count"], (int, float))
    ]
    first_degraded_leads = [
        int(hit["first_degraded_ok_before_final_dead_end"]["frames_before_hit"])
        for hit in hits
        if hit["first_degraded_ok_before_final_dead_end"] is not None
    ]
    final_warning_leads = [
        int(hit["final_degraded_ok_run"]["lead_frames"])
        for hit in hits
        if hit["final_degraded_ok_run"]["lead_frames"] is not None
    ]
    context_counts = Counter(str(hit["source_context"]) for hit in hits)

    return {
        "schema": AUDIT_SCHEMA,
        "parameters": {
            "window_frames": window_frames,
            "narrow_hard_count_maximum": narrow_max,
            "window_never_crosses_previous_physical_hit": True,
        },
        "evidence_boundary": {
            "evaluation_only": True,
            "may_define_future_collection_hypothesis": True,
            "may_supply_training_rows": False,
            "may_supply_counterfactual_labels": False,
            "may_select_or_promote_candidate": False,
            "promotion_metric": "fresh interleaved complete natural original-retail Wine Stage HIT count",
        },
        "totals": {
            "trace_rows_with_frames": len(ordered),
            "physical_hits": len(hits),
            "degraded_horizon_ok_runs": len(degraded_runs),
            "hits_with_final_dead_end": sum(
                hit["final_dead_end"]["rows"] > 0 for hit in hits
            ),
            "hits_with_narrow_ok_row": sum(hit["narrow_ok_rows"] > 0 for hit in hits),
            "hits_with_medium_ok_row": sum(hit["medium_ok_rows"] > 0 for hit in hits),
            "hits_with_degraded_horizon_ok_row": sum(
                hit["degraded_horizon_ok_rows"] > 0 for hit in hits
            ),
            "hits_with_stale_retry": sum(hit["stale_retry_rows"] > 0 for hit in hits),
            "hits_with_input_lease": sum(hit["input_lease_rows"] > 0 for hit in hits),
            "final_dead_end_lead_frames": {
                "minimum": min(final_leads) if final_leads else None,
                "median": float(statistics.median(final_leads)) if final_leads else None,
                "maximum": max(final_leads) if final_leads else None,
            },
            "last_ok_hard_count_before_final_dead_end": {
                "minimum": min(last_ok_widths) if last_ok_widths else None,
                "median": float(statistics.median(last_ok_widths)) if last_ok_widths else None,
                "maximum": max(last_ok_widths) if last_ok_widths else None,
                "histogram": dict(sorted(Counter(last_ok_widths).items())),
            },
            "first_degraded_horizon_lead_frames": {
                "minimum": min(first_degraded_leads) if first_degraded_leads else None,
                "median": float(statistics.median(first_degraded_leads))
                if first_degraded_leads
                else None,
                "maximum": max(first_degraded_leads) if first_degraded_leads else None,
            },
            "final_contiguous_warning_lead_frames": {
                "minimum": min(final_warning_leads) if final_warning_leads else None,
                "median": float(statistics.median(final_warning_leads))
                if final_warning_leads
                else None,
                "maximum": max(final_warning_leads) if final_warning_leads else None,
            },
            "hit_context_counts": dict(sorted(context_counts.items())),
        },
        "collection_trigger_diagnostics": trigger_diagnostics,
        "hits": hits,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--window-frames", type=int, default=120)
    parser.add_argument("--narrow-max", type=int, default=4)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = audit_hit_windows(
        load_trace(args.trace),
        window_frames=args.window_frames,
        narrow_max=args.narrow_max,
    )
    result["trace"] = {
        "path": str(args.trace),
        "sha256": _sha256(args.trace),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result["totals"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
