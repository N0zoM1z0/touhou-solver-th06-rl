#!/usr/bin/env python3
"""Classify post-Stage TH06 infra failures without tuning route policy."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

from th06_rl.corpus import expand_compact
from th06_rl.th06.control_capture import decode_control_snapshot
from th06_rl.th06.donor import enable_donor_imports


def _rows(paths):
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _stream_paths(run_dir: Path, manifest: dict, stream: str) -> list[Path]:
    return [
        run_dir / item["path"]
        for item in manifest.get("shards", ())
        if item.get("stream") == stream
    ]


def _selected_paths(
    run_dir: Path,
    manifest: dict,
    stream: str,
    sequences: set[int],
) -> list[Path]:
    if not sequences:
        return []
    return [
        run_dir / item["path"]
        for item in manifest.get("shards", ())
        if item.get("stream") == stream
        and any(
            int(item["first_sequence"]) <= sequence <= int(item["last_sequence"])
            for sequence in sequences
        )
    ]


def _overlaps_player(snapshot, box) -> bool:
    left, top, right, bottom = box
    return not (
        right < snapshot.x - snapshot.half_width
        or left > snapshot.x + snapshot.half_width
        or bottom < snapshot.y - snapshot.half_height
        or top > snapshot.y + snapshot.half_height
    )


def _collision_evidence(before, after, elapsed: int) -> dict[str, object]:
    enable_donor_imports()
    from th06.hazards.bullets import hazard_box
    from th06.hazards.enemies import future_boxes
    from th06.hazards.lasers import future_hazards, signed_laser_clearance

    before_bullet_slots = {item.slot for item in before.bullets}
    after_overlaps = [
        item.slot
        for item in after.bullets
        if item.state == 1
        and _overlaps_player(after, (
            item.x - item.half_width,
            item.y - item.half_height,
            item.x + item.half_width,
            item.y + item.half_height,
        ))
    ]
    projected_bullets = [
        item.slot
        for item in before.bullets
        if _overlaps_player(after, hazard_box(item, max(1, elapsed)))
    ]
    projected_enemy = []
    for index, enemy in enumerate(before.enemies):
        boxes = future_boxes(enemy, max(1, elapsed))
        if boxes and _overlaps_player(after, boxes[-1]):
            projected_enemy.append(index)
    projected_lasers = []
    for laser in before.lasers:
        frames = future_hazards(laser, max(1, elapsed))
        if frames and any(
            signed_laser_clearance(
                after.x,
                after.y,
                after.half_width,
                after.half_height,
                hazard,
            ) <= 0.0
            for hazard in frames[-1]
        ):
            projected_lasers.append(laser.slot)
    return {
        "after_overlapping_bullet_slots": after_overlaps,
        "new_after_overlapping_bullet_slots": [
            slot for slot in after_overlaps if slot not in before_bullet_slots
        ],
        "projected_observed_bullet_slots": projected_bullets,
        "projected_enemy_indices": projected_enemy,
        "projected_laser_slots": projected_lasers,
    }


def _decode_frame(row: dict, objects: dict[str, object]):
    raw = expand_compact(row["snapshot"], objects)
    if raw.get("capture_tier") != "control-v1":
        enable_donor_imports()
        from th06.barrage_lab.corpus import decode_snapshot

        return decode_snapshot(raw)
    return decode_control_snapshot(raw)


def audit(run_dir: Path) -> dict[str, object]:
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    objects = {
        row["object_id"]: row["payload"]
        for row in _rows(_stream_paths(run_dir, manifest, "objects"))
    }
    events = list(_rows(_stream_paths(run_dir, manifest, "events")))
    hit_sequences = {
        int(row["sequence"])
        for row in events
        if row.get("event") == "life-lost"
    }
    wanted_frames = hit_sequences | {sequence + 1 for sequence in hit_sequences}
    frame_rows = {
        int(row["sequence"]): row
        for row in _rows(
            _selected_paths(run_dir, manifest, "frames", wanted_frames)
        )
        if int(row["sequence"]) in wanted_frames
    }
    transition_rows = {
        int(row["sequence"]): row
        for row in _rows(
            _selected_paths(run_dir, manifest, "transitions", hit_sequences)
        )
        if int(row["sequence"]) in hit_sequences
    }
    findings = []
    for sequence in sorted(hit_sequences):
        before_row = frame_rows.get(sequence)
        after_row = frame_rows.get(sequence + 1)
        transition = transition_rows.get(sequence)
        if before_row is None or after_row is None or transition is None:
            findings.append({
                "sequence": sequence,
                "classification": "corpus-linkage-missing",
            })
            continue
        before = _decode_frame(before_row, objects)
        after = _decode_frame(after_row, objects)
        elapsed = int(transition["outcome_terms"]["elapsed_frames"])
        evidence = _collision_evidence(before, after, elapsed)
        decision = before_row["decision"]
        hard_actions = {item[0] for item in decision.get("hard_actions", ())}
        published = decision.get("published_action")
        if elapsed != 1:
            classification = "latency-observation-gap"
        elif transition["outcome_terms"].get("control_dead_end"):
            classification = "hard-safe-set-empty"
        elif published is None:
            classification = "action-not-published"
        elif evidence["new_after_overlapping_bullet_slots"]:
            classification = "new-hazard-after-observation"
        elif (
            published in hard_actions
            and (
                evidence["projected_observed_bullet_slots"]
                or evidence["projected_enemy_indices"]
                or evidence["projected_laser_slots"]
            )
        ):
            classification = "safety-counterexample-candidate"
        else:
            classification = "unresolved-hit-needs-local-trace"
        findings.append({
            "sequence": sequence,
            "frame_before": before.frame,
            "frame_after": after.frame,
            "source_context": before_row["scope"]["phase_id"],
            "classification": classification,
            "published_action": published,
            "hard_actions": sorted(hard_actions),
            "capture_ms": decision.get("capture_ms"),
            "solve_ms": decision.get("solve_ms"),
            "elapsed_frames": elapsed,
            "collision_evidence": evidence,
        })

    summary = manifest.get("summary") or {}
    categories: dict[str, int] = {}
    for finding in findings:
        key = str(finding["classification"])
        categories[key] = categories.get(key, 0) + 1
    bomb_events = sum(row.get("event") == "bomb-used" for row in events)
    expected = run["metadata"]
    scope_prefix = "/".join(str(expected[key]) for key in (
        "difficulty", "character", "shot_type", "stage"
    )) + "/"
    scope_pollution = [
        item.get("scope")
        for item in summary.get("phases", ())
        if not str(item.get("scope", "")).startswith(scope_prefix)
    ]
    integrity_errors = []
    if manifest.get("dropped_records", 0):
        integrity_errors.append("dropped-records")
    if manifest.get("complete") is not True:
        integrity_errors.append("storage-incomplete")
    if manifest.get("stage_trajectory_complete") is not True:
        integrity_errors.append("stage-trajectory-incomplete")
    if bomb_events:
        integrity_errors.append("bomb-observed")
    if scope_pollution:
        integrity_errors.append("scope-pollution")
    counterexamples = categories.get("safety-counterexample-candidate", 0)
    unresolved = categories.get("unresolved-hit-needs-local-trace", 0)
    return {
        "schema_version": "th06-rl-infra-audit-v1",
        "run_id": manifest.get("run_id"),
        "scope": {key: expected[key] for key in (
            "difficulty", "character", "shot_type", "stage"
        )},
        "stage_completed": manifest.get("stage_trajectory_complete"),
        "physical_hits": len(hit_sequences),
        "hit_classifications": categories,
        "integrity_errors": integrity_errors,
        "scope_pollution": scope_pollution,
        "anchor_records": int(manifest.get("records", {}).get("anchors", 0)),
        "latency": {
            "capture": summary.get("capture_timing"),
            "solve": summary.get("solve_timing"),
            "stale_retry_rate": summary.get("stale_retry_rate"),
            "observation_gap_rate": summary.get("observation_gap_rate"),
            "capture_over_frame_budget_rate": summary.get(
                "capture_over_frame_budget_rate"
            ),
        },
        "infra_stable_for_learning": bool(
            not integrity_errors and not counterexamples and not unresolved
        ),
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir)
    output = args.output or args.run_dir / "infra-audit.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
