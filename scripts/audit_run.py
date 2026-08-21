#!/usr/bin/env python3
"""Audit factual Wine episodes and the observed-hazard shield boundary."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import heapq
import json
import math
from pathlib import Path
import struct

from th06_rl.corpus import expand_compact
from th06_rl.episode_dataset import EpisodeDatasetError, validate_episode
from th06_rl.retail.hazards.enemies import future_boxes
from th06_rl.retail.hazards.lasers import future_hazards, signed_laser_clearance
from th06_rl.retail.model import action_from_input
from th06_rl.th06.control_capture import OFFLINE_FACT_SCHEMA, decode_control_snapshot
from th06_rl.th06.observed_bullets import hazard_box


_MAX_COUNTEREXAMPLES = 64


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


def _decode_frame(row: dict, objects: dict[str, object]):
    raw = expand_compact(row["snapshot"], objects)
    return decode_control_snapshot(raw)


def _overlaps_player(snapshot, box) -> bool:
    left, top, right, bottom = box
    return not (
        right < snapshot.x - snapshot.half_width
        or left > snapshot.x + snapshot.half_width
        or bottom < snapshot.y - snapshot.half_height
        or top > snapshot.y + snapshot.half_height
    )


def _collision_evidence(before, after, elapsed: int) -> dict[str, object]:
    """Classify a HIT using only hazards factual at the prior root."""
    before_bullet_slots = {item.slot for item in before.bullets}
    after_overlaps = [
        item.slot
        for item in after.bullets
        if item.state == 1
        and _overlaps_player(
            after,
            (
                item.x - item.half_width,
                item.y - item.half_height,
                item.x + item.half_width,
                item.y + item.half_height,
            ),
        )
    ]
    projected_bullets = [
        item.slot
        for item in before.bullets
        if _overlaps_player(after, hazard_box(item, max(1, elapsed)))
    ]
    projected_enemies = []
    for index, enemy in enumerate(before.enemies):
        boxes = future_boxes(enemy, max(1, elapsed))
        if boxes and _overlaps_player(after, boxes[-1]):
            projected_enemies.append(index)
    after_enemy_overlaps = [
        index
        for index, enemy in enumerate(after.enemies)
        if _overlaps_player(
            after,
            (
                enemy.x - enemy.half_width,
                enemy.y - enemy.half_height,
                enemy.x + enemy.half_width,
                enemy.y + enemy.half_height,
            ),
        )
    ]
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
            )
            <= 0.0
            for hazard in frames[-1]
        ):
            projected_lasers.append(laser.slot)
    return {
        "after_overlapping_bullet_slots": after_overlaps,
        "new_after_overlapping_bullet_slots": [
            slot for slot in after_overlaps if slot not in before_bullet_slots
        ],
        "projected_observed_bullet_slots": projected_bullets,
        "projected_enemy_indices": projected_enemies,
        "after_overlapping_enemy_indices": after_enemy_overlaps,
        "new_after_overlapping_enemy_indices": (
            after_enemy_overlaps if not before.enemies else []
        ),
        "projected_laser_slots": projected_lasers,
    }


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _f32_bits(value: float) -> int:
    return struct.unpack("<I", struct.pack("<f", value))[0]


def _new_player_successor_parity() -> dict[str, object]:
    return {
        "candidate_links": 0,
        "checked_links": 0,
        "bit_exact_links": 0,
        "mismatches": 0,
        "max_axis_error": 0.0,
        "skipped": {
            "observation_gap": 0,
            "stage_boundary": 0,
            "player_not_active": 0,
            "time_stopped": 0,
        },
        "counterexamples": [],
    }


def _measure_player_successor(before, after, sequence: int, parity) -> None:
    """Verify sampled input against the next factual player center."""
    parity["candidate_links"] += 1
    elapsed = int(after.frame) - int(before.frame)
    skipped = parity["skipped"]
    if elapsed != 1:
        skipped["observation_gap"] += 1
        return
    if before.stage != after.stage:
        skipped["stage_boundary"] += 1
        return
    if before.player_state not in (0, 3):
        skipped["player_not_active"] += 1
        return
    if after.time_stopped:
        skipped["time_stopped"] += 1
        return
    sampled = action_from_input(after.input_mask)
    diagonal = sampled.dx != 0 and sampled.dy != 0
    speed = (
        before.focus_diagonal_speed if diagonal else before.focus_speed
    ) if sampled.focused else (
        before.normal_diagonal_speed if diagonal else before.normal_speed
    )
    expected_x = min(
        376.0,
        max(8.0, _f32(_f32(before.x) + _f32(sampled.dx * speed))),
    )
    expected_y = min(
        432.0,
        max(16.0, _f32(_f32(before.y) + _f32(sampled.dy * speed))),
    )
    actual_x = _f32(after.x)
    actual_y = _f32(after.y)
    axis_error = max(abs(expected_x - actual_x), abs(expected_y - actual_y))
    parity["checked_links"] += 1
    parity["max_axis_error"] = max(parity["max_axis_error"], axis_error)
    if (
        _f32_bits(expected_x) == _f32_bits(actual_x)
        and _f32_bits(expected_y) == _f32_bits(actual_y)
    ):
        parity["bit_exact_links"] += 1
        return
    parity["mismatches"] += 1
    if len(parity["counterexamples"]) < _MAX_COUNTEREXAMPLES:
        parity["counterexamples"].append(
            {
                "sequence": sequence,
                "frame": int(before.frame),
                "next_frame": int(after.frame),
                "sampled_action": sampled.name,
                "before": [before.x, before.y],
                "expected": [expected_x, expected_y],
                "actual": [actual_x, actual_y],
                "axis_error": axis_error,
            }
        )


def _audit_player_successors(run_dir, manifest, objects) -> dict[str, object]:
    parity = _new_player_successor_parity()
    previous = None
    previous_sequence = -1
    for row in _rows(_stream_paths(run_dir, manifest, "frames")):
        current = _decode_frame(row, objects)
        if previous is not None:
            _measure_player_successor(
                previous,
                current,
                previous_sequence,
                parity,
            )
        previous = current
        previous_sequence = int(row["sequence"])
    return {
        "method": "contiguous-player-center-successor-v1",
        "arithmetic_comparison": "float32-bit-exact",
        "input_semantics": "next-completed-root-sampled-input",
        **parity,
    }


def _audit_dense_shield_parity(
    run_dir: Path,
    manifest: dict,
    objects: dict[str, object],
    native_library: Path | None,
) -> dict[str, object] | None:
    if native_library is None:
        return None
    samples = (manifest.get("summary") or {}).get("dense_frame_samples", ())
    sequences = {int(item["sequence"]) for item in samples}
    sample_source = "manifest-dense-samples"
    if sequences:
        rows = {
            int(row["sequence"]): row
            for row in _rows(
                _selected_paths(run_dir, manifest, "frames", sequences)
            )
            if int(row["sequence"]) in sequences
        }
    else:
        sample_source = "streaming-densest-64"
        densest = []
        for row in _rows(_stream_paths(run_dir, manifest, "frames")):
            count = int(row["snapshot"].get("live_bullet_count", 0))
            entry = (count, int(row["sequence"]), row)
            if len(densest) < 64:
                heapq.heappush(densest, entry)
            else:
                heapq.heappushpop(densest, entry)
        rows = {sequence: row for _count, sequence, row in densest}
        sequences = set(rows)
    from th06_rl.native import Aabb, LaserRect, NativeKernel, PackedHazards
    from th06_rl.th06.source import core_action_from_input, kinematics_from_snapshot

    kernel = NativeKernel(native_library)
    unsafe = []
    conservative = []
    checked = 0
    skipped = 0
    for sequence in sorted(sequences):
        row = rows.get(sequence)
        if row is None:
            unsafe.append({"sequence": sequence, "reason": "frame-row-missing"})
            continue
        snapshot = _decode_frame(row, objects)
        if snapshot.player_state not in (0, 3) or snapshot.in_menu:
            continue
        decision = row.get("decision") or {}
        if decision.get("shield_contract") != "observed-hazard-kinematics-v1":
            skipped += 1
            continue
        aabb_frames = decision.get("shield_aabb_frames", ())
        laser_frames = decision.get("shield_laser_frames", ())
        margin = decision.get("shield_collision_margin")
        if margin != 0.35 or len(aabb_frames) < 4 or len(laser_frames) < 4:
            unsafe.append(
                {"sequence": sequence, "reason": "shield-primitives-incomplete"}
            )
            continue
        hazards = PackedHazards(
            tuple(
                tuple(Aabb(*map(float, hazard)) for hazard in frame)
                for frame in aabb_frames[:4]
            ),
            tuple(
                tuple(LaserRect(*map(float, hazard)) for hazard in frame)
                for frame in laser_frames[:4]
            ),
        )
        replayed = {
            item.action.name
            for item in kernel.certify_actions(
                x=snapshot.x,
                y=snapshot.y,
                half_width=snapshot.half_width,
                half_height=snapshot.half_height,
                kinematics=kinematics_from_snapshot(snapshot),
                current_action=core_action_from_input(snapshot.input_mask),
                hazards=hazards,
                collision_margin=0.35,
            )
        }
        recorded = {
            str(item[0]) for item in decision.get("shield_actions", ())
        }
        if recorded - replayed:
            unsafe.append(
                {"sequence": sequence, "extra": sorted(recorded - replayed)}
            )
        if replayed - recorded:
            conservative.append(
                {"sequence": sequence, "missing": sorted(replayed - recorded)}
            )
        checked += 1
    return {
        "method": "stored-observed-primitives-native-replay-v1",
        "sample_source": sample_source,
        "checked": checked,
        "required_collision_margin": 0.35,
        "uncommitted_samples_skipped": skipped,
        "unsafe_divergences": unsafe,
        "conservative_divergences": conservative,
    }


def audit(
    run_dir: Path,
    *,
    native_library: Path | None = None,
) -> dict[str, object]:
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
    wanted = hit_sequences | {sequence + 1 for sequence in hit_sequences}
    frame_rows = {
        int(row["sequence"]): row
        for row in _rows(_selected_paths(run_dir, manifest, "frames", wanted))
        if int(row["sequence"]) in wanted
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
            findings.append(
                {"sequence": sequence, "classification": "corpus-linkage-missing"}
            )
            continue
        before = _decode_frame(before_row, objects)
        after = _decode_frame(after_row, objects)
        elapsed = int(transition["outcome_terms"]["elapsed_frames"])
        evidence = _collision_evidence(before, after, elapsed)
        decision = before_row.get("decision") or {}
        shield_actions = {
            str(item[0]) for item in decision.get("shield_actions", ())
        }
        reason = str(decision.get("reason", ""))
        published = decision.get("published_action")
        if elapsed != 1:
            classification = "observation-gap"
        elif reason.startswith("control-dead-end:") or not shield_actions:
            classification = "observed-shield-empty"
        elif published is None:
            classification = "action-not-published"
        elif (
            evidence["new_after_overlapping_bullet_slots"]
            or evidence["new_after_overlapping_enemy_indices"]
        ):
            classification = "future-unobserved-hazard"
        elif (
            evidence["projected_observed_bullet_slots"]
            or evidence["projected_enemy_indices"]
            or evidence["projected_laser_slots"]
        ):
            classification = "observed-shield-counterexample-candidate"
        else:
            classification = "policy-outcome"
        findings.append(
            {
                "sequence": sequence,
                "frame_before": before.frame,
                "frame_after": after.frame,
                "classification": classification,
                "decision_reason": reason,
                "published_action": published,
                "shield_actions": sorted(shield_actions),
                "elapsed_frames": elapsed,
                "collision_evidence": evidence,
            }
        )

    summary = manifest.get("summary") or {}
    expected = run.get("metadata") or {}
    online_contract = expected.get("online_contract") or {}
    episode_unit = str(expected.get("episode_unit", "practice-stage"))
    raw_stages = expected.get("expected_stages") or [expected.get("stage")]
    if (
        episode_unit not in {"practice-stage", "route"}
        or not isinstance(raw_stages, list)
        or not raw_stages
        or any(not isinstance(stage, int) or not 1 <= stage <= 6 for stage in raw_stages)
    ):
        raise ValueError("physical episode scope metadata is invalid")
    expected_stages = tuple(raw_stages)
    scope_prefixes = tuple(
        "/".join(
            str(value)
            for value in (
                expected["difficulty"],
                expected["character"],
                expected["shot_type"],
                stage,
            )
        )
        + "/"
        for stage in expected_stages
    )
    phase_rows = summary.get("phases", ())
    scope_pollution = [
        item.get("scope")
        for item in phase_rows
        if not str(item.get("scope", "")).startswith(scope_prefixes)
    ]
    observed_stages = {
        int(str(item.get("scope", "")).split("/", 4)[3])
        for item in phase_rows
        if str(item.get("scope", "")).startswith(scope_prefixes)
    }
    outcome = manifest.get("run_outcome") or {}
    bomb_events = sum(row.get("event") == "bomb-used" for row in events)
    policy_fallback_sequences = [
        int(row["sequence"])
        for row in _rows(_stream_paths(run_dir, manifest, "frames"))
        if isinstance(row.get("decision"), dict)
        and row["decision"].get("policy_id") == "reactive-baseline-policy-error"
    ]
    raw_policy_failures = outcome.get("policy_failures")
    policy_last_error = outcome.get("policy_last_error")
    policy_contract_valid = (
        isinstance(raw_policy_failures, int)
        and not isinstance(raw_policy_failures, bool)
        and raw_policy_failures >= 0
        and (
            (raw_policy_failures == 0 and policy_last_error is None)
            or (
                raw_policy_failures > 0
                and isinstance(policy_last_error, str)
                and bool(policy_last_error)
            )
        )
    )
    frame_records = int((manifest.get("records") or {}).get("frames", 0))
    dataset_admission = {
        "checked_frames": 0,
        "checked_transitions": 0,
        "passes": False,
        "error": None,
    }
    if frame_records:
        try:
            counts = validate_episode(run_dir)
            dataset_admission["checked_frames"] = counts["frames"]
            dataset_admission["checked_transitions"] = counts["transitions"]
            dataset_admission["passes"] = True
        except (EpisodeDatasetError, OSError, ValueError) as error:
            dataset_admission["error"] = str(error)
    else:
        dataset_admission["error"] = "episode contains no factual frames"

    shield_parity = _audit_dense_shield_parity(
        run_dir, manifest, objects, native_library
    )
    player_parity = _audit_player_successors(run_dir, manifest, objects)
    integrity_errors = []
    if not (
        online_contract.get("algorithm") == "observed-shield4-paused-publication-v1"
        and online_contract.get("shield_contract") == "observed-hazard-kinematics-v1"
        and online_contract.get("publication_epoch")
        == "coherent-root-process-suspended-v1"
        and online_contract.get("shield_horizon") == 4
        and online_contract.get("predicts_future_births") is False
    ):
        integrity_errors.append("observed-shield-contract-invalid")
    if online_contract.get("factual_state_schema") != OFFLINE_FACT_SCHEMA:
        integrity_errors.append("offline-factual-state-incomplete")
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
    if episode_unit == "route" and observed_stages != set(expected_stages):
        integrity_errors.append("route-stage-coverage")
    if outcome.get("physical_hits") != len(hit_sequences):
        integrity_errors.append("physical-hit-count-mismatch")
    if not policy_contract_valid:
        integrity_errors.append("policy-failure-evidence-invalid")
    elif raw_policy_failures:
        integrity_errors.append("policy-callback-failure")
    if (
        policy_contract_valid
        and raw_policy_failures != len(policy_fallback_sequences)
    ):
        integrity_errors.append("policy-failure-conservation")
    if frame_records and not dataset_admission["passes"]:
        integrity_errors.append("episode-dataset-admission-failed")
    if shield_parity and shield_parity["unsafe_divergences"]:
        integrity_errors.append("native-shield-replay-divergence")
    if frame_records > 1 and not player_parity["checked_links"]:
        integrity_errors.append("player-successor-parity-unavailable")
    if player_parity["mismatches"]:
        integrity_errors.append("player-successor-parity-counterexample")

    categories: dict[str, int] = {}
    for finding in findings:
        category = str(finding["classification"])
        categories[category] = categories.get(category, 0) + 1
    return {
        "schema_version": "th06-rl-infra-audit-v2",
        "run_id": manifest.get("run_id"),
        "scope": {
            **{
                key: expected[key]
                for key in ("difficulty", "character", "shot_type", "stage")
            },
            "episode_unit": episode_unit,
            "expected_stages": list(expected_stages),
            "observed_stages": sorted(observed_stages),
        },
        "stage_completed": manifest.get("stage_trajectory_complete"),
        "physical_hits": len(hit_sequences),
        "bomb_events": bomb_events,
        "hit_classifications": categories,
        "integrity_errors": integrity_errors,
        "scope_pollution": scope_pollution,
        "dense_shield_parity": shield_parity,
        "player_successor_parity": player_parity,
        "policy_callback_failures": {
            "count": raw_policy_failures if policy_contract_valid else None,
            "last_error": policy_last_error,
            "fallback_frames": len(policy_fallback_sequences),
            "fallback_sequences": policy_fallback_sequences[:128],
            "fallback_sequences_truncated": max(
                0, len(policy_fallback_sequences) - 128
            ),
            "conserved": bool(
                policy_contract_valid
                and raw_policy_failures == len(policy_fallback_sequences)
            ),
        },
        "episode_dataset_admission": dataset_admission,
        "latency": {
            "capture": summary.get("capture_timing"),
            "solve": summary.get("solve_timing"),
            "stale_retry_rate": summary.get("stale_retry_rate"),
            "observation_gap_rate": summary.get("observation_gap_rate"),
            "capture_over_frame_budget_rate": summary.get(
                "capture_over_frame_budget_rate"
            ),
        },
        # HIT classifications are gameplay outcomes, not infrastructure errors.
        "infra_stable_for_learning": not integrity_errors,
        "findings": findings,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--native-library", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir, native_library=args.native_library)
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return 0 if not report["integrity_errors"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
