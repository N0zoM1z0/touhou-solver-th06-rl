#!/usr/bin/env python3
"""Classify post-Stage TH06 infra failures without tuning route policy."""

from __future__ import annotations

import argparse
import gzip
import heapq
import json
import math
from pathlib import Path

from th06_rl.corpus import expand_compact
from th06_rl.retail.barrage_lab.corpus import decode_snapshot
from th06_rl.retail.hazards.enemies import future_boxes
from th06_rl.retail.hazards.lasers import (
    LaserHazard,
    future_hazards,
    signed_laser_clearance,
)
from th06_rl.th06.control_capture import decode_control_snapshot
from th06_rl.th06.source_dataset import SourceDatasetError, iter_source_frames
from th06_rl.th06.observed_bullets import hazard_box


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
    after_enemy_overlaps = [
        index
        for index, enemy in enumerate(after.enemies)
        if _overlaps_player(after, (
            enemy.x - enemy.half_width,
            enemy.y - enemy.half_height,
            enemy.x + enemy.half_width,
            enemy.y + enemy.half_height,
        ))
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
        "after_overlapping_enemy_indices": after_enemy_overlaps,
        "new_after_overlapping_enemy_indices": (
            after_enemy_overlaps if not before.enemies else []
        ),
        "projected_laser_slots": projected_lasers,
    }


def _decode_frame(row: dict, objects: dict[str, object]):
    raw = expand_compact(row["snapshot"], objects)
    if not str(raw.get("capture_tier", "")).startswith("control-v"):
        return decode_snapshot(raw)
    return decode_control_snapshot(raw)


# This tolerance belongs only to the offline comparison between Python double
# arithmetic and source float32 roots. It is not a collision margin and never
# enters online certification.
_SUCCESSOR_FLOAT_TOLERANCE = 1e-3
_MAX_SUCCESSOR_COUNTEREXAMPLES = 128


def _reachable_envelope(snapshot, steps: int, margin: float):
    speed = max(snapshot.normal_speed, snapshot.focus_speed)
    return (
        max(8.0, snapshot.x - speed * steps)
        - snapshot.half_width - margin,
        max(16.0, snapshot.y - speed * steps)
        - snapshot.half_height - margin,
        min(376.0, snapshot.x + speed * steps)
        + snapshot.half_width + margin,
        min(432.0, snapshot.y + speed * steps)
        + snapshot.half_height + margin,
    )


def _aabb_intersects(left, right) -> bool:
    return not (
        left[2] < right[0]
        or left[0] > right[2]
        or left[3] < right[1]
        or left[1] > right[3]
    )


def _aabb_contains(outer, inner, tolerance=_SUCCESSOR_FLOAT_TOLERANCE) -> bool:
    return (
        outer[0] <= inner[0] + tolerance
        and outer[1] <= inner[1] + tolerance
        and outer[2] >= inner[2] - tolerance
        and outer[3] >= inner[3] - tolerance
    )


def _laser_corners(laser) -> tuple[tuple[float, float], ...]:
    half_x = max(0.0, laser.size_x) / 2.0
    half_y = max(0.0, laser.size_y) / 2.0
    cosine = math.cos(laser.angle)
    sine = math.sin(laser.angle)
    result = []
    for local_x in (
        laser.center_offset - half_x,
        laser.center_offset + half_x,
    ):
        for local_y in (-half_y, half_y):
            result.append((
                laser.origin_x + cosine * local_x - sine * local_y,
                laser.origin_y + sine * local_x + cosine * local_y,
            ))
    return tuple(result)


def _laser_aabb(laser) -> tuple[float, float, float, float]:
    corners = _laser_corners(laser)
    return (
        min(point[0] for point in corners),
        min(point[1] for point in corners),
        max(point[0] for point in corners),
        max(point[1] for point in corners),
    )


def _point_in_laser(point, laser, tolerance=_SUCCESSOR_FLOAT_TOLERANCE) -> bool:
    dx = point[0] - laser.origin_x
    dy = point[1] - laser.origin_y
    cosine = math.cos(laser.angle)
    sine = math.sin(laser.angle)
    local_x = cosine * dx + sine * dy
    local_y = cosine * dy - sine * dx
    return (
        laser.center_offset - laser.size_x / 2.0 - tolerance
        <= local_x
        <= laser.center_offset + laser.size_x / 2.0 + tolerance
        and -laser.size_y / 2.0 - tolerance
        <= local_y
        <= laser.size_y / 2.0 + tolerance
    )


def _laser_is_covered(laser, committed_aabbs, committed_lasers) -> bool:
    corners = _laser_corners(laser)
    actual_aabb = _laser_aabb(laser)
    return any(
        _aabb_contains(box, actual_aabb) for box in committed_aabbs
    ) or any(
        all(_point_in_laser(point, candidate) for point in corners)
        for candidate in committed_lasers
    )


def _retained_post_update_laser_hazards(laser):
    """Recover collision geometry from a retained post-BulletManager root.

    The exact source ticks the timer after collision. Natural state
    transitions reset it before that tick. A forced clear and a natural
    state-1 -> state-2 transition with zero hitbox delay are indistinguishable
    in the retained scalar root, so that one case is explicitly unavailable
    rather than guessed.
    """
    if laser.timer <= 0:
        return (), "timer-not-yet-ticked"
    full_length = max(0.0, laser.end_offset - laser.start_offset)
    prior_timer = laser.timer - 1
    prior_timer_float = laser.timer_float - 1.0

    if laser.state == 0:
        if prior_timer < laser.hitbox_start_time:
            return (), "not-collidable"
        if laser.flags & 1:
            size_x = full_length
        else:
            res = min(laser.start_time, 30)
            width_now = (
                prior_timer_float * laser.width / max(1, laser.start_time)
                if laser.start_time - res < prior_timer
                else 1.2
            )
            size_x = width_now / 2.0
    elif laser.state == 1:
        # startTime==0 lasers are born directly in state 1. Otherwise timer 1
        # witnesses the source's state-0 fallthrough and its midpoint bug.
        size_x = (
            full_length
            if laser.start_time == 0 or laser.timer > 1 or laser.flags & 1
            else laser.width / 2.0
        )
    elif laser.state == 2:
        if laser.timer == 1:
            if laser.hitbox_end_delay <= 0:
                return (), "ambiguous-zero-delay-state-transition"
            # The natural transition performed the state-1 full-length test
            # before falling through to the despawn branch.
            size_x = full_length
        else:
            if prior_timer >= laser.hitbox_end_delay:
                return (), "not-collidable"
            if laser.flags & 1:
                size_x = full_length
            else:
                width_now = laser.width
                if laser.despawn_duration > 0:
                    width_now -= (
                        prior_timer_float * laser.width
                        / laser.despawn_duration
                    )
                size_x = max(0.0, width_now / 2.0)
    else:
        return (), "invalid-state"
    return (LaserHazard(
        laser.x,
        laser.y,
        laser.angle,
        (laser.end_offset - laser.start_offset) / 2.0 + laser.start_offset,
        max(0.0, size_x),
        laser.width / 2.0,
    ),), "checked"


def _audit_source_successor_coverage(
    run_dir: Path,
    manifest: dict,
    objects: dict[str, object],
) -> dict[str, object]:
    """Check committed Hard frames against the following factual Wine root.

    This is intentionally a one-sided coverage check. It can disprove a
    source envelope by finding a retained physical hazard outside it; it does
    not call an envelope complete merely because hazards that retired during
    the update are absent from the next root.
    """
    paths = _stream_paths(run_dir, manifest, "frames")
    skipped = {
        "non_control_v4": 0,
        "source_uncommitted": 0,
        "stage_boundary": 0,
        "outside_hard_horizon": 0,
    }
    laser_unavailable: dict[str, int] = {}
    checked_links = 0
    actual_aabbs_checked = 0
    actual_lasers_checked = 0
    uncovered_aabbs = 0
    uncovered_lasers = 0
    counterexamples = []
    previous_row = None
    previous_snapshot = None

    for row in _rows(paths):
        current_snapshot = _decode_frame(row, objects)
        if previous_row is None:
            previous_row = row
            previous_snapshot = current_snapshot
            continue
        assert previous_snapshot is not None
        before = previous_snapshot
        after = current_snapshot
        decision = previous_row.get("decision") or {}
        tier = str(getattr(before, "capture_tier", ""))
        if tier != "control-v4" or str(
            getattr(after, "capture_tier", "")
        ) != "control-v4":
            skipped["non_control_v4"] += 1
        elif decision.get("source_commitment") != "source-complete-hard-v1":
            skipped["source_uncommitted"] += 1
        elif before.stage != after.stage:
            skipped["stage_boundary"] += 1
        else:
            elapsed = int(after.frame) - int(before.frame)
            if not 1 <= elapsed <= 4:
                skipped["outside_hard_horizon"] += 1
            else:
                aabb_frames = decision.get("source_hard_aabb_frames", ())
                laser_frames = decision.get("source_hard_laser_frames", ())
                margin = decision.get("hard_collision_margin")
                if (
                    margin is None
                    or len(aabb_frames) < elapsed
                    or len(laser_frames) < elapsed
                ):
                    if len(counterexamples) < _MAX_SUCCESSOR_COUNTEREXAMPLES:
                        counterexamples.append({
                            "sequence": int(previous_row["sequence"]),
                            "frame": int(before.frame),
                            "elapsed": elapsed,
                            "kind": "committed-primitives-missing",
                        })
                    uncovered_aabbs += 1
                else:
                    committed_aabbs = tuple(
                        tuple(map(float, item))
                        for item in aabb_frames[elapsed - 1]
                    )
                    committed_lasers = tuple(
                        LaserHazard(*map(float, item))
                        for item in laser_frames[elapsed - 1]
                    )
                    reachable = _reachable_envelope(
                        before, elapsed, float(margin)
                    )
                    actual_aabbs = [
                        (
                            f"bullet:{bullet.slot}",
                            (
                                bullet.x - bullet.half_width,
                                bullet.y - bullet.half_height,
                                bullet.x + bullet.half_width,
                                bullet.y + bullet.half_height,
                            ),
                        )
                        for bullet in after.bullets
                        if bullet.state == 1
                    ]
                    actual_aabbs.extend(
                        (
                            f"enemy:{index}",
                            (
                                enemy.x - enemy.half_width,
                                enemy.y - enemy.half_height,
                                enemy.x + enemy.half_width,
                                enemy.y + enemy.half_height,
                            ),
                        )
                        for index, enemy in enumerate(after.enemies)
                    )
                    for identity, actual in actual_aabbs:
                        if not _aabb_intersects(actual, reachable):
                            continue
                        actual_aabbs_checked += 1
                        if any(
                            _aabb_contains(candidate, actual)
                            for candidate in committed_aabbs
                        ):
                            continue
                        uncovered_aabbs += 1
                        if len(counterexamples) < _MAX_SUCCESSOR_COUNTEREXAMPLES:
                            counterexamples.append({
                                "sequence": int(previous_row["sequence"]),
                                "frame": int(before.frame),
                                "next_frame": int(after.frame),
                                "kind": identity,
                                "actual": list(actual),
                                "reachable_envelope": list(reachable),
                            })
                    for laser in after.lasers:
                        hazards, reason = _retained_post_update_laser_hazards(
                            laser
                        )
                        if reason != "checked":
                            laser_unavailable[reason] = (
                                laser_unavailable.get(reason, 0) + 1
                            )
                        for actual in hazards:
                            if not _aabb_intersects(
                                _laser_aabb(actual), reachable
                            ):
                                continue
                            actual_lasers_checked += 1
                            if _laser_is_covered(
                                actual, committed_aabbs, committed_lasers
                            ):
                                continue
                            uncovered_lasers += 1
                            if len(counterexamples) < _MAX_SUCCESSOR_COUNTEREXAMPLES:
                                counterexamples.append({
                                    "sequence": int(previous_row["sequence"]),
                                    "frame": int(before.frame),
                                    "next_frame": int(after.frame),
                                    "kind": f"laser:{laser.slot}",
                                    "actual": [
                                        actual.origin_x,
                                        actual.origin_y,
                                        actual.angle,
                                        actual.center_offset,
                                        actual.size_x,
                                        actual.size_y,
                                    ],
                                    "reachable_envelope": list(reachable),
                                })
                checked_links += 1
        previous_row = row
        previous_snapshot = current_snapshot

    return {
        "method": "retained-next-root-one-sided-coverage-v1",
        "float_comparison_tolerance": _SUCCESSOR_FLOAT_TOLERANCE,
        "checked_links": checked_links,
        "actual_aabbs_checked": actual_aabbs_checked,
        "actual_lasers_checked": actual_lasers_checked,
        "uncovered_aabbs": uncovered_aabbs,
        "uncovered_lasers": uncovered_lasers,
        "skipped_links": skipped,
        "retained_laser_geometry_unavailable": laser_unavailable,
        "counterexamples_truncated": (
            uncovered_aabbs + uncovered_lasers > len(counterexamples)
        ),
        "counterexamples": counterexamples,
    }


def _audit_dense_hard_parity(
    run_dir: Path,
    manifest: dict,
    objects: dict[str, object],
    native_library: Path | None,
) -> dict[str, object] | None:
    if native_library is None:
        return None
    samples = (manifest.get("summary") or {}).get("dense_frame_samples", ())
    sequences = {int(item["sequence"]) for item in samples}
    rows: dict[int, dict]
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
        # Compatibility for the first control-v1 run, which predates compact
        # dense sample indices. Stream once and retain only 64 encoded roots;
        # never materialize the full Stage corpus in RAM.
        sample_source = "streaming-fallback"
        densest: list[tuple[int, int, dict]] = []
        for row in _rows(_stream_paths(run_dir, manifest, "frames")):
            sequence = int(row["sequence"])
            encoded = row["snapshot"].get("live_bullet_count")
            if encoded is None:
                bullets = row["snapshot"].get("bullets", ())
                encoded = (
                    len(bullets.get("rows", ()))
                    if isinstance(bullets, dict)
                    else len(bullets)
                )
            entry = (int(encoded), sequence, row)
            if len(densest) < 64:
                heapq.heappush(densest, entry)
            else:
                heapq.heappushpop(densest, entry)
        rows = {sequence: row for _count, sequence, row in densest}
        sequences = set(rows)
    from th06_rl.native import Aabb, LaserRect, NativeKernel, PackedHazards
    from th06_rl.th06.source import (
        core_action_from_input,
        kinematics_from_snapshot,
        lower_observed_hazards,
    )

    kernel = NativeKernel(native_library)
    unsafe = []
    conservative = []
    checked = 0
    explicit_margins = 0
    legacy_inferred_margins = 0
    for sequence in sorted(sequences):
        row = rows.get(sequence)
        if row is None:
            unsafe.append({
                "sequence": sequence,
                "reason": "dense-frame-row-missing",
            })
            continue
        snapshot = _decode_frame(row, objects)
        if snapshot.player_state not in (0, 3) or snapshot.in_menu:
            continue
        recorded = {
            item[0] for item in row["decision"].get("hard_actions", ())
        }
        recorded_margin = row["decision"].get("hard_collision_margin")
        source_commitment = row["decision"].get("source_commitment")
        committed_hazards = None
        if source_commitment == "source-complete-hard-v1":
            aabb_frames = row["decision"].get("source_hard_aabb_frames", ())
            laser_frames = row["decision"].get("source_hard_laser_frames", ())
            if len(aabb_frames) < 4 or len(laser_frames) < 4:
                unsafe.append({
                    "sequence": sequence,
                    "frame": snapshot.frame,
                    "reason": "source-commitment-primitives-missing",
                })
                continue
            committed_hazards = PackedHazards(
                tuple(
                    tuple(Aabb(*map(float, hazard)) for hazard in frame)
                    for frame in aabb_frames[:4]
                ),
                tuple(
                    tuple(LaserRect(*map(float, hazard)) for hazard in frame)
                    for frame in laser_frames[:4]
                ),
            )

        def replay(margin: float) -> set[str]:
            if committed_hazards is not None:
                hazards = committed_hazards
            else:
                forecast = lower_observed_hazards(
                    snapshot, 4, collision_margin=margin
                )
                hazards = PackedHazards(
                    forecast.hazards.aabb_frames[:4],
                    forecast.hazards.laser_frames[:4],
                )
            certified = kernel.certify_actions(
                x=snapshot.x,
                y=snapshot.y,
                half_width=snapshot.half_width,
                half_height=snapshot.half_height,
                kinematics=kinematics_from_snapshot(snapshot),
                current_action=core_action_from_input(snapshot.input_mask),
                hazards=hazards,
                collision_margin=margin,
            )
            return {item.action.name for item in certified}

        if recorded_margin is not None:
            margin = float(recorded_margin)
            full = replay(margin)
            explicit_margins += 1
        else:
            # control-v1/v2 did not serialize which branch of the documented
            # 0.35 -> 0.0 fallback produced the Hard set. Reproduce both and
            # select only an exact match; never report the margin-0 fallback
            # as an unsafe divergence against the default margin.
            conservative_full = replay(0.35)
            exact_full = replay(0.0)
            if recorded == conservative_full:
                margin, full = 0.35, conservative_full
            elif recorded == exact_full:
                margin, full = 0.0, exact_full
            else:
                margin, full = 0.0, exact_full
            legacy_inferred_margins += 1
        unsafe_extra = sorted(recorded - full)
        missing_safe = sorted(full - recorded)
        if unsafe_extra:
            unsafe.append({
                "sequence": sequence,
                "frame": snapshot.frame,
                "bullets": snapshot.live_bullet_count,
                "collision_margin": margin,
                "recorded_but_not_full_safe": unsafe_extra,
            })
        if missing_safe:
            conservative.append({
                "sequence": sequence,
                "frame": snapshot.frame,
                "collision_margin": margin,
                "full_safe_but_not_recorded": missing_safe,
            })
        checked += 1
    return {
        "sample_source": sample_source,
        "checked": checked,
        "explicit_collision_margins": explicit_margins,
        "legacy_inferred_collision_margins": legacy_inferred_margins,
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
    anchors = list(_rows(_stream_paths(run_dir, manifest, "anchors")))
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
        decision_reason = str(decision.get("reason", ""))
        if elapsed != 1:
            classification = "latency-observation-gap"
        elif (
            transition["outcome_terms"].get("control_dead_end")
            or decision_reason.startswith("control-dead-end:")
            or not hard_actions
        ):
            classification = "hard-safe-set-empty"
        elif decision_reason == "stale-retry":
            classification = "latency-stale-publication"
        elif published is None:
            classification = "action-not-published"
        elif (
            evidence["new_after_overlapping_bullet_slots"]
            or evidence["new_after_overlapping_enemy_indices"]
        ):
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
            "decision_reason": decision_reason,
            "published_action": published,
            "hard_actions": sorted(hard_actions),
            "capture_ms": decision.get("capture_ms"),
            "solve_ms": decision.get("solve_ms"),
            "elapsed_frames": elapsed,
            "collision_evidence": evidence,
        })

    summary = manifest.get("summary") or {}
    dense_hard_parity = _audit_dense_hard_parity(
        run_dir,
        manifest,
        objects,
        native_library,
    )
    source_successor_coverage = _audit_source_successor_coverage(
        run_dir,
        manifest,
        objects,
    )
    categories: dict[str, int] = {}
    for finding in findings:
        key = str(finding["classification"])
        categories[key] = categories.get(key, 0) + 1
    bomb_events = sum(row.get("event") == "bomb-used" for row in events)
    expected = run["metadata"]
    episode_unit = str(expected.get("episode_unit", "practice-stage"))
    raw_stages = expected.get("expected_stages")
    if raw_stages is None:
        raw_stages = [expected.get("stage")]
    if (
        episode_unit not in {"practice-stage", "route"}
        or not isinstance(raw_stages, list)
        or not raw_stages
        or any(not isinstance(stage, int) or not 1 <= stage <= 6 for stage in raw_stages)
    ):
        raise ValueError("physical episode scope metadata is invalid")
    expected_stages = tuple(raw_stages)
    scope_prefixes = tuple(
        "/".join(str(value) for value in (
            expected["difficulty"],
            expected["character"],
            expected["shot_type"],
            stage,
        )) + "/"
        for stage in expected_stages
    )
    scope_pollution = [
        item.get("scope")
        for item in summary.get("phases", ())
        if not str(item.get("scope", "")).startswith(scope_prefixes)
    ]
    observed_stages = {
        int(str(item.get("scope", "")).split("/", 4)[3])
        for item in summary.get("phases", ())
        if str(item.get("scope", "")).startswith(scope_prefixes)
    }
    anchor_stages = {
        int(scope["stage"])
        for row in anchors
        if isinstance((scope := row.get("scope")), dict)
        and isinstance(scope.get("stage"), int)
        and str(scope.get("key", "")).startswith(scope_prefixes)
    }
    missing_anchor_stages = observed_stages - anchor_stages
    integrity_errors = []
    planner = (run.get("metadata") or {}).get("planner") or {}
    if planner.get("source_commitment") != "source-complete-hard-v1":
        integrity_errors.append("source-authority-incomplete")
    if (
        planner.get("factual_state_schema")
        != "th06-1.02h-offline-facts-v2"
    ):
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
    if missing_anchor_stages:
        integrity_errors.append("source-anchor-stage-coverage")
    outcome = manifest.get("run_outcome") or {}
    if outcome.get("physical_hits") != len(hit_sequences):
        integrity_errors.append("physical-hit-count-mismatch")
    if dense_hard_parity and dense_hard_parity["unsafe_divergences"]:
        integrity_errors.append("dense-hard-parity-unsafe-divergence")
    frame_records = int((manifest.get("records") or {}).get("frames", 0))
    source_dataset_admission = {
        "checked_frames": 0,
        "passes": frame_records == 0,
        "error": None,
    }
    if frame_records:
        try:
            source_dataset_admission["checked_frames"] = sum(
                1 for _bundle in iter_source_frames(run_dir)
            )
            source_dataset_admission["passes"] = True
        except (SourceDatasetError, OSError, ValueError) as error:
            source_dataset_admission["error"] = str(error)
            integrity_errors.append("source-dataset-not-self-contained")
    if frame_records > 1 and not source_successor_coverage["checked_links"]:
        integrity_errors.append("source-successor-coverage-unavailable")
    if (
        source_successor_coverage["uncovered_aabbs"]
        or source_successor_coverage["uncovered_lasers"]
    ):
        integrity_errors.append("source-successor-coverage-counterexample")
    counterexamples = categories.get("safety-counterexample-candidate", 0)
    unresolved = categories.get("unresolved-hit-needs-local-trace", 0)
    return {
        "schema_version": "th06-rl-infra-audit-v1",
        "run_id": manifest.get("run_id"),
        "scope": {
            **{key: expected[key] for key in (
                "difficulty", "character", "shot_type", "stage"
            )},
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
        "anchor_records": int(manifest.get("records", {}).get("anchors", 0)),
        "source_anchor_coverage": {
            "anchored_stages": sorted(anchor_stages),
            "missing_observed_stages": sorted(missing_anchor_stages),
        },
        "dense_hard_parity": dense_hard_parity,
        "source_successor_coverage": source_successor_coverage,
        "source_dataset_admission": source_dataset_admission,
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
    parser.add_argument("--native-library", type=Path)
    args = parser.parse_args()
    report = audit(args.run_dir, native_library=args.native_library)
    output = args.output or args.run_dir / "infra-audit.json"
    output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
