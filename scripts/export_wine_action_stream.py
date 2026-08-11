#!/usr/bin/env python3
"""Export one verified retail-Wine first-failure prefix for source replay.

The output recreates the observed Stage RNG state, the pre-controller released
input interval, each factual Bomb-free movement publication, and any exact
dialogue input samples retained by frame-schema v5.  It is a platform/delivery
diagnostic only: unsampled observation gaps remain uncertain, and the portable
source runtime is not the shipped executable.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from th06_rl.offline import ACTION_SET
from th06_rl.wine_risk import (
    FROZEN_INCUMBENT_POLICY_ID,
    FirstFailurePrefix,
    load_first_failure_prefix,
)

try:
    from run_source_platform_differential import (
        ACTION_STREAM_SCHEMA,
        DELIVERY_CONTRACT,
        ActionSegment,
        DialogueInputSegment,
        SourceActionStream,
    )
except ModuleNotFoundError:  # Imported as scripts.export_wine_action_stream.
    from scripts.run_source_platform_differential import (
        ACTION_STREAM_SCHEMA,
        DELIVERY_CONTRACT,
        ActionSegment,
        DialogueInputSegment,
        SourceActionStream,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _frame(reference: object) -> int:
    parts = str(reference).rsplit(":f", 1)
    if len(parts) != 2:
        raise ValueError(f"snapshot reference has no frame: {reference!r}")
    try:
        frame = int(parts[1])
    except ValueError as error:
        raise ValueError(f"snapshot reference has an invalid frame: {reference!r}") from error
    if frame < 1:
        raise ValueError(f"snapshot frame must be positive: {reference!r}")
    return frame


def _verified_stream_rows(
    run_directory: Path,
    manifest: Mapping[str, Any],
    stream: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TypeError("retail manifest shard list is invalid")
    selected = [
        shard for shard in shards
        if isinstance(shard, dict) and shard.get("stream") == stream
    ]
    selected.sort(key=lambda shard: int(shard.get("first_sequence", -1)))
    if not selected:
        raise ValueError(f"retail manifest has no {stream} shards")
    rows: list[dict[str, Any]] = []
    evidence: list[dict[str, Any]] = []
    for shard in selected:
        name = shard.get("path")
        if not isinstance(name, str) or not name or Path(name).name != name:
            raise ValueError(f"unsafe retail {stream} shard path")
        path = run_directory / name
        if not path.is_file():
            raise ValueError(f"retail {stream} shard is absent: {name}")
        actual_sha256 = _sha256(path)
        if actual_sha256 != shard.get("sha256"):
            raise ValueError(f"retail {stream} shard SHA-256 mismatch: {name}")
        if path.stat().st_size != int(shard.get("compressed_bytes", -1)):
            raise ValueError(f"retail {stream} shard byte count mismatch: {name}")
        shard_rows: list[dict[str, Any]] = []
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"invalid retail {stream} JSON at {name}:{line_number}: {error}"
                    ) from error
                if not isinstance(value, dict):
                    raise TypeError(
                        f"retail {stream} row is not an object at {name}:{line_number}"
                    )
                shard_rows.append(value)
        if len(shard_rows) != int(shard.get("records", -1)):
            raise ValueError(f"retail {stream} shard record count mismatch: {name}")
        expected_first = len(rows)
        expected_last = expected_first + len(shard_rows) - 1
        if (
            int(shard.get("first_sequence", -1)) != expected_first
            or int(shard.get("last_sequence", -1)) != expected_last
        ):
            raise ValueError(f"non-contiguous retail {stream} shard: {name}")
        for offset, row in enumerate(shard_rows):
            if row.get("sequence") != expected_first + offset:
                raise ValueError(
                    f"non-contiguous retail {stream} row at sequence "
                    f"{expected_first + offset}"
                )
        rows.extend(shard_rows)
        evidence.append(
            {
                "path": name,
                "sha256": actual_sha256,
                "compressed_bytes": path.stat().st_size,
                "records": len(shard_rows),
            }
        )
    expected_records = manifest.get("records")
    if (
        not isinstance(expected_records, Mapping)
        or int(expected_records.get(stream, -1)) != len(rows)
    ):
        raise ValueError(f"retail {stream} total does not match the manifest")
    return rows, evidence


def _inverse_rng_step(seed: int) -> int:
    if not 0 <= seed <= 0xFFFF:
        raise ValueError("RNG seed must fit u16")
    unrotated = ((seed >> 2) | (seed << 14)) & 0xFFFF
    return ((unrotated + 0x6553) & 0xFFFF) ^ 0x9630


def _forward_rng_step(seed: int) -> int:
    mixed = ((seed ^ 0x9630) - 0x6553) & 0xFFFF
    return ((mixed << 2) | (mixed >> 14)) & 0xFFFF


def recover_stage_rng_seed(observed_seed: int, generation: int) -> int:
    if not 0 <= observed_seed <= 0xFFFF:
        raise ValueError("observed RNG seed must fit u16")
    if not 0 <= generation <= 10_000_000:
        raise ValueError("observed RNG generation is invalid")
    recovered = observed_seed
    for _ in range(generation):
        recovered = _inverse_rng_step(recovered)
    checked = recovered
    for _ in range(generation):
        checked = _forward_rng_step(checked)
    if checked != observed_seed:
        raise AssertionError("RNG inversion did not reproduce the observed seed")
    return recovered


def _movement_action(input_mask: object) -> str:
    if type(input_mask) is not int or not 0 <= input_mask <= 0xFFFF:
        raise ValueError("retail input mask is invalid")
    if input_mask & 0x02:
        raise ValueError("Bomb bit is present in retail replay input")
    if input_mask & ~(0x01 | 0x04 | 0x10 | 0x20 | 0x40 | 0x80):
        raise ValueError(f"unsupported retail battle input bits: 0x{input_mask:04X}")
    if input_mask & 0x10 and input_mask & 0x20:
        raise ValueError("retail input contains contradictory vertical directions")
    if input_mask & 0x40 and input_mask & 0x80:
        raise ValueError("retail input contains contradictory horizontal directions")
    vertical = "up" if input_mask & 0x10 else "down" if input_mask & 0x20 else ""
    horizontal = "left" if input_mask & 0x40 else "right" if input_mask & 0x80 else ""
    core = "_".join(part for part in (vertical, horizontal) if part) or "stay"
    return core if input_mask & 0x04 else core + "_fast"


def _rle(actions: Sequence[str]) -> tuple[ActionSegment, ...]:
    segments: list[ActionSegment] = []
    for action in actions:
        if segments and segments[-1].action == action:
            previous = segments[-1]
            segments[-1] = ActionSegment(previous.count + 1, action)
        else:
            segments.append(ActionSegment(1, action))
    return tuple(segments)


_DIALOGUE_SAMPLE_KEYS = {
    "game_frame",
    "current_input_mask",
    "previous_input_mask",
    "published_input_mask",
    "held_repeat",
    "held_frames",
    "active",
    "skippable",
    "pulsed_shoot",
}
_DIALOGUE_ALLOWED_INPUT = 0x01 | 0x04 | 0x10 | 0x20 | 0x40 | 0x80 | 0x100


def _dialogue_mask(value: object, *, field: str) -> int:
    if type(value) is not int or not 0 <= value <= 0xFFFF:
        raise ValueError(f"retail dialogue {field} is not a u16")
    if value & 0x02 or value & ~_DIALOGUE_ALLOWED_INPUT:
        raise ValueError(f"retail dialogue {field} is forbidden or Bomb-bearing")
    return value


def _dialogue_input_segments(
    frames: Sequence[Mapping[str, Any]],
    *,
    requested_tick: int,
) -> tuple[tuple[DialogueInputSegment, ...], dict[str, int | None]]:
    """Recover only input ticks established by passive retail evidence.

    ``current_input_mask`` establishes the sampled game frame and
    ``previous_input_mask`` establishes the immediately preceding frame.  When
    both are equal, the retail held counter additionally proves that many
    immediately preceding frames had the same mask.  Backfill is clipped at
    the previous coherent battle snapshot, so it never crosses an observed
    state boundary.
    """

    exact: dict[int, int] = {}
    directly_sampled: set[int] = set()
    previous_sampled: set[int] = set()
    held_backfilled: set[int] = set()
    sample_records = 0
    previous_sample_frame = -1

    def retain(frame: int, mask: int, evidence: set[int]) -> None:
        if frame < 1 or frame > requested_tick:
            return
        prior = exact.get(frame)
        if prior is not None and prior != mask:
            raise ValueError(
                f"conflicting retail dialogue input evidence at frame {frame}: "
                f"0x{prior:04X} != 0x{mask:04X}"
            )
        exact[frame] = mask
        evidence.add(frame)

    for index, frame_row in enumerate(frames):
        decision = frame_row.get("decision")
        snapshot = frame_row.get("snapshot")
        if not isinstance(decision, Mapping) or not isinstance(snapshot, Mapping):
            raise TypeError(f"retail frame {index} lacks snapshot/decision evidence")
        samples = decision.get("dialogue_delivery", [])
        if not isinstance(samples, list):
            raise TypeError(f"retail dialogue delivery at frame row {index} is not a list")
        if not samples:
            continue
        carrier_frame = int(snapshot.get("frame", -1))
        previous_coherent_frame = -1
        if index:
            previous_snapshot = frames[index - 1].get("snapshot")
            if not isinstance(previous_snapshot, Mapping):
                raise TypeError(f"retail frame {index - 1} lacks snapshot evidence")
            previous_coherent_frame = int(previous_snapshot.get("frame", -1))
        for sample_index, sample in enumerate(samples):
            if not isinstance(sample, Mapping) or set(sample) != _DIALOGUE_SAMPLE_KEYS:
                raise ValueError(
                    f"retail dialogue sample {index}:{sample_index} has an invalid schema"
                )
            game_frame = sample.get("game_frame")
            held_frames = sample.get("held_frames")
            held_repeat = sample.get("held_repeat")
            if type(game_frame) is not int or game_frame < 1:
                raise ValueError("retail dialogue game_frame is invalid")
            if not previous_coherent_frame <= game_frame <= carrier_frame:
                raise ValueError(
                    f"retail dialogue frame {game_frame} lies outside its carrier gap"
                )
            if game_frame < previous_sample_frame:
                raise ValueError("retail dialogue samples are not globally frame ordered")
            if type(held_frames) is not int or not 0 <= held_frames <= 0xFFFF:
                raise ValueError("retail dialogue held_frames is invalid")
            if type(held_repeat) is not int or not 0 <= held_repeat <= 0xFFFF:
                raise ValueError("retail dialogue held_repeat is invalid")
            for field in ("active", "skippable", "pulsed_shoot"):
                if type(sample.get(field)) is not bool:
                    raise ValueError(f"retail dialogue {field} is not a boolean")
            current = _dialogue_mask(
                sample.get("current_input_mask"), field="current_input_mask"
            )
            previous = _dialogue_mask(
                sample.get("previous_input_mask"), field="previous_input_mask"
            )
            _dialogue_mask(
                sample.get("published_input_mask"), field="published_input_mask"
            )
            sample_records += 1
            retain(game_frame, current, directly_sampled)
            if game_frame - 1 > previous_coherent_frame:
                retain(game_frame - 1, previous, previous_sampled)
            if current == previous:
                start = max(previous_coherent_frame + 1, game_frame - held_frames)
                for inferred_frame in range(start, game_frame):
                    retain(inferred_frame, current, held_backfilled)
            previous_sample_frame = game_frame

    segments: list[DialogueInputSegment] = []
    for retail_frame, mask in sorted(exact.items()):
        # Headless NextInput() is called while runtime ``ticks`` still names
        # the preceding observation. The returned input becomes the coherent
        # input in trace/game frame ``ticks + 1``.
        frame = retail_frame - 1
        if (
            segments
            and frame == segments[-1].start_tick + segments[-1].count
            and mask == segments[-1].input_mask
        ):
            previous = segments[-1]
            segments[-1] = DialogueInputSegment(
                previous.start_tick, previous.count + 1, mask
            )
        else:
            segments.append(DialogueInputSegment(frame, 1, mask))
    return tuple(segments), {
        "sample_records": sample_records,
        "sampled_current_frames": len(directly_sampled),
        "sampled_previous_frames": len(previous_sampled),
        "held_backfilled_frames": len(
            held_backfilled - directly_sampled - previous_sampled
        ),
        "exact_frames": len(exact),
        "segments": len(segments),
        "first_exact_frame": min(exact, default=None),
        "last_exact_frame": max(exact, default=None),
        "first_runtime_input_tick": min(exact) - 1 if exact else None,
        "last_runtime_input_tick": max(exact) - 1 if exact else None,
    }


def build_retail_action_stream(
    prefix: FirstFailurePrefix,
    transitions: Sequence[Mapping[str, Any]],
    frames: Sequence[Mapping[str, Any]],
    *,
    max_source_tick: int | None = None,
    initial_seed: int = 0,
    shard_evidence: Mapping[str, Any] | None = None,
) -> SourceActionStream:
    if not 0 <= initial_seed <= 0xFFFF:
        raise ValueError("initial seed must fit u16")
    if len(transitions) != prefix.transitions or len(frames) != len(transitions) + 1:
        raise ValueError("retail frame/transition counts do not form one complete prefix")
    first_snapshot = frames[0].get("snapshot")
    first_decision = frames[0].get("decision")
    if not isinstance(first_snapshot, Mapping) or not isinstance(first_decision, Mapping):
        raise TypeError("retail prefix lacks its first snapshot/decision")
    first_frame = int(first_snapshot.get("frame", -1))
    if first_frame < 1:
        raise ValueError("retail prefix first frame is invalid")
    if first_snapshot.get("input_mask") != 0:
        raise ValueError("retail replay requires a released first-capture input")
    prelude_action = first_decision.get("current_action")
    if prelude_action != "stay_fast" or _movement_action(first_snapshot["input_mask"]) != prelude_action:
        raise ValueError("retail replay first capture is not released stay_fast")
    observed_rng_seed = int(first_snapshot.get("rng_seed", -1))
    observed_rng_generation = int(first_snapshot.get("rng_generation", -1))
    stage_rng_seed = recover_stage_rng_seed(
        observed_rng_seed,
        observed_rng_generation,
    )
    scope = tuple(
        int(first_snapshot.get(name, -1))
        for name in ("difficulty", "character", "shot_type", "stage")
    )
    if scope != prefix.scope:
        raise ValueError(f"retail first snapshot scope mismatch: {scope} != {prefix.scope}")

    delivery_intervals: list[tuple[int, int, str, str]] = []
    gaps: list[dict[str, int]] = []
    first_publication_frame: int | None = None
    previous_target = first_frame
    for sequence, transition in enumerate(transitions):
        frame_row = frames[sequence]
        next_frame_row = frames[sequence + 1]
        snapshot = frame_row.get("snapshot")
        decision = frame_row.get("decision")
        next_snapshot = next_frame_row.get("snapshot")
        next_decision = next_frame_row.get("decision")
        if (
            not isinstance(snapshot, Mapping)
            or not isinstance(decision, Mapping)
            or not isinstance(next_snapshot, Mapping)
            or not isinstance(next_decision, Mapping)
        ):
            raise TypeError(f"retail replay row {sequence} lacks snapshot/decision evidence")
        source_frame = _frame(transition.get("snapshot_ref"))
        target_frame = _frame(transition.get("next_snapshot_ref"))
        outcome = transition.get("outcome_terms")
        if not isinstance(outcome, Mapping):
            raise TypeError(f"retail replay transition {sequence} lacks outcome terms")
        elapsed = int(outcome.get("elapsed_frames", -1))
        if (
            transition.get("sequence") != sequence
            or frame_row.get("sequence") != sequence
            or next_frame_row.get("sequence") != sequence + 1
            or source_frame != previous_target
            or int(snapshot.get("frame", -1)) != source_frame
            or int(next_snapshot.get("frame", -1)) != target_frame
            or target_frame - source_frame != elapsed
            or elapsed < 1
        ):
            raise ValueError(f"retail replay linkage is incoherent at sequence {sequence}")
        if outcome.get("bomb_used") is not False:
            raise ValueError(f"Bomb-free outcome is not established at sequence {sequence}")
        current_action = decision.get("current_action")
        if not isinstance(current_action, str) or current_action not in ACTION_SET:
            raise ValueError(f"retail current action is invalid at sequence {sequence}")
        if _movement_action(snapshot.get("input_mask")) != current_action:
            raise ValueError(f"retail current action/input disagree at sequence {sequence}")
        next_current_action = next_decision.get("current_action")
        if (
            not isinstance(next_current_action, str)
            or next_current_action not in ACTION_SET
            or _movement_action(next_snapshot.get("input_mask")) != next_current_action
        ):
            raise ValueError(
                f"retail next current action/input disagree at sequence {sequence}"
            )
        published = transition.get("published_action")
        decision_published = decision.get("published_action")
        if published is None:
            if decision_published is not None:
                raise ValueError(f"retail publication evidence disagrees at sequence {sequence}")
            unobserved_action = current_action
        else:
            if (
                not isinstance(published, str)
                or published not in ACTION_SET
                or decision_published != published
            ):
                raise ValueError(f"retail published action is invalid at sequence {sequence}")
            unobserved_action = published
            if first_publication_frame is None:
                first_publication_frame = source_frame
        if transition.get("proposed_action") not in ACTION_SET:
            raise ValueError(f"retail proposed action is invalid at sequence {sequence}")
        if elapsed != 1:
            gaps.append({"sequence": sequence, "source_frame": source_frame, "frames": elapsed})
        # A publication is an intent, not evidence that the game sampled the
        # new bridge mask on the next frame. The target coherent snapshot is
        # authoritative for its own tick. Only unobserved interior ticks use
        # the best available publication/current-action inference.
        delivery_intervals.append(
            (source_frame, target_frame, unobserved_action, next_current_action)
        )
        previous_target = target_frame

    terminal_frame = delivery_intervals[-1][1]
    if first_publication_frame is None:
        raise ValueError("retail replay prefix has no successful battle publication")
    if terminal_frame != prefix.failure_frame:
        raise ValueError("retail replay terminal frame does not match the first failure")
    requested_tick = terminal_frame if max_source_tick is None else max_source_tick
    if not first_frame <= requested_tick <= terminal_frame:
        raise ValueError(
            f"max source tick must be in {first_frame}..{terminal_frame}"
        )

    # The runtime first asks for input while producing trace tick 2. Therefore
    # max_source_tick - 1 delivered actions exactly cover trace tick 2 through
    # max_source_tick; one final padding action satisfies the conservative
    # action-stream coverage contract and is never consumed.
    actions = [prelude_action] * (first_frame - 1)
    for (
        source_frame,
        target_frame,
        unobserved_action,
        target_action,
    ) in delivery_intervals:
        if source_frame >= requested_tick:
            break
        covered_target = min(target_frame, requested_tick)
        covered_frames = covered_target - source_frame
        if covered_target == target_frame:
            actions.extend([unobserved_action] * (covered_frames - 1))
            actions.append(target_action)
        else:
            actions.extend([unobserved_action] * covered_frames)
        if target_frame >= requested_tick:
            break
    if len(actions) != requested_tick - 1:
        raise AssertionError(
            f"retail replay covers {len(actions)} actions for tick {requested_tick}"
        )
    actions.append(actions[-1])

    included_gaps = [gap for gap in gaps if gap["source_frame"] < requested_tick]
    dialogue_gaps = [gap for gap in included_gaps if gap["frames"] >= 8]
    dialogue_control_after_tick = (
        max(dialogue_gaps, key=lambda gap: gap["frames"])["source_frame"]
        if dialogue_gaps
        else requested_tick
    )
    dialogue_inputs, dialogue_evidence = _dialogue_input_segments(
        frames,
        requested_tick=requested_tick,
    )
    if dialogue_inputs:
        known_limit = (
            "Observed battle snapshots and retained dialogue current/previous/held "
            "ticks are exact. Other unobserved interior battle inputs remain inferred; "
            "source rendering, numeric, and RNG paths remain non-retail."
        )
    else:
        known_limit = (
            "Observed battle input is exact, but interior actions and dialogue "
            "release/tap edges inside capture gaps are absent from this corpus."
        )
    return SourceActionStream(
        difficulty=prefix.scope[0],
        character=prefix.scope[1],
        shot_type=prefix.scope[2],
        stage=prefix.scope[3],
        initial_seed=initial_seed,
        stage_rng_seed=stage_rng_seed,
        max_ticks=requested_tick,
        auto_shoot=True,
        auto_shoot_after_tick=first_publication_frame,
        retail_dialogue_control=True,
        retail_dialogue_control_after_tick=dialogue_control_after_tick,
        retail_dialogue_inputs=dialogue_inputs,
        segments=_rle(actions),
        description=(
            "Verified original-retail Wine movement prefix for source platform/delivery "
            "differential; never promotion or training evidence"
        ),
        provenance={
            "kind": "verified-original-retail-wine-first-failure-action-prefix",
            "run_id": prefix.run_id,
            "run_directory": str(prefix.run_dir),
            "manifest_sha256": prefix.manifest_sha256,
            "run_sha256": prefix.run_sha256,
            "retail_executable_sha256": prefix.executable_sha256,
            "native_kernel_sha256": prefix.native_kernel_sha256,
            "controller_code_commit": prefix.code_commit,
            "failure_kind": prefix.failure_kind,
            "first_retail_frame": first_frame,
            "last_retail_frame": requested_tick,
            "observed_rng_seed": observed_rng_seed,
            "observed_rng_generation": observed_rng_generation,
            "recovered_stage_rng_seed": stage_rng_seed,
            "prelude_actions": first_frame - 1,
            "first_battle_publication_frame": first_publication_frame,
            "delivered_actions": requested_tick - 1,
            "coverage_padding_actions": 1,
            "delivery_reconstruction": (
                "coherent target snapshot action at every observed frame; successful "
                "publication or current action only for unobserved interior frames"
            ),
            "observation_gaps": included_gaps,
            "maximum_observation_gap": max(
                (gap["frames"] for gap in included_gaps), default=1
            ),
            "retail_dialogue_control_after_tick": dialogue_control_after_tick,
            "retail_dialogue_control_selection": (
                "source frame of the first maximum gap among observed gaps of at least "
                "eight frames; source GUI state remains the runtime dialogue gate"
            ),
            "retail_dialogue_input_evidence": dialogue_evidence,
            "known_limit": known_limit,
            "shards": dict(shard_evidence or {}),
        },
    )


def export_wine_action_stream(
    run_directory: Path,
    *,
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
    expected_policy_id: str = FROZEN_INCUMBENT_POLICY_ID,
    max_source_tick: int | None = None,
    initial_seed: int = 0,
) -> SourceActionStream:
    run_directory = run_directory.resolve()
    prefix = load_first_failure_prefix(
        run_directory,
        expected_scope=expected_scope,
        expected_executable_sha256=expected_executable_sha256,
        expected_native_kernel_sha256=expected_native_kernel_sha256,
        expected_policy_id=expected_policy_id,
    )
    manifest = _object(run_directory / "manifest.json")
    transitions, transition_evidence = _verified_stream_rows(
        run_directory, manifest, "transitions"
    )
    frames, frame_evidence = _verified_stream_rows(run_directory, manifest, "frames")
    return build_retail_action_stream(
        prefix,
        transitions,
        frames,
        max_source_tick=max_source_tick,
        initial_seed=initial_seed,
        shard_evidence={
            "transitions": transition_evidence,
            "frames": frame_evidence,
        },
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_directory", type=Path)
    parser.add_argument("--expected-executable-sha256", required=True)
    parser.add_argument("--expected-native-kernel-sha256", required=True)
    parser.add_argument(
        "--expected-policy-id",
        default=FROZEN_INCUMBENT_POLICY_ID,
    )
    parser.add_argument("--difficulty", type=int, default=3)
    parser.add_argument("--character", type=int, default=0)
    parser.add_argument("--shot-type", type=int, default=0)
    parser.add_argument("--stage", type=int, default=6)
    parser.add_argument("--max-source-tick", type=int)
    parser.add_argument("--initial-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    try:
        stream = export_wine_action_stream(
            args.run_directory,
            expected_scope=(
                args.difficulty,
                args.character,
                args.shot_type,
                args.stage,
            ),
            expected_executable_sha256=args.expected_executable_sha256,
            expected_native_kernel_sha256=args.expected_native_kernel_sha256,
            expected_policy_id=args.expected_policy_id,
            max_source_tick=args.max_source_tick,
            initial_seed=args.initial_seed,
        )
    except (KeyError, OSError, TypeError, ValueError) as error:
        parser.error(str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(stream.as_object(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "schema": ACTION_STREAM_SCHEMA,
                "delivery_contract": DELIVERY_CONTRACT,
                "output": str(args.output.resolve()),
                "actions": stream.action_count,
                "segments": len(stream.segments),
                "stage_rng_seed": stream.stage_rng_seed,
                "auto_shoot_after_tick": stream.auto_shoot_after_tick,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
