"""Algorithm-neutral access to source-authoritative control-v5 corpus rows.

This loader deliberately returns the dense factual root and its active
immutable source anchor separately.  It validates that no live ECL/timeline
pointer or visual Enemy geometry depends on the terminated Wine process; it
does not silently substitute the anchor's old dynamic state for the frame's
raw records.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import struct
from typing import Iterator

from ..corpus import expand_compact
from ..retail import native
from ..retail.barrage_lab.corpus import decode_snapshot
from ..retail.model import Snapshot
from .control_capture import (
    CONTROL_CAPTURE_TIER,
    OFFLINE_FACT_SCHEMA,
    SOURCE_RECORD_SCHEMA,
    ControlSnapshot,
    decode_control_snapshot,
)


class SourceDatasetError(RuntimeError):
    """A corpus row cannot be admitted as self-contained source evidence."""


@dataclass(frozen=True)
class SourceFrameBundle:
    sequence: int
    snapshot_id: str
    control: ControlSnapshot
    anchor: Snapshot
    anchor_sequence: int
    anchor_reason: str
    scope: dict[str, object]
    decision: dict[str, object]


def _stream_paths(
    run_dir: Path,
    manifest: dict[str, object],
    stream: str,
) -> tuple[Path, ...]:
    shards = [
        shard
        for shard in manifest.get("shards", ())
        if isinstance(shard, dict) and shard.get("stream") == stream
    ]
    shards.sort(key=lambda shard: int(shard.get("first_sequence", -1)))
    paths = []
    for shard in shards:
        path = (run_dir / str(shard["path"])).resolve()
        if path.parent != run_dir or not path.is_file():
            raise SourceDatasetError(f"invalid {stream} shard path {path}")
        digest = hashlib.sha256()
        compressed_bytes = 0
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
                compressed_bytes += len(chunk)
        if (
            digest.hexdigest() != shard.get("sha256")
            or compressed_bytes != int(shard.get("compressed_bytes", -1))
        ):
            raise SourceDatasetError(f"{stream} shard digest disagrees with manifest")
        paths.append(path)
    return tuple(paths)


def _rows(paths: tuple[Path, ...]) -> Iterator[dict[str, object]]:
    for path in paths:
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise SourceDatasetError(f"non-object row in {path}")
                yield row


def _source_addresses(snapshot: Snapshot) -> frozenset[int]:
    addresses = {
        instruction.address for instruction in snapshot.timeline_instructions
    }
    addresses.update(
        instruction.address for instruction in snapshot.timeline_ecl_program
    )
    for spawner in snapshot.spawners:
        addresses.update(
            instruction.address for instruction in spawner.ecl_program
        )
    return frozenset(addresses)


def _timeline_pointer(control: ControlSnapshot) -> int:
    offset = (
        native.ENEMY_TIMELINE_INSTRUCTION_OFFSET
        - native.ENEMY_ARRAY_OFFSET
        - native.ENEMY_COUNT * native.ENEMY_STRIDE
    )
    return struct.unpack_from("<I", control.raw_enemy_manager_tail, offset)[0]


def _required_enemy_addresses(
    control: ControlSnapshot,
    anchor: Snapshot,
) -> frozenset[int]:
    required: set[int] = set()
    record_size = 2 + native.ENEMY_STRIDE
    subroutines = anchor.ecl_subroutines
    for record_offset in range(0, len(control.raw_enemy_records), record_size):
        base = record_offset + 2
        slot = struct.unpack_from(
            "<H", control.raw_enemy_records, record_offset
        )[0]

        def context_address(relative: int) -> None:
            address = struct.unpack_from(
                "<I", control.raw_enemy_records, base + relative
            )[0]
            if address:
                required.add(address)

        context_address(native.ENEMY_ECL_CONTEXT_OFFSET)
        stack_depth = struct.unpack_from(
            "<i",
            control.raw_enemy_records,
            base + native.ENEMY_ECL_STACK_DEPTH_OFFSET,
        )[0]
        if not 0 <= stack_depth <= native.ENEMY_ECL_STACK_CAPACITY:
            raise SourceDatasetError(
                f"frame {control.frame} enemy {slot} has invalid ECL stack depth"
            )
        for stack_index in range(stack_depth):
            context_address(
                native.ENEMY_ECL_CONTEXT_OFFSET
                + native.ENEMY_ECL_CONTEXT_SIZE * (stack_index + 1)
            )

        death_callback = struct.unpack_from(
            "<i",
            control.raw_enemy_records,
            base + native.ENEMY_DEATH_CALLBACK_SUB_OFFSET,
        )[0]
        life_threshold, life_callback, timer_threshold, timer_callback = (
            struct.unpack_from(
                "<iiii",
                control.raw_enemy_records,
                base + native.ENEMY_LIFE_CALLBACK_THRESHOLD_OFFSET,
            )
        )
        interrupts = struct.unpack_from(
            "<" + "i" * 8,
            control.raw_enemy_records,
            base + native.ENEMY_INTERRUPTS_OFFSET,
        )
        callback_subs = [death_callback, *interrupts]
        if life_threshold >= 0:
            callback_subs.append(life_callback)
        if timer_threshold >= 0:
            callback_subs.append(timer_callback)
        for sub_id in callback_subs:
            if sub_id < 0:
                continue
            if sub_id >= len(subroutines):
                raise SourceDatasetError(
                    f"frame {control.frame} enemy {slot} has invalid ECL sub {sub_id}"
                )
            required.add(subroutines[sub_id])
    return frozenset(required)


def validate_frame_authority(
    control: ControlSnapshot,
    anchor: Snapshot,
    *,
    same_pause: bool = True,
) -> None:
    """Fail closed unless one dense row is self-contained with its anchor."""
    if control.capture_tier != CONTROL_CAPTURE_TIER:
        raise SourceDatasetError(
            f"unsupported dense tier {control.capture_tier!r}; "
            f"{CONTROL_CAPTURE_TIER} required"
        )
    if (
        control.source_record_schema != SOURCE_RECORD_SCHEMA
        or control.factual_state_schema != OFFLINE_FACT_SCHEMA
    ):
        raise SourceDatasetError("dense source/factual schema is not authoritative")
    if (
        control.stage != anchor.stage
        or control.difficulty != anchor.difficulty
        or control.character != anchor.character
        or control.shot_type
        != (
            anchor.player_attack.shot_type
            if anchor.player_attack is not None
            else -1
        )
    ):
        raise SourceDatasetError("dense frame and source anchor scopes disagree")
    if tuple(control.ecl_ex_function_addresses) != tuple(
        anchor.ecl_ex_function_addresses
    ):
        raise SourceDatasetError("EX callback dispatch table disagrees with anchor")
    if same_pause and control.repeat_star_state != anchor.repeat_star_state:
        raise SourceDatasetError(
            "repeating-star globals disagree with the same-pause anchor"
        )

    addresses = _source_addresses(anchor)
    timeline_pointer = _timeline_pointer(control)
    if timeline_pointer and timeline_pointer not in addresses:
        raise SourceDatasetError(
            f"frame {control.frame} timeline pointer lacks anchor coverage"
        )
    missing = _required_enemy_addresses(control, anchor) - addresses
    if missing:
        sample = ", ".join(f"0x{address:08X}" for address in sorted(missing)[:4])
        raise SourceDatasetError(
            f"frame {control.frame} ECL pointers lack anchor coverage: {sample}"
        )


def iter_source_frames(run_dir: Path) -> Iterator[SourceFrameBundle]:
    """Yield validated dense frames with the exact active source anchor."""
    run_dir = run_dir.resolve()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest.get("complete") is not True:
        raise SourceDatasetError("incomplete corpus cannot enter training")
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    metadata = run.get("metadata") or {}
    planner = metadata.get("planner") or {}
    if (
        run.get("run_id") != manifest.get("run_id")
        or metadata.get("executable_sha256") != native.TARGET_SHA256
        or planner.get("algorithm") != "source-hard4-paused-publication-v2"
        or planner.get("source_commitment") != "source-complete-hard-v1"
        or planner.get("publication_epoch") != "source-root-process-suspended-v1"
        or planner.get("factual_state_schema") != OFFLINE_FACT_SCHEMA
        or planner.get("hard_horizon") != 4
        or planner.get("learner_feature_horizon") != 4
        or planner.get("minimum_collision_margin") != 0.35
        or planner.get("zero_margin_fallback") is not False
    ):
        raise SourceDatasetError("run metadata does not bind the current authority contract")
    objects = {
        str(row["object_id"]): row["payload"]
        for row in _rows(_stream_paths(run_dir, manifest, "objects"))
    }
    anchors_by_ref: dict[str, list[tuple[int, str, Snapshot]]] = {}
    for row in _rows(_stream_paths(run_dir, manifest, "anchors")):
        reference = row.get("control_snapshot_ref")
        if not isinstance(reference, str):
            raise SourceDatasetError("source anchor lacks an exact control-frame reference")
        snapshot = decode_snapshot(expand_compact(row["snapshot"], objects))
        anchors_by_ref.setdefault(reference, []).append((
            int(row["sequence"]),
            str(row.get("reason", "")),
            snapshot,
        ))

    active_by_stage: dict[int, tuple[int, str, Snapshot]] = {}
    seen_stage: set[int] = set()
    for row in _rows(_stream_paths(run_dir, manifest, "frames")):
        snapshot_id = str(row["snapshot_id"])
        control = decode_control_snapshot(
            expand_compact(row["snapshot"], objects)
        )
        attached = sorted(anchors_by_ref.get(snapshot_id, ()), key=lambda item: item[0])
        for anchor in attached:
            active_by_stage[anchor[2].stage] = anchor
        if control.stage not in seen_stage:
            if not attached or all(reason != "stage-root" for _, reason, _ in attached):
                raise SourceDatasetError(
                    f"stage {control.stage} first dense frame lacks a same-frame stage root"
                )
            seen_stage.add(control.stage)
        active = active_by_stage.get(control.stage)
        if active is None:
            raise SourceDatasetError(
                f"frame {control.frame} has no active stage source root"
            )
        anchor_sequence, anchor_reason, anchor = active
        validate_frame_authority(
            control,
            anchor,
            same_pause=any(item == active for item in attached),
        )
        yield SourceFrameBundle(
            int(row["sequence"]),
            snapshot_id,
            control,
            anchor,
            anchor_sequence,
            anchor_reason,
            dict(row.get("scope") or {}),
            dict(row.get("decision") or {}),
        )
