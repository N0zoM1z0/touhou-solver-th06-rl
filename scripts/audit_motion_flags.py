#!/usr/bin/env python3
"""Census source motion flags in complete TH06-RL corpus runs.

The report distinguishes every occupied physical bullet slot retained in the
raw control snapshot from the smaller reachable subset used by the online
Hard gate.  It is an offline audit and never runs in the resident controller.
"""

from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
from dataclasses import fields
import gzip
import json
from pathlib import Path
import struct
import sys

from th06_rl.th06.donor import enable_donor_imports
from th06_rl.th06.observed_bullets import classify_ex_flags


enable_donor_imports()
from th06 import native  # noqa: E402
from th06.model import Bullet, Laser  # noqa: E402

try:
    import msgspec
except ImportError:  # pragma: no cover - optional audit accelerator
    msgspec = None

_JSON_DECODER = msgspec.json.Decoder() if msgspec is not None else None


def _decode_json(line: bytes):
    return (
        _JSON_DECODER.decode(line)
        if _JSON_DECODER is not None
        else json.loads(line)
    )


def _frame_paths(run_dir: Path, manifest: dict) -> list[Path]:
    return [
        run_dir / shard["path"]
        for shard in manifest.get("shards", ())
        if shard.get("stream") == "frames"
    ]


def _state_support(flags: int, state: int) -> str:
    if state == 1:
        return classify_ex_flags(flags)
    if state in (2, 3, 4):
        fired_support = classify_ex_flags(flags)
        return (
            "source-bounded-spawn-transition"
            if fired_support.startswith("source-exact-")
            else "conservative-spawn-transition"
        )
    if state == 5:
        return "non-collidable-despawn"
    return "unknown-state-fail-closed"


def _counter_rows(counter: Counter, contexts: dict) -> list[dict[str, object]]:
    rows = []
    for (flags, state), count in sorted(counter.items()):
        scope_counts = contexts[(flags, state)]
        rows.append({
            "ex_flags": flags,
            "ex_flags_hex": f"0x{flags:X}",
            "state": state,
            "support": _state_support(flags, state),
            "slot_observations": count,
            "top_source_contexts": [
                {"phase_id": phase, "observations": observations}
                for phase, observations in scope_counts.most_common(8)
            ],
        })
    return rows


def audit(root: Path) -> dict[str, object]:
    bullet_layout = [field.name for field in fields(Bullet)]
    laser_layout = [field.name for field in fields(Laser)]
    bullet_flag_index = bullet_layout.index("ex_flags")
    bullet_state_index = bullet_layout.index("state")
    laser_flag_index = laser_layout.index("flags")
    laser_state_index = laser_layout.index("state")
    laser_motion_index = laser_layout.index("motion_known")

    raw_counts: Counter = Counter()
    reachable_counts: Counter = Counter()
    raw_contexts: dict = defaultdict(Counter)
    reachable_contexts: dict = defaultdict(Counter)
    laser_counts: Counter = Counter()
    run_names = []
    total_frames = 0
    raw_record_size = 6 + native.BULLET_STRIDE - native.BULLET_SIZE_OFFSET
    tail_flag_offset = native.BULLET_EX_FLAGS_OFFSET - native.BULLET_SIZE_OFFSET
    tail_state_offset = native.BULLET_STATE_OFFSET - native.BULLET_SIZE_OFFSET

    for run_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        manifest_path = run_dir / "manifest.json"
        if not manifest_path.is_file():
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete") is not True:
            continue
        run_names.append(run_dir.name)
        for path in _frame_paths(run_dir, manifest):
            with gzip.open(path, "rb") as source:
                for line in source:
                    total_frames += 1
                    row = _decode_json(line)
                    snapshot = row["snapshot"]
                    phase_id = str(row.get("scope", {}).get("phase_id", "unknown"))

                    bullets = snapshot.get("bullets")
                    if isinstance(bullets, dict) and bullets.get("codec") == "dataclass-rows-v1":
                        for values in bullets.get("rows", ()):
                            flags = int(values[bullet_flag_index])
                            state = int(values[bullet_state_index])
                            key = (flags, state)
                            reachable_counts[key] += 1
                            reachable_contexts[key][phase_id] += 1

                    raw = snapshot.get("raw_bullet_tails")
                    if isinstance(raw, dict) and raw.get("codec") == "bytes-base64-v1":
                        payload = base64.b64decode(raw["data"], validate=True)
                        if len(payload) % raw_record_size:
                            raise RuntimeError(
                                f"invalid raw bullet tail length in {path}: {len(payload)}"
                            )
                        for offset in range(0, len(payload), raw_record_size):
                            tail = offset + 6
                            flags = struct.unpack_from(
                                "<H", payload, tail + tail_flag_offset
                            )[0]
                            state = struct.unpack_from(
                                "<H", payload, tail + tail_state_offset
                            )[0]
                            key = (flags, state)
                            raw_counts[key] += 1
                            raw_contexts[key][phase_id] += 1

                    lasers = snapshot.get("lasers")
                    if isinstance(lasers, dict) and lasers.get("codec") == "dataclass-rows-v1":
                        for values in lasers.get("rows", ()):
                            laser_counts[(
                                int(values[laser_flag_index]),
                                int(values[laser_state_index]),
                                bool(values[laser_motion_index]),
                            )] += 1
        print(f"audited {run_dir.name}", file=sys.stderr, flush=True)

    return {
        "schema_version": "th06-rl-motion-flag-audit-v1",
        "source_provenance": {
            "bullet_update": "GensokyoClub-th06/src/BulletManager.cpp:710-909",
            "bullet_spawn": "GensokyoClub-th06/src/BulletManager.cpp:178-378",
            "bounds": "GensokyoClub-th06/src/GameManager.cpp:94-113",
        },
        "runs": run_names,
        "frame_records": total_frames,
        "all_occupied_bullet_slots": _counter_rows(raw_counts, raw_contexts),
        "online_reachable_bullet_slots": _counter_rows(
            reachable_counts, reachable_contexts
        ),
        "laser_modes": [
            {
                "flags": flags,
                "flags_hex": f"0x{flags:X}",
                "state": state,
                "motion_known": motion_known,
                "support": (
                    "source-exact-observed"
                    if not flags & ~3 and motion_known
                    else "source-exact-static-angle"
                    if not flags & ~3
                    else "unknown-flag-fail-closed"
                ),
                "slot_observations": count,
            }
            for (flags, state, motion_known), count in sorted(laser_counts.items())
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    report = audit(args.root)
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
