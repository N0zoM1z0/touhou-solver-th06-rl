#!/usr/bin/env python3
"""Read-only one-shot probe for a currently running exact TH06 process."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th06_rl.th06.control_capture import read_control_snapshot
from th06_rl.th06.donor import enable_donor_imports


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game-dir", type=Path, required=True)
    args = parser.parse_args()
    enable_donor_imports()
    import th06.native as native

    process = native.attach_exact(args.game_dir.resolve())
    try:
        result: dict[str, object] = {
            "pid": process.pid,
            "frame": native.read_game_frame(process),
            "supervisor": native.read_supervisor_state(process),
            "dialogue": native.read_dialogue_state(process),
        }
        try:
            snapshot = read_control_snapshot(process)
            result["control"] = {
                "frame": snapshot.frame,
                "in_menu": snapshot.in_menu,
                "time_stopped": snapshot.time_stopped,
                "player_state": snapshot.player_state,
                "input_mask": snapshot.input_mask,
                "timeline_time": snapshot.timeline_time,
                "source_context": snapshot.source_context,
                "bullets": snapshot.live_bullet_count,
            }
        except Exception as error:
            result["control_error"] = f"{type(error).__name__}: {error}"
        print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    finally:
        process.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
