#!/usr/bin/env python3
"""Read-only one-shot probe for a currently running exact TH06 process."""

from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path
import struct
from ctypes import wintypes

from th06_rl.th06.control_capture import read_control_snapshot
from th06_rl.th06.donor import enable_donor_imports


def _window_state(pid: int) -> dict[str, object] | None:
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    windows: list[int] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    @callback_type
    def visit(hwnd, _parameter):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            windows.append(int(hwnd))
        return True

    user32.EnumWindows(visit, 0)
    if not windows:
        return None
    hwnd = windows[0]
    return {
        "hwnd": hwnd,
        "minimized": bool(user32.IsIconic(hwnd)),
        "foreground": int(user32.GetForegroundWindow()) == hwnd,
    }


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
            "window": _window_state(process.pid),
        }
        flags = process.read(
            native.ADDR_GAME_MANAGER + native.GAME_FLAGS_OFFSET,
            6,
        )
        result["game_flags"] = list(flags)
        result["time_stopped"] = bool(process.read(
            native.ADDR_GAME_MANAGER + native.GAME_TIME_STOPPED_OFFSET,
            1,
        )[0])
        result["bullet_time"] = struct.unpack(
            "<i",
            process.read(
                native.ADDR_BULLET_MANAGER
                + native.BULLET_MANAGER_TIME_OFFSET
                + 8,
                4,
            ),
        )[0]
        result["player_state"] = process.read(
            native.ADDR_PLAYER + native.PLAYER_STATE_OFFSET,
            1,
        )[0]
        result["input_mask"] = struct.unpack(
            "<H", process.read(native.ADDR_CURRENT_INPUT, 2)
        )[0]
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
