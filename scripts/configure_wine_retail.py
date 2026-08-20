#!/usr/bin/env python3
"""Set the source-defined TH06 1.02h 32-bit windowed launcher options."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile


CONFIG_SIZE = 0x38
CANONICAL_CONFIG_NAME = "東方紅魔郷.cfg"
# ControllerMapping is 18 bytes; MSVC aligns the following i32 at 0x14.
CONTROLLER_MAPPING = (0, 1, 2, 4, -1, -1, -1, -1, 3)
VERSION_OFFSET = 0x14
VERSION_102H = 0x102
LIFE_COUNT_OFFSET = 0x18
BOMB_COUNT_OFFSET = 0x19
COLOR_MODE_16BIT_OFFSET = 0x1A
MUSIC_MODE_OFFSET = 0x1B
PLAY_SOUNDS_OFFSET = 0x1C
DEFAULT_DIFFICULTY_OFFSET = 0x1D
WINDOWED_OFFSET = 0x1E
FRAMESKIP_OFFSET = 0x1F
PAD_X_AXIS_OFFSET = 0x20
PAD_Y_AXIS_OFFSET = 0x22
OPTIONS_OFFSET = 0x34


def source_default_config() -> bytes:
    """Reconstruct LoadConfig's 1.02h defaults before custom.exe resolves them."""
    payload = bytearray(CONFIG_SIZE)
    struct.pack_into("<9h", payload, 0, *CONTROLLER_MAPPING)
    struct.pack_into("<I", payload, VERSION_OFFSET, VERSION_102H)
    payload[LIFE_COUNT_OFFSET] = 2
    payload[BOMB_COUNT_OFFSET] = 3
    payload[COLOR_MODE_16BIT_OFFSET] = 0xFF
    payload[MUSIC_MODE_OFFSET] = 1  # WAV; the shipped archive includes BGM WAVs.
    payload[PLAY_SOUNDS_OFFSET] = 1
    payload[DEFAULT_DIFFICULTY_OFFSET] = 1
    payload[WINDOWED_OFFSET] = 0
    payload[FRAMESKIP_OFFSET] = 0
    struct.pack_into("<hh", payload, PAD_X_AXIS_OFFSET, 600, 600)
    # GCOS_USE_D3D_HW_TEXTURE_BLENDING is the source default option bit.
    struct.pack_into("<I", payload, OPTIONS_OFFSET, 1)
    return bytes(payload)


def configure_windowed(payload: bytes) -> bytes:
    if len(payload) != CONFIG_SIZE:
        raise ValueError(f"TH06 config must be {CONFIG_SIZE} bytes")
    version = struct.unpack_from("<I", payload, VERSION_OFFSET)[0]
    if version != VERSION_102H:
        raise ValueError(f"TH06 config version is 0x{version:x}, expected 0x102")
    if payload[WINDOWED_OFFSET] not in (0, 1):
        raise ValueError("TH06 windowed byte is invalid")
    if payload[COLOR_MODE_16BIT_OFFSET] not in (0, 1, 0xFF):
        raise ValueError("TH06 color-mode byte is invalid")
    configured = bytearray(payload)
    # 0xff is the first-launch auto-detect sentinel. LoadConfig rejects it on
    # the next launch before window creation and resets the whole structure to
    # fullscreen. custom.exe resolves the sentinel to the selected 32-bit mode.
    configured[COLOR_MODE_16BIT_OFFSET] = 0
    configured[WINDOWED_OFFSET] = 1
    return bytes(configured)


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_directory", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help="create a missing canonical cfg from source-defined 1.02h defaults",
    )
    args = parser.parse_args()

    game_directory = args.game_directory.resolve()
    canonical = game_directory / CANONICAL_CONFIG_NAME
    candidates = sorted(game_directory.glob("*.cfg"))
    if canonical.is_file():
        path = canonical
        initialized = False
    elif len(candidates) == 1:
        path = candidates[0]
        initialized = False
    elif not candidates and args.initialize:
        path = canonical
        before = source_default_config()
        initialized = True
    else:
        parser.error(
            "canonical TH06 cfg is absent and fallback is ambiguous; "
            f"found {len(candidates)} cfg files"
        )
    if not initialized:
        before = path.read_bytes()
    after = configure_windowed(before)
    changed = initialized or after != before
    if changed:
        _atomic_write(path, after)

    report = {
        "schema": "th06-rl-wine-retail-config-v1",
        "path": str(path),
        "size": len(after),
        "version": VERSION_102H,
        "color_mode_16bit_before": int(before[COLOR_MODE_16BIT_OFFSET]),
        "color_mode_16bit_after": int(after[COLOR_MODE_16BIT_OFFSET]),
        "windowed_before": int(before[WINDOWED_OFFSET]),
        "windowed_after": int(after[WINDOWED_OFFSET]),
        "initialized": initialized,
        "changed": changed,
        "sha256_before": _sha256(before),
        "sha256_after": _sha256(after),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
