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
VERSION_OFFSET = 0x14
VERSION_102H = 0x102
COLOR_MODE_16BIT_OFFSET = 0x1A
WINDOWED_OFFSET = 0x1E


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_directory", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    game_directory = args.game_directory.resolve()
    canonical = game_directory / CANONICAL_CONFIG_NAME
    candidates = sorted(game_directory.glob("*.cfg"))
    if canonical.is_file():
        path = canonical
    elif len(candidates) == 1:
        path = candidates[0]
    else:
        parser.error(
            "canonical TH06 cfg is absent and fallback is ambiguous; "
            f"found {len(candidates)} cfg files"
        )
    before = path.read_bytes()
    after = configure_windowed(before)
    changed = after != before
    if changed:
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.write(after)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    report = {
        "schema": "th06-rl-wine-retail-config-v1",
        "path": str(path),
        "size": len(after),
        "version": VERSION_102H,
        "color_mode_16bit_before": int(before[COLOR_MODE_16BIT_OFFSET]),
        "color_mode_16bit_after": int(after[COLOR_MODE_16BIT_OFFSET]),
        "windowed_before": int(before[WINDOWED_OFFSET]),
        "windowed_after": int(after[WINDOWED_OFFSET]),
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
