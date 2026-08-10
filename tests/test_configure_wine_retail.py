from __future__ import annotations

import struct

import pytest

from scripts.configure_wine_retail import (
    COLOR_MODE_16BIT_OFFSET,
    CONFIG_SIZE,
    VERSION_OFFSET,
    WINDOWED_OFFSET,
    configure_windowed,
)


def _config(
    *, version: int = 0x102, windowed: int = 0, color_mode: int = 0xFF
) -> bytes:
    payload = bytearray(CONFIG_SIZE)
    struct.pack_into("<I", payload, VERSION_OFFSET, version)
    payload[COLOR_MODE_16BIT_OFFSET] = color_mode
    payload[WINDOWED_OFFSET] = windowed
    return bytes(payload)


def test_configure_windowed_changes_only_the_official_launcher_bytes() -> None:
    before = _config()
    after = configure_windowed(before)

    assert after[WINDOWED_OFFSET] == 1
    assert after[COLOR_MODE_16BIT_OFFSET] == 0
    changed = {
        index for index, (left, right) in enumerate(zip(before, after, strict=True))
        if left != right
    }
    assert changed == {COLOR_MODE_16BIT_OFFSET, WINDOWED_OFFSET}


def test_configure_windowed_is_idempotent() -> None:
    before = _config(windowed=1, color_mode=0)
    assert configure_windowed(before) == before


@pytest.mark.parametrize("payload", [b"", bytes(CONFIG_SIZE - 1)])
def test_configure_windowed_rejects_wrong_size(payload: bytes) -> None:
    with pytest.raises(ValueError, match="must be"):
        configure_windowed(payload)


def test_configure_windowed_rejects_wrong_version() -> None:
    with pytest.raises(ValueError, match="expected 0x102"):
        configure_windowed(_config(version=0x101))
