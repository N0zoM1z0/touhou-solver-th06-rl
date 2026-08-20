from __future__ import annotations

import struct

import pytest

from scripts.configure_wine_retail import (
    BOMB_COUNT_OFFSET,
    COLOR_MODE_16BIT_OFFSET,
    CONFIG_SIZE,
    CONTROLLER_MAPPING,
    DEFAULT_DIFFICULTY_OFFSET,
    FRAMESKIP_OFFSET,
    LIFE_COUNT_OFFSET,
    MUSIC_MODE_OFFSET,
    OPTIONS_OFFSET,
    PAD_X_AXIS_OFFSET,
    PAD_Y_AXIS_OFFSET,
    PLAY_SOUNDS_OFFSET,
    VERSION_OFFSET,
    WINDOWED_OFFSET,
    configure_windowed,
    source_default_config,
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


def test_source_default_config_matches_the_exact_102h_layout() -> None:
    payload = source_default_config()

    assert len(payload) == CONFIG_SIZE
    assert struct.unpack_from("<9h", payload, 0) == CONTROLLER_MAPPING
    assert payload[18:20] == b"\0\0"
    assert struct.unpack_from("<I", payload, VERSION_OFFSET)[0] == 0x102
    assert payload[LIFE_COUNT_OFFSET] == 2
    assert payload[BOMB_COUNT_OFFSET] == 3
    assert payload[COLOR_MODE_16BIT_OFFSET] == 0xFF
    assert payload[MUSIC_MODE_OFFSET] == 1
    assert payload[PLAY_SOUNDS_OFFSET] == 1
    assert payload[DEFAULT_DIFFICULTY_OFFSET] == 1
    assert payload[WINDOWED_OFFSET] == 0
    assert payload[FRAMESKIP_OFFSET] == 0
    assert struct.unpack_from("<h", payload, PAD_X_AXIS_OFFSET)[0] == 600
    assert struct.unpack_from("<h", payload, PAD_Y_AXIS_OFFSET)[0] == 600
    assert payload[0x24:0x34] == bytes(16)
    assert struct.unpack_from("<I", payload, OPTIONS_OFFSET)[0] == 1
