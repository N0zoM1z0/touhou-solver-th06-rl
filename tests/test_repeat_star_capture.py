import math
import struct

import pytest

from th06_rl.retail import native


class _Memory:
    def __init__(self, values: dict[int, bytes]):
        self.values = values

    def read(self, address: int, size: int) -> bytes:
        value = self.values[address]
        assert len(value) == size
        return value


def test_repeat_star_globals_use_exact_retail_addresses() -> None:
    process = _Memory({
        native.ADDR_STAR_ANGLE_TABLE: struct.pack(
            "<6f", 0.1, 0.2, 0.3, 0.4, 0.5, 0.6
        ),
        native.ADDR_STAR_ENEMY_POSITION: struct.pack(
            "<3f", 100.0, 120.0, 7.0
        ),
        native.ADDR_STAR_PLAYER_POSITION: struct.pack(
            "<3f", 192.0, 400.0, 9.0
        ),
    })

    state = native._read_repeat_star_state(process)

    assert state.angles == pytest.approx((0.1, 0.2, 0.3, 0.4, 0.5, 0.6))
    assert (state.enemy_x, state.enemy_y) == (100.0, 120.0)
    assert (state.player_x, state.player_y) == (192.0, 400.0)
    assert state.angles_known


def test_repeat_star_capture_rejects_nonfinite_source_state() -> None:
    process = _Memory({
        native.ADDR_STAR_ANGLE_TABLE: struct.pack(
            "<6f", math.nan, 0.0, 0.0, 0.0, 0.0, 0.0
        ),
        native.ADDR_STAR_ENEMY_POSITION: struct.pack("<3f", 0.0, 0.0, 0.0),
        native.ADDR_STAR_PLAYER_POSITION: struct.pack("<3f", 0.0, 0.0, 0.0),
    })

    with pytest.raises(RuntimeError, match="non-finite repeating-star"):
        native._read_repeat_star_state(process)
