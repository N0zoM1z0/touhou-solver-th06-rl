import struct

import pytest

from th06_rl.th06.control_capture import (
    _completed_calc_lag,
    _read_bulk_view,
    read_passive_input_delivery,
)
from th06_rl.retail import native


def test_bulk_view_falls_back_to_ordinary_reader():
    class Process:
        @staticmethod
        def read(address, size):
            assert address == 0x1234
            assert size == 4
            return b"TH06"

    view = _read_bulk_view(Process(), 0x1234, 4)
    assert isinstance(view, memoryview)
    assert bytes(view) == b"TH06"


def test_active_calc_phase_uses_initial_equal_clock_witness():
    assert _completed_calc_lag(
        None,
        None,
        stage=5,
        game_frame=120,
        bullet_time=120,
        passive=False,
    ) == (5, 0, True)


def test_time_stop_rebases_completed_calc_lag():
    assert _completed_calc_lag(
        5,
        0,
        stage=5,
        game_frame=420,
        bullet_time=120,
        passive=True,
    ) == (5, 300, True)
    assert _completed_calc_lag(
        5,
        300,
        stage=5,
        game_frame=421,
        bullet_time=121,
        passive=False,
    ) == (5, 300, True)


def test_priority_window_is_one_frame_beyond_completed_lag():
    assert _completed_calc_lag(
        5,
        300,
        stage=5,
        game_frame=421,
        bullet_time=120,
        passive=False,
    ) == (5, 300, False)


def test_unobserved_nonzero_midstage_lag_fails_closed():
    assert _completed_calc_lag(
        None,
        None,
        stage=5,
        game_frame=420,
        bullet_time=120,
        passive=False,
    ) == (5, None, False)


def test_stage_change_does_not_reuse_prior_lag():
    assert _completed_calc_lag(
        4,
        300,
        stage=5,
        game_frame=1,
        bullet_time=1,
        passive=False,
    ) == (5, 0, True)


def test_passive_input_delivery_reads_one_coherent_game_frame(monkeypatch):
    frames = iter((123, 123))
    monkeypatch.setattr(native, "read_game_frame", lambda _process: next(frames))

    block = bytearray(14)
    for offset, value in ((0, 0x101), (4, 0x001), (8, 3), (12, 7)):
        struct.pack_into("<H", block, offset, value)

    class Process:
        @staticmethod
        def read(address, size):
            assert address == native.ADDR_CURRENT_INPUT
            assert size == 14
            return bytes(block)

    assert read_passive_input_delivery(Process()) == (123, 0x101, 1, 3, 7)


def test_passive_input_delivery_rejects_repeated_cross_frame_reads(monkeypatch):
    frames = iter(range(16))
    monkeypatch.setattr(native, "read_game_frame", lambda _process: next(frames))

    class Process:
        @staticmethod
        def read(_address, size):
            return bytes(size)

    with pytest.raises(RuntimeError, match="crossed game frames"):
        read_passive_input_delivery(Process())
