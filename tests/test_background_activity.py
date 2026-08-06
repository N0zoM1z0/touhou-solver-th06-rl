import ctypes
import struct

import pytest

from th06_rl.th06.background_activity import (
    ADDR_GAME_WINDOW,
    BackgroundActivityLease,
    _decode_activity,
)


class _MemoryKernel:
    def __init__(self, process):
        self.process = process

    def WriteProcessMemory(self, _handle, address, buffer, size, written):
        self.process.memory[address.value] = ctypes.string_at(buffer, size)
        ctypes.cast(written, ctypes.POINTER(ctypes.c_size_t))[0] = size
        return True


class _MemoryProcess:
    handle = 1

    def __init__(self):
        self.memory = {ADDR_GAME_WINDOW + 8: struct.pack("<i", 0)}
        self.kernel32 = _MemoryKernel(self)

    def read(self, address, _size):
        return self.memory[address]


def test_activity_source_flag_accepts_boolean_wparam_values():
    assert _decode_activity(b"\x00\x00\x00\x00") == 0
    assert _decode_activity(b"\x01\x00\x00\x00") == 1


def test_activity_source_flag_rejects_incoherent_values():
    with pytest.raises(RuntimeError):
        _decode_activity(b"\x02\x00\x00\x00")
    with pytest.raises(RuntimeError):
        _decode_activity(b"\x00")


def test_activity_lease_reactivates_through_native_process_handle():
    lease = BackgroundActivityLease.__new__(BackgroundActivityLease)
    lease.process = _MemoryProcess()
    lease.reactivations = 0

    assert lease.maintain() is True
    assert lease.maintain() is False
    assert lease.reactivations == 1
    assert lease.process.read(lease.address, 4) == struct.pack("<i", 1)
