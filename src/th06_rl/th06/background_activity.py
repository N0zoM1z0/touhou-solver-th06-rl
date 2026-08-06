"""Reversible source-state lease that keeps windowed TH06 calculating."""

from __future__ import annotations

import ctypes
import os
import struct
from ctypes import wintypes


# Authoritative provenance:
#   reference/GensokyoClub-th06/config/globals.csv: _g_GameWindow
#   reference/GensokyoClub-th06/src/GameWindow.hpp: lastActiveAppValue
# GameWindow::Render returns before RunCalcChain whenever this field is zero.
ADDR_GAME_WINDOW = 0x006C6BD4
GAME_WINDOW_HWND_OFFSET = 0x0
GAME_WINDOW_LAST_ACTIVE_OFFSET = 0x8


def _decode_activity(raw: bytes) -> int:
    if len(raw) != 4:
        raise RuntimeError("short GameWindow activity read")
    value = struct.unpack("<i", raw)[0]
    if value not in (0, 1):
        raise RuntimeError(f"invalid GameWindow activity value {value}")
    return value


def _write_verified(process, address: int, data: bytes) -> None:
    """Write through the donor process handle and verify the exact bytes."""
    buffer = ctypes.create_string_buffer(data)
    written = ctypes.c_size_t()
    if not process.kernel32.WriteProcessMemory(
        process.handle,
        ctypes.c_void_p(address),
        buffer,
        len(data),
        ctypes.byref(written),
    ) or written.value != len(data):
        raise ctypes.WinError(ctypes.get_last_error())
    if process.read(address, len(data)) != data:
        raise RuntimeError(
            f"GameWindow activity write did not verify at {address:#x}"
        )


class BackgroundActivityLease:
    """Keep source calc active while attached, then restore real app state."""

    def __init__(self, process) -> None:
        if os.name != "nt":
            raise RuntimeError("background activity lease requires Windows")
        self.process = process
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [
            wintypes.HWND,
            ctypes.POINTER(wintypes.DWORD),
        ]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.reactivations = 0

    @property
    def address(self) -> int:
        return ADDR_GAME_WINDOW + GAME_WINDOW_LAST_ACTIVE_OFFSET

    def maintain(self) -> bool:
        """Return true only when an inactive source flag was reactivated."""
        current = _decode_activity(self.process.read(self.address, 4))
        if current:
            return False
        _write_verified(self.process, self.address, struct.pack("<i", 1))
        if _decode_activity(self.process.read(self.address, 4)) != 1:
            raise RuntimeError("GameWindow activity publication was not retained")
        self.reactivations += 1
        return True

    def _target_application_is_active(self) -> bool:
        hwnd = self.user32.GetForegroundWindow()
        if not hwnd:
            return False
        owner = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        return owner.value == self.process.pid

    def release(self) -> None:
        """Restore the value WindowProc would publish for the current app."""
        if not self.process.handle:
            return
        # WAIT_TIMEOUT proves the exact process is still live. If it already
        # exited, there is no source state left to restore.
        if self.process.kernel32.WaitForSingleObject(self.process.handle, 0) != 258:
            return
        desired = 1 if self._target_application_is_active() else 0
        _write_verified(self.process, self.address, struct.pack("<i", desired))
