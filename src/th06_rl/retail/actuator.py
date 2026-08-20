"""Foreground-guarded physical keyboard output. There is no Bomb mapping."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes

from .model import (
    BUTTON_DOWN,
    BUTTON_FOCUS,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_SHOOT,
    BUTTON_UP,
    Action,
)


ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class KeybdInput(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class HardwareInput(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]


class InputUnion(ctypes.Union):
    _fields_ = [("mi", MouseInput), ("ki", KeybdInput), ("hi", HardwareInput)]


class Input(ctypes.Structure):
    _fields_ = [("type", wintypes.DWORD), ("union", InputUnion)]


class Keyboard:
    SCANCODES = {
        "shoot": (0x2C, False),
        "focus": (0x2A, False),
        "skip": (0x1D, False),
        "menu": (0x01, False),
        "up": (0x48, True),
        "down": (0x50, True),
        "left": (0x4B, True),
        "right": (0x4D, True),
    }

    def __init__(self, pid: int):
        self.pid = pid
        self.user32 = ctypes.WinDLL("user32", use_last_error=True)
        self.user32.SendInput.argtypes = [wintypes.UINT, ctypes.POINTER(Input), ctypes.c_int]
        self.user32.SendInput.restype = wintypes.UINT
        self.user32.GetForegroundWindow.restype = wintypes.HWND
        self.user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
        self.user32.GetWindowThreadProcessId.restype = wintypes.DWORD
        self.held: set[str] = set()
        self.base_desired: set[str] = set()
        self.auxiliary_desired: set[str] = set()
        self.suppressed: set[str] = set()

    def foreground(self) -> bool:
        hwnd = self.user32.GetForegroundWindow()
        pid = wintypes.DWORD()
        self.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        return pid.value == self.pid

    def _input(self, key: str, down: bool) -> Input:
        scan, extended = self.SCANCODES[key]
        flags = 0x0008 | (0x0001 if extended else 0) | (0 if down else 0x0002)
        return Input(type=1, union=InputUnion(ki=KeybdInput(0, scan, flags, 0, 0)))

    def _events(self, events: tuple[tuple[str, bool], ...]) -> None:
        if not events:
            return
        inputs = (Input * len(events))(*(self._input(key, down) for key, down in events))
        if self.user32.SendInput(len(inputs), inputs, ctypes.sizeof(Input)) != len(inputs):
            raise ctypes.WinError(ctypes.get_last_error())

    def _event(self, key: str, down: bool) -> None:
        self._events(((key, down),))

    def _sync(self) -> tuple[tuple[str, bool], ...]:
        desired = (self.base_desired | self.auxiliary_desired) - self.suppressed
        events = tuple((key, False) for key in sorted(self.held - desired)) + tuple(
            (key, True) for key in sorted(desired - self.held)
        )
        # Send the complete release-then-press transition as one Win32 input
        # transaction so the game cannot sample an avoidable inter-call mask.
        self._events(events)
        self.held = desired
        return events

    @property
    def base_input_mask(self) -> int:
        mask = 0
        mask |= BUTTON_SHOOT if "shoot" in self.base_desired else 0
        mask |= BUTTON_FOCUS if "focus" in self.base_desired else 0
        mask |= BUTTON_UP if "up" in self.base_desired else 0
        mask |= BUTTON_DOWN if "down" in self.base_desired else 0
        mask |= BUTTON_LEFT if "left" in self.base_desired else 0
        mask |= BUTTON_RIGHT if "right" in self.base_desired else 0
        return mask

    def apply(self, action: Action) -> tuple[tuple[str, bool], ...]:
        desired = {"shoot"}
        if action.focused:
            desired.add("focus")
        if action.dx < 0:
            desired.add("left")
        elif action.dx > 0:
            desired.add("right")
        if action.dy < 0:
            desired.add("up")
        elif action.dy > 0:
            desired.add("down")
        self.base_desired = desired
        return self._sync()

    def set_auxiliary(self, key: str, enabled: bool) -> None:
        if key not in self.SCANCODES:
            raise ValueError(f"unsupported auxiliary key: {key}")
        if enabled:
            self.auxiliary_desired.add(key)
        else:
            self.auxiliary_desired.discard(key)
        self._sync()

    def set_suppressed(self, key: str, suppressed: bool) -> None:
        """Temporarily suppress a desired key without blocking the agent loop."""
        if key not in self.SCANCODES:
            raise ValueError(f"unsupported suppressed key: {key}")
        if suppressed:
            self.suppressed.add(key)
        else:
            self.suppressed.discard(key)
        self._sync()

    def tap(self, key: str, hold_seconds: float = 0.05) -> None:
        if key not in self.SCANCODES:
            raise ValueError(f"unsupported key: {key}")
        if not self.foreground():
            raise RuntimeError("TH06 is not foreground for menu input")
        self._event(key, True)
        try:
            time.sleep(hold_seconds)
        finally:
            self._event(key, False)
        time.sleep(0.12)

    def release_all(self) -> tuple[tuple[str, bool], ...]:
        events = tuple((key, False) for key in sorted(self.held))
        self._events(events)
        self.held.clear()
        self.base_desired.clear()
        self.auxiliary_desired.clear()
        self.suppressed.clear()
        return events
