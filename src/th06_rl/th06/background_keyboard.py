"""Small Keyboard-compatible publisher backed by the TH06 call-site bridge."""

from __future__ import annotations

import time

from .background_input import KEY_MASKS, BackgroundInputBridge


class BackgroundKeyboard:
    """Publish complete Bomb-free input masks without foreground key events."""

    def __init__(self, pid: int, bridge: BackgroundInputBridge) -> None:
        if pid != bridge.process.pid:
            raise ValueError("keyboard and input bridge PIDs differ")
        self.pid = pid
        self.bridge = bridge
        self.held: set[str] = set()
        self.base_desired: set[str] = set()
        self.auxiliary_desired: set[str] = set()
        self.suppressed: set[str] = set()

    def foreground(self) -> bool:
        # The bridge is deliberately background-capable and bound to one exact
        # process. Foreground state is not input authority here.
        return True

    def _sync(self) -> tuple[tuple[str, bool], ...]:
        desired = (self.base_desired | self.auxiliary_desired) - self.suppressed
        unknown = desired - KEY_MASKS.keys()
        if unknown:
            raise ValueError(f"unsupported TH06 keys: {sorted(unknown)}")
        events = tuple((key, False) for key in sorted(self.held - desired)) + tuple(
            (key, True) for key in sorted(desired - self.held)
        )
        # One verified DWORD publication replaces the donor's sequence of
        # SendInput events, so TH06 cannot sample a transition prefix.
        self.bridge.set_keys(desired)
        self.held = set(desired)
        return events

    @property
    def base_input_mask(self) -> int:
        mask = 0
        for key in self.base_desired:
            mask |= KEY_MASKS[key]
        return mask

    @property
    def published_input_mask(self) -> int:
        mask = 0
        for key in self.held:
            mask |= KEY_MASKS[key]
        return mask

    def apply(self, action) -> tuple[tuple[str, bool], ...]:
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
        if key not in KEY_MASKS:
            raise ValueError(f"unsupported auxiliary key: {key}")
        if enabled:
            self.auxiliary_desired.add(key)
        else:
            self.auxiliary_desired.discard(key)
        self._sync()

    def set_suppressed(self, key: str, suppressed: bool) -> None:
        if key not in KEY_MASKS:
            raise ValueError(f"unsupported suppressed key: {key}")
        if suppressed:
            self.suppressed.add(key)
        else:
            self.suppressed.discard(key)
        self._sync()

    def tap(self, key: str, hold_seconds: float = 0.05) -> None:
        if key not in KEY_MASKS:
            raise ValueError(f"unsupported key: {key}")
        self.set_auxiliary(key, True)
        try:
            time.sleep(hold_seconds)
        finally:
            self.set_auxiliary(key, False)
        time.sleep(0.12)

    def release_all(self) -> tuple[tuple[str, bool], ...]:
        events = tuple((key, False) for key in sorted(self.held))
        self.base_desired.clear()
        self.auxiliary_desired.clear()
        self.suppressed.clear()
        self.bridge.release_all()
        self.held.clear()
        return events
