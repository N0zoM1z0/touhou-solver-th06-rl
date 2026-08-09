from __future__ import annotations

from dataclasses import dataclass

import pytest

from th06_rl.th06.background_input import BUTTON_BOMB
from th06_rl.th06.background_keyboard import BackgroundKeyboard


class Bridge:
    def __init__(self, pid: int = 77) -> None:
        self.process = type("Process", (), {"pid": pid})()
        self.keys = []
        self.releases = 0

    def set_keys(self, names: set[str]) -> None:
        self.keys.append(set(names))

    def release_all(self) -> None:
        self.releases += 1


@dataclass(frozen=True)
class Action:
    dx: int
    dy: int
    focused: bool


def test_apply_publishes_one_complete_background_mask() -> None:
    bridge = Bridge()
    keyboard = BackgroundKeyboard(77, bridge)

    events = keyboard.apply(Action(-1, 1, True))

    assert bridge.keys == [{"shoot", "focus", "left", "down"}]
    assert set(events) == {
        ("shoot", True),
        ("focus", True),
        ("left", True),
        ("down", True),
    }
    assert not keyboard.base_input_mask & BUTTON_BOMB
    assert keyboard.published_input_mask == keyboard.base_input_mask


def test_release_clears_all_bridge_state() -> None:
    bridge = Bridge()
    keyboard = BackgroundKeyboard(77, bridge)
    keyboard.apply(Action(1, 0, False))

    events = keyboard.release_all()

    assert set(events) == {
        ("shoot", False),
        ("right", False),
    }
    assert bridge.releases == 1
    assert keyboard.base_input_mask == 0


def test_keyboard_rejects_a_different_process() -> None:
    with pytest.raises(ValueError, match="PIDs differ"):
        BackgroundKeyboard(1, Bridge(pid=2))
