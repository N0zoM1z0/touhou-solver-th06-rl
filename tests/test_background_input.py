from __future__ import annotations

import struct

import pytest

from th06_rl.th06.background_input import (
    ALLOWED_INPUT_MASK,
    BUTTON_BOMB,
    CONTROL_OFFSET,
    HOOK_ADDRESS,
    build_hook,
    build_stub,
    call_target,
    keys_to_input_mask,
)
import th06_rl.th06.menu as menu
from th06_rl.th06.menu import BACKGROUND_TAP_SECONDS, _settled_tap, _tap


def test_call_site_hook_keeps_one_complete_instruction() -> None:
    cave = 0x12340000
    hook = build_hook(cave)
    assert len(hook) == 5
    assert hook[0] == 0xE8
    assert call_target(HOOK_ADDRESS, hook) == cave


def test_stub_masks_bomb_inside_target_process() -> None:
    cave = 0x12340000
    stub = build_stub(cave)
    assert stub[:1] == b"\xA1"
    assert struct.unpack("<I", stub[1:5])[0] == cave + CONTROL_OFFSET
    assert stub[5:6] == b"\x25"
    assert struct.unpack("<I", stub[6:10])[0] == ALLOWED_INPUT_MASK
    assert not ALLOWED_INPUT_MASK & BUTTON_BOMB
    assert stub[10:] == b"\xC3"


def test_public_key_encoder_cannot_represent_bomb() -> None:
    assert not keys_to_input_mask({"shoot", "focus", "left"}) & BUTTON_BOMB
    with pytest.raises(ValueError):
        keys_to_input_mask({"bomb"})


def test_menu_tap_spans_a_background_throttled_update() -> None:
    calls = []

    class Keyboard:
        def tap(self, key, *, hold_seconds):
            calls.append((key, hold_seconds))

    _tap(Keyboard(), "shoot")

    assert calls == [("shoot", BACKGROUND_TAP_SECONDS)]
    assert BACKGROUND_TAP_SECONDS > 0.25


def test_settled_menu_tap_waits_for_a_post_release_game_tick(monkeypatch) -> None:
    states = iter(((7, 3, 65), (7, 3, 65), (7, 3, 66)))
    calls = []

    class Keyboard:
        def tap(self, key, *, hold_seconds):
            calls.append((key, hold_seconds))

    monkeypatch.setattr(menu, "read_menu_state", lambda _process: next(states))
    monkeypatch.setattr(menu.time, "sleep", lambda _seconds: None)

    _settled_tap(object(), Keyboard(), "shoot")

    assert calls == [("shoot", BACKGROUND_TAP_SECONDS)]


def test_settled_menu_tap_maintains_background_activity(monkeypatch) -> None:
    states = iter(((1, 0, 61), (1, 0, 61), (1, 0, 62)))
    maintained = []

    class Keyboard:
        def set_auxiliary(self, key, enabled):
            auxiliary.append((key, enabled))

    auxiliary = []
    monkeypatch.setattr(menu, "read_menu_state", lambda _process: next(states))
    monkeypatch.setattr(menu.time, "sleep", lambda _seconds: None)

    _settled_tap(
        object(),
        Keyboard(),
        "shoot",
        maintain_activity=lambda: maintained.append(True),
    )

    assert len(maintained) >= 2
    assert auxiliary == [("shoot", True), ("shoot", False)]
