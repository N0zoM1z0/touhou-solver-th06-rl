"""Independent native dialogue sensing and physical Ctrl/Skip control."""

from __future__ import annotations

import time
from dataclasses import dataclass

from .actuator import Keyboard
from .native import NativeProcess, read_dialogue_state


@dataclass(frozen=True)
class DialogueState:
    active: bool
    skippable: bool
    pulsed_shoot: bool


class DialogueSkipper:
    def __init__(self, process: NativeProcess, keyboard: Keyboard):
        self.process = process
        self.keyboard = keyboard
        self.last_shoot_pulse = 0.0
        self.shoot_release_until: float | None = None

    def update(self, gameplay_context: bool) -> DialogueState:
        if gameplay_context:
            active, skippable = read_dialogue_state(self.process)
        else:
            active, skippable = False, False
        pulsed = False
        self.keyboard.set_auxiliary("skip", active and skippable)
        now = time.monotonic()
        if self.shoot_release_until is not None and (
            now >= self.shoot_release_until or not active or skippable
        ):
            self.keyboard.set_suppressed("shoot", False)
            self.shoot_release_until = None
            pulsed = True
        if (
            active
            and not skippable
            and self.shoot_release_until is None
            and now - self.last_shoot_pulse >= 0.25
        ):
            # GuiImpl::RunMsg requires a new WAS_PRESSED(SHOOT) edge for an
            # unskippable WAIT. Z is normally held, so begin a non-blocking
            # 50 ms release; a later update supplies the fresh press edge.
            self.keyboard.set_suppressed("shoot", True)
            self.shoot_release_until = now + 0.05
            self.last_shoot_pulse = now
        state = DialogueState(active, skippable, pulsed)
        return state

    def release(self) -> None:
        self.keyboard.set_auxiliary("skip", False)
        self.keyboard.set_suppressed("shoot", False)
        self.shoot_release_until = None
