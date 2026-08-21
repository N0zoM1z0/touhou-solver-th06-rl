"""Keep one physical movement command in flight at a time."""

from __future__ import annotations

from dataclasses import dataclass

from .model import (
    Action,
    BUTTON_DOWN,
    BUTTON_FOCUS,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_UP,
)


_CONTROL_MASK = BUTTON_FOCUS | BUTTON_UP | BUTTON_DOWN | BUTTON_LEFT | BUTTON_RIGHT
INPUT_PICKUP_MAX_FRAMES = 2


def _action_mask(action: Action) -> int:
    mask = BUTTON_FOCUS if action.focused else 0
    if action.dx < 0:
        mask |= BUTTON_LEFT
    elif action.dx > 0:
        mask |= BUTTON_RIGHT
    if action.dy < 0:
        mask |= BUTTON_UP
    elif action.dy > 0:
        mask |= BUTTON_DOWN
    return mask


@dataclass(frozen=True)
class LeaseStatus:
    action: Action | None = None
    timed_out: bool = False
    # Whole-mask pickup branches still possible after the paused root.  The
    # game can retain the prior mask for a bounded number of updates and then
    # sample the target mask, but it cannot observe a partial key transition.
    delivery_delays: tuple[int, ...] = (0, 1, 2, 3)


class InputLease:
    """Hold a desired direction until the game reports that it sampled it."""

    def __init__(self) -> None:
        self.desired: Action | None = None
        self.source: Action | None = None
        self.issued_frame: int | None = None

    def status(self, native_input: int, frame: int) -> LeaseStatus:
        if self.desired is None or self.issued_frame is None:
            return LeaseStatus()
        if native_input & _CONTROL_MASK == _action_mask(self.desired):
            self.cleared()
            return LeaseStatus()
        elapsed = frame - self.issued_frame
        if elapsed < 0 or elapsed >= INPUT_PICKUP_MAX_FRAMES:
            return LeaseStatus(timed_out=True)
        if self.source is None:
            return LeaseStatus(action=self.desired)
        observed_mask = native_input & _CONTROL_MASK
        if observed_mask != _action_mask(self.source):
            return LeaseStatus(timed_out=True)
        # If the source mask is still visible, the next update may retain that
        # whole mask or atomically sample the target.  The issue-age timeout is
        # separate and fails closed at its measured boundary.
        return LeaseStatus(
            action=self.desired,
            delivery_delays=(0, 1),
        )

    def issued(
        self,
        frame: int,
        action: Action,
        source: Action | None = None,
    ) -> None:
        self.desired = action
        self.source = source
        self.issued_frame = frame

    def cleared(self) -> None:
        self.desired = None
        self.source = None
        self.issued_frame = None
