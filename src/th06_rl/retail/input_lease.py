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
    SafeAction,
)
from .safety import transition_input_masks


_CONTROL_MASK = BUTTON_FOCUS | BUTTON_UP | BUTTON_DOWN | BUTTON_LEFT | BUTTON_RIGHT
INPUT_PICKUP_MAX_FRAMES = 2
SEND_INPUT_CROSSING_MAX_FRAMES = 1
BASE_CERTIFIED_DELIVERY_MAX_FRAMES = (
    INPUT_PICKUP_MAX_FRAMES + SEND_INPUT_CROSSING_MAX_FRAMES
)


def bounded_delivery_age(snapshot_frame: int, issue_frame: int) -> int | None:
    """Return an age still covered by the hard pickup branches."""
    age = issue_frame - snapshot_frame
    if 0 <= age <= INPUT_PICKUP_MAX_FRAMES:
        return age
    return None


def changed_action_delivery_supported(
    delivery_age: int,
    current: Action,
    proposed: Action,
    certified_max_delay: int = BASE_CERTIFIED_DELIVERY_MAX_FRAMES,
) -> bool:
    """Whether publication is covered by the available pickup proof."""
    return proposed == current or (
        required_changed_action_delivery_delay(delivery_age)
        <= certified_max_delay
    )


def required_changed_action_delivery_delay(delivery_age: int) -> int:
    """Worst snapshot-relative pickup delay for a command sent at this age."""
    if delivery_age < 0:
        raise ValueError("delivery age cannot be negative")
    return (
        delivery_age
        + SEND_INPUT_CROSSING_MAX_FRAMES
        + INPUT_PICKUP_MAX_FRAMES
    )


def covered_current_retry(
    snapshot_frame: int,
    observed_frame: int,
    horizon: int,
    current: Action,
    safe_actions: tuple[SafeAction, ...],
) -> bool:
    """Whether a late frame is still inside the certified current-input hold."""
    age = observed_frame - snapshot_frame
    return (
        age >= INPUT_PICKUP_MAX_FRAMES + 1
        and age < horizon
        and any(candidate.action == current for candidate in safe_actions)
    )


@dataclass(frozen=True)
class LeaseStatus:
    action: Action | None = None
    timed_out: bool = False
    # Delivery branches still possible on the next game update.  Once a
    # release/press prefix from the already-issued SendInput batch is visible,
    # that batch is known to have crossed the game update and only its settled
    # target can follow.  Treating the prefix as a fresh command invents a
    # second pickup window that was never issued.
    delivery_delays: tuple[int, ...] = (0, 1, 2, 3)


def _issued_mask(action: Action) -> int:
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


class InputLease:
    """Hold a desired direction until the game reports that it sampled it."""

    def __init__(self) -> None:
        self.desired: Action | None = None
        self.source: Action | None = None
        self.issued_frame: int | None = None

    def status(self, native_input: int, frame: int) -> LeaseStatus:
        if self.desired is None or self.issued_frame is None:
            return LeaseStatus()
        if native_input & _CONTROL_MASK == _issued_mask(self.desired):
            self.cleared()
            return LeaseStatus()
        elapsed = frame - self.issued_frame
        if elapsed < 0 or elapsed >= INPUT_PICKUP_MAX_FRAMES:
            return LeaseStatus(timed_out=True)
        if self.source is None:
            return LeaseStatus(action=self.desired)
        observed_mask = native_input & _CONTROL_MASK
        prefixes = transition_input_masks(self.source, self.desired)
        if observed_mask in prefixes:
            return LeaseStatus(action=self.desired, delivery_delays=(0,))
        if observed_mask != _issued_mask(self.source):
            return LeaseStatus(timed_out=True)
        # If the original state is still visible, the next update may retain
        # it, observe one sorted-key prefix, or observe the settled target.
        # The issue-age timeout remains separate and fails closed if pickup
        # has still not completed at its measured boundary.
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
