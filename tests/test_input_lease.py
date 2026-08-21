from __future__ import annotations

from th06_rl.retail.input_lease import InputLease
from th06_rl.retail.model import (
    ACTIONS,
    BUTTON_FOCUS,
    BUTTON_LEFT,
    BUTTON_RIGHT,
    BUTTON_UP,
)


BY_NAME = {action.name: action for action in ACTIONS}


def test_lease_accepts_only_source_or_atomic_target_mask() -> None:
    lease = InputLease()
    lease.issued(
        frame=10,
        action=BY_NAME["up_right"],
        source=BY_NAME["left"],
    )

    source = lease.status(BUTTON_FOCUS | BUTTON_LEFT, frame=11)
    assert source.action == BY_NAME["up_right"]
    assert source.delivery_delays == (0, 1)
    assert not source.timed_out

    # This used to be accepted as a sorted SendInput release prefix.  The
    # in-process bridge publishes one DWORD, so a partial mask is impossible.
    partial = lease.status(BUTTON_FOCUS, frame=11)
    assert partial.timed_out
    assert partial.action is None


def test_lease_clears_after_target_mask_is_sampled() -> None:
    lease = InputLease()
    lease.issued(
        frame=10,
        action=BY_NAME["up_right"],
        source=BY_NAME["left"],
    )

    target = lease.status(
        BUTTON_FOCUS | BUTTON_UP | BUTTON_RIGHT,
        frame=11,
    )
    assert target.action is None
    assert not target.timed_out
    assert lease.desired is None
