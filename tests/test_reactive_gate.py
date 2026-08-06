from __future__ import annotations

from th06_rl.native import ACTIONS, NativeCertifiedAction
from th06_rl.th06.controller import _reactive_baseline


BY_NAME = {action.name: action for action in ACTIONS}


def certified(name: str, clearance: float, x: float, y: float):
    return NativeCertifiedAction(BY_NAME[name], clearance, x, y)


def test_reactive_fallback_prioritizes_physical_clearance() -> None:
    result = _reactive_baseline(
        (
            certified("stay", 3.0, 192.0, 384.0),
            certified("left", 9.0, 180.0, 384.0),
        ),
        BY_NAME["stay"],
    )

    assert result.action == BY_NAME["left"]


def test_reactive_fallback_uses_boundary_reserve_only_after_safety() -> None:
    result = _reactive_baseline(
        (
            certified("left", 9.0, 10.0, 384.0),
            certified("right", 9.0, 30.0, 384.0),
        ),
        BY_NAME["left"],
    )

    assert result.action == BY_NAME["right"]
