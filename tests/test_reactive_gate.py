from __future__ import annotations

from th06_rl.native import ACTIONS, NativeCertifiedAction
from th06_rl.th06.controller import (
    _certify_hard_with_source_fallback,
    _reactive_baseline,
)


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


class _RecordingKernel:
    def __init__(self, conservative, source_exact):
        self.results = {0.35: conservative, 0.0: source_exact}
        self.margins = []

    def certify_actions(self, **kwargs):
        margin = kwargs["collision_margin"]
        self.margins.append(margin)
        return self.results[margin]


def test_hard_gate_keeps_conservative_margin_when_nonempty() -> None:
    expected = (certified("left", 1.0, 180.0, 384.0),)
    kernel = _RecordingKernel(expected, ())

    result, margin = _certify_hard_with_source_fallback(kernel, token="root")

    assert result == expected
    assert margin == 0.35
    assert kernel.margins == [0.35]


def test_hard_gate_uses_source_exact_geometry_only_after_margin_closure() -> None:
    expected = (certified("right", 0.2, 196.0, 384.0),)
    kernel = _RecordingKernel((), expected)

    result, margin = _certify_hard_with_source_fallback(kernel, token="root")

    assert result == expected
    assert margin == 0.0
    assert kernel.margins == [0.35, 0.0]
