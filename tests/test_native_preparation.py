from __future__ import annotations

import math

import pytest

from th06_rl.core.model import Kinematics
from th06_rl.native import (
    ACTIONS,
    Aabb,
    LaserRect,
    NativeKernel,
    PackedHazards,
)


BY_NAME = {action.name: action for action in ACTIONS}
KINEMATICS = Kinematics(4.0, 2.0, 2.828427, 1.414214)


def certify_stay(aabb: Aabb | None):
    frame = () if aabb is None else (aabb,)
    return NativeKernel().certify_actions(
        x=192.0,
        y=384.0,
        half_width=2.0,
        half_height=2.0,
        kinematics=KINEMATICS,
        current_action=BY_NAME["stay"],
        hazards=PackedHazards(
            aabb_frames=(frame,) * 4,
            laser_frames=((),) * 4,
        ),
        candidates=(BY_NAME["stay"],),
    )


def test_prepared_prefix_reuses_the_exact_native_buffers() -> None:
    hazards = PackedHazards(
        aabb_frames=(
            (Aabb(1.0, 2.0, 3.0, 4.0),),
            (Aabb(5.0, 6.0, 7.0, 8.0),),
        ),
        laser_frames=(
            (LaserRect(9.0, 10.0, 0.5, 11.0, 12.0, 13.0),),
            (LaserRect(14.0, 15.0, 1.5, 16.0, 17.0, 18.0),),
        ),
    )

    prepared = NativeKernel.prepare_hazards(hazards)
    shield = prepared.prefix(1)

    assert prepared.horizon == 2
    assert shield.horizon == 1
    assert all(
        hard_buffer is full_buffer
        for hard_buffer, full_buffer in zip(
            shield.native_args,
            prepared.native_args,
            strict=True,
        )
    )
    assert list(prepared.aabb_offsets) == [0, 1, 2]
    assert list(prepared.laser_offsets) == [0, 1, 2]


def test_prepared_prefix_cannot_expand_or_be_empty() -> None:
    hazards = PackedHazards(
        aabb_frames=((), ()),
        laser_frames=((), ()),
    )
    prepared = NativeKernel.prepare_hazards(hazards)

    with pytest.raises(ValueError):
        prepared.prefix(0)
    with pytest.raises(ValueError):
        prepared.prefix(3)


def test_native_clearance_defers_distance_without_changing_value() -> None:
    certified = certify_stay(Aabb(197.0, 390.0, 198.0, 391.0))

    assert len(certified) == 1
    assert math.isclose(certified[0].min_clearance, 5.0)
    assert math.isinf(certify_stay(None)[0].min_clearance)


def test_native_collision_margin_keeps_exact_diagonal_boundary() -> None:
    outside = certify_stay(Aabb(194.2, 386.3, 195.0, 387.0))
    inside = certify_stay(Aabb(194.2, 386.2, 195.0, 387.0))

    assert len(outside) == 1
    assert not inside


def test_delivery_delay_uses_only_atomic_source_and_target_masks() -> None:
    hazards = PackedHazards(
        # A fabricated key-release prefix would stay at x=192 and collide.
        # Atomic pickup holds the complete left mask for frame 1 (x=190), then
        # applies the complete right mask, so the candidate is admissible.
        aabb_frames=((Aabb(194.0, 383.0, 195.0, 385.0),), (), (), ()),
        laser_frames=((), (), (), ()),
    )

    certified = NativeKernel().certify_actions(
        x=192.0,
        y=384.0,
        half_width=2.0,
        half_height=2.0,
        kinematics=KINEMATICS,
        current_action=BY_NAME["left"],
        hazards=hazards,
        candidates=(BY_NAME["right"],),
        delivery_delays=(1,),
    )

    assert [item.action.name for item in certified] == ["right"]
    assert math.isclose(certified[0].final_x, 196.0)
