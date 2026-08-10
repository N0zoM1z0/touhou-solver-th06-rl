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
    PlayerAimedBullet,
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
    hard = prepared.prefix(1)

    assert prepared.horizon == 2
    assert hard.horizon == 1
    assert all(
        hard_buffer is full_buffer
        for hard_buffer, full_buffer in zip(
            hard.native_args,
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


def test_native_action_profile_retains_checkpoint_clearance() -> None:
    frames = [()] * 12
    frames[7] = (Aabb(197.0, 390.0, 198.0, 391.0),)
    profile = NativeKernel().profile_actions(
        x=192.0,
        y=384.0,
        half_width=2.0,
        half_height=2.0,
        kinematics=KINEMATICS,
        current_action=BY_NAME["stay"],
        hazards=PackedHazards(
            aabb_frames=tuple(frames),
            laser_frames=((),) * 12,
        ),
        candidates=(BY_NAME["stay"],),
    )[0]

    assert math.isinf(profile.min_clearances[0])
    assert profile.min_clearances[1:] == (5.0, 5.0)


def test_native_player_aim_turn_is_coupled_to_each_candidate() -> None:
    hazards = PackedHazards(
        aabb_frames=((),) * 4,
        laser_frames=((),) * 4,
        player_aimed_bullets=(PlayerAimedBullet(
            x=100.0,
            y=80.0,
            vx=0.0,
            vy=0.0,
            half_width=1.0,
            half_height=1.0,
            speed=0.0,
            angle=0.0,
            turn_speed=10.0,
            direction_rotation=0.0,
            timer_float=0.0,
            timer=0,
            direction_interval=0,
            direction_num_times=0,
            direction_max_times=1,
        ),),
    )

    certified = NativeKernel().certify_actions(
        x=100.0,
        y=100.0,
        half_width=1.0,
        half_height=1.0,
        kinematics=KINEMATICS,
        current_action=BY_NAME["stay"],
        hazards=hazards,
        delivery_delays=(0,),
        collision_margin=0.0,
    )
    names = {item.action.name for item in certified}

    # The final aim turn hits a stationary candidate on frame two. Mutually
    # exclusive horizontal targets retain their own trajectories and escape.
    assert "stay" not in names
    assert {"left_fast", "right_fast"} <= names


def test_candidate_coupled_aim_is_rejected_by_feature_only_profile() -> None:
    hazards = PackedHazards(
        aabb_frames=((),) * 4,
        laser_frames=((),) * 4,
        player_aimed_bullets=(PlayerAimedBullet(
            100.0, 80.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0,
            10.0, 0.0, 0.0, 0, 0, 0, 1,
        ),),
    )

    with pytest.raises(ValueError, match="Hard-certification only"):
        NativeKernel().profile_actions(
            x=100.0,
            y=100.0,
            half_width=1.0,
            half_height=1.0,
            kinematics=KINEMATICS,
            current_action=BY_NAME["stay"],
            hazards=hazards,
            candidates=(BY_NAME["stay"],),
            checkpoints=(4,),
        )
