from __future__ import annotations

import math
import struct

import pytest

from th06_rl.retail.model import (
    BulletPattern,
    EclInstruction,
    EnemySpawner,
    Laser,
    Snapshot,
    StageTimelineInstruction,
)
from th06_rl.th06.source import AuthorityUnavailable, lower_source_forecast


def _instruction(
    address: int,
    time: int,
    opcode: int,
    size: int = 12,
) -> tuple[EclInstruction, bytearray]:
    raw = bytearray(size)
    struct.pack_into("<ihh", raw, 0, time, opcode, size)
    raw[9] = 0xFF
    return EclInstruction(address, time, opcode, size, 0xFF, raw.hex()), raw


def _with_raw(instruction: EclInstruction, raw: bytearray) -> EclInstruction:
    return EclInstruction(
        instruction.address,
        instruction.time,
        instruction.opcode,
        instruction.offset_to_next,
        instruction.skip_for_difficulty,
        raw.hex(),
    )


def _spawner(first: EclInstruction, end: EclInstruction, **changes) -> EnemySpawner:
    values = dict(
        slot=0,
        x=192.0,
        y=400.0,
        velocity_x=0.0,
        velocity_y=0.0,
        angle=0.0,
        angular_velocity=0.0,
        speed=0.0,
        acceleration=0.0,
        movement_mode=0,
        movement_ease=0,
        invert_x=False,
        move_interp_x=0.0,
        move_interp_y=0.0,
        move_start_x=0.0,
        move_start_y=0.0,
        move_timer=0,
        move_timer_float=0.0,
        move_start_time=0,
        shoot_offset_x=0.0,
        shoot_offset_y=0.0,
        bullet_rank_speed_low=0.0,
        bullet_rank_speed_high=0.0,
        bullet_rank_amount1_low=0,
        bullet_rank_amount1_high=0,
        bullet_rank_amount2_low=0,
        bullet_rank_amount2_high=0,
        life=100,
        shooting_disabled=True,
        interval=0,
        timer=0,
        timer_float=0.0,
        pattern=None,
        ecl_time=0,
        ecl_time_float=0.0,
        ecl_ints=(0,) * 8,
        ecl_floats=(0.0,) * 4,
        ecl_compare=0,
        repeat_ex_index=None,
        next_instruction=first,
        ecl_program=(first, end),
    )
    values.update(changes)
    return EnemySpawner(**values)


def _snapshot(spawner: EnemySpawner, *, lasers=()) -> Snapshot:
    timeline_end = StageTimelineInstruction(
        0x20000, -1, 0, 0, 8, "00" * 8
    )
    return Snapshot(
        frame=1,
        stage=1,
        player_state=0,
        x=192.0,
        y=400.0,
        half_width=1.25,
        half_height=1.25,
        normal_speed=4.0,
        focus_speed=2.0,
        normal_diagonal_speed=2.8,
        focus_diagonal_speed=1.4,
        frame_multiplier=1.0,
        input_mask=1,
        bullets=(),
        laser_count=len(lasers),
        in_menu=False,
        time_stopped=False,
        replay_or_demo=False,
        lasers=tuple(lasers),
        spawners=(spawner,),
        difficulty=3,
        rank=0,
        bullet_sizes=((1.0, 1.0),),
        timeline_instructions=(timeline_end,),
        timeline_complete=True,
    )


def test_source_commitment_contains_same_frame_shootnow_birth() -> None:
    shoot, _raw = _instruction(0x10000, 0, 80)
    end, _raw = _instruction(0x1000C, -1, 0)
    pattern = BulletPattern(
        0, 0.0, 0.0, 1.0, 1.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 0, 0x4, 1.0, 1.0,
    )

    forecast = lower_source_forecast(
        _snapshot(_spawner(shoot, end, pattern=pattern)), 4
    )

    assert forecast.source_coverage == 4
    assert all(frame for frame in forecast.hazards.aabb_frames)
    first = forecast.hazards.aabb_frames[0][0]
    assert (first.left, first.top, first.right, first.bottom) == pytest.approx(
        (190.0, 398.0, 194.0, 402.0)
    )


def test_source_commitment_contains_same_frame_enemy_teleport() -> None:
    move, raw = _instruction(0x10000, 0, 43, 24)
    struct.pack_into("<fff", raw, 12, 192.0, 400.0, 0.0)
    move = _with_raw(move, raw)
    end, _raw = _instruction(0x10018, -1, 0)
    spawner = _spawner(
        move,
        end,
        x=0.0,
        y=0.0,
        hitbox_half_width=4.0,
        hitbox_half_height=4.0,
        interactable=True,
        collidable=True,
        has_been_in_bounds=True,
        sprite_half_width=8.0,
        sprite_half_height=8.0,
    )

    first = lower_source_forecast(
        _snapshot(spawner), 4
    ).hazards.aabb_frames[0][0]

    assert (first.left, first.top, first.right, first.bottom) == pytest.approx(
        (188.0, 396.0, 196.0, 404.0)
    )


def test_source_commitment_contains_ecl_laser_rotation() -> None:
    rotate, raw = _instruction(0x10000, 0, 88, 28)
    struct.pack_into("<i", raw, 12, 0)
    struct.pack_into("<f", raw, 16, math.pi / 2.0)
    rotate = _with_raw(rotate, raw)
    end, _raw = _instruction(0x1001C, -1, 0)
    spawner = _spawner(
        rotate,
        end,
        x=192.0,
        y=100.0,
        laser_slots=(0,) + (-1,) * 31,
    )
    laser = Laser(
        192.0, 100.0, 0.0, 0.0, 100.0, 100.0, 8.0, 0.0,
        0, 0, 100, 10, 0, 10, 10.0, 0, 1, slot=0,
    )

    angles = {
        hazard.angle
        for hazard in lower_source_forecast(
            _snapshot(spawner, lasers=(laser,)), 4
        ).hazards.laser_frames[0]
    }

    # The unchanged live projection is conservatively retained alongside the
    # source-mutated laser. The important property is that rotation is not
    # omitted from Hard authority.
    assert 0.0 in angles
    assert any(angle == pytest.approx(math.pi / 2.0) for angle in angles)


def test_source_commitment_fails_closed_on_unsupported_ecl() -> None:
    # SETVARSELFZ is explicitly unmodelled because the compact source world
    # does not capture the enemy z coordinate.
    unsupported, _raw = _instruction(0x10000, 0, 12)
    end, _raw = _instruction(0x1000C, -1, 0)

    with pytest.raises(AuthorityUnavailable, match="coverage ended"):
        lower_source_forecast(_snapshot(_spawner(unsupported, end)), 4)
