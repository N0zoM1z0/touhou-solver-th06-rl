from __future__ import annotations

import math
import struct
from dataclasses import replace

import pytest

from th06_rl.retail.model import (
    Bullet,
    BulletPattern,
    EclInstruction,
    EnemySpawner,
    Laser,
    RepeatStarState,
    Snapshot,
    StageTimelineInstruction,
)
from th06_rl.retail.hazards.ecl import forecast_ecl_births
from th06_rl.retail.hazards.rng import RngState
from th06_rl.retail.hazards.world import forecast_world_births
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


def _snapshot(spawner: EnemySpawner, *, lasers=(), bullets=()) -> Snapshot:
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
        bullets=tuple(bullets),
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
        repeat_star_state=RepeatStarState(
            (0.0,) * 6, 192.0, 128.0, 192.0, 400.0
        ),
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
        (189.0, 397.0, 195.0, 403.0)
    )


def test_math_normalize_angle_reads_its_integer_variable_id() -> None:
    random_angle, raw = _instruction(0x10000, 0, 9, 24)
    struct.pack_into("<iff", raw, 12, -10005, math.tau, -math.pi)
    random_angle = _with_raw(random_angle, raw)
    normalize, raw = _instruction(0x10018, 0, 26, 16)
    struct.pack_into("<i", raw, 12, -10005)
    normalize = _with_raw(normalize, raw)
    shoot, raw = _instruction(0x10028, 0, 68, 44)
    struct.pack_into("<hh", raw, 12, 0, 0)
    struct.pack_into("<ii", raw, 16, 1, 1)
    struct.pack_into("<ffff", raw, 24, 1.0, 0.3, -10005.0, 0.0)
    struct.pack_into("<I", raw, 40, 0)
    shoot = _with_raw(shoot, raw)
    shoot_now, _raw = _instruction(0x10054, 0, 80)
    end, _raw = _instruction(0x10060, -1, 0)
    spawner = _spawner(
        random_angle,
        end,
        ecl_program=(random_angle, normalize, shoot, shoot_now, end),
    )

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 400.0),),
        3,
        0,
        ((1.0, 1.0),),
        rng=RngState(0x349E, 0),
        model_player_damage=False,
    )

    assert forecast.covered_frames == 1
    assert forecast.reason == ""
    assert len(forecast.births[0]) == 1
    assert forecast.next_spawner is not None
    assert all(math.isfinite(value) for value in forecast.next_spawner.ecl_floats)


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


def test_source_commitment_unions_exact_cirno_stop_position() -> None:
    ex_call, raw = _instruction(0x10000, 3, 121, 20)
    struct.pack_into("<Ii", raw, 12, 0, 0)
    ex_call = _with_raw(ex_call, raw)
    end, _raw = _instruction(0x10014, -1, 0)
    bullet = Bullet(
        190.0,
        390.0,
        2.0,
        1.0,
        1.0,
        1.0,
        1,
        ex_flags=0x4,
        speed=math.sqrt(5.0),
        sprite_half_width=8.0,
        sprite_half_height=8.0,
    )

    forecast = lower_source_forecast(
        _snapshot(_spawner(ex_call, end), bullets=(bullet,)), 4
    )
    assert forecast.bullet_stop_frames == (3,)
    assert forecast.bullet_release_frames == ()
    centers = {
        ((box.left + box.right) / 2.0, (box.top + box.bottom) / 2.0)
        for box in forecast.hazards.aabb_frames[3]
    }

    # No-damage and candidate-damage branches are unioned by Hard authority.
    # Keep both the ordinary fourth update and the exact position after three
    # updates where EX_CALL 0,param 0 zeros velocity before BulletManager.
    assert (198.0, 394.0) in centers
    assert (196.0, 393.0) in centers


def test_source_commitment_encloses_random_cirno_release_acceleration() -> None:
    ex_call, raw = _instruction(0x10000, 0, 121, 20)
    struct.pack_into("<Ii", raw, 12, 0, 1)
    ex_call = _with_raw(ex_call, raw)
    end, _raw = _instruction(0x10014, -1, 0)
    bullet = Bullet(
        190.0, 390.0, 2.0, 1.0, 1.0, 1.0, 1,
        ex_flags=0x4,
        speed=math.sqrt(5.0),
        sprite_half_width=8.0,
        sprite_half_height=8.0,
    )

    forecast = lower_source_forecast(
        _snapshot(_spawner(ex_call, end), bullets=(bullet,)), 4
    )

    assert forecast.source_coverage == 4
    assert forecast.bullet_stop_frames == ()
    assert forecast.bullet_release_frames == (0,)
    assert any(
        (box.left, box.top, box.right, box.bottom)
        == pytest.approx((196.9, 392.9, 199.1, 395.1))
        for box in forecast.hazards.aabb_frames[3]
    )


def test_source_commitment_rejects_release_over_dynamic_live_bullet() -> None:
    ex_call, raw = _instruction(0x10000, 0, 121, 20)
    struct.pack_into("<Ii", raw, 12, 0, 1)
    ex_call = _with_raw(ex_call, raw)
    end, _raw = _instruction(0x10014, -1, 0)
    bullet = Bullet(
        190.0, 390.0, 0.0, 0.0, 1.0, 1.0, 1,
        ex_flags=0x10,
        acceleration_duration=10,
    )

    with pytest.raises(AuthorityUnavailable, match="global bullet mutation"):
        lower_source_forecast(
            _snapshot(_spawner(ex_call, end), bullets=(bullet,)), 4
        )


def test_source_commitment_encloses_random_area_external_shot() -> None:
    ex_call, raw = _instruction(0x10000, 0, 121, 20)
    struct.pack_into("<Ii", raw, 12, 1, 128)
    ex_call = _with_raw(ex_call, raw)
    end, _raw = _instruction(0x10014, -1, 0)
    pattern = BulletPattern(
        0, 0.0, 0.0, 1.0, 1.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 0, 0x4, 1.0, 1.0,
    )

    forecast = lower_source_forecast(
        _snapshot(_spawner(ex_call, end, pattern=pattern)), 4
    )

    assert forecast.source_coverage == 4
    first = forecast.hazards.aabb_frames[0][0]
    assert first.left <= 126.0
    assert first.right >= 258.0
    assert first.top <= 350.0
    assert first.bottom >= 450.0


def _repeat_star_instruction(address: int, time: int) -> EclInstruction:
    instruction, raw = _instruction(address, time, 122, 16)
    struct.pack_into("<i", raw, 0x0C, 2)
    return _with_raw(instruction, raw)


def _repeat_star_pattern() -> BulletPattern:
    return BulletPattern(
        0, 0.0, 0.0, 1.0, 2.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 1, 0, 1.0, 1.0,
    )


def test_future_repeat_star_instruction_covers_the_full_hard_horizon() -> None:
    repeat = _repeat_star_instruction(0x12000, 3)
    end, _raw = _instruction(0x12010, -1, 0)
    spawner = _spawner(
        repeat,
        end,
        pattern=_repeat_star_pattern(),
        ecl_ints=(0, 0, 0, 12, 0, 0, 0, 0),
        ecl_floats=(0.0, 0.0, 0.0, 10.0),
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),) * 4
    )

    assert forecast.covered_frames == 4
    assert not forecast.reason
    assert len(forecast.births[3]) == 5


def test_repeat_star_hard_envelope_keeps_full_origin_and_speed_support() -> None:
    repeat = _repeat_star_instruction(0x13000, 0)
    end, _raw = _instruction(0x13010, -1, 0)
    star_state = RepeatStarState(
        (0.0,) * 6, 192.0, 128.0, 192.0, 400.0
    )
    spawner = _spawner(
        repeat,
        end,
        pattern=_repeat_star_pattern(),
        ecl_ints=(0, 0, 0, 12, 0, 0, 0, 0),
        ecl_floats=(0.0, 0.0, 0.0, 10.0),
    )

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 400.0),) * 4,
        3,
        0,
        ((1.0, 1.0),),
        radial_births=True,
        abstract_rng=True,
        model_player_damage=False,
        repeat_star_state=star_state,
    )

    assert forecast.covered_frames == 4
    assert len(forecast.births[0]) == 5
    assert all(bullet.half_width == 11.0 for bullet in forecast.births[0])
    assert all(bullet.half_height == 11.0 for bullet in forecast.births[0])
    assert all(bullet.speed == 3.0 for bullet in forecast.births[0])
    assert forecast.next_spawner is not None
    assert forecast.next_spawner.repeat_ex_index == 2
    assert forecast.next_spawner.ecl_ints[2] == 4
    assert forecast.repeat_star_state is not None
    assert forecast.repeat_star_state.angles_known is False


def test_repeat_star_nominal_replay_consumes_exact_shared_rng_order() -> None:
    repeat = _repeat_star_instruction(0x14000, 0)
    end, _raw = _instruction(0x14010, -1, 0)
    star_state = RepeatStarState(
        (0.0,) * 6, 0.0, 0.0, 0.0, 0.0
    )
    spawner = _spawner(
        repeat,
        end,
        pattern=_repeat_star_pattern(),
        ecl_ints=(0, 0, 0, 12, 0, 0, 0, 0),
        ecl_floats=(0.0, 0.0, 0.0, 10.0),
    )
    rng = RngState(1234, 20)

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 400.0),),
        3,
        0,
        ((1.0, 1.0),),
        rng=rng,
        model_player_damage=False,
        repeat_star_state=star_state,
    )

    assert forecast.covered_frames == 1
    assert len(forecast.births[0]) == 5
    # Six f32 draws, each sourced from two GetRandomU16 calls.
    assert rng.generation_count == 32
    assert all(1.0 <= bullet.speed <= 3.0 for bullet in forecast.births[0])
    assert all(
        math.isclose(
            math.hypot(bullet.x - 192.0, bullet.y - 400.0),
            10.0,
            rel_tol=1e-5,
        )
        for bullet in forecast.births[0]
    )
    assert forecast.repeat_star_state is not None
    assert forecast.repeat_star_state.angles_known is True


def test_nominal_repeat_star_globals_flow_in_enemy_slot_order() -> None:
    end, _raw = _instruction(0x15000, -1, 0)
    first = _spawner(
        end,
        end,
        x=100.0,
        y=100.0,
        repeat_ex_index=2,
        pattern=_repeat_star_pattern(),
        ecl_ints=(0, 0, 0, 12, 0, 0, 0, 0),
        ecl_floats=(0.0, 0.0, 0.0, 10.0),
    )
    second = _spawner(
        end,
        end,
        slot=1,
        x=300.0,
        y=200.0,
        repeat_ex_index=2,
        pattern=_repeat_star_pattern(),
        ecl_ints=(0, 0, 6, 12, 0, 0, 0, 0),
        ecl_floats=(0.0, 0.0, 0.0, 0.0),
    )
    snapshot = replace(
        _snapshot(first),
        spawners=(first, second),
        rng_seed=1234,
        rng_generation=20,
    )

    forecast = forecast_world_births(
        snapshot,
        ((192.0, 400.0),),
        rng_mode="nominal",
    )

    assert forecast.covered_frames == 1
    assert len(forecast.births[0]) == 10
    assert all(
        bullet.x == pytest.approx(104.6, abs=1e-4)
        and bullet.y == pytest.approx(115.0, abs=1e-4)
        for bullet in forecast.births[0][5:]
    )
    assert forecast.continuation is not None
    assert forecast.continuation.rng_generation == 42
    assert forecast.continuation.repeat_star_state is not None
    assert forecast.continuation.repeat_star_state.enemy_x == 100.0
    assert forecast.continuation.repeat_star_state.enemy_y == 100.0


def test_hard_rejects_two_shared_repeat_star_writers() -> None:
    end, _raw = _instruction(0x16000, -1, 0)
    first = _spawner(end, end, repeat_ex_index=2)
    second = _spawner(end, end, slot=1, repeat_ex_index=2)
    snapshot = replace(_snapshot(first), spawners=(first, second))

    forecast = forecast_world_births(
        snapshot, ((192.0, 400.0),) * 4
    )

    assert forecast.covered_frames == 0
    assert "shared repeating-star globals" in forecast.reason


def test_hard_timeline_child_cannot_own_shared_repeat_star_globals() -> None:
    live_end, _raw = _instruction(0x17000, -1, 0)
    repeat = _repeat_star_instruction(0x17100, 0)
    child_end, _raw = _instruction(0x17110, -1, 0)
    spawn_raw = bytearray(24)
    struct.pack_into("<fffhh", spawn_raw, 8, 192.0, 128.0, 0.0, 100, -1)
    spawn = StageTimelineInstruction(
        0x20000, 0, 0, 0, len(spawn_raw), spawn_raw.hex()
    )
    timeline_end = StageTimelineInstruction(
        0x20018, -1, 0, 0, 8, "00" * 8
    )
    snapshot = replace(
        _snapshot(_spawner(live_end, live_end)),
        boss_present=False,
        timeline_instructions=(spawn, timeline_end),
        timeline_ecl_program=(repeat, child_end),
        ecl_subroutines=(repeat.address,),
    )

    forecast = forecast_world_births(
        snapshot, ((192.0, 400.0),) * 4
    )

    assert forecast.covered_frames == 0
    assert "captured shared globals" in forecast.reason
