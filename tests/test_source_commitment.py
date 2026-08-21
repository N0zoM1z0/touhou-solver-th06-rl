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
    FloatInterval,
    Laser,
    PlayerAttackState,
    RepeatStarState,
    Snapshot,
    StageTimelineInstruction,
)
from th06_rl.retail.hazards.ecl import forecast_ecl_births
from th06_rl.retail.hazards.rng import RngState
from th06_rl.retail.hazards.world import (
    _patchouli_shottype_vars,
    forecast_world_births,
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


def test_hard_carries_random_bullet_effect_support_across_frames() -> None:
    random_rotation, raw = _instruction(0x10000, 0, 9, 24)
    struct.pack_into("<iff", raw, 12, -10005, math.pi, -math.pi / 2.0)
    random_rotation = _with_raw(random_rotation, raw)
    effects, raw = _instruction(0x10018, 0, 82, 44)
    struct.pack_into("<iiii", raw, 12, 90, 1, -1, -1)
    struct.pack_into("<ffff", raw, 28, -10005.0, 1.8, -1.0, -1.0)
    effects = _with_raw(effects, raw)
    shoot_random, raw = _instruction(0x10044, 0, 75, 44)
    struct.pack_into("<hhii", raw, 12, 1, 0, 12, 1)
    struct.pack_into("<ffffI", raw, 24, 2.4, 1.0, math.pi, 0.0, 0x43)
    shoot_random = _with_raw(shoot_random, raw)
    shoot_now, _raw = _instruction(0x10070, 1, 80)
    end, _raw = _instruction(0x1007C, -1, 0)
    spawner = _spawner(
        random_rotation,
        end,
        shooting_disabled=False,
        ecl_program=(random_rotation, effects, shoot_random, shoot_now, end),
    )

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 400.0),) * 2,
        3,
        0,
        ((1.0, 1.0), (8.0, 8.0)),
        radial_births=True,
        abstract_rng=True,
        model_player_damage=False,
    )

    assert forecast.covered_frames == 2
    assert forecast.reason == ""
    assert len(forecast.births[0]) == 1
    assert len(forecast.births[1]) == 1
    bullet = forecast.births[1][0]
    assert bullet.turn_speed == pytest.approx(1.8)
    assert bullet.direction_rotation == 0.0
    assert forecast.next_spawner is not None
    support = forecast.next_spawner.bullet_effect_floats[0]
    assert isinstance(support, FloatInterval)
    assert support.low == pytest.approx(-math.pi / 2.0)
    assert support.high == pytest.approx(math.pi / 2.0)


def test_exact_forecast_rejects_random_bullet_effect_support() -> None:
    random_rotation, raw = _instruction(0x10100, 0, 9, 24)
    struct.pack_into("<iff", raw, 12, -10005, math.pi, -math.pi / 2.0)
    random_rotation = _with_raw(random_rotation, raw)
    effects, raw = _instruction(0x10118, 0, 82, 44)
    struct.pack_into("<iiii", raw, 12, 90, 1, -1, -1)
    struct.pack_into("<ffff", raw, 28, -10005.0, 1.8, -1.0, -1.0)
    effects = _with_raw(effects, raw)
    end, _raw = _instruction(0x10144, -1, 0)
    spawner = _spawner(
        random_rotation,
        end,
        ecl_program=(random_rotation, effects, end),
    )

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 400.0),),
        3,
        0,
        ((1.0, 1.0),),
        abstract_rng=True,
        model_player_damage=False,
    )

    assert forecast.covered_frames == 0
    assert forecast.reason == "uncertain bullet effects need a hard envelope"


def test_hard_unions_future_player_comparison_branches() -> None:
    compare, raw = _instruction(0x10200, 0, 28, 20)
    struct.pack_into("<ff", raw, 12, -10019.0, 200.0)
    compare = _with_raw(compare, raw)
    jump_greater, raw = _instruction(0x10214, 0, 32, 20)
    struct.pack_into("<ii", raw, 12, 0, 0x40)
    jump_greater = _with_raw(jump_greater, raw)
    shoot, raw = _instruction(0x10228, 0, 67, 44)
    struct.pack_into("<hhii", raw, 12, 0, 0, 1, 1)
    struct.pack_into("<ffffI", raw, 24, 1.0, 1.0, 0.0, 0.0, 0)
    shoot = _with_raw(shoot, raw)
    wait, _raw = _instruction(0x10254, 10, 0)
    spawner = _spawner(
        compare,
        wait,
        shooting_disabled=False,
        ecl_program=(compare, jump_greater, shoot, wait),
    )

    hard = forecast_ecl_births(
        spawner,
        ((192.0, 300.0),) * 4,
        3,
        0,
        ((1.0, 1.0),),
        allow_player_variables=False,
        radial_births=True,
        abstract_rng=True,
        model_player_damage=False,
    )
    below = forecast_ecl_births(
        spawner,
        ((192.0, 100.0),),
        3,
        0,
        ((1.0, 1.0),),
        model_player_damage=False,
    )
    above = forecast_ecl_births(
        spawner,
        ((192.0, 300.0),),
        3,
        0,
        ((1.0, 1.0),),
        model_player_damage=False,
    )

    assert hard.covered_frames == 4
    assert hard.reason == ""
    assert len(hard.births[0]) == 1
    assert len(below.births[0]) == 1
    assert above.births[0] == ()


def test_unreached_laser_graph_keeps_world_comparison_branch_union() -> None:
    compare, raw = _instruction(0x10280, 0, 28, 20)
    struct.pack_into("<ff", raw, 12, -10019.0, 200.0)
    compare = _with_raw(compare, raw)
    jump_greater, raw = _instruction(0x10294, 0, 32, 20)
    struct.pack_into("<ii", raw, 12, 0, 0x40)
    jump_greater = _with_raw(jump_greater, raw)
    shoot, raw = _instruction(0x102A8, 0, 67, 44)
    struct.pack_into("<hhii", raw, 12, 0, 0, 1, 1)
    struct.pack_into("<ffffI", raw, 24, 1.0, 1.0, 0.0, 0.0, 0)
    shoot = _with_raw(shoot, raw)
    wait, _raw = _instruction(0x102D4, 10, 0)
    create = _laser_create_instruction(0x102E0, time=10)
    end, _raw = _instruction(0x10320, -1, 0)
    spawner = _spawner(
        compare,
        end,
        shooting_disabled=False,
        ecl_program=(compare, jump_greater, shoot, wait, create, end),
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 300.0),) * 4
    )

    assert forecast.covered_frames == 4
    assert forecast.reason == ""
    assert forecast.laser_births == 0
    assert forecast.laser_effect_worlds == 0


def test_nonabstract_future_player_comparison_fails_closed() -> None:
    compare, raw = _instruction(0x10300, 0, 28, 20)
    struct.pack_into("<ff", raw, 12, -10019.0, 200.0)
    compare = _with_raw(compare, raw)
    wait, _raw = _instruction(0x10314, 10, 0)
    spawner = _spawner(compare, wait, ecl_program=(compare, wait))

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 300.0),),
        3,
        0,
        ((1.0, 1.0),),
        allow_player_variables=False,
        radial_births=True,
        model_player_damage=False,
    )

    assert forecast.covered_frames == 0
    assert forecast.reason == (
        "uncertain ECL comparison needs a hard branch union"
    )


def test_hard_stops_before_ecl_reads_damage_uncertain_life() -> None:
    compare, raw = _instruction(0x10100, 1, 27, 20)
    struct.pack_into("<ii", raw, 0x0C, -10024, 100)
    compare = _with_raw(compare, raw)
    end, _raw = _instruction(0x10114, -1, 0)
    spawner = _spawner(
        compare,
        end,
        interactable=True,
        damageable=True,
        life=100,
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),) * 2
    )

    assert forecast.covered_frames == 1
    assert "candidate player damage" in forecast.reason


def test_framewise_laser_world_keeps_damage_uncertainty() -> None:
    rotate, raw = _instruction(0x10120, 0, 88, 28)
    struct.pack_into("<i", raw, 0x0C, 0)
    struct.pack_into("<f", raw, 0x10, 0.0)
    rotate = _with_raw(rotate, raw)
    compare, raw = _instruction(0x1013C, 1, 27, 20)
    struct.pack_into("<ii", raw, 0x0C, -10024, 100)
    compare = _with_raw(compare, raw)
    end, _raw = _instruction(0x10150, -1, 0)
    spawner = _spawner(
        rotate,
        end,
        interactable=True,
        damageable=True,
        life=100,
        laser_slots=(0,) + (-1,) * 31,
        ecl_program=(rotate, compare, end),
    )
    laser = Laser(
        192.0, 100.0, 0.0, 0.0, 100.0, 100.0, 8.0, 0.0,
        0, 0, 100, 10, 0, 10, 10.0, 0, 1, slot=0,
    )

    forecast = forecast_world_births(
        _snapshot(spawner, lasers=(laser,)), ((192.0, 400.0),) * 2
    )

    assert forecast.covered_frames == 1
    assert "candidate player damage" in forecast.reason


def test_exact_life_set_clears_damage_uncertainty_before_ecl_read() -> None:
    life_set = _set_int_instruction(0x10140, 1, 111, 123)
    compare, raw = _instruction(0x10150, 1, 27, 20)
    struct.pack_into("<ii", raw, 0x0C, -10024, 123)
    compare = _with_raw(compare, raw)
    end, _raw = _instruction(0x10164, -1, 0)
    spawner = _spawner(
        life_set,
        end,
        interactable=True,
        damageable=True,
        life=100,
        ecl_program=(life_set, compare, end),
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),) * 2
    )

    assert forecast.covered_frames == 2


def test_life_callback_union_expands_integer_rng_in_no_callback_path() -> None:
    random_int, raw = _instruction(0x10180, 3, 6, 20)
    struct.pack_into("<ii", raw, 0x0C, -10001, 3)
    random_int = _with_raw(random_int, raw)
    main_end, _raw = _instruction(0x10194, -1, -1)
    callback_end, _raw = _instruction(0x10200, -1, -1)
    spawner = _spawner(
        random_int,
        main_end,
        ecl_program=(random_int, main_end, callback_end),
        ecl_subroutines=(0x10180, 0x10200),
        life=950,
        interactable=True,
        damageable=True,
        life_callback_threshold=900,
        life_callback_sub=1,
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),) * 4
    )

    assert forecast.covered_frames == 4
    assert forecast.reason == ""


def test_callback_prefix_rejects_divergent_rng_continuations() -> None:
    random_int, raw = _instruction(0x10220, 0, 6, 20)
    struct.pack_into("<ii", raw, 0x0C, -10001, 3)
    random_int = _with_raw(random_int, raw)
    wait, _raw = _instruction(0x10234, 10, 0)
    main_end, _raw = _instruction(0x10240, -1, -1)
    callback_end, _raw = _instruction(0x10300, -1, -1)
    spawner = _spawner(
        random_int,
        main_end,
        ecl_program=(random_int, wait, main_end, callback_end),
        ecl_subroutines=(0x10220, 0x10300),
        life=950,
        interactable=True,
        damageable=True,
        life_callback_threshold=900,
        life_callback_sub=1,
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),) * 2
    )

    assert forecast.covered_frames == 1
    assert "unrepresentable live continuations" in forecast.reason


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


def test_source_commitment_replays_external_shots_from_live_laser_angles() -> None:
    ex_call, raw = _instruction(0x11000, 0, 121, 20)
    # ExInsStage4Func12 ignores its instruction parameter.
    struct.pack_into("<Ii", raw, 12, 12, 37)
    ex_call = _with_raw(ex_call, raw)
    end, _raw = _instruction(0x11014, -1, 0)
    pattern = BulletPattern(
        0, 0.0, 0.0, 1.0, 1.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 0, 0x4, 1.0, 1.0,
    )
    spawner = _spawner(
        ex_call,
        end,
        pattern=pattern,
        # Duplicate pointers shoot twice; slot 1 is a stale pointer and is
        # skipped because no in-use laser occupies it.
        laser_slots=(0, 0, 1) + (-1,) * 29,
    )
    angle = math.pi / 4.0
    laser = Laser(
        192.0, 100.0, angle, 0.0, 100.0, 100.0, 8.0, 0.0,
        0, 0, 100, 10, 0, 10, 10.0, 0, 1, slot=0,
    )

    forecast = forecast_world_births(
        _snapshot(spawner, lasers=(laser,)), ((192.0, 400.0),)
    )

    assert forecast.covered_frames == 1
    assert forecast.reason == ""
    assert len(forecast.births[0]) == 2
    cosine = struct.unpack("<f", struct.pack("<f", math.cos(angle)))[0]
    sine = struct.unpack("<f", struct.pack("<f", math.sin(angle)))[0]
    offset_x = struct.unpack("<f", struct.pack("<f", cosine * 64.0))[0]
    offset_y = struct.unpack("<f", struct.pack("<f", -sine * offset_x))[0]
    assert all(
        (bullet.x, bullet.y)
        == pytest.approx((192.0 + offset_x, 400.0 + offset_y))
        for bullet in forecast.births[0]
    )


def test_external_laser_shot_requests_the_mutable_laser_world() -> None:
    ex_call, raw = _instruction(0x11100, 0, 121, 20)
    struct.pack_into("<Ii", raw, 12, 12, 0)
    ex_call = _with_raw(ex_call, raw)
    end, _raw = _instruction(0x11114, -1, 0)
    pattern = BulletPattern(
        0, 0.0, 0.0, 1.0, 1.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 0, 0x4, 1.0, 1.0,
    )

    forecast = forecast_ecl_births(
        _spawner(
            ex_call,
            end,
            pattern=pattern,
            laser_slots=(0,) + (-1,) * 31,
        ),
        ((192.0, 400.0),),
        3,
        0,
        ((1.0, 1.0),),
        radial_births=True,
        abstract_rng=True,
    )

    assert forecast.covered_frames == 0
    assert "laser world" in forecast.reason


@pytest.mark.parametrize(
    ("character", "shot_type", "expected"),
    (
        (0, 0, (0, 3, 1)),
        (0, 1, (2, 3, 4)),
        (1, 0, (1, 4, 0)),
        (1, 1, (4, 2, 3)),
    ),
)
def test_patchouli_external_call_sets_exact_character_shot_vars(
    character: int,
    shot_type: int,
    expected: tuple[int, int, int],
) -> None:
    ex_call, raw = _instruction(0x11120, 0, 121, 20)
    struct.pack_into("<Ii", raw, 12, 3, 37)
    ex_call = _with_raw(ex_call, raw)
    wait, _raw = _instruction(0x11134, 10, 0)
    attack = PlayerAttackState(
        shots=(),
        last_enemy_hit_x=0.0,
        last_enemy_hit_y=0.0,
        orb_state=0,
        is_focus=False,
        focus_timer_previous=0,
        focus_timer=0,
        focus_timer_float=0.0,
        fire_timer_previous=0,
        fire_timer=0,
        fire_timer_float=0.0,
        orb_positions=((0.0, 0.0), (0.0, 0.0)),
        shot_type=shot_type,
        bomb_active=False,
        spell_active=False,
    )
    snapshot = replace(
        _snapshot(_spawner(ex_call, wait)),
        character=character,
        player_attack=attack,
    )
    variables = _patchouli_shottype_vars(snapshot)
    spawner = replace(snapshot.spawners[0], patchouli_shottype_vars=variables)

    forecast = forecast_ecl_births(
        spawner,
        ((192.0, 400.0),),
        3,
        0,
        ((1.0, 1.0),),
        model_player_damage=False,
    )

    assert variables == expected
    assert forecast.covered_frames == 1
    assert forecast.next_spawner is not None
    assert forecast.next_spawner.ecl_ints[1:4] == expected


def test_patchouli_external_call_fails_without_shot_scope() -> None:
    ex_call, raw = _instruction(0x11120, 0, 121, 20)
    struct.pack_into("<Ii", raw, 12, 3, 0)
    ex_call = _with_raw(ex_call, raw)
    wait, _raw = _instruction(0x11134, 10, 0)

    forecast = forecast_ecl_births(
        _spawner(ex_call, wait),
        ((192.0, 400.0),),
        3,
        0,
        ((1.0, 1.0),),
        model_player_damage=False,
    )

    assert forecast.covered_frames == 0
    assert "shot type" in forecast.reason


def _laser_create_instruction(address: int, *, time: int = 0) -> EclInstruction:
    instruction, raw = _instruction(address, time, 85, 64)
    struct.pack_into("<fffff", raw, 0x10, 0.0, 0.0, 0.0, 100.0, 10.0)
    struct.pack_into("<f", raw, 0x24, 4.0)
    struct.pack_into("<iiiiii", raw, 0x28, 0, 60, 10, 0, 0, 0)
    return _with_raw(instruction, raw)


def _laser_rotate_instruction(address: int) -> EclInstruction:
    instruction, raw = _instruction(address, 0, 88, 20)
    struct.pack_into("<if", raw, 0x0C, 0, 0.1)
    return _with_raw(instruction, raw)


def test_one_laser_world_replays_stale_pointer_reuse_in_source_order() -> None:
    rotate_missing = _laser_rotate_instruction(0x11140)
    create = _laser_create_instruction(0x11154)
    rotate_aliased = _laser_rotate_instruction(0x11194)
    end, _raw = _instruction(0x111A8, -1, 0)
    spawner = _spawner(
        rotate_missing,
        end,
        ecl_program=(rotate_missing, create, rotate_aliased, end),
        laser_slots=(0,) + (-1,) * 31,
        laser_store=1,
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),)
    )

    assert forecast.covered_frames == 1
    assert forecast.reason == ""
    assert forecast.laser_births == 1
    assert forecast.laser_effect_worlds == 1
    assert forecast.missing_laser_dereferences == (0,)
    assert forecast.laser_hazards[0][0].angle == pytest.approx(0.1)


def test_one_create_only_world_keeps_candidate_damage_branch_union() -> None:
    create = _laser_create_instruction(0x111B0)
    wait, _raw = _instruction(0x111F0, 10, 0)
    callback_wait, _raw = _instruction(0x111FC, 10, 0)
    spawner = _spawner(
        create,
        wait,
        life=100,
        interactable=True,
        damageable=True,
        life_callback_threshold=99,
        life_callback_sub=1,
        ecl_program=(create, wait, callback_wait),
        ecl_subroutines=(create.address, callback_wait.address),
    )

    forecast = forecast_world_births(
        _snapshot(spawner), ((192.0, 400.0),) * 4
    )

    assert forecast.covered_frames == 4
    assert forecast.reason == ""
    # With no second owner, the source-conservative static beam union is
    # sufficient; forcing one shared mutable state would erase the valid
    # candidate-damage callback branches.
    assert forecast.laser_births == 0
    assert forecast.laser_effect_worlds == 0
    assert any(forecast.body_hazards)


def test_cross_emitter_stale_pointer_reuse_remains_fail_closed() -> None:
    create = _laser_create_instruction(0x111A0)
    creator_end, _raw = _instruction(0x111E0, -1, 0)
    creator = _spawner(create, creator_end)
    rotate = _laser_rotate_instruction(0x111EC)
    stale_end, _raw = _instruction(0x11200, -1, 0)
    stale = _spawner(
        rotate,
        stale_end,
        slot=1,
        laser_slots=(0,) + (-1,) * 31,
    )
    snapshot = replace(
        _snapshot(creator),
        spawners=(creator, stale),
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),))

    assert forecast.covered_frames == 0
    assert forecast.laser_effect_worlds == 2
    assert forecast.reason == (
        "future laser allocation may alias a stale ECL pointer"
    )


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


def _set_int_instruction(
    address: int,
    time: int,
    opcode: int,
    value: int,
) -> EclInstruction:
    instruction, raw = _instruction(address, time, opcode, 16)
    struct.pack_into("<i", raw, 0x0C, value)
    return _with_raw(instruction, raw)


def _timeline_spawn(
    address: int,
    time: int,
    sub_id: int,
) -> StageTimelineInstruction:
    raw = bytearray(24)
    struct.pack_into("<fffhh", raw, 8, 192.0, 128.0, 0.0, 100, -1)
    return StageTimelineInstruction(
        address, time, sub_id, 0, len(raw), raw.hex()
    )


def _timeline_kill_all_snapshot(
    child_program: tuple[EclInstruction, ...],
    child_subroutines: tuple[int, ...],
    *,
    live: EnemySpawner | None = None,
    prior_spawn: bool = False,
    target_sub_id: int = 0,
) -> Snapshot:
    live_end, _raw = _instruction(0x1A000, -1, 0)
    root = live if live is not None else _spawner(live_end, live_end)
    timeline = []
    if prior_spawn:
        timeline.append(_timeline_spawn(0x2A000, 1, 0))
    timeline.append(_timeline_spawn(0x2A018, 3, target_sub_id))
    timeline.append(
        StageTimelineInstruction(0x2A030, -1, 0, 0, 8, "00" * 8)
    )
    return replace(
        _snapshot(root),
        boss_present=False,
        timeline_instructions=tuple(timeline),
        timeline_ecl_program=child_program,
        ecl_subroutines=child_subroutines,
    )


def test_timeline_kill_all_allows_source_ordered_noninteractive_child() -> None:
    # Stage 4 sub21 has this material ordering: INTERACTABLE(0), KILLALL,
    # then DEATHCALLBACKSUB.  SpawnEnemy initialized the callback to -1, so
    # kill-all cannot invoke it; the later non-interactive state also cannot
    # take the ordinary EnemyManager death path.
    interact = _set_int_instruction(0x18000, 0, 117, 0)
    kill_all, _raw = _instruction(0x18010, 0, 96)
    callback = _set_int_instruction(0x1801C, 0, 108, 1)
    wait, _raw = _instruction(0x1802C, -1, 0)
    snapshot = _timeline_kill_all_snapshot(
        (interact, kill_all, callback, wait),
        (interact.address, wait.address),
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 5


def test_timeline_kill_all_rejects_active_self_callback() -> None:
    callback = _set_int_instruction(0x18100, 0, 108, 1)
    kill_all, _raw = _instruction(0x18110, 0, 96)
    wait, _raw = _instruction(0x1811C, -1, 0)
    snapshot = _timeline_kill_all_snapshot(
        (callback, kill_all, wait),
        (callback.address, wait.address),
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "active death callback" in forecast.reason


def test_timeline_kill_all_rejects_delayed_interactive_callback() -> None:
    kill_all, _raw = _instruction(0x18200, 0, 96)
    callback = _set_int_instruction(0x1820C, 0, 108, 1)
    wait, _raw = _instruction(0x1821C, -1, 0)
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, callback, wait),
        (kill_all.address, wait.address),
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "assigned later" in forecast.reason


def test_timeline_kill_all_rejects_same_call_enemy_creation() -> None:
    create, raw = _instruction(0x18240, 0, 95, 32)
    struct.pack_into("<iff", raw, 0x0C, 1, 192.0, 128.0)
    struct.pack_into("<hh", raw, 0x1C, 100, -1)
    create = _with_raw(create, raw)
    kill_all, _raw = _instruction(0x18260, 0, 96)
    wait, _raw = _instruction(0x1826C, -1, 0)
    snapshot = _timeline_kill_all_snapshot(
        (create, kill_all, wait),
        (create.address, wait.address),
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "ENEMYCREATE/ENEMYKILLALL" in forecast.reason


def test_timeline_kill_all_rejects_live_slot_created_in_prefix() -> None:
    kill_all, _raw = _instruction(0x18280, 0, 96)
    child_wait, _raw = _instruction(0x1828C, -1, 0)
    create, raw = _instruction(0x19280, 0, 95, 32)
    struct.pack_into("<iff", raw, 0x0C, 1, 192.0, 128.0)
    struct.pack_into("<hh", raw, 0x1C, 100, -1)
    create = _with_raw(create, raw)
    live_wait, _raw = _instruction(0x192A0, -1, 0)
    live = _spawner(
        create,
        live_wait,
        ecl_program=(create, live_wait),
        ecl_subroutines=(create.address, live_wait.address),
    )
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, child_wait),
        (kill_all.address, child_wait.address),
        live=live,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 0
    assert "inserts a live slot" in forecast.reason


def test_timeline_kill_all_rejects_active_live_slot_callback() -> None:
    kill_all, _raw = _instruction(0x18300, 0, 96)
    wait, _raw = _instruction(0x1830C, -1, 0)
    live_end, _raw = _instruction(0x19300, -1, 0)
    live = _spawner(
        live_end,
        live_end,
        life=10_000,
        death_callback_sub=1,
        ecl_subroutines=(live_end.address, live_end.address),
    )
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, wait),
        (kill_all.address, wait.address),
        live=live,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "ENEMYKILLALL" in forecast.reason


def test_timeline_kill_all_replays_interactive_callback_on_next_frame() -> None:
    kill_all, _raw = _instruction(0x18340, 0, 96)
    child_wait, _raw = _instruction(0x1834C, -1, 0)
    live_wait, _raw = _instruction(0x19340, -1, 0)
    callback_shoot, _raw = _instruction(0x1934C, 0, 80)
    callback_wait, _raw = _instruction(0x19358, -1, 0)
    pattern = BulletPattern(
        0, 0.0, 0.0, 1.0, 1.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 0, 0x4, 1.0, 1.0,
    )
    live = _spawner(
        live_wait,
        callback_wait,
        life=10_000,
        interactable=True,
        has_been_in_bounds=True,
        death_mode=1,
        death_callback_sub=1,
        pattern=pattern,
        ecl_program=(live_wait, callback_shoot, callback_wait),
        ecl_subroutines=(live_wait.address, callback_shoot.address),
    )
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, child_wait),
        (kill_all.address, child_wait.address),
        live=live,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 5
    assert len(forecast.births[3]) == 0
    assert len(forecast.births[4]) == 1


def test_timeline_kill_all_target_ecl_reads_forced_zero_life() -> None:
    kill_all, _raw = _instruction(0x18380, 0, 96)
    child_wait, _raw = _instruction(0x1838C, -1, 0)
    compare, raw = _instruction(0x19380, 3, 27, 20)
    struct.pack_into("<ii", raw, 0x0C, -10024, 0)
    compare = _with_raw(compare, raw)
    jump_not_equal, raw = _instruction(0x19394, 3, 34, 20)
    struct.pack_into("<ii", raw, 0x0C, 3, 32)
    jump_not_equal = _with_raw(jump_not_equal, raw)
    shoot, _raw = _instruction(0x193A8, 3, 80)
    live_wait, _raw = _instruction(0x193B4, -1, 0)
    callback_wait, _raw = _instruction(0x193C0, 30, 0)
    pattern = BulletPattern(
        0, 0.0, 0.0, 1.0, 1.0,
        (0.0,) * 4, (0,) * 4,
        1, 1, 0, 0x4, 1.0, 1.0,
    )
    live = _spawner(
        compare,
        callback_wait,
        life=10_000,
        interactable=True,
        has_been_in_bounds=True,
        death_mode=1,
        death_callback_sub=1,
        pattern=pattern,
        ecl_program=(
            compare,
            jump_not_equal,
            shoot,
            live_wait,
            callback_wait,
        ),
        ecl_subroutines=(compare.address, callback_wait.address),
    )
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, child_wait),
        (kill_all.address, child_wait.address),
        live=live,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 5
    # The ordinary retained branch sees life 10000 and jumps over SHOOTNOW;
    # this birth can only come from the exact timeline-forced life-zero path.
    assert len(forecast.births[3]) == 1


def test_timeline_kill_all_rejects_forced_shared_laser_creation() -> None:
    kill_all, _raw = _instruction(0x183D0, 0, 96)
    child_wait, _raw = _instruction(0x183DC, -1, 0)
    laser, raw = _instruction(0x193D0, 3, 85, 64)
    struct.pack_into("<fffff", raw, 0x10, 0.0, 0.0, 0.0, 100.0, 10.0)
    struct.pack_into("<f", raw, 0x24, 4.0)
    struct.pack_into("<iiiiii", raw, 0x28, 0, 60, 10, 0, 0, 0)
    laser = _with_raw(laser, raw)
    live_wait, _raw = _instruction(0x19410, -1, 0)
    live = _spawner(
        laser,
        live_wait,
        life=10_000,
        interactable=True,
        has_been_in_bounds=True,
        ecl_program=(laser, live_wait),
    )
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, child_wait),
        (kill_all.address, child_wait.address),
        live=live,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "shared laser world" in forecast.reason


def test_timeline_kill_all_rejects_live_callback_assigned_in_prefix() -> None:
    kill_all, _raw = _instruction(0x18400, 0, 96)
    wait, _raw = _instruction(0x1840C, -1, 0)
    live_callback = _set_int_instruction(0x19400, 0, 108, 1)
    live_wait, _raw = _instruction(0x19410, -1, 0)
    live = _spawner(
        live_callback,
        live_wait,
        life=10_000,
        ecl_program=(live_callback, live_wait),
        ecl_subroutines=(live_callback.address, live_wait.address),
    )
    snapshot = _timeline_kill_all_snapshot(
        (kill_all, wait),
        (kill_all.address, wait.address),
        live=live,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "ENEMYKILLALL" in forecast.reason


def test_timeline_kill_all_does_not_hide_later_boss_coverage_failure() -> None:
    child_kill_all, _raw = _instruction(0x18440, 0, 96)
    child_wait, _raw = _instruction(0x1844C, -1, 0)
    unsupported, raw = _instruction(0x19440, 4, 122, 20)
    struct.pack_into("<Ii", raw, 0x0C, 99, 0)
    unsupported = _with_raw(unsupported, raw)
    boss = _spawner(
        unsupported,
        unsupported,
        life=10_000,
        is_boss=True,
        boss_id=0,
        ecl_program=(unsupported,),
    )
    snapshot = _timeline_kill_all_snapshot(
        (child_kill_all, child_wait),
        (child_kill_all.address, child_wait.address),
        live=boss,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 6)

    assert forecast.covered_frames == 4
    assert "emitter 0" in forecast.reason


def test_timeline_kill_all_rejects_prior_compact_timeline_child() -> None:
    neutral, _raw = _instruction(0x18500, -1, 0)
    kill_all, _raw = _instruction(0x18510, 0, 96)
    wait, _raw = _instruction(0x1851C, -1, 0)
    snapshot = _timeline_kill_all_snapshot(
        (neutral, kill_all, wait),
        (neutral.address, kill_all.address),
        prior_spawn=True,
        target_sub_id=1,
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 5)

    assert forecast.covered_frames == 3
    assert "ENEMYKILLALL" in forecast.reason


def test_live_kill_all_rejects_callback_assigned_by_another_slot() -> None:
    callback = _set_int_instruction(0x19600, 0, 108, 1)
    callback_wait, _raw = _instruction(0x19610, -1, 0)
    callback_owner = _spawner(
        callback,
        callback_wait,
        slot=0,
        life=10_000,
        ecl_program=(callback, callback_wait),
        ecl_subroutines=(callback.address, callback_wait.address),
    )
    kill_all, _raw = _instruction(0x19700, 1, 96)
    kill_wait, _raw = _instruction(0x1970C, -1, 0)
    killer = _spawner(
        kill_all,
        kill_wait,
        slot=1,
        life=10_000,
        is_boss=True,
        boss_id=0,
        ecl_program=(kill_all, kill_wait),
    )
    snapshot = replace(
        _snapshot(callback_owner),
        spawners=(callback_owner, killer),
    )

    forecast = forecast_world_births(snapshot, ((192.0, 400.0),) * 4)

    assert forecast.covered_frames == 1
    assert "ENEMYKILLALL" in forecast.reason


def test_live_boss_kill_all_is_exact_noop_without_external_target() -> None:
    kill_all, _raw = _instruction(0x19800, 1, 96)
    wait, _raw = _instruction(0x1980C, -1, 0)
    boss = _spawner(
        kill_all,
        wait,
        is_boss=True,
        boss_id=0,
        ecl_program=(kill_all, wait),
    )

    forecast = forecast_world_births(
        _snapshot(boss),
        ((192.0, 400.0),) * 4,
    )

    assert forecast.covered_frames == 4
    assert forecast.reason == ""


def test_first_live_slot_boss_kill_all_replays_later_simple_targets() -> None:
    kill_all, _raw = _instruction(0x19840, 1, 96)
    boss_wait, _raw = _instruction(0x1984C, -1, 0)
    target_wait, _raw = _instruction(0x19880, -1, 0)
    boss = _spawner(
        kill_all,
        boss_wait,
        slot=0,
        life=10_000,
        is_boss=True,
        boss_id=0,
        ecl_program=(kill_all, boss_wait),
    )
    target = _spawner(
        target_wait,
        target_wait,
        slot=1,
        life=2_000,
        interactable=True,
        has_been_in_bounds=True,
        death_mode=0,
        death_callback_sub=-1,
    )
    snapshot = replace(
        _snapshot(boss),
        spawners=(boss, target),
    )

    forecast = forecast_world_births(
        snapshot,
        ((192.0, 400.0),) * 4,
    )

    assert forecast.covered_frames == 4
    assert forecast.reason == ""


def test_live_boss_kill_all_rejects_same_call_created_target() -> None:
    create, raw = _instruction(0x19900, 1, 95, 32)
    struct.pack_into("<iff", raw, 0x0C, 1, 192.0, 128.0)
    struct.pack_into("<hh", raw, 0x1C, 100, -1)
    create = _with_raw(create, raw)
    kill_all, _raw = _instruction(0x19920, 1, 96)
    wait, _raw = _instruction(0x1992C, -1, 0)
    boss = _spawner(
        create,
        wait,
        is_boss=True,
        boss_id=0,
        ecl_program=(create, kill_all, wait),
        ecl_subroutines=(create.address, wait.address),
    )

    forecast = forecast_world_births(
        _snapshot(boss),
        ((192.0, 400.0),) * 4,
    )

    assert forecast.covered_frames == 1
    assert "ENEMYCREATE/ENEMYKILLALL" in forecast.reason


def test_live_boss_kill_all_rejects_prior_update_created_target() -> None:
    create, raw = _instruction(0x19940, 0, 95, 32)
    struct.pack_into("<iff", raw, 0x0C, 1, 192.0, 128.0)
    struct.pack_into("<hh", raw, 0x1C, 100, -1)
    create = _with_raw(create, raw)
    kill_all, _raw = _instruction(0x19960, 1, 96)
    wait, _raw = _instruction(0x1996C, -1, 0)
    boss = _spawner(
        create,
        wait,
        is_boss=True,
        boss_id=0,
        ecl_program=(create, kill_all, wait),
        ecl_subroutines=(create.address, wait.address),
    )

    forecast = forecast_world_births(
        _snapshot(boss),
        ((192.0, 400.0),) * 4,
    )

    assert forecast.covered_frames == 1
    assert "ENEMYCREATE/ENEMYKILLALL" in forecast.reason


def test_live_boss_kill_all_rejects_timeline_created_target() -> None:
    kill_all, _raw = _instruction(0x19A00, 1, 96)
    wait, _raw = _instruction(0x19A0C, -1, 0)
    child_wait, _raw = _instruction(0x19B00, -1, 0)
    boss = _spawner(
        kill_all,
        wait,
        is_boss=True,
        boss_id=0,
        ecl_program=(kill_all, wait),
    )
    timeline_end = StageTimelineInstruction(
        0x2B018, -1, 0, 0, 8, "00" * 8
    )
    snapshot = replace(
        _snapshot(boss),
        boss_present=False,
        timeline_instructions=(
            _timeline_spawn(0x2B000, 0, 0),
            timeline_end,
        ),
        timeline_ecl_program=(child_wait,),
        ecl_subroutines=(child_wait.address,),
    )

    forecast = forecast_world_births(
        snapshot,
        ((192.0, 400.0),) * 4,
    )

    assert forecast.covered_frames == 1
    assert "ENEMYKILLALL" in forecast.reason
