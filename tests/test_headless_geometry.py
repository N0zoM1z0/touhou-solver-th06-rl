from __future__ import annotations

from copy import deepcopy
import math

import pytest

from th06_rl.headless_geometry import (
    HeadlessAuthorityUnavailable,
    HEADLESS_DELIVERY_CONTRACT,
    HEADLESS_DELIVERY_DELAYS,
    OBSERVATION_SCHEMA,
    certify_headless_actions,
    certify_lowered_headless_actions,
    lower_headless_hard_hazards,
    lower_headless_hazards,
    reactive_headless_action,
)
from th06_rl.native import ACTIONS


def observation() -> dict[str, object]:
    return {
        "schema": OBSERVATION_SCHEMA,
        "tick": 100,
        "terminal_reason": None,
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 6},
        "input": 0x04,
        "player": {
            "x": 192.0,
            "y": 384.0,
            "state": 0,
            "half_width": 1.25,
            "half_height": 1.25,
            "focused": True,
        },
        "bullets": [],
        "lasers": [],
        "enemies": [],
    }


def bullet(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "x": 192.0,
        "y": 370.0,
        "vx": 0.0,
        "vy": 4.0,
        "half_width": 1.0,
        "half_height": 1.0,
        "sprite_half_width": 4.0,
        "sprite_half_height": 4.0,
        "state": 1,
        "ex_flags": 0,
        "acceleration_x": 0.0,
        "acceleration_y": 0.0,
        "speed": 4.0,
        "angle": 1.57079632679,
        "curve_speed_acceleration": 0.0,
        "curve_angular_velocity": 0.0,
        "turn_speed": 0.0,
        "direction_rotation": 0.0,
        "timer": 10,
        "timer_float": 10.0,
        "acceleration_duration": 0,
        "direction_interval": 0,
        "direction_num_times": 0,
        "direction_max_times": 0,
    }
    result.update(overrides)
    return result


def laser(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "slot": 0,
        "x": 0.0,
        "y": 0.0,
        "angle": 0.0,
        "angular_velocity": 0.0,
        "angle_tracked": True,
        "angle_initialized": False,
        "start": 0.0,
        "end": 32.0,
        "start_length": 32.0,
        "width": 8.0,
        "speed": 0.0,
        "start_time": 0,
        "graze_delay": 0,
        "duration": 120,
        "end_time": 30,
        "graze_interval": 10,
        "timer": 1,
        "flags": 0,
        "state": 1,
    }
    result.update(overrides)
    return result


def interpolating_enemy(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "x": 198.1742706298828,
        "y": 120.0,
        "vx": -0.35784912109375,
        "vy": 19.646591186523438,
        "axis_vx": -0.35784912109375,
        "axis_vy": 19.646591186523438,
        "angle": 1.5890085697174072,
        "angular_velocity": 0.0,
        "speed": 1.5,
        "acceleration": 0.0,
        "movement_mode": 2,
        "movement_ease": 1,
        "invert_x": False,
        "move_interp_x": -76.91134643554688,
        "move_interp_y": 57.565704345703125,
        "move_start_x": 268.9113464355469,
        "move_start_y": 86.43429565429688,
        "move_timer": 33,
        "move_timer_float": 33.0,
        "move_start_time": 120,
        "hitbox_width": 26.666666,
        "hitbox_height": 37.333332,
        "contact_active": True,
    }
    result.update(overrides)
    return result


def test_empty_headless_world_certifies_the_full_bombless_vocabulary() -> None:
    certified = certify_headless_actions(observation())

    assert len(certified) == 18
    assert "bomb" not in {item.action.name for item in certified}
    assert reactive_headless_action(observation(), certified).name == "up_fast"


def test_enemy_interpolation_state_prevents_false_constant_velocity_body() -> None:
    value = observation()
    value["enemies"] = [interpolating_enemy()]

    frames = lower_headless_hard_hazards(value).aabb_frames
    centers_y = [
        (frame[0].top + frame[0].bottom) / 2.0
        for frame in frames
    ]

    assert centers_y == pytest.approx(
        [
            139.64659118652344,
            139.90643880208333,
            140.1582887585958,
            140.40214347839355,
        ]
    )

    legacy = observation()
    enemy = interpolating_enemy()
    for name in (
        "axis_vx",
        "axis_vy",
        "angle",
        "angular_velocity",
        "speed",
        "acceleration",
        "movement_mode",
        "movement_ease",
        "invert_x",
        "move_interp_x",
        "move_interp_y",
        "move_start_x",
        "move_start_y",
        "move_timer",
        "move_timer_float",
        "move_start_time",
    ):
        del enemy[name]
    legacy["enemies"] = [enemy]
    legacy_frames = lower_headless_hard_hazards(legacy).aabb_frames
    legacy_centers_y = [
        (frame[0].top + frame[0].bottom) / 2.0
        for frame in legacy_frames
    ]

    assert legacy_centers_y[-1] > centers_y[-1] + 58.0


def test_partial_enemy_interpolation_state_fails_closed() -> None:
    value = observation()
    enemy = interpolating_enemy()
    del enemy["move_start_time"]
    value["enemies"] = [enemy]

    with pytest.raises(HeadlessAuthorityUnavailable, match="partial enemy"):
        lower_headless_hard_hazards(value)


def test_linear_source_bullet_removes_a_colliding_first_action() -> None:
    value = observation()
    value["bullets"] = [bullet(y=366.0)]

    names = {item.action.name for item in certify_headless_actions(value)}

    assert "stay" not in names
    assert "left_fast" in names
    assert "right_fast" in names


def test_spawn_state_encloses_every_possible_firing_tick() -> None:
    value = observation()
    value["bullets"] = [bullet(state=3)]

    frames = lower_headless_hazards(value).aabb_frames

    assert not frames[0]
    assert all(len(frame) == 1 for frame in frames[1:])
    assert frames[-1][0].bottom >= frames[1][0].bottom


def test_player_aim_turn_only_encloses_hard_reachable_targets() -> None:
    value = observation()
    value["player"] = {
        **value["player"],  # type: ignore[dict-item]
        "x": 45.4141846,
        "y": 39.119915,
    }
    value["bullets"] = [bullet(
        x=55.0,
        y=42.3366013,
        vx=-0.0835062414,
        vy=-0.0498731174,
        speed=3.890625,
        angle=-2.6032064,
        ex_flags=0x84,
        turn_speed=4.0,
        timer=40,
        timer_float=40.0,
        direction_interval=40,
        direction_num_times=0,
        direction_max_times=1,
    )]

    frames = lower_headless_hazards(value).aabb_frames

    # The source can only aim at positions produced by the declared movement,
    # delivery-delay, and key-transition product.  The old arbitrary 360-degree
    # box was about 12 px wide on frame one; the reachable target cone is tight.
    assert frames[0][0].right - frames[0][0].left < 4.0
    assert frames[0][0].bottom - frames[0][0].top < 6.0

    player = value["player"]
    assert isinstance(player, dict)
    for target_x in (player["x"] - 4.0, player["x"], player["x"] + 4.0):
        for target_y in (player["y"] - 4.0, player["y"], player["y"] + 4.0):
            angle = math.atan2(target_y - 42.3366013, target_x - 55.0)
            vx = math.cos(angle) * 4.0
            vy = math.sin(angle) * 4.0
            x = 55.0
            y = 42.3366013
            for frame in frames:
                x += vx
                y += vy
                assert frame[0].left <= x - 1.0 <= frame[0].right
                assert frame[0].left <= x + 1.0 <= frame[0].right
                assert frame[0].top <= y - 1.0 <= frame[0].bottom
                assert frame[0].top <= y + 1.0 <= frame[0].bottom


def test_hard_lowering_routes_final_player_aim_to_native_candidate_paths() -> None:
    value = observation()
    value["bullets"] = [bullet(
        ex_flags=0x84,
        direction_interval=40,
        direction_num_times=0,
        direction_max_times=1,
    )]

    hazards = lower_headless_hard_hazards(value)

    assert len(hazards.player_aimed_bullets) == 1
    assert all(not frame for frame in hazards.aabb_frames)


def test_headless_step_delivery_contract_is_exactly_synchronous() -> None:
    assert HEADLESS_DELIVERY_CONTRACT == "synchronous-step-v1"
    assert HEADLESS_DELIVERY_DELAYS == (0,)


def test_lowered_certificate_can_audit_an_explicit_delivery_contract() -> None:
    value = observation()
    hazards = lower_headless_hard_hazards(value)
    observed: list[tuple[tuple[int, ...], tuple[object, ...]]] = []

    class RecordingKernel:
        def certify_actions(self, **kwargs):
            observed.append((kwargs["delivery_delays"], kwargs["candidates"]))
            return ()

    certify_lowered_headless_actions(
        value,
        hazards,
        kernel=RecordingKernel(),  # type: ignore[arg-type]
        candidates=ACTIONS[:2],
        delivery_delays=(0, 1, 2, 3),
    )

    assert observed == [((0, 1, 2, 3), ACTIONS[:2])]


def test_hard_lowering_keeps_multiturn_player_aim_fail_closed() -> None:
    value = observation()
    value["bullets"] = [bullet(
        ex_flags=0x84,
        direction_interval=40,
        direction_num_times=0,
        direction_max_times=2,
    )]

    hazards = lower_headless_hard_hazards(value)

    assert not hazards.player_aimed_bullets
    assert any(frame for frame in hazards.aabb_frames)


def test_hard_lowering_rejects_incoherent_player_aim_counter() -> None:
    value = observation()
    value["bullets"] = [bullet(
        ex_flags=0x84,
        direction_num_times=1,
        direction_max_times=1,
    )]

    with pytest.raises(HeadlessAuthorityUnavailable, match="player-aim"):
        lower_headless_hard_hazards(value)


def test_unknown_ex_motion_fails_closed() -> None:
    value = observation()
    value["bullets"] = [bullet(ex_flags=0x1000)]

    with pytest.raises(HeadlessAuthorityUnavailable, match="unknown.*EX"):
        certify_headless_actions(value)


def test_new_laser_may_be_ignored_only_before_its_hitbox_horizon() -> None:
    value = observation()
    value["lasers"] = [laser(
        angle_tracked=False,
        state=0,
        timer=0,
        start_time=20,
        graze_delay=10,
    )]

    assert all(not frame for frame in lower_headless_hazards(value).laser_frames)

    dangerous = deepcopy(value)
    dangerous["lasers"][0]["graze_delay"] = 3  # type: ignore[index]
    with pytest.raises(HeadlessAuthorityUnavailable, match="angular history"):
        lower_headless_hazards(dangerous)

    initialized = deepcopy(dangerous)
    initialized["lasers"][0]["angle_initialized"] = True  # type: ignore[index]
    assert any(lower_headless_hazards(initialized).laser_frames)


def test_laser_initialization_evidence_is_bounded_to_source_timer_reset() -> None:
    value = observation()
    value["lasers"] = [laser(
        angle_tracked=False,
        angle_initialized=True,
        timer=2,
    )]

    with pytest.raises(HeadlessAuthorityUnavailable, match="angular history"):
        lower_headless_hazards(value)


def test_bomb_in_observed_input_fails_closed() -> None:
    value = observation()
    value["input"] = 0x02

    with pytest.raises(HeadlessAuthorityUnavailable, match="Bomb"):
        certify_headless_actions(value)
