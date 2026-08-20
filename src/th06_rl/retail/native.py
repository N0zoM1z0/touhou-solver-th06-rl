"""Exact TH06 1.02h process identity, patching, and native sensing."""

from __future__ import annotations

import ctypes
import hashlib
import math
import os
import struct
import time
from collections import deque
from ctypes import wintypes
from dataclasses import dataclass
from pathlib import Path

from .model import (
    Bullet,
    BulletPattern,
    EnemyBody,
    EnemyEclContext,
    EnemySpawner,
    EclInstruction,
    ItemState,
    Laser,
    MessageInstruction,
    PLAYER_ALIVE,
    PLAYER_INVULNERABLE,
    PlayerAttackState,
    PlayerShot,
    Snapshot,
    StageTimelineInstruction,
)
from .hazards.timeline import timeline_message_index


TARGET_EXE = "th06.exe"
TARGET_SHA256 = "9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245"
PROCESS_ACCESS = 0x00100000 | 0x0001 | 0x0400 | 0x1000 | 0x0010 | 0x0020 | 0x0008

IMAGE_BASE = 0x400000
ADDR_LIFE_PATCH = 0x428DEC
ADDR_GAME_MANAGER = 0x69BCA0
ADDR_CURRENT_INPUT = 0x69D904
ADDR_RNG = 0x69D8F8
ADDR_CHAIN = 0x69D918
ADDR_SUPERVISOR = 0x6C6D18
ADDR_PLAYER = 0x6CA628
ADDR_FRAME_MULTIPLIER = 0x6C6EC0
ADDR_ENEMY_MANAGER = 0x4B79C8
ADDR_EFFECT_MANAGER = 0x487FE0
ADDR_ECL_EX_TABLE = 0x476220
ADDR_ECL_MANAGER = 0x487E50
ADDR_ENEMY_CALC_CHAIN = 0x5A5FB4
ADDR_BULLET_MANAGER = 0x5A5FF8
ADDR_BULLET_ARRAY = 0x5AB5F8
ADDR_LASER_ARRAY = 0x691FF8
ADDR_MAIN_MENU = 0x6D46C0
ADDR_GUI = 0x69BC30
ADDR_ITEM_MANAGER = 0x69E268

SUPERVISOR_STATES_OFFSET = 0x188
CHAIN_ROOT_NEXT_OFFSET = 0x14
CHAIN_ELEM_CALLBACK_OFFSET = 0x4
CHAIN_ELEM_NEXT_OFFSET = 0x14
CHAIN_ELEM_ARG_OFFSET = 0x1C
CHAIN_ELEM_SIZE = 0x20
RESULT_SCREEN_ON_UPDATE = 0x42D98E
RESULT_SCREEN_STATE_SIZE = 0x34

GAME_TIME_STOPPED_OFFSET = 0x2C
GAME_DIFFICULTY_OFFSET = 0x10
GAME_FLAGS_OFFSET = 0x181F
GAME_FRAMES_OFFSET = 0x1A30
GAME_STAGE_OFFSET = 0x1A34
GAME_RANK_OFFSET = 0x1A70
GAME_MAX_RANK_OFFSET = 0x1A74
GAME_MIN_RANK_OFFSET = 0x1A78
GAME_SUBRANK_OFFSET = 0x1A7C
GAME_CURRENT_POWER_OFFSET = 0x1810
GAME_LIVES_REMAINING_OFFSET = 0x181A
GAME_CHARACTER_OFFSET = 0x181D
GAME_SHOT_TYPE_OFFSET = 0x181E
PLAYER_POSITION_OFFSET = 0x440
PLAYER_HITBOX_TOP_LEFT_OFFSET = 0x458
PLAYER_HITBOX_BOTTOM_RIGHT_OFFSET = 0x464
PLAYER_ORBS_POSITION_OFFSET = 0x4A0
PLAYER_STATE_OFFSET = 0x9E0
PLAYER_ORB_STATE_OFFSET = 0x9E2
PLAYER_FOCUS_OFFSET = 0x9E3
PLAYER_FOCUS_TIMER_OFFSET = 0x9E8
PLAYER_SPEEDS_OFFSET = 0x9F4
PLAYER_LAST_ENEMY_HIT_OFFSET = 0xA1C
PLAYER_BULLETS_OFFSET = 0xA28
PLAYER_BULLET_COUNT = 80
PLAYER_BULLET_STRIDE = 0x158
PLAYER_BULLET_POSITION_OFFSET = 0x110
PLAYER_BULLET_SIZE_OFFSET = 0x11C
PLAYER_BULLET_VELOCITY_OFFSET = 0x128
PLAYER_BULLET_SIDEWAYS_OFFSET = 0x130
PLAYER_BULLET_HOMING_SPEED_OFFSET = 0x138
PLAYER_BULLET_TIMER_OFFSET = 0x140
PLAYER_BULLET_DAMAGE_OFFSET = 0x14C
PLAYER_BULLET_STATE_OFFSET = 0x14E
PLAYER_BULLET_TYPE_OFFSET = 0x150
PLAYER_BULLET_LASER_INDEX_OFFSET = 0x152
PLAYER_BULLET_SPAWN_POSITION_OFFSET = 0x154
PLAYER_FIRE_TIMER_OFFSET = 0x75A8
PLAYER_BOMB_ACTIVE_OFFSET = 0x75C8

ANM_VM_TIMER_OFFSET = 0x30
ANM_VM_SCRIPT_OFFSET = 0xB4
ANM_VM_SPRITE_OFFSET = 0xC0
ANM_LOADED_SPRITE_HEIGHT_OFFSET = 0x2C

BULLET_COUNT = 640
BULLET_STRIDE = 0x5C4
BULLET_TEMPLATE_COUNT = 16
BULLET_TEMPLATE_STRIDE = 0x560
BULLET_MANAGER_SIZE = 0xF5C18
BULLET_MANAGER_TIME_OFFSET = 0xF5C08
BULLET_TEMPLATE_SIZE_OFFSET = 0x550
BULLET_SIZE_OFFSET = 0x550
BULLET_POSITION_OFFSET = 0x560
BULLET_VELOCITY_OFFSET = 0x56C
BULLET_ACCELERATION_OFFSET = 0x578
BULLET_SPEED_OFFSET = 0x584
BULLET_ACCEL_SPEED_OFFSET = 0x588
BULLET_TURN_SPEED_OFFSET = 0x58C
BULLET_ANGLE_OFFSET = 0x590
BULLET_CURVE_ANGULAR_VELOCITY_OFFSET = 0x594
BULLET_DIRECTION_ROTATION_OFFSET = 0x598
BULLET_TIMER_SUBFRAME_OFFSET = 0x5A0
BULLET_TIMER_CURRENT_OFFSET = 0x5A4
BULLET_ACCELERATION_DURATION_OFFSET = 0x5A8
BULLET_DIRECTION_INTERVAL_OFFSET = 0x5AC
BULLET_DIRECTION_NUM_TIMES_OFFSET = 0x5B0
BULLET_DIRECTION_MAX_TIMES_OFFSET = 0x5B4
BULLET_EX_FLAGS_OFFSET = 0x5B8
BULLET_STATE_OFFSET = 0x5BE
BULLET_OUT_OF_BOUNDS_FRAMES_OFFSET = 0x5C0
BULLET_IS_GRAZED_OFFSET = 0x5C3
LASER_COUNT = 64
LASER_STRIDE = 0x270
LASER_POSITION_OFFSET = 0x220
LASER_ANGLE_OFFSET = 0x22C
LASER_START_OFFSET = 0x230
LASER_END_OFFSET = 0x234
LASER_START_LENGTH_OFFSET = 0x238
LASER_WIDTH_OFFSET = 0x23C
LASER_SPEED_OFFSET = 0x240
LASER_START_TIME_OFFSET = 0x244
LASER_HITBOX_START_TIME_OFFSET = 0x248
LASER_DURATION_OFFSET = 0x24C
LASER_DESPAWN_DURATION_OFFSET = 0x250
LASER_HITBOX_END_DELAY_OFFSET = 0x254
LASER_IN_USE_OFFSET = 0x258
LASER_TIMER_SUBFRAME_OFFSET = 0x260
LASER_TIMER_OFFSET = 0x264
LASER_FLAGS_OFFSET = 0x268
LASER_STATE_OFFSET = 0x26C
ENEMY_MANAGER_SIZE = 0xEE5EC
ENEMY_ARRAY_OFFSET = 0xED0
ENEMY_COUNT = 256
ENEMY_STRIDE = 0xEC8
ENEMY_ECL_CONTEXT_OFFSET = 0x990
ENEMY_ECL_CONTEXT_SIZE = 0x4C
ENEMY_ECL_STACK_CAPACITY = 8
ENEMY_ECL_STACK_DEPTH_OFFSET = 0xC3C
ENEMY_INTERRUPTS_OFFSET = 0xC48
ENEMY_RUN_INTERRUPT_OFFSET = 0xC68
ENEMY_POSITION_OFFSET = 0xC6C
ENEMY_HITBOX_OFFSET = 0xC78
ENEMY_AXIS_SPEED_OFFSET = 0xC84
ENEMY_ANGLE_OFFSET = 0xC90
ENEMY_ANGULAR_VELOCITY_OFFSET = 0xC94
ENEMY_SPEED_OFFSET = 0xC98
ENEMY_ACCELERATION_OFFSET = 0xC9C
ENEMY_SHOOT_OFFSET = 0xCA0
ENEMY_MOVE_INTERP_OFFSET = 0xCAC
ENEMY_MOVE_START_OFFSET = 0xCB8
ENEMY_MOVE_TIMER_SUBFRAME_OFFSET = 0xCC8
ENEMY_MOVE_TIMER_OFFSET = 0xCCC
ENEMY_MOVE_START_TIME_OFFSET = 0xCD0
ENEMY_BULLET_RANK_SPEED_LOW_OFFSET = 0xCD4
ENEMY_BULLET_RANK_SPEED_HIGH_OFFSET = 0xCD8
ENEMY_BULLET_RANK_AMOUNT1_LOW_OFFSET = 0xCDC
ENEMY_BULLET_RANK_AMOUNT1_HIGH_OFFSET = 0xCDE
ENEMY_BULLET_RANK_AMOUNT2_LOW_OFFSET = 0xCE0
ENEMY_BULLET_RANK_AMOUNT2_HIGH_OFFSET = 0xCE2
ENEMY_LIFE_OFFSET = 0xCE4
ENEMY_BOSS_TIMER_SUBFRAME_OFFSET = 0xCF4
ENEMY_BOSS_TIMER_OFFSET = 0xCF8
ENEMY_BULLET_PROPS_OFFSET = 0xD00
ENEMY_SHOOT_INTERVAL_OFFSET = 0xD54
ENEMY_SHOOT_TIMER_SUBFRAME_OFFSET = 0xD5C
ENEMY_SHOOT_TIMER_OFFSET = 0xD60
ENEMY_LASER_POINTERS_OFFSET = 0xDB8
ENEMY_LASER_STORE_OFFSET = 0xE38
ENEMY_DEATH_ANM_OFFSET = 0xE3C
ENEMY_BOSS_ID_OFFSET = 0xE40
ENEMY_FLAGS_OFFSET = 0xE50
ENEMY_LOWER_MOVE_LIMIT_OFFSET = 0xE60
ENEMY_UPPER_MOVE_LIMIT_OFFSET = 0xE68
ENEMY_LIFE_CALLBACK_THRESHOLD_OFFSET = 0xEA8
ENEMY_LIFE_CALLBACK_SUB_OFFSET = 0xEAC
ENEMY_TIMER_CALLBACK_THRESHOLD_OFFSET = 0xEB0
ENEMY_TIMER_CALLBACK_SUB_OFFSET = 0xEB4
ENEMY_DEATH_CALLBACK_SUB_OFFSET = 0xC44
ENEMY_TIMELINE_INSTRUCTION_OFFSET = 0xEE5DC
ENEMY_TIMELINE_TIMER_OFFSET = 0xEE5E0
ENEMY_BOSSES_OFFSET = 0xEE598
ENEMY_RANDOM_ITEM_SPAWN_INDEX_OFFSET = 0xEE5B8
ENEMY_RANDOM_ITEM_TABLE_INDEX_OFFSET = 0xEE5BA
ENEMY_SPELL_ACTIVE_OFFSET = 0xEE5C8
EFFECT_ACTIVE_COUNT_OFFSET = 0x4
EFFECT_ARRAY_OFFSET = 0x8
EFFECT_COUNT = 512
EFFECT_STRIDE = 0x17C
EFFECT_TIMER_PREVIOUS_OFFSET = 0x164
EFFECT_TIMER_CURRENT_OFFSET = 0x16C
EFFECT_IN_USE_OFFSET = 0x178
EFFECT_ID_OFFSET = 0x179
ITEM_ACTIVE_COUNT_OFFSET = 0x28948
ITEM_NEXT_INDEX_OFFSET = 0x28944
ITEM_COUNT = 512
ITEM_STRIDE = 0x144
ITEM_POSITION_OFFSET = 0x110
ITEM_START_POSITION_OFFSET = 0x11C
ITEM_TARGET_POSITION_OFFSET = 0x128
ITEM_TIMER_OFFSET = 0x134
ITEM_TYPE_OFFSET = 0x140
ITEM_IN_USE_OFFSET = 0x141
ITEM_STATE_OFFSET = 0x143
ECL_EX_COUNT = 17
ECL_PROGRAM_INSTRUCTION_LIMIT = 256
ECL_SUBROUTINE_LIMIT = 512
ECL_TIMELINE_INSTRUCTION_LIMIT = 4096
ECL_TIMELINE_SNAPSHOT_LIMIT = 96
MSG_PROGRAM_INSTRUCTION_LIMIT = 4096
MSG_WAIT_FRAME_LIMIT = 65536
MAIN_MENU_CURSOR_OFFSET = 0x81A0
MAIN_MENU_STATE_OFFSET = 0x81F0
MAIN_MENU_TIMER_OFFSET = 0x81F4
GUI_IMPL_MSG_OFFSET = 0x2534
GUI_MSG_INSTRUCTION_OFFSET = 0x4
GUI_MSG_INDEX_OFFSET = 0x8
GUI_MSG_IGNORE_WAIT_OFFSET = 0x6A0
GUI_MSG_SKIPPABLE_OFFSET = 0x6A4
GUI_BOSS_PRESENT_OFFSET = 0x20


@dataclass(frozen=True)
class ResultScreenState:
    address: int
    frame_timer: int
    state: int
    cursor: int
    replay_number: int
    selected_character: int


class NativeDecodeError(RuntimeError):
    def __init__(self, message: str, evidence: dict):
        super().__init__(message)
        self.evidence = evidence


class _SnapshotEpochChanged(RuntimeError):
    pass


class _SnapshotReadTorn(RuntimeError):
    pass


class _SnapshotPhaseIncomplete(RuntimeError):
    def __init__(self, game_frame: int, bullet_time: int):
        super().__init__(
            f"calc chain incomplete at game frame {game_frame}: "
            f"bullet manager is at {bullet_time}"
        )
        self.game_frame = game_frame
        self.bullet_time = bullet_time


def _decode_timeline_boss_slots(
    boss_pointers: tuple[int, ...],
) -> tuple[int, ...]:
    """Map EnemyManager's raw boss pointers without inferring boss identity."""
    pool_start = ADDR_ENEMY_MANAGER + ENEMY_ARRAY_OFFSET
    pool_end = pool_start + ENEMY_COUNT * ENEMY_STRIDE
    slots = []
    for boss_id, pointer in enumerate(boss_pointers):
        if pointer == 0:
            slots.append(-1)
            continue
        relative = pointer - pool_start
        if not pool_start <= pointer < pool_end or relative % ENEMY_STRIDE:
            raise RuntimeError(
                f"invalid timeline boss pointer 0x{pointer:08X} "
                f"at boss id {boss_id}"
            )
        slots.append(relative // ENEMY_STRIDE)
    return tuple(slots)


def _decode_enemy_boss_id(
    enemy_pool: bytes,
    base: int,
    index: int,
    is_boss: bool,
    timeline_boss_slots: tuple[int, ...],
) -> int:
    """Decode Enemy's own identity and only reject a true torn BOSSSET."""
    if not is_boss:
        return -1
    boss_id = enemy_pool[base + ENEMY_BOSS_ID_OFFSET]
    if boss_id >= len(timeline_boss_slots):
        raise _SnapshotReadTorn(
            f"invalid published boss id {boss_id} at enemy slot {index}"
        )
    if timeline_boss_slots[boss_id] != index:
        raise _SnapshotReadTorn(
            f"incoherent true boss pointer at enemy slot {index}"
        )
    return boss_id


def _read_sprite_dimensions(
    process: "NativeProcess",
    pointers: set[int],
) -> dict[int, tuple[float, float]]:
    """Read immutable ``AnmLoadedSprite`` width/height for live VMs."""
    dimensions = {}
    for pointer in sorted(pointers):
        if not 0x10000 <= pointer < 0x80000000:
            raise RuntimeError(f"invalid active sprite pointer 0x{pointer:08X}")
        height, width = struct.unpack(
            "<ff",
            process.read(
                pointer + ANM_LOADED_SPRITE_HEIGHT_OFFSET,
                8,
            ),
        )
        if (
            not math.isfinite(width)
            or not math.isfinite(height)
            or not 0.0 < width <= 4096.0
            or not 0.0 < height <= 4096.0
        ):
            raise RuntimeError(
                f"invalid active sprite geometry at 0x{pointer:08X}"
            )
        dimensions[pointer] = (width, height)
    return dimensions


def _decode_player_attack(
    player: bytes,
    sprite_dimensions: dict[int, tuple[float, float]],
    *,
    shot_type: int,
    spell_active: bool,
) -> PlayerAttackState:
    """Decode the authoritative ``Player`` attack tail copied this epoch."""
    relative = lambda absolute: absolute - PLAYER_POSITION_OFFSET
    shots = []
    for slot in range(PLAYER_BULLET_COUNT):
        base = relative(PLAYER_BULLETS_OFFSET) + slot * PLAYER_BULLET_STRIDE
        state = struct.unpack_from(
            "<h", player, base + PLAYER_BULLET_STATE_OFFSET
        )[0]
        if state == 0:
            continue
        bullet_type = struct.unpack_from(
            "<h", player, base + PLAYER_BULLET_TYPE_OFFSET
        )[0]
        sprite_pointer = struct.unpack_from(
            "<I", player, base + ANM_VM_SPRITE_OFFSET
        )[0]
        try:
            sprite_width, sprite_height = sprite_dimensions[sprite_pointer]
        except KeyError as exc:
            raise RuntimeError(
                f"player bullet slot {slot} has an unresolved sprite"
            ) from exc
        x, y = struct.unpack_from(
            "<ff", player, base + PLAYER_BULLET_POSITION_OFFSET
        )
        size_x, size_y = struct.unpack_from(
            "<ff", player, base + PLAYER_BULLET_SIZE_OFFSET
        )
        vx, vy = struct.unpack_from(
            "<ff", player, base + PLAYER_BULLET_VELOCITY_OFFSET
        )
        sideways_motion = struct.unpack_from(
            "<f", player, base + PLAYER_BULLET_SIDEWAYS_OFFSET
        )[0]
        homing_speed = struct.unpack_from(
            "<f", player, base + PLAYER_BULLET_HOMING_SPEED_OFFSET
        )[0]
        timer_previous, timer_subframe, timer = struct.unpack_from(
            "<ifi", player, base + PLAYER_BULLET_TIMER_OFFSET
        )
        anm_previous, anm_subframe, anm_timer = struct.unpack_from(
            "<ifi", player, base + ANM_VM_TIMER_OFFSET
        )
        damage, _state, _type, laser_index, spawn_position = struct.unpack_from(
            "<hhhhh", player, base + PLAYER_BULLET_DAMAGE_OFFSET
        )
        anm_script = struct.unpack_from(
            "<h", player, base + ANM_VM_SCRIPT_OFFSET
        )[0]
        numbers = (
            x,
            y,
            size_x,
            size_y,
            vx,
            vy,
            sideways_motion,
            homing_speed,
            timer_subframe,
            anm_subframe,
            sprite_width,
            sprite_height,
        )
        if (
            state not in (1, 2)
            or bullet_type not in (0, 1, 2, 3)
            or not all(math.isfinite(value) for value in numbers)
            or not 0.0 < size_x <= 4096.0
            or not 0.0 < size_y <= 4096.0
            or not -32768 <= damage <= 32767
            or not -1_000_000 <= timer_previous <= 1_000_000
            or not -1_000_000 <= timer <= 1_000_000
            or not -1_000_000 <= anm_previous <= 1_000_000
            or not -1_000_000 <= anm_timer <= 1_000_000
            or not -1 <= anm_script < 0x1000
        ):
            raise RuntimeError(f"invalid player bullet state at slot {slot}")
        shots.append(PlayerShot(
            slot=slot,
            x=x,
            y=y,
            half_width=size_x / 2.0,
            half_height=size_y / 2.0,
            vx=vx,
            vy=vy,
            homing_speed=homing_speed,
            timer_previous=timer_previous,
            timer=timer,
            timer_float=timer + timer_subframe,
            damage=damage,
            state=state,
            bullet_type=bullet_type,
            anm_script=anm_script,
            anm_timer=anm_timer,
            anm_timer_float=anm_timer + anm_subframe,
            sprite_half_width=sprite_width / 2.0,
            sprite_half_height=sprite_height / 2.0,
            sideways_motion=sideways_motion,
            laser_index=laser_index,
            spawn_position_index=spawn_position,
        ))

    last_hit_x, last_hit_y = struct.unpack_from(
        "<ff", player, relative(PLAYER_LAST_ENEMY_HIT_OFFSET)
    )
    orb_positions = tuple(
        struct.unpack_from(
            "<ff",
            player,
            relative(PLAYER_ORBS_POSITION_OFFSET) + index * 12,
        )
        for index in range(2)
    )
    focus_previous, focus_subframe, focus_timer = struct.unpack_from(
        "<ifi", player, relative(PLAYER_FOCUS_TIMER_OFFSET)
    )
    fire_previous, fire_subframe, fire_timer = struct.unpack_from(
        "<ifi", player, relative(PLAYER_FIRE_TIMER_OFFSET)
    )
    orb_state = player[relative(PLAYER_ORB_STATE_OFFSET)]
    is_focus = bool(player[relative(PLAYER_FOCUS_OFFSET)])
    bomb_active = bool(struct.unpack_from(
        "<I", player, relative(PLAYER_BOMB_ACTIVE_OFFSET)
    )[0])
    attack_numbers = (
        last_hit_x,
        last_hit_y,
        focus_subframe,
        fire_subframe,
        *(coordinate for pair in orb_positions for coordinate in pair),
    )
    if (
        not all(math.isfinite(value) for value in attack_numbers)
        or orb_state not in range(5)
        or shot_type not in (0, 1)
        or not -1_000_000 <= focus_previous <= 1_000_000
        or not -1_000_000 <= focus_timer <= 1_000_000
        or not -1_000_000 <= fire_previous <= 1_000_000
        or not -1_000_000 <= fire_timer <= 1_000_000
    ):
        raise RuntimeError("invalid player attack state")
    return PlayerAttackState(
        shots=tuple(shots),
        last_enemy_hit_x=last_hit_x,
        last_enemy_hit_y=last_hit_y,
        orb_state=orb_state,
        is_focus=is_focus,
        focus_timer_previous=focus_previous,
        focus_timer=focus_timer,
        focus_timer_float=focus_timer + focus_subframe,
        fire_timer_previous=fire_previous,
        fire_timer=fire_timer,
        fire_timer_float=fire_timer + fire_subframe,
        orb_positions=orb_positions,
        shot_type=shot_type,
        bomb_active=bomb_active,
        spell_active=spell_active,
    )


def _decode_bullet_tail(
    tail: bytes,
    slot: int,
    sprite_size: tuple[float, float] | None = None,
) -> Bullet | None:
    relative = lambda absolute: absolute - BULLET_SIZE_OFFSET
    state = struct.unpack_from("<H", tail, relative(BULLET_STATE_OFFSET))[0]
    if state == 0:
        return None
    size_x, size_y = struct.unpack_from("<ff", tail, 0)
    bx, by = struct.unpack_from("<ff", tail, relative(BULLET_POSITION_OFFSET))
    vx, vy = struct.unpack_from("<ff", tail, relative(BULLET_VELOCITY_OFFSET))
    ax, ay = struct.unpack_from("<ff", tail, relative(BULLET_ACCELERATION_OFFSET))
    speed, accel_speed, turn_speed = struct.unpack_from(
        "<fff", tail, relative(BULLET_SPEED_OFFSET)
    )
    angle = struct.unpack_from("<f", tail, relative(BULLET_ANGLE_OFFSET))[0]
    curve_angular_velocity = struct.unpack_from(
        "<f", tail, relative(BULLET_CURVE_ANGULAR_VELOCITY_OFFSET)
    )[0]
    direction_rotation = struct.unpack_from(
        "<f", tail, relative(BULLET_DIRECTION_ROTATION_OFFSET)
    )[0]
    timer_subframe = struct.unpack_from(
        "<f", tail, relative(BULLET_TIMER_SUBFRAME_OFFSET)
    )[0]
    timer = struct.unpack_from("<i", tail, relative(BULLET_TIMER_CURRENT_OFFSET))[0]
    acceleration_duration = struct.unpack_from(
        "<i", tail, relative(BULLET_ACCELERATION_DURATION_OFFSET)
    )[0]
    direction_interval, direction_num_times, direction_max_times = struct.unpack_from(
        "<iii", tail, relative(BULLET_DIRECTION_INTERVAL_OFFSET)
    )
    ex_flags = struct.unpack_from("<H", tail, relative(BULLET_EX_FLAGS_OFFSET))[0]
    out_of_bounds_frames = struct.unpack_from(
        "<H", tail, relative(BULLET_OUT_OF_BOUNDS_FRAMES_OFFSET)
    )[0]
    is_grazed = bool(tail[relative(BULLET_IS_GRAZED_OFFSET)])
    if sprite_size is None:
        # Focused decoder tests predate visual-sprite capture.  Runtime
        # capture always supplies the authoritative AnmLoadedSprite size.
        sprite_size = (size_x, size_y)
    sprite_width, sprite_height = sprite_size
    numbers = (
        size_x,
        size_y,
        bx,
        by,
        vx,
        vy,
        ax,
        ay,
        speed,
        accel_speed,
        turn_speed,
        angle,
        curve_angular_velocity,
        direction_rotation,
        timer_subframe,
        sprite_width,
        sprite_height,
    )
    if state not in (1, 2, 3, 4, 5) or not all(
        math.isfinite(value) for value in numbers
    ) or not (
        0.0 < size_x <= 256.0
        and 0.0 < size_y <= 256.0
        and 0.0 < sprite_width <= 4096.0
        and 0.0 < sprite_height <= 4096.0
        and out_of_bounds_frames < 0x100
    ):
        raise NativeDecodeError(
            f"invalid bullet geometry at slot {slot}",
            {
                "slot": slot,
                "state": state,
                "values": tuple(repr(value) for value in numbers),
                "tail_hex": tail.hex(),
            },
        )
    if state == 1 and ex_flags & 0x1C0 and not (
        timer >= 0
        and direction_interval > 0
        and 0 <= direction_num_times < direction_max_times <= 0x10000
    ):
        raise NativeDecodeError(
            f"invalid bullet direction schedule at slot {slot}",
            {
                "slot": slot,
                "state": state,
                "ex_flags": ex_flags,
                "timer": timer,
                "direction_interval": direction_interval,
                "direction_num_times": direction_num_times,
                "direction_max_times": direction_max_times,
                "tail_hex": tail.hex(),
            },
        )
    invalid_acceleration_duration = (
        ex_flags & 0x10 and not 0 < acceleration_duration <= 99999
    ) or (
        ex_flags & 0x20 and not -99999 <= acceleration_duration <= 99999
    )
    if invalid_acceleration_duration:
        raise NativeDecodeError(
            f"invalid bullet acceleration duration at slot {slot}",
            {
                "slot": slot,
                "state": state,
                "ex_flags": ex_flags,
                "timer": timer,
                "acceleration_duration": acceleration_duration,
                "tail_hex": tail.hex(),
            },
        )
    return Bullet(
        bx,
        by,
        vx,
        vy,
        size_x / 2.0,
        size_y / 2.0,
        state,
        ex_flags=ex_flags,
        acceleration=math.hypot(ax, ay) + abs(accel_speed),
        speed=speed,
        turn_speed=turn_speed,
        acceleration_x=ax,
        acceleration_y=ay,
        angle=angle,
        direction_rotation=direction_rotation,
        timer=timer,
        timer_float=timer + timer_subframe,
        acceleration_duration=acceleration_duration,
        direction_interval=direction_interval,
        direction_num_times=direction_num_times,
        direction_max_times=direction_max_times,
        curve_speed_acceleration=accel_speed,
        curve_angular_velocity=curve_angular_velocity,
        slot=slot,
        is_grazed=is_grazed,
        sprite_half_width=sprite_width / 2.0,
        sprite_half_height=sprite_height / 2.0,
        out_of_bounds_frames=out_of_bounds_frames,
    )


def _kernel32():
    if os.name != "nt":
        raise RuntimeError("native TH06 access requires Windows Python")
    api = ctypes.WinDLL("kernel32", use_last_error=True)
    api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    api.OpenProcess.restype = wintypes.HANDLE
    api.CloseHandle.argtypes = [wintypes.HANDLE]
    api.CloseHandle.restype = wintypes.BOOL
    api.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    api.TerminateProcess.restype = wintypes.BOOL
    api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    api.WaitForSingleObject.restype = wintypes.DWORD
    api.ReadProcessMemory.argtypes = [
        wintypes.HANDLE, wintypes.LPCVOID, wintypes.LPVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    api.ReadProcessMemory.restype = wintypes.BOOL
    api.WriteProcessMemory.argtypes = [
        wintypes.HANDLE, wintypes.LPVOID, wintypes.LPCVOID, ctypes.c_size_t, ctypes.POINTER(ctypes.c_size_t)
    ]
    api.WriteProcessMemory.restype = wintypes.BOOL
    api.VirtualProtectEx.argtypes = [
        wintypes.HANDLE, wintypes.LPVOID, ctypes.c_size_t, wintypes.DWORD, ctypes.POINTER(wintypes.DWORD)
    ]
    api.VirtualProtectEx.restype = wintypes.BOOL
    api.FlushInstructionCache.argtypes = [wintypes.HANDLE, wintypes.LPCVOID, ctypes.c_size_t]
    api.FlushInstructionCache.restype = wintypes.BOOL
    api.CreateToolhelp32Snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    api.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    api.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE, wintypes.DWORD, wintypes.LPWSTR, ctypes.POINTER(wintypes.DWORD)
    ]
    api.QueryFullProcessImageNameW.restype = wintypes.BOOL
    return api


class NativeProcess:
    def __init__(self, pid: int):
        self.kernel32 = _kernel32()
        # SYNCHRONIZE is required to verify TerminateProcess completion with
        # WaitForSingleObject during trial cleanup.
        self.handle = self.kernel32.OpenProcess(PROCESS_ACCESS, False, pid)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        self.pid = pid
        self.ecl_instruction_cache: dict[int, EclInstruction] = {}
        self.ecl_program_cache: dict[int, tuple[EclInstruction, ...]] = {}
        self.ecl_timeline_instruction_cache: dict[
            int, StageTimelineInstruction
        ] = {}
        self.ecl_timeline_cache: dict[
            int, tuple[StageTimelineInstruction, ...]
        ] = {}
        self.ecl_timeline_program_cache: dict[
            tuple[int, ...], tuple[EclInstruction, ...]
        ] = {}
        self.ecl_cache_stage: int | None = None
        self.ecl_subroutines: tuple[int, ...] = ()
        self.ecl_subroutine_traits: dict[int, tuple[bool, bool]] = {}
        self.message_program_cache: dict[
            tuple[int, int], tuple[MessageInstruction, ...]
        ] = {}

    def close(self) -> None:
        if self.handle:
            self.kernel32.CloseHandle(self.handle)
            self.handle = None

    def terminate(self) -> None:
        """Stop only the exact, identity-verified trial process we attached."""
        if not self.handle:
            return
        if not self.kernel32.TerminateProcess(self.handle, 0):
            raise ctypes.WinError(ctypes.get_last_error())
        if self.kernel32.WaitForSingleObject(self.handle, 5000) != 0:
            raise RuntimeError(f"TH06 pid {self.pid} did not terminate within 5 seconds")

    def read(self, address: int, size: int) -> bytes:
        buffer = ctypes.create_string_buffer(size)
        count = ctypes.c_size_t()
        if not self.kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(address), buffer, size, ctypes.byref(count)
        ) or count.value != size:
            raise ctypes.WinError(ctypes.get_last_error())
        return buffer.raw

    def patch_lives(self) -> str:
        old = self.read(ADDR_LIFE_PATCH, 1)
        if old == b"\x00":
            return "already-patched"
        if old != b"\x01":
            raise RuntimeError(f"unexpected life-patch byte: {old.hex()}")
        old_protection = wintypes.DWORD()
        if not self.kernel32.VirtualProtectEx(
            self.handle, ctypes.c_void_p(ADDR_LIFE_PATCH), 1, 0x40, ctypes.byref(old_protection)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            data = ctypes.create_string_buffer(b"\x00")
            written = ctypes.c_size_t()
            if not self.kernel32.WriteProcessMemory(
                self.handle, ctypes.c_void_p(ADDR_LIFE_PATCH), data, 1, ctypes.byref(written)
            ) or written.value != 1:
                raise ctypes.WinError(ctypes.get_last_error())
            if not self.kernel32.FlushInstructionCache(self.handle, ctypes.c_void_p(ADDR_LIFE_PATCH), 1):
                raise ctypes.WinError(ctypes.get_last_error())
        finally:
            restored = wintypes.DWORD()
            self.kernel32.VirtualProtectEx(
                self.handle, ctypes.c_void_p(ADDR_LIFE_PATCH), 1, old_protection.value, ctypes.byref(restored)
            )
        if self.read(ADDR_LIFE_PATCH, 1) != b"\x00":
            raise RuntimeError("life patch did not verify")
        return "patched-01-to-00"

    def set_diagnostic_rng_seed(self, seed: int) -> tuple[int, int]:
        """Fix only the source RNG initial state for reproducible trials."""
        if not 0 <= seed <= 0xFFFF:
            raise ValueError("diagnostic RNG seed must fit u16")
        before = self.read(ADDR_RNG, 8)
        old_seed, old_generation = struct.unpack("<HxxI", before)
        replacement = bytearray(before)
        struct.pack_into("<H", replacement, 0, seed)
        struct.pack_into("<I", replacement, 4, 0)
        data = ctypes.create_string_buffer(bytes(replacement))
        written = ctypes.c_size_t()
        if not self.kernel32.WriteProcessMemory(
            self.handle,
            ctypes.c_void_p(ADDR_RNG),
            data,
            len(replacement),
            ctypes.byref(written),
        ) or written.value != len(replacement):
            raise ctypes.WinError(ctypes.get_last_error())
        if self.read(ADDR_RNG, 8) != bytes(replacement):
            raise RuntimeError("diagnostic RNG seed did not verify")
        return old_seed, old_generation

    def read_ecl_instruction(self, address: int) -> EclInstruction:
        cached = self.ecl_instruction_cache.get(address)
        if cached is not None:
            return cached
        if not 0x10000 <= address < 0x80000000:
            raise RuntimeError(f"invalid ECL instruction pointer 0x{address:08X}")
        header = self.read(address, 12)
        instruction_time, opcode, offset_to_next = struct.unpack_from(
            "<ihh", header
        )
        if instruction_time < 0:
            raw = header
        elif not 12 <= offset_to_next <= 4096:
            raise RuntimeError(f"invalid ECL instruction size at 0x{address:08X}")
        else:
            raw = self.read(address, offset_to_next)
        instruction = EclInstruction(
            address,
            instruction_time,
            opcode,
            offset_to_next,
            header[9],
            raw.hex(),
        )
        self.ecl_instruction_cache[address] = instruction
        return instruction


def _read_ecl_program(
    process: NativeProcess,
    start_address: int,
) -> tuple[EclInstruction, ...]:
    """Capture a bounded immutable instruction graph, including jump targets."""
    program_cache = getattr(process, "ecl_program_cache", None)
    if program_cache is not None:
        cached = program_cache.get(start_address)
        if cached is not None:
            return cached
    pending = deque((start_address,))
    found: dict[int, EclInstruction] = {}
    while pending and len(found) < ECL_PROGRAM_INSTRUCTION_LIMIT:
        address = pending.popleft()
        if not address or address in found:
            continue
        instruction = process.read_ecl_instruction(address)
        found[address] = instruction
        if instruction.time < 0:
            continue
        pending.append(address + instruction.offset_to_next)
        if instruction.opcode in (2, 3, 29, 30, 31, 32, 33, 34):
            raw = bytes.fromhex(instruction.raw_hex)
            jump_offset = struct.unpack_from("<i", raw, 0x10)[0]
            pending.append(address + jump_offset)
        if instruction.opcode == 35 or 37 <= instruction.opcode <= 42:
            raw = bytes.fromhex(instruction.raw_hex)
            sub_id = struct.unpack_from("<i", raw, 0x0C)[0]
            if not 0 <= sub_id < len(process.ecl_subroutines):
                raise RuntimeError(f"invalid ECL subroutine id {sub_id}")
            pending.append(process.ecl_subroutines[sub_id])
        if instruction.opcode == 95:
            raw = bytes.fromhex(instruction.raw_hex)
            sub_id = struct.unpack_from("<i", raw, 0x0C)[0]
            if not 0 <= sub_id < len(process.ecl_subroutines):
                raise RuntimeError(f"invalid spawned ECL subroutine id {sub_id}")
            pending.append(process.ecl_subroutines[sub_id])
        if instruction.opcode in (108, 109, 114, 116):
            raw = bytes.fromhex(instruction.raw_hex)
            sub_id = struct.unpack_from("<i", raw, 0x0C)[0]
            if not 0 <= sub_id < len(process.ecl_subroutines):
                raise RuntimeError(f"invalid ECL callback subroutine id {sub_id}")
            pending.append(process.ecl_subroutines[sub_id])
    program = tuple(found[address] for address in sorted(found))
    if program_cache is not None:
        program_cache[start_address] = program
    return program


def _read_ecl_subroutines(process: NativeProcess) -> tuple[int, ...]:
    ecl_file, sub_table = struct.unpack(
        "<II", process.read(ADDR_ECL_MANAGER, 8)
    )
    if not ecl_file or not sub_table:
        return ()
    sub_count = struct.unpack("<h", process.read(ecl_file, 2))[0]
    if not 0 <= sub_count <= ECL_SUBROUTINE_LIMIT:
        raise RuntimeError(f"invalid ECL subroutine count {sub_count}")
    addresses = struct.unpack(
        "<" + "I" * sub_count,
        process.read(sub_table, sub_count * 4),
    ) if sub_count else ()
    if any(not 0x10000 <= address < 0x80000000 for address in addresses):
        raise RuntimeError("invalid ECL subroutine pointer")
    return tuple(addresses)


def _read_stage_timeline(
    process: NativeProcess,
    start_address: int,
) -> tuple[StageTimelineInstruction, ...]:
    """Read the immutable remaining source stage timeline once per pointer."""
    if not start_address:
        return ()
    if not 0x10000 <= start_address < 0x80000000:
        raise RuntimeError(
            f"invalid ECL timeline pointer 0x{start_address:08X}"
        )
    timeline_cache = getattr(process, "ecl_timeline_cache", None)
    if timeline_cache is not None:
        cached = timeline_cache.get(start_address)
        if cached is not None:
            return cached
    instruction_cache = getattr(
        process, "ecl_timeline_instruction_cache", None
    )
    result: list[StageTimelineInstruction] = []
    address = start_address
    for _ in range(ECL_TIMELINE_INSTRUCTION_LIMIT):
        instruction = (
            instruction_cache.get(address)
            if instruction_cache is not None
            else None
        )
        if instruction is None:
            header = process.read(address, 8)
            time_value, arg0, opcode, size = struct.unpack("<hhhh", header)
            if time_value < 0:
                raw = header
            else:
                # The encoded timeline legitimately uses an 8-byte header for
                # argument-free opcodes and larger records for source argument
                # views. EnemyManager walks the encoded ``size`` field, so
                # retain that source behavior rather than assuming one stride.
                if not 0x08 <= size <= 0x100:
                    raise RuntimeError(
                        "invalid ECL timeline instruction size "
                        f"{size} at 0x{address:08X}"
                    )
                raw = process.read(address, size)
            instruction = StageTimelineInstruction(
                address,
                time_value,
                arg0,
                opcode,
                size,
                raw.hex(),
            )
            if instruction_cache is not None:
                instruction_cache[address] = instruction
        result.append(instruction)
        if instruction.time < 0:
            timeline = tuple(result)
            if timeline_cache is not None:
                timeline_cache[start_address] = timeline
            return timeline
        address += instruction.size
    raise RuntimeError("ECL stage timeline exceeds bounded instruction limit")


def _timeline_subroutine_traits(
    process: NativeProcess,
    instructions: tuple[StageTimelineInstruction, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Classify source ECL graphs without embedding route strategy.

    The first bit records a graph that can configure or directly create an
    enemy bullet/laser source.  The second records BOSSSET with a non-negative
    slot.  Calls and registered callbacks are already included by the bounded
    immutable program reader.
    """
    spawn_subs = sorted({
        instruction.arg0
        for instruction in instructions
        if 0 <= instruction.opcode <= 7
        and 0 <= instruction.arg0 < len(process.ecl_subroutines)
    })
    for sub_id in spawn_subs:
        if sub_id in process.ecl_subroutine_traits:
            continue
        program = _read_ecl_program(
            process,
            process.ecl_subroutines[sub_id],
        )
        emits = any(
            67 <= instruction.opcode <= 86 for instruction in program
        )
        sets_boss = False
        for instruction in program:
            if instruction.opcode != 101:
                continue
            raw = bytes.fromhex(instruction.raw_hex)
            if len(raw) >= 16 and struct.unpack_from("<i", raw, 0x0C)[0] >= 0:
                sets_boss = True
                break
        process.ecl_subroutine_traits[sub_id] = (emits, sets_boss)
    return (
        tuple(
            sub_id for sub_id in spawn_subs
            if process.ecl_subroutine_traits[sub_id][0]
        ),
        tuple(
            sub_id for sub_id in spawn_subs
            if process.ecl_subroutine_traits[sub_id][1]
        ),
    )


def _timeline_ecl_program(
    process: NativeProcess,
    instructions: tuple[StageTimelineInstruction, ...],
) -> tuple[EclInstruction, ...]:
    """Capture immutable ECL graphs referenced by visible timeline spawns."""
    sub_ids = tuple(sorted({
        instruction.arg0
        for instruction in instructions
        if 0 <= instruction.opcode <= 7
    }))
    cache = getattr(process, "ecl_timeline_program_cache", None)
    if cache is not None and sub_ids in cache:
        return cache[sub_ids]
    program_by_address: dict[int, EclInstruction] = {}
    for sub_id in sub_ids:
        if not 0 <= sub_id < len(process.ecl_subroutines):
            raise RuntimeError(f"invalid timeline ECL subroutine id {sub_id}")
        for instruction in _read_ecl_program(
            process,
            process.ecl_subroutines[sub_id],
        ):
            program_by_address.setdefault(instruction.address, instruction)
    program = tuple(
        program_by_address[address]
        for address in sorted(program_by_address)
    )
    if cache is not None:
        cache[sub_ids] = program
    return program


def _with_installed_interrupt_programs(
    process: NativeProcess,
    program: tuple[EclInstruction, ...],
    interrupts: tuple[int, ...],
) -> tuple[EclInstruction, ...]:
    """Add source graphs reachable through a live Enemy interrupt table."""
    program_by_address = {
        instruction.address: instruction for instruction in program
    }
    for sub_id in sorted(set(interrupts)):
        if sub_id < 0:
            continue
        for instruction in _read_ecl_program(
            process,
            process.ecl_subroutines[sub_id],
        ):
            program_by_address.setdefault(instruction.address, instruction)
    return tuple(
        program_by_address[address]
        for address in sorted(program_by_address)
    )


def _read_message_program(
    process: NativeProcess,
    msg_file: int,
    msg_index: int,
) -> tuple[MessageInstruction, ...]:
    """Read one immutable ``.msg`` program from Gui's relocated table."""
    if not msg_file:
        return ()
    if not 0x10000 <= msg_file < 0x80000000:
        raise RuntimeError(f"invalid GUI message file pointer 0x{msg_file:08X}")
    cache = getattr(process, "message_program_cache", None)
    key = (msg_file, msg_index)
    if cache is not None and key in cache:
        return cache[key]
    num_programs = struct.unpack("<i", process.read(msg_file, 4))[0]
    if not 0 <= num_programs <= 1024:
        raise RuntimeError(f"invalid GUI message program count {num_programs}")
    if not 0 <= msg_index < num_programs:
        return ()
    address = struct.unpack(
        "<I", process.read(msg_file + 4 + msg_index * 4, 4)
    )[0]
    if not 0x10000 <= address < 0x80000000:
        raise RuntimeError(
            f"invalid GUI message {msg_index} pointer 0x{address:08X}"
        )
    result: list[MessageInstruction] = []
    for _ in range(MSG_PROGRAM_INSTRUCTION_LIMIT):
        header = process.read(address, 4)
        time_value, opcode, arg_size = struct.unpack("<HBB", header)
        raw = header + (process.read(address + 4, arg_size) if arg_size else b"")
        result.append(
            MessageInstruction(
                address,
                time_value,
                opcode,
                arg_size,
                raw.hex(),
            )
        )
        if opcode == 0:
            program = tuple(result)
            if cache is not None:
                cache[key] = program
            return program
        address += 4 + arg_size
        if not 0x10000 <= address < 0x80000000:
            raise RuntimeError(
                f"GUI message {msg_index} walked to invalid address "
                f"0x{address:08X}"
            )
    raise RuntimeError(
        f"GUI message {msg_index} exceeds bounded instruction limit"
    )


def _message_minimum_waits(
    program: tuple[MessageInstruction, ...],
    *,
    current_instruction: int | None = None,
    timer: int = 0,
    frames_elapsed: int = 0,
    dialogue_skippable: bool = True,
    gui_before_enemy: bool = True,
) -> int:
    """Interpret GuiImpl::RunMsg and bound priority-9 MSGWAIT stalls.

    Future input is optimistic but source-legal: Ctrl is considered held only
    after WAITSKIPPABLE enables it, and a fresh Shoot edge is available as
    soon as WAIT's mandatory first eight frames permit one.  This yields a
    lower bound on stalls and therefore an earliest safe timeline transition.
    """
    if not program:
        return 0
    start = 0
    if current_instruction is not None:
        for index, instruction in enumerate(program):
            if instruction.address == current_instruction:
                start = index
                break
        else:
            return 0
    pointer = start
    permanently_blocked = False

    def run_gui() -> bool | None:
        nonlocal pointer, timer, frames_elapsed, dialogue_skippable
        nonlocal permanently_blocked
        if not 0 <= pointer < len(program):
            return None
        if dialogue_skippable:
            timer = program[pointer].time
        while pointer < len(program) and timer >= program[pointer].time:
            instruction = program[pointer]
            raw = bytes.fromhex(instruction.raw_hex)
            if len(raw) != 4 + instruction.arg_size:
                return None
            if instruction.opcode == 0:  # MSGDELETE
                return True
            if instruction.opcode in (3, 8):  # TEXTDIALOGUE/TEXTINTRO
                frames_elapsed = 0
            elif instruction.opcode == 4:  # WAIT
                if instruction.arg_size < 4:
                    return None
                wait_frames = struct.unpack_from("<i", raw, 4)[0]
                can_advance = dialogue_skippable or (
                    frames_elapsed >= wait_frames
                    or frames_elapsed >= 8
                )
                if not can_advance:
                    frames_elapsed += 1
                    return False
            elif instruction.opcode == 6:  # ECLRESUME
                return True
            elif instruction.opcode in (10, 11):  # MSGHALT/STAGEEND
                permanently_blocked = True
                return False
            elif instruction.opcode == 13:  # WAITSKIPPABLE
                if instruction.arg_size < 4:
                    return None
                dialogue_skippable = bool(
                    struct.unpack_from("<i", raw, 4)[0]
                )
            pointer += 1
        timer += 1
        if timer < 60 and dialogue_skippable:
            timer = 60
        return False

    if gui_before_enemy:
        released = run_gui()
        if released is None:
            return 0
        if released:
            return 0
        if permanently_blocked:
            return MSG_WAIT_FRAME_LIMIT
    for waits in range(1, MSG_WAIT_FRAME_LIMIT + 1):
        released = run_gui()
        if released is None:
            return 0
        if released:
            return waits
        if permanently_blocked:
            return MSG_WAIT_FRAME_LIMIT
    return MSG_WAIT_FRAME_LIMIT


def _message_opening_guarantees_wait(
    program: tuple[MessageInstruction, ...],
    *,
    skip_pressed: bool,
) -> bool:
    """Compatibility predicate for focused tests and older callers."""
    del skip_pressed
    return _message_minimum_waits(program) > 0


def _timeline_message_delays(
    process: NativeProcess,
    msg_file: int,
    instructions: tuple[StageTimelineInstruction, ...],
    *,
    stage: int,
    difficulty: int,
    character: int,
    input_mask: int,
) -> tuple[tuple[int, int], ...]:
    """Compile minimum MSGWAIT delays for visible timeline MSGREADs."""
    del input_mask
    indices = {
        message_index
        for instruction in instructions
        if (
            message_index := timeline_message_index(
                instruction,
                stage,
                difficulty,
                character,
            )
        ) is not None
    }
    return tuple(sorted(
        (message_index, waits)
        for message_index in indices
        if (waits := _message_minimum_waits(
            _read_message_program(process, msg_file, message_index)
        )) > 0
    ))


def _current_message_waits(
    process: NativeProcess,
    msg_file: int,
    current_index: int,
    current_instruction: int,
    ignore_wait_counter: int,
    timer: int,
    frames_elapsed: int,
    dialogue_skippable: bool,
) -> int:
    """Prove remaining waits for an already-active captured message."""
    if ignore_wait_counter > 0 or current_index < 0 or not current_instruction:
        return 0
    program = _read_message_program(process, msg_file, current_index)
    # BulletManager proves hazards through priority 11, while Gui runs at 12.
    # The same frame can expose either side of RunMsg.  Take the smaller bound
    # across both interpretations to preserve safety at that phase boundary.
    return min(
        _message_minimum_waits(
            program,
            current_instruction=current_instruction,
            timer=timer,
            frames_elapsed=frames_elapsed,
            dialogue_skippable=dialogue_skippable,
            gui_before_enemy=True,
        ),
        _message_minimum_waits(
            program,
            current_instruction=current_instruction,
            timer=timer,
            frames_elapsed=frames_elapsed,
            dialogue_skippable=dialogue_skippable,
            gui_before_enemy=False,
        ),
    )


def _process_candidates(exe_name: str) -> list[tuple[int, str]]:
    api = _kernel32()

    class ProcessEntry(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    api.Process32FirstW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    api.Process32FirstW.restype = wintypes.BOOL
    api.Process32NextW.argtypes = [wintypes.HANDLE, ctypes.POINTER(ProcessEntry)]
    api.Process32NextW.restype = wintypes.BOOL
    snapshot = api.CreateToolhelp32Snapshot(0x00000002, 0)
    if snapshot == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    result: list[tuple[int, str]] = []
    try:
        entry = ProcessEntry()
        entry.dwSize = ctypes.sizeof(entry)
        more = api.Process32FirstW(snapshot, ctypes.byref(entry))
        while more:
            if entry.szExeFile.lower() == exe_name.lower():
                handle = api.OpenProcess(0x1000, False, entry.th32ProcessID)
                if handle:
                    try:
                        size = wintypes.DWORD(32768)
                        path = ctypes.create_unicode_buffer(size.value)
                        if api.QueryFullProcessImageNameW(handle, 0, path, ctypes.byref(size)):
                            result.append((entry.th32ProcessID, path.value))
                    finally:
                        api.CloseHandle(handle)
            more = api.Process32NextW(snapshot, ctypes.byref(entry))
    finally:
        api.CloseHandle(snapshot)
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def attach_exact(game_dir: Path, timeout: float = 20.0) -> NativeProcess:
    expected = os.path.normcase(os.path.abspath(str(game_dir / TARGET_EXE)))
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        exact = [
            (pid, path)
            for pid, path in _process_candidates(TARGET_EXE)
            if os.path.normcase(os.path.abspath(path)) == expected
        ]
        if len(exact) > 1:
            raise RuntimeError(f"multiple exact TH06 processes: {[pid for pid, _ in exact]}")
        if len(exact) == 1:
            pid, image_path = exact[0]
            if _sha256(Path(image_path)) != TARGET_SHA256:
                raise RuntimeError("TH06 executable SHA-256 mismatch")
            process = NativeProcess(pid)
            try:
                if process.read(IMAGE_BASE, 2) != b"MZ":
                    raise RuntimeError("TH06 image-base verification failed")
                return process
            except Exception:
                process.close()
                raise
        time.sleep(0.25)
    raise RuntimeError(f"exact target not found at {game_dir / TARGET_EXE}")


def _read_snapshot_once(
    process: NativeProcess,
    capture_epoch=None,
    bullet_read_retries: int = 0,
) -> Snapshot:
    game = process.read(ADDR_GAME_MANAGER + GAME_FLAGS_OFFSET, GAME_STAGE_OFFSET + 4 - GAME_FLAGS_OFFSET)
    frame = struct.unpack_from("<I", game, GAME_FRAMES_OFFSET - GAME_FLAGS_OFFSET)[0]
    stage = struct.unpack_from("<i", game, GAME_STAGE_OFFSET - GAME_FLAGS_OFFSET)[0]
    if process.ecl_cache_stage != stage:
        process.ecl_instruction_cache.clear()
        process.ecl_program_cache.clear()
        process.ecl_timeline_instruction_cache.clear()
        process.ecl_timeline_cache.clear()
        process.ecl_timeline_program_cache.clear()
        process.ecl_subroutine_traits.clear()
        process.message_program_cache.clear()
        process.ecl_cache_stage = stage
        process.ecl_subroutines = _read_ecl_subroutines(process)
    # The source layouts place EnemyManager's runtime array, BulletManager's
    # templates/bullets, lasers, and trailing timer in one mapped interval.
    # Read the trailing timer once before that large copy as well.  A single
    # ReadProcessMemory call is not an atomic snapshot: the calc chain can run
    # from EnemyManager into BulletManager while Windows is copying the range,
    # leaving an early enemy state beside later bullets and a completed timer.
    # Requiring the leading witness to match the copied trailing witness
    # rejects that otherwise invisible same-game-frame tear.
    bullet_time_before_pool = struct.unpack(
        "<i",
        process.read(
            ADDR_BULLET_MANAGER + BULLET_MANAGER_TIME_OFFSET + 8,
            4,
        ),
    )[0]
    pool_start = ADDR_ENEMY_MANAGER + ENEMY_ARRAY_OFFSET
    pool_end = ADDR_BULLET_MANAGER + BULLET_MANAGER_SIZE
    native_pools = process.read(pool_start, pool_end - pool_start)
    enemy_pool = native_pools[:ENEMY_COUNT * ENEMY_STRIDE]
    template_offset = ADDR_BULLET_MANAGER - pool_start
    bullet_templates = native_pools[
        template_offset:
        template_offset + BULLET_TEMPLATE_COUNT * BULLET_TEMPLATE_STRIDE
    ]
    bullet_offset = ADDR_BULLET_ARRAY - pool_start
    pool = native_pools[bullet_offset:]
    ex_function_table = process.read(
        ADDR_ECL_EX_TABLE,
        ECL_EX_COUNT * 4,
    )
    manager_relative = lambda absolute: (
        ADDR_ENEMY_MANAGER + absolute - pool_start
    )
    timeline_instruction_address = struct.unpack_from(
        "<I",
        native_pools,
        manager_relative(ENEMY_TIMELINE_INSTRUCTION_OFFSET),
    )[0]
    timeline_previous, timeline_subframe, timeline_time = struct.unpack_from(
        "<ifi",
        native_pools,
        manager_relative(ENEMY_TIMELINE_TIMER_OFFSET),
    )
    if (
        not math.isfinite(timeline_subframe)
        or not 0.0 <= timeline_subframe < 1.0
        or timeline_time < -1000
        or timeline_time >= 10_000_000
    ):
        raise RuntimeError("invalid source stage timeline timer")

    bullet_manager_relative = ADDR_BULLET_MANAGER - pool_start
    (
        _bullet_time_previous,
        bullet_time_subframe,
        bullet_time,
    ) = struct.unpack_from(
        "<ifi",
        native_pools,
        bullet_manager_relative + BULLET_MANAGER_TIME_OFFSET,
    )
    if (
        not math.isfinite(bullet_time_subframe)
        or not 0.0 <= bullet_time_subframe < 1.0
        or bullet_time < 0
        or bullet_time >= 10_000_000
    ):
        raise RuntimeError("invalid source bullet-manager timer")
    if bullet_time_before_pool != bullet_time:
        raise _SnapshotReadTorn(
            "bullet-manager phase changed while copying enemy/hazard pools: "
            f"{bullet_time_before_pool}->{bullet_time}"
        )

    game = process.read(
        ADDR_GAME_MANAGER + GAME_FLAGS_OFFSET,
        GAME_STAGE_OFFSET + 4 - GAME_FLAGS_OFFSET,
    )
    game_menu, retry_menu, gameplay_active, _completed, _practice, demo_mode = game[0:6]
    captured_frame = struct.unpack_from(
        "<I", game, GAME_FRAMES_OFFSET - GAME_FLAGS_OFFSET
    )[0]
    captured_stage = struct.unpack_from(
        "<i", game, GAME_STAGE_OFFSET - GAME_FLAGS_OFFSET
    )[0]
    if captured_frame != frame or captured_stage != stage:
        raise _SnapshotEpochChanged
    # Despite its name, GameManager::OnUpdate sets isInMenu=1 during normal
    # gameplay and 0 while its pause/retry menu breaks the calc chain.
    in_menu = bool(game_menu or retry_menu or not gameplay_active or demo_mode)

    # Read separately stored dynamic state only after the bulk hazard capture.
    # If the next calc update begins during these reads, capture_epoch observes
    # the GameManager increment and discards the snapshot.
    difficulty = struct.unpack(
        "<i", process.read(ADDR_GAME_MANAGER + GAME_DIFFICULTY_OFFSET, 4)
    )[0]
    rank, max_rank, min_rank, subrank = struct.unpack(
        "<iiii",
        process.read(ADDR_GAME_MANAGER + GAME_RANK_OFFSET, 16),
    )
    rng_seed, rng_generation = struct.unpack(
        "<HxxI", process.read(ADDR_RNG, 8)
    )
    time_stopped = bool(process.read(ADDR_GAME_MANAGER + GAME_TIME_STOPPED_OFFSET, 1)[0])
    is_replay = bool(struct.unpack("<I", process.read(ADDR_GAME_MANAGER + 0x1C, 4))[0])
    frame_multiplier = struct.unpack("<f", process.read(ADDR_FRAME_MULTIPLIER, 4))[0]
    input_mask = struct.unpack("<H", process.read(ADDR_CURRENT_INPUT, 2))[0]
    current_power = struct.unpack(
        "<H", process.read(ADDR_GAME_MANAGER + GAME_CURRENT_POWER_OFFSET, 2)
    )[0]
    lives_remaining = struct.unpack(
        "<b", process.read(ADDR_GAME_MANAGER + GAME_LIVES_REMAINING_OFFSET, 1)
    )[0]
    if not 0 <= lives_remaining <= 8:
        raise RuntimeError("invalid source remaining-life count")
    reported_effect_count = struct.unpack(
        "<i", process.read(ADDR_EFFECT_MANAGER + EFFECT_ACTIVE_COUNT_OFFSET, 4)
    )[0]
    effect_pool = process.read(
        ADDR_EFFECT_MANAGER + EFFECT_ARRAY_OFFSET,
        EFFECT_COUNT * EFFECT_STRIDE,
    )
    effect_active_upper_bound = 0
    pending_effect_rng_ids = []
    for slot in range(EFFECT_COUNT):
        base = slot * EFFECT_STRIDE
        if not effect_pool[base + EFFECT_IN_USE_OFFSET]:
            continue
        effect_id = effect_pool[base + EFFECT_ID_OFFSET]
        if not 0 <= effect_id < 20:
            raise RuntimeError(f"invalid source effect id {effect_id}")
        effect_active_upper_bound += 1
        previous = struct.unpack_from(
            "<i", effect_pool, base + EFFECT_TIMER_PREVIOUS_OFFSET
        )[0]
        current = struct.unpack_from(
            "<i", effect_pool, base + EFFECT_TIMER_CURRENT_OFFSET
        )[0]
        if previous == -999 and current == 0:
            pending_effect_rng_ids.append(effect_id)
    item_pool = process.read(
        ADDR_ITEM_MANAGER,
        ITEM_ACTIVE_COUNT_OFFSET + 4,
    )
    item_next_index, reported_item_count = struct.unpack_from(
        "<II", item_pool, ITEM_NEXT_INDEX_OFFSET
    )
    if not 0 <= item_next_index < ITEM_COUNT:
        raise RuntimeError("invalid source item next index")
    item_states = []
    for slot in range(ITEM_COUNT):
        base = slot * ITEM_STRIDE
        if not item_pool[base + ITEM_IN_USE_OFFSET]:
            continue
        item_type = struct.unpack_from(
            "<b", item_pool, base + ITEM_TYPE_OFFSET
        )[0]
        state = item_pool[base + ITEM_STATE_OFFSET]
        x, y = struct.unpack_from(
            "<ff", item_pool, base + ITEM_POSITION_OFFSET
        )
        start_x, start_y = struct.unpack_from(
            "<ff", item_pool, base + ITEM_START_POSITION_OFFSET
        )
        target_x, target_y = struct.unpack_from(
            "<ff", item_pool, base + ITEM_TARGET_POSITION_OFFSET
        )
        timer_previous, timer_subframe, timer = struct.unpack_from(
            "<ifi", item_pool, base + ITEM_TIMER_OFFSET
        )
        numbers = (
            x, y, start_x, start_y, target_x, target_y, timer_subframe,
        )
        if (
            item_type not in range(7)
            or state not in range(3)
            or not all(math.isfinite(value) for value in numbers)
            or not -1000 <= timer_previous < 1_000_000
            or not -1000 <= timer < 1_000_000
        ):
            raise RuntimeError(f"invalid source item state at slot {slot}")
        item_states.append(ItemState(
            slot,
            x,
            y,
            start_x,
            start_y,
            target_x,
            target_y,
            timer_previous,
            timer,
            timer + timer_subframe,
            item_type,
            state,
        ))
    item_active_upper_bound = len(item_states)
    if not 0 <= reported_effect_count <= 512:
        raise RuntimeError("invalid source effect active count")
    if not (
        min_rank <= rank <= max_rank
        and 0 <= subrank < 100
    ):
        raise RuntimeError("invalid source rank state")
    if not 0 <= reported_item_count <= 512:
        raise RuntimeError("invalid source item active count")
    character, shot_type = process.read(
        ADDR_GAME_MANAGER + GAME_CHARACTER_OFFSET, 2
    )
    if character not in (0, 1):
        raise RuntimeError(f"invalid player character {character}")
    if shot_type not in (0, 1):
        raise RuntimeError(f"invalid player shot type {shot_type}")
    gui_impl = struct.unpack("<I", process.read(ADDR_GUI + 4, 4))[0]
    boss_present_raw = process.read(
        ADDR_GUI + GUI_BOSS_PRESENT_OFFSET, 1
    )[0]
    if boss_present_raw not in (0, 1):
        raise RuntimeError(
            f"invalid GUI boss-present flag {boss_present_raw}"
        )
    if gui_impl and not 0x10000 <= gui_impl < 0x80000000:
        raise RuntimeError(f"invalid GuiImpl pointer 0x{gui_impl:08X}")
    if gui_impl:
        msg_address = gui_impl + GUI_IMPL_MSG_OFFSET
        (
            msg_file,
            current_message_instruction,
            current_message_index,
            message_timer_previous,
            message_timer_subframe,
            message_timer,
            message_frames_elapsed,
        ) = struct.unpack(
            "<IIiifii", process.read(msg_address, 0x1C)
        )
        ignore_wait_counter = struct.unpack(
            "<I", process.read(msg_address + GUI_MSG_IGNORE_WAIT_OFFSET, 4)
        )[0]
        message_skippable = bool(
            process.read(msg_address + GUI_MSG_SKIPPABLE_OFFSET, 1)[0]
        )
        if not math.isfinite(message_timer_subframe):
            raise RuntimeError("non-finite GUI message timer")
        if not (
            -1000000 <= message_timer_previous <= 1000000
            and -1000000 <= message_timer <= 1000000
            and 0 <= message_frames_elapsed <= 1000000
        ):
            raise RuntimeError("invalid GUI message timing state")
    else:
        msg_file = 0
        current_message_instruction = 0
        current_message_index = -1
        ignore_wait_counter = 0
        message_timer = 0
        message_frames_elapsed = 0
        message_skippable = False

    player = process.read(
        ADDR_PLAYER + PLAYER_POSITION_OFFSET,
        PLAYER_BOMB_ACTIVE_OFFSET + 4 - PLAYER_POSITION_OFFSET,
    )
    relative = lambda absolute: absolute - PLAYER_POSITION_OFFSET
    x, y = struct.unpack_from("<ff", player, relative(PLAYER_POSITION_OFFSET))
    hit_left, hit_top = struct.unpack_from("<ff", player, relative(PLAYER_HITBOX_TOP_LEFT_OFFSET))
    hit_right, hit_bottom = struct.unpack_from("<ff", player, relative(PLAYER_HITBOX_BOTTOM_RIGHT_OFFSET))
    player_state = player[relative(PLAYER_STATE_OFFSET)]
    normal_speed, focus_speed, normal_diagonal, focus_diagonal = struct.unpack_from(
        "<ffff", player, relative(PLAYER_SPEEDS_OFFSET)
    )
    values = (
        x, y, hit_left, hit_top, hit_right, hit_bottom,
        normal_speed, focus_speed, normal_diagonal, focus_diagonal, frame_multiplier,
    )
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite player or timing state")
    half_width = max(0.0, (hit_right - hit_left) / 2.0)
    half_height = max(0.0, (hit_bottom - hit_top) / 2.0)
    active_geometry = (
        0.5 <= half_width <= 8.0
        and 0.5 <= half_height <= 8.0
        and 0.5 <= focus_speed <= 8.0
        and 0.5 <= normal_speed <= 8.0
        and -64.0 <= x <= 448.0
        and -64.0 <= y <= 512.0
    )
    if player_state in (PLAYER_ALIVE, PLAYER_INVULNERABLE) and not active_geometry:
        in_menu = True

    player_sprite_pointers = {
        struct.unpack_from(
            "<I",
            player,
            relative(PLAYER_BULLETS_OFFSET)
            + slot * PLAYER_BULLET_STRIDE
            + ANM_VM_SPRITE_OFFSET,
        )[0]
        for slot in range(PLAYER_BULLET_COUNT)
        if struct.unpack_from(
            "<h",
            player,
            relative(PLAYER_BULLETS_OFFSET)
            + slot * PLAYER_BULLET_STRIDE
            + PLAYER_BULLET_STATE_OFFSET,
        )[0]
    }
    enemy_sprite_pointers = {
        struct.unpack_from(
            "<I",
            enemy_pool,
            slot * ENEMY_STRIDE + ANM_VM_SPRITE_OFFSET,
        )[0]
        for slot in range(ENEMY_COUNT)
        if enemy_pool[slot * ENEMY_STRIDE + ENEMY_FLAGS_OFFSET] & 0x80
    }
    bullet_sprite_pointer_by_slot: dict[int, int] = {}
    for slot in range(BULLET_COUNT):
        base = slot * BULLET_STRIDE
        state = struct.unpack_from("<H", pool, base + BULLET_STATE_OFFSET)[0]
        if state == 0:
            continue
        pointer = struct.unpack_from(
            "<I", pool, base + ANM_VM_SPRITE_OFFSET
        )[0]
        if not 0x10000 <= pointer < 0x80000000:
            raise _SnapshotReadTorn(
                f"bullet slot {slot} has an unpublished sprite pointer"
            )
        bullet_sprite_pointer_by_slot[slot] = pointer
    sprite_dimensions = _read_sprite_dimensions(
        process,
        player_sprite_pointers
        | enemy_sprite_pointers
        | set(bullet_sprite_pointer_by_slot.values()),
    )
    spell_active = bool(struct.unpack_from(
        "<i",
        native_pools,
        manager_relative(ENEMY_SPELL_ACTIVE_OFFSET),
    )[0])
    random_item_spawn_index = struct.unpack_from(
        "<H",
        native_pools,
        manager_relative(ENEMY_RANDOM_ITEM_SPAWN_INDEX_OFFSET),
    )[0]
    random_item_table_index = struct.unpack_from(
        "<H",
        native_pools,
        manager_relative(ENEMY_RANDOM_ITEM_TABLE_INDEX_OFFSET),
    )[0]
    if random_item_table_index >= 32:
        raise RuntimeError("invalid source random item table index")
    player_attack = _decode_player_attack(
        player,
        sprite_dimensions,
        shot_type=shot_type,
        spell_active=spell_active,
    )

    # Both timers are initialized to zero for a stage. At the supported 1x
    # rate GameManager increments at priority 4, while BulletManager advances
    # this timer only after bullet/laser updates and collision at priority 11.
    # Equality therefore rejects the otherwise invisible priority-4..10 phase.
    if (
        not in_menu
        and not time_stopped
        and 0.99 <= frame_multiplier <= 1.01
        and bullet_time != frame
    ):
        raise _SnapshotPhaseIncomplete(frame, bullet_time)
    if capture_epoch is not None:
        capture_epoch(frame)

    remaining_timeline = _read_stage_timeline(
        process,
        timeline_instruction_address,
    )
    timeline_instructions = remaining_timeline[:ECL_TIMELINE_SNAPSHOT_LIMIT]
    timeline_complete = bool(
        timeline_instructions
        and timeline_instructions[-1].time < 0
    )
    timeline_emitter_subs, timeline_boss_subs = (
        _timeline_subroutine_traits(process, timeline_instructions)
    )
    timeline_ecl_program = _timeline_ecl_program(
        process,
        timeline_instructions,
    )
    timeline_message_delays = _timeline_message_delays(
        process,
        msg_file,
        timeline_instructions,
        stage=stage,
        difficulty=difficulty,
        character=character,
        input_mask=input_mask,
    )
    timeline_current_message_waits = _current_message_waits(
        process,
        msg_file,
        current_message_index,
        current_message_instruction,
        ignore_wait_counter,
        message_timer,
        message_frames_elapsed,
        message_skippable,
    )

    bullets: list[Bullet] = []
    despawning_bullets: list[Bullet] = []
    for index in range(BULLET_COUNT):
        base = index * BULLET_STRIDE
        tail = bytes(pool[base + BULLET_SIZE_OFFSET : base + BULLET_STRIDE])
        sprite_size = (
            sprite_dimensions[bullet_sprite_pointer_by_slot[index]]
            if index in bullet_sprite_pointer_by_slot
            else None
        )
        bullet = _decode_bullet_tail(tail, index, sprite_size)
        if bullet is None:
            continue
        if bullet.state == 5:
            despawning_bullets.append(bullet)
        else:
            bullets.append(bullet)

    laser_base = ADDR_LASER_ARRAY - ADDR_BULLET_ARRAY
    lasers: list[Laser] = []
    for index in range(LASER_COUNT):
        base = laser_base + index * LASER_STRIDE
        if not struct.unpack_from("<i", pool, base + LASER_IN_USE_OFFSET)[0]:
            continue
        lx, ly = struct.unpack_from("<ff", pool, base + LASER_POSITION_OFFSET)
        angle, start_offset, end_offset, start_length, width, laser_speed = struct.unpack_from(
            "<ffffff", pool, base + LASER_ANGLE_OFFSET
        )
        start_time, hitbox_start_time, duration, despawn_duration, hitbox_end_delay = struct.unpack_from(
            "<iiiii", pool, base + LASER_START_TIME_OFFSET
        )
        timer_subframe = struct.unpack_from("<f", pool, base + LASER_TIMER_SUBFRAME_OFFSET)[0]
        timer = struct.unpack_from("<i", pool, base + LASER_TIMER_OFFSET)[0]
        flags = struct.unpack_from("<H", pool, base + LASER_FLAGS_OFFSET)[0]
        state = pool[base + LASER_STATE_OFFSET]
        laser_numbers = (
            lx,
            ly,
            angle,
            start_offset,
            end_offset,
            start_length,
            width,
            laser_speed,
            timer_subframe,
        )
        laser_times = (
            start_time,
            hitbox_start_time,
            duration,
            despawn_duration,
            hitbox_end_delay,
            timer,
        )
        if (
            not all(math.isfinite(value) for value in laser_numbers)
            or not 0.0 < width <= 1024.0
            or not 0.0 <= start_length <= 4096.0
            or not all(0 <= value < 10_000_000 for value in laser_times)
            or state not in (0, 1, 2)
        ):
            raise RuntimeError(f"invalid laser state at slot {index}")
        lasers.append(
            Laser(
                lx,
                ly,
                angle,
                start_offset,
                end_offset,
                start_length,
                width,
                laser_speed,
                start_time,
                hitbox_start_time,
                duration,
                despawn_duration,
                hitbox_end_delay,
                timer,
                timer + timer_subframe,
                flags,
                state,
                slot=index,
            )
        )

    # EnemyManager ends exactly at its separately mapped calc-chain global:
    # 0x4B79C8 + 0xEE5EC == 0x5A5FB4. The source layout puts the 256 runtime
    # slots after two archive pointers and one 0xEC8-byte template.
    bullet_sizes = tuple(
        tuple(
            value / 2.0 for value in struct.unpack_from(
                "<ff",
                bullet_templates,
                index * BULLET_TEMPLATE_STRIDE + BULLET_TEMPLATE_SIZE_OFFSET,
            )
        )
        for index in range(BULLET_TEMPLATE_COUNT)
    )
    ex_function_addresses = struct.unpack(
        "<" + "I" * ECL_EX_COUNT,
        ex_function_table,
    )
    ex_index_by_address = {
        address: index for index, address in enumerate(ex_function_addresses)
    }
    boss_pointers = struct.unpack_from(
        "<" + "I" * 8,
        native_pools,
        manager_relative(ENEMY_BOSSES_OFFSET),
    )
    timeline_boss_slots = _decode_timeline_boss_slots(boss_pointers)
    enemies: list[EnemyBody] = []
    spawners: list[EnemySpawner] = []
    for index in range(ENEMY_COUNT):
        base = index * ENEMY_STRIDE
        flags0, flags1, flags2 = struct.unpack_from(
            "<BBB", enemy_pool, base + ENEMY_FLAGS_OFFSET
        )
        if not flags0 & 0x80:
            continue
        ex, ey = struct.unpack_from("<ff", enemy_pool, base + ENEMY_POSITION_OFFSET)
        velocity_x, velocity_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_AXIS_SPEED_OFFSET
        )
        angle, angular_velocity, enemy_speed, enemy_acceleration = struct.unpack_from(
            "<ffff", enemy_pool, base + ENEMY_ANGLE_OFFSET
        )
        move_interp_x, move_interp_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_MOVE_INTERP_OFFSET
        )
        move_start_x, move_start_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_MOVE_START_OFFSET
        )
        move_subframe = struct.unpack_from(
            "<f", enemy_pool, base + ENEMY_MOVE_TIMER_SUBFRAME_OFFSET
        )[0]
        move_timer = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_MOVE_TIMER_OFFSET
        )[0]
        move_start_time = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_MOVE_START_TIME_OFFSET
        )[0]
        movement_mode = flags0 & 0x03
        movement_ease = (flags0 >> 2) & 0x07
        motion_numbers = (
            ex,
            ey,
            velocity_x,
            velocity_y,
            angle,
            angular_velocity,
            enemy_speed,
            enemy_acceleration,
            move_interp_x,
            move_interp_y,
            move_start_x,
            move_start_y,
            move_subframe,
        )
        if (
            not all(math.isfinite(value) for value in motion_numbers)
            or movement_mode == 2 and (
                movement_ease > 4 or move_start_time <= 0 or move_timer < 0
            )
        ):
            raise RuntimeError(f"invalid occupied enemy motion at slot {index}")
        motion = (
            ex,
            ey,
            velocity_x,
            velocity_y,
            angle,
            angular_velocity,
            enemy_speed,
            enemy_acceleration,
            movement_mode,
            movement_ease,
            bool(flags0 & 0x40),
            move_interp_x,
            move_interp_y,
            move_start_x,
            move_start_y,
            move_timer,
            move_timer + move_subframe,
            move_start_time,
        )
        lethal = (
            flags1 & 0x01
            and flags1 & 0x02
            and flags1 & 0x04
            and not flags2 & 0x08
        )
        hitbox_x, hitbox_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_HITBOX_OFFSET
        )
        lower_move_x, lower_move_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_LOWER_MOVE_LIMIT_OFFSET
        )
        upper_move_x, upper_move_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_UPPER_MOVE_LIMIT_OFFSET
        )
        if (
            not math.isfinite(hitbox_x)
            or not math.isfinite(hitbox_y)
            or not 0.0 <= hitbox_x <= 1024.0
            or not 0.0 <= hitbox_y <= 1024.0
        ):
            raise RuntimeError(
                f"invalid occupied enemy geometry at slot {index}"
            )
        if flags2 & 0x01 and (
            not all(math.isfinite(value) for value in (
                lower_move_x, lower_move_y, upper_move_x, upper_move_y
            ))
            or lower_move_x > upper_move_x
            or lower_move_y > upper_move_y
        ):
            raise RuntimeError(
                f"invalid occupied enemy move bounds at slot {index}"
            )
        if lethal:
            enemies.append(EnemyBody(
                ex,
                ey,
                hitbox_x / 3.0,
                hitbox_y / 3.0,
                *motion[2:],
            ))

        life = struct.unpack_from("<i", enemy_pool, base + ENEMY_LIFE_OFFSET)[0]
        death_anm1, death_anm2, death_anm3, item_drop = struct.unpack_from(
            "<BBBb", enemy_pool, base + ENEMY_DEATH_ANM_OFFSET
        )
        boss_timer_subframe = struct.unpack_from(
            "<f", enemy_pool, base + ENEMY_BOSS_TIMER_SUBFRAME_OFFSET
        )[0]
        boss_timer = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_BOSS_TIMER_OFFSET
        )[0]
        death_callback_sub = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_DEATH_CALLBACK_SUB_OFFSET
        )[0]
        (
            life_callback_threshold,
            life_callback_sub,
            timer_callback_threshold,
            timer_callback_sub,
        ) = struct.unpack_from(
            "<iiii", enemy_pool, base + ENEMY_LIFE_CALLBACK_THRESHOLD_OFFSET
        )
        bullet_rank_speed_low, bullet_rank_speed_high = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_BULLET_RANK_SPEED_LOW_OFFSET
        )
        (
            bullet_rank_amount1_low,
            bullet_rank_amount1_high,
            bullet_rank_amount2_low,
            bullet_rank_amount2_high,
        ) = struct.unpack_from(
            "<hhhh", enemy_pool, base + ENEMY_BULLET_RANK_AMOUNT1_LOW_OFFSET
        )
        interval = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_SHOOT_INTERVAL_OFFSET
        )[0]
        shooting_disabled = bool(flags0 & 0x20)
        shoot_offset_x, shoot_offset_y = struct.unpack_from(
            "<ff", enemy_pool, base + ENEMY_SHOOT_OFFSET
        )
        props = base + ENEMY_BULLET_PROPS_OFFSET
        sprite = struct.unpack_from("<h", enemy_pool, props)[0]
        angle1, angle2, speed1, speed2 = struct.unpack_from(
            "<ffff", enemy_pool, props + 0x10
        )
        ex_floats = struct.unpack_from("<ffff", enemy_pool, props + 0x20)
        ex_ints = struct.unpack_from("<iiii", enemy_pool, props + 0x30)
        count1, count2, aim_mode = struct.unpack_from(
            "<hhH", enemy_pool, props + 0x44
        )
        bullet_flags = struct.unpack_from("<I", enemy_pool, props + 0x4C)[0]
        shoot_subframe = struct.unpack_from(
            "<f", enemy_pool, base + ENEMY_SHOOT_TIMER_SUBFRAME_OFFSET
        )[0]
        shoot_timer = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_SHOOT_TIMER_OFFSET
        )[0]
        pattern_numbers = (
            shoot_offset_x,
            shoot_offset_y,
            angle1,
            angle2,
            speed1,
            speed2,
            *ex_floats,
            shoot_subframe,
        )
        props_valid = (
            all(math.isfinite(value) for value in pattern_numbers)
            and 0 <= sprite < BULLET_TEMPLATE_COUNT
            and 0 <= aim_mode <= 8
            and 0 < count1 <= BULLET_COUNT
            and 0 < count2 <= BULLET_COUNT
            and count1 * count2 <= BULLET_COUNT
        )
        active_periodic = life > 0 and interval > 0 and not shooting_disabled
        if (
            (active_periodic and not props_valid)
            or not -10_000_000 < interval < 10_000_000
            or not 0 <= shoot_timer < 10_000_000
            or active_periodic and shoot_timer > interval
            or not 0.0 <= shoot_subframe < 1.0
        ):
            raise RuntimeError(
                f"invalid periodic bullet shooter at enemy slot {index}"
            )
        bullet_pattern = None
        if props_valid:
            template = sprite * BULLET_TEMPLATE_STRIDE
            bullet_half_width, bullet_half_height = bullet_sizes[sprite]
            if (
                not math.isfinite(bullet_half_width)
                or not math.isfinite(bullet_half_height)
                or not 0.0 < bullet_half_width <= 256.0
                or not 0.0 < bullet_half_height <= 256.0
            ):
                if active_periodic:
                    raise RuntimeError(
                        f"invalid bullet template {sprite} for enemy slot {index}"
                    )
            else:
                bullet_pattern = BulletPattern(
                    sprite,
                    angle1,
                    angle2,
                    speed1,
                    speed2,
                    tuple(ex_floats),
                    tuple(ex_ints),
                    count1,
                    count2,
                    aim_mode,
                    bullet_flags,
                    bullet_half_width,
                    bullet_half_height,
                )

        def decode_ecl_context(offset: int) -> EnemyEclContext:
            instruction_address = struct.unpack_from(
                "<I", enemy_pool, offset
            )[0]
            subframe = struct.unpack_from("<f", enemy_pool, offset + 0x08)[0]
            current_time = struct.unpack_from("<i", enemy_pool, offset + 0x0C)[0]
            repeat_function = struct.unpack_from(
                "<I", enemy_pool, offset + 0x10
            )[0]
            integers = (
                *struct.unpack_from("<iiii", enemy_pool, offset + 0x14),
                *struct.unpack_from("<iiii", enemy_pool, offset + 0x34),
            )
            floats = struct.unpack_from("<ffff", enemy_pool, offset + 0x24)
            compare = struct.unpack_from("<i", enemy_pool, offset + 0x44)[0]
            if (
                not math.isfinite(subframe)
                or not 0.0 <= subframe < 1.0
                or not all(math.isfinite(value) for value in floats)
            ):
                raise RuntimeError(f"invalid ECL context at enemy slot {index}")
            repeat_index = (
                ex_index_by_address.get(repeat_function, -1)
                if repeat_function
                else None
            )
            return EnemyEclContext(
                instruction_address,
                current_time,
                current_time + subframe,
                tuple(integers),
                tuple(floats),
                compare,
                repeat_index,
            )

        context = base + ENEMY_ECL_CONTEXT_OFFSET
        current_context = decode_ecl_context(context)
        stack_depth = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_ECL_STACK_DEPTH_OFFSET
        )[0]
        if not 0 <= stack_depth <= ENEMY_ECL_STACK_CAPACITY:
            raise RuntimeError(f"invalid ECL stack depth at enemy slot {index}")
        ecl_stack = tuple(
            decode_ecl_context(
                context + ENEMY_ECL_CONTEXT_SIZE * (stack_index + 1)
            )
            for stack_index in range(stack_depth)
        )
        current_instruction_address = current_context.instruction_address
        ecl_time = current_context.time
        ecl_subframe = current_context.time_float - current_context.time
        ecl_ints = current_context.ints
        ecl_floats = current_context.floats
        ecl_compare = current_context.compare
        repeat_ex_index = current_context.repeat_ex_index
        next_instruction = None
        ecl_program = ()
        if current_instruction_address:
            next_instruction = process.read_ecl_instruction(
                current_instruction_address
            )
            ecl_program = _read_ecl_program(
                process, current_instruction_address
            )
        program_by_address = {
            instruction.address: instruction for instruction in ecl_program
        }
        for saved_context in ecl_stack:
            if saved_context.instruction_address:
                for instruction in _read_ecl_program(
                    process, saved_context.instruction_address
                ):
                    program_by_address.setdefault(
                        instruction.address, instruction
                    )
        active_callbacks = (
            death_callback_sub,
            life_callback_sub if life_callback_threshold >= 0 else -1,
            timer_callback_sub if timer_callback_threshold >= 0 else -1,
        )
        for callback_sub in active_callbacks:
            if callback_sub < 0:
                continue
            if callback_sub >= len(process.ecl_subroutines):
                raise RuntimeError(
                    f"invalid active ECL callback subroutine {callback_sub}"
                )
            callback_address = process.ecl_subroutines[callback_sub]
            for instruction in _read_ecl_program(process, callback_address):
                program_by_address.setdefault(instruction.address, instruction)
        ecl_program = tuple(program_by_address.values())

        interrupts = struct.unpack_from(
            "<" + "i" * 8,
            enemy_pool,
            base + ENEMY_INTERRUPTS_OFFSET,
        )
        run_interrupt = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_RUN_INTERRUPT_OFFSET
        )[0]
        if (
            any(not -1 <= sub_id < len(process.ecl_subroutines)
                for sub_id in interrupts)
            or not -1 <= run_interrupt < 8
        ):
            raise RuntimeError(
                f"invalid ECL interrupt state at enemy slot {index}"
            )
        laser_pointers = struct.unpack_from(
            "<" + "I" * 32,
            enemy_pool,
            base + ENEMY_LASER_POINTERS_OFFSET,
        )
        laser_slots = []
        for pointer in laser_pointers:
            if pointer == 0:
                laser_slots.append(-1)
                continue
            relative_laser = pointer - ADDR_LASER_ARRAY
            if (
                relative_laser < 0
                or relative_laser % LASER_STRIDE
                or relative_laser // LASER_STRIDE >= LASER_COUNT
            ):
                raise RuntimeError(
                    f"invalid laser pointer 0x{pointer:08X} "
                    f"at enemy slot {index}"
                )
            laser_slots.append(relative_laser // LASER_STRIDE)
        laser_store = struct.unpack_from(
            "<i", enemy_pool, base + ENEMY_LASER_STORE_OFFSET
        )[0]
        if not 0 <= laser_store < 32:
            raise RuntimeError(
                f"invalid laser store {laser_store} at enemy slot {index}"
            )
        # ENEMYINTERRUPTSET may already be behind the live instruction
        # pointer, while the stage timeline can still set runInterrupt on a
        # boss.  The table is authoritative state, so seed every installed
        # target graph instead of relying only on future fallthrough.
        ecl_program = _with_installed_interrupt_programs(
            process,
            ecl_program,
            tuple(interrupts),
        )
        is_boss = bool(flags1 & 0x08)
        boss_id = _decode_enemy_boss_id(
            enemy_pool,
            base,
            index,
            is_boss,
            timeline_boss_slots,
        )

        enemy_sprite_pointer = struct.unpack_from(
            "<I", enemy_pool, base + ANM_VM_SPRITE_OFFSET
        )[0]
        enemy_sprite_width, enemy_sprite_height = (
            sprite_dimensions[enemy_sprite_pointer]
        )
        spawners.append(EnemySpawner(
            index,
            *motion,
            shoot_offset_x,
            shoot_offset_y,
            bullet_rank_speed_low,
            bullet_rank_speed_high,
            bullet_rank_amount1_low,
            bullet_rank_amount1_high,
            bullet_rank_amount2_low,
            bullet_rank_amount2_high,
            life,
            shooting_disabled,
            interval,
            shoot_timer,
            shoot_timer + shoot_subframe,
            bullet_pattern,
            ecl_time,
            ecl_time + ecl_subframe,
            tuple(ecl_ints),
            tuple(ecl_floats),
            ecl_compare,
            repeat_ex_index,
            next_instruction,
            ecl_program,
            ecl_stack,
            hitbox_x / 3.0,
            hitbox_y / 3.0,
            bool(flags1 & 0x01),
            bool(flags1 & 0x02),
            bool(flags2 & 0x08),
            bool(flags2 & 0x04),
            process.ecl_subroutines,
            lower_move_x,
            lower_move_y,
            upper_move_x,
            upper_move_y,
            bool(flags2 & 0x01),
            boss_timer,
            boss_timer + boss_timer_subframe,
            death_callback_sub,
            life_callback_threshold,
            life_callback_sub,
            timer_callback_threshold,
            timer_callback_sub,
            is_boss,
            bool(flags2 & 0x10),
            bool(flags1 & 0x10),
            tuple(ex_floats),
            tuple(ex_ints),
            (flags1 >> 5) & 0x07,
            boss_id,
            tuple(interrupts),
            run_interrupt,
            bool(flags1 & 0x04),
            enemy_sprite_width / 2.0,
            enemy_sprite_height / 2.0,
            death_anm1,
            death_anm2,
            death_anm3,
            item_drop,
            tuple(laser_slots),
            laser_store,
        ))
    return Snapshot(
        frame, stage, player_state, x, y, half_width, half_height,
        normal_speed, focus_speed, normal_diagonal, focus_diagonal,
        frame_multiplier, input_mask, tuple(bullets), len(lasers), in_menu, time_stopped,
        bool(is_replay or demo_mode), tuple(lasers), tuple(enemies),
        tuple(despawning_bullets), bullet_read_retries, tuple(spawners),
        difficulty, rank, bullet_sizes, rng_seed, rng_generation,
        current_power, timeline_time, timeline_time + timeline_subframe,
        timeline_instructions, timeline_complete,
        timeline_emitter_subs, timeline_boss_subs,
        process.ecl_subroutines, timeline_ecl_program,
        character, timeline_message_delays,
        timeline_current_message_waits,
        player_attack,
        effect_active_upper_bound,
        item_active_upper_bound,
        random_item_spawn_index,
        random_item_table_index,
        current_message_index >= 0,
        subrank,
        max_rank,
        min_rank,
        tuple(pending_effect_rng_ids),
        tuple(item_states),
        item_next_index,
        timeline_boss_slots,
        lives_remaining,
        timeline_previous,
        bool(boss_present_raw),
    )


def read_snapshot(process: NativeProcess) -> Snapshot:
    """Return one frame-coherent native snapshot or fail closed."""
    observed_epochs = []
    observed_phases = []
    observed_torn_reads = []
    last_decode_error = None
    bullet_read_retries = 0
    for _attempt in range(8):
        before = read_game_frame(process)
        epoch_checked = False

        def capture_epoch(snapshot_frame):
            nonlocal epoch_checked
            after = read_game_frame(process)
            observed_epochs.append((before, snapshot_frame, after))
            epoch_checked = True
            if before != snapshot_frame or snapshot_frame != after:
                raise _SnapshotEpochChanged

        try:
            snapshot = _read_snapshot_once(
                process,
                capture_epoch,
                bullet_read_retries,
            )
        except _SnapshotEpochChanged:
            continue
        except _SnapshotPhaseIncomplete as error:
            observed_phases.append((error.game_frame, error.bullet_time))
            continue
        except _SnapshotReadTorn as error:
            observed_torn_reads.append(str(error))
            continue
        except NativeDecodeError as error:
            # SpawnSingleBullet publishes state before geometry and velocity.
            # Discard the whole captured pool instead of mixing a later tail
            # into it, then retry from a fresh epoch.
            bullet_read_retries += 1
            last_decode_error = error
            continue
        if not epoch_checked:
            # Keep the helper independently mockable in focused tests.
            after = read_game_frame(process)
            observed_epochs.append((before, snapshot.frame, after))
            if before != snapshot.frame or snapshot.frame != after:
                continue
        return snapshot
    if last_decode_error is not None:
        evidence = dict(last_decode_error.evidence)
        evidence["read_retries"] = bullet_read_retries
        evidence["observed_epochs"] = observed_epochs
        evidence["observed_phases"] = observed_phases
        evidence["observed_torn_reads"] = observed_torn_reads
        raise NativeDecodeError(str(last_decode_error), evidence)
    if observed_torn_reads:
        raise NativeDecodeError(
            "snapshot remained internally incoherent after retries",
            {
                "observed_epochs": observed_epochs,
                "observed_phases": observed_phases,
                "observed_torn_reads": observed_torn_reads,
            },
        )
    raise NativeDecodeError(
        "native state changed or remained inside an incomplete calc phase "
        "throughout snapshot reads",
        {
            "observed_epochs": observed_epochs,
            "observed_phases": observed_phases,
        },
    )


def read_game_frame(process: NativeProcess) -> int:
    """Read the source-defined stage frame at the physical command boundary."""
    return struct.unpack(
        "<I", process.read(ADDR_GAME_MANAGER + GAME_FRAMES_OFFSET, 4)
    )[0]


def read_menu_state(process: NativeProcess) -> tuple[int, int, int]:
    block = process.read(
        ADDR_MAIN_MENU + MAIN_MENU_CURSOR_OFFSET,
        MAIN_MENU_TIMER_OFFSET + 4 - MAIN_MENU_CURSOR_OFFSET,
    )
    cursor = struct.unpack_from("<i", block, 0)[0]
    state = struct.unpack_from("<i", block, MAIN_MENU_STATE_OFFSET - MAIN_MENU_CURSOR_OFFSET)[0]
    timer = struct.unpack_from("<i", block, MAIN_MENU_TIMER_OFFSET - MAIN_MENU_CURSOR_OFFSET)[0]
    return state, cursor, timer


def read_supervisor_state(process: NativeProcess) -> tuple[int, int]:
    wanted, current = struct.unpack(
        "<ii", process.read(ADDR_SUPERVISOR + SUPERVISOR_STATES_OFFSET, 8)
    )
    if not 0 <= wanted <= 10 or not 0 <= current <= 10:
        raise RuntimeError(f"invalid Supervisor states: wanted={wanted}, current={current}")
    return wanted, current


def read_result_screen(process: NativeProcess) -> ResultScreenState | None:
    """Find ResultScreen through the source-defined calc chain."""
    pointer = struct.unpack(
        "<I", process.read(ADDR_CHAIN + CHAIN_ROOT_NEXT_OFFSET, 4)
    )[0]
    visited: set[int] = set()
    for _ in range(64):
        if pointer == 0:
            return None
        if pointer in visited or not 0x10000 <= pointer < 0x80000000:
            raise RuntimeError(f"invalid calc-chain pointer: 0x{pointer:08X}")
        visited.add(pointer)
        elem = process.read(pointer, CHAIN_ELEM_SIZE)
        callback = struct.unpack_from("<I", elem, CHAIN_ELEM_CALLBACK_OFFSET)[0]
        if callback == RESULT_SCREEN_ON_UPDATE:
            address = struct.unpack_from("<I", elem, CHAIN_ELEM_ARG_OFFSET)[0]
            if not 0x10000 <= address < 0x80000000:
                raise RuntimeError(f"invalid ResultScreen pointer: 0x{address:08X}")
            block = process.read(address, RESULT_SCREEN_STATE_SIZE)
            frame_timer, state, cursor = struct.unpack_from("<iii", block, 0x4)
            replay_number = struct.unpack_from("<i", block, 0x1C)[0]
            selected_character = struct.unpack_from("<i", block, 0x20)[0]
            if not 0 <= frame_timer < 10_000_000 or not 0 <= state <= 17:
                raise RuntimeError(
                    f"invalid ResultScreen state: timer={frame_timer}, state={state}"
                )
            return ResultScreenState(
                address, frame_timer, state, cursor, replay_number, selected_character
            )
        pointer = struct.unpack_from("<I", elem, CHAIN_ELEM_NEXT_OFFSET)[0]
    raise RuntimeError("calc chain exceeded 64 elements")


def read_dialogue_state(process: NativeProcess) -> tuple[bool, bool]:
    impl = struct.unpack("<I", process.read(ADDR_GUI + 4, 4))[0]
    if impl == 0:
        return False, False
    msg = impl + GUI_IMPL_MSG_OFFSET
    current_index = struct.unpack("<i", process.read(msg + GUI_MSG_INDEX_OFFSET, 4))[0]
    skippable = bool(process.read(msg + GUI_MSG_SKIPPABLE_OFFSET, 1)[0])
    return current_index >= 0, skippable
