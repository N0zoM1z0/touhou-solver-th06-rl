"""Bounded, fail-closed ECL bullet-birth forecasting.

The forecaster interprets only source-audited emission instructions and the
small amount of control flow needed to reach them. Its coverage result is part
of the contract: callers must not treat frames after the first unsupported
instruction as modeled.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache
import math
import struct

from ..model import (
    Bullet,
    BulletPattern,
    EnemyEclContext,
    EnemySpawner,
    EclInstruction,
    Laser,
)
from .births import UnsupportedBirthModel, spawn_pattern, spawn_pattern_envelope
from .enemies import finish_motion_values, interpolation_progress
from .lasers import LaserHazard, advance_laser, future_hazards
from .rng import RngState


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


OPCODE_NOP = 0
OPCODE_JUMP = 2
OPCODE_JUMPDEC = 3
OPCODE_SET_INT = 4
OPCODE_SET_FLOAT = 5
OPCODE_SET_INT_RANDOM = 6
OPCODE_SET_INT_RANDOM_MIN = 7
OPCODE_SET_FLOAT_RANDOM = 8
OPCODE_SET_FLOAT_RANDOM_MIN = 9
OPCODE_SET_SELF_X = 10
OPCODE_SET_SELF_Y = 11
OPCODE_SET_SELF_Z = 12
OPCODE_MATH_INT_ADD = 13
OPCODE_MATH_INT_SUBTRACT = 14
OPCODE_MATH_INT_MULTIPLY = 15
OPCODE_MATH_INT_DIVIDE = 16
OPCODE_MATH_INT_MODULO = 17
OPCODE_MATH_INCREMENT = 18
OPCODE_MATH_DECREMENT = 19
OPCODE_MATH_FLOAT_ADD = 20
OPCODE_MATH_FLOAT_SUBTRACT = 21
OPCODE_MATH_FLOAT_MULTIPLY = 22
OPCODE_MATH_FLOAT_DIVIDE = 23
OPCODE_MATH_FLOAT_MODULO = 24
OPCODE_MATH_ATAN2 = 25
OPCODE_MATH_NORMALIZE_ANGLE = 26
OPCODE_COMPARE_INT = 27
OPCODE_COMPARE_FLOAT = 28
OPCODE_JUMP_LESS = 29
OPCODE_JUMP_LESS_EQUAL = 30
OPCODE_JUMP_EQUAL = 31
OPCODE_JUMP_GREATER = 32
OPCODE_JUMP_GREATER_EQUAL = 33
OPCODE_JUMP_NOT_EQUAL = 34
OPCODE_CALL = 35
OPCODE_RETURN = 36
OPCODE_CALL_LESS = 37
OPCODE_CALL_LESS_EQUAL = 38
OPCODE_CALL_EQUAL = 39
OPCODE_CALL_GREATER = 40
OPCODE_CALL_GREATER_EQUAL = 41
OPCODE_CALL_NOT_EQUAL = 42
OPCODE_MOVE_POSITION = 43
OPCODE_MOVE_AT_PLAYER = 51
OPCODE_MOVE_RANDOM = 49
OPCODE_MOVE_RANDOM_IN_BOUNDS = 50
OPCODE_MOVE_DIR_TIME_FIRST = 52
OPCODE_MOVE_POSITION_TIME_FIRST = 56
OPCODE_MOVE_TIME_FIRST = 61
OPCODE_MOVE_TIME_LAST = 64
OPCODE_MOVE_BOUNDS_SET = 65
OPCODE_MOVE_BOUNDS_DISABLE = 66
OPCODE_BULLET_FIRST = 67
OPCODE_BULLET_LAST = 75
OPCODE_SHOOT_INTERVAL = 76
OPCODE_SHOOT_INTERVAL_DELAYED = 77
OPCODE_SHOOT_DISABLED = 78
OPCODE_SHOOT_ENABLED = 79
OPCODE_SHOOT_NOW = 80
OPCODE_SHOOT_OFFSET = 81
OPCODE_BULLET_EFFECTS = 82
OPCODE_BULLET_CANCEL = 83
OPCODE_BULLET_SOUND = 84
OPCODE_LASER_CREATE = 85
OPCODE_LASER_CREATE_AIMED = 86
OPCODE_LASER_INDEX = 87
OPCODE_LASER_ROTATE = 88
OPCODE_LASER_ROTATE_FROM_PLAYER = 89
OPCODE_LASER_OFFSET = 90
OPCODE_LASER_TEST = 91
OPCODE_LASER_CANCEL = 92
OPCODE_SPELL_START = 93
OPCODE_SPELL_END = 94
OPCODE_ENEMY_CREATE = 95
OPCODE_ENEMY_KILL_ALL = 96
OPCODE_ANIMATION_MAIN = 97
OPCODE_ANIMATION_POSES = 98
OPCODE_ANIMATION_SLOT = 99
OPCODE_ANIMATION_DEATH = 100
OPCODE_BOSS_SET = 101
OPCODE_SPELL_EFFECT = 102
OPCODE_HITBOX_SET = 103
OPCODE_COLLIDABLE_FLAG = 104
OPCODE_DAMAGEABLE_FLAG = 105
OPCODE_EFFECT_SOUND = 106
OPCODE_DEATH_FLAG = 107
OPCODE_DEATH_CALLBACK = 108
OPCODE_INTERRUPT_SET = 109
OPCODE_INTERRUPT = 110
OPCODE_LIFE_SET = 111
OPCODE_BOSS_TIMER_SET = 112
OPCODE_LIFE_CALLBACK_THRESHOLD = 113
OPCODE_LIFE_CALLBACK_SUB = 114
OPCODE_TIMER_CALLBACK_THRESHOLD = 115
OPCODE_TIMER_CALLBACK_SUB = 116
OPCODE_EFFECT_PARTICLE = 118
OPCODE_INTERACTABLE_FLAG = 117
OPCODE_DROP_ITEMS = 119
OPCODE_ANIMATION_ROTATION = 120
OPCODE_EX_CALL = 121
OPCODE_EX_REPEAT = 122
OPCODE_TIME_SET = 123
OPCODE_DROP_ITEM_ID = 124
OPCODE_STAGE_UNPAUSE = 125
OPCODE_BOSS_LIFE_COUNT = 126
OPCODE_DEBUG_WATCH = 127
OPCODE_ANIMATION_INTERRUPT_MAIN = 128
OPCODE_ANIMATION_INTERRUPT_SLOT = 129
OPCODE_CALL_STACK_DISABLED = 130
OPCODE_BULLET_RANK_INFLUENCE = 131
OPCODE_INVISIBLE_FLAG = 132
OPCODE_BOSS_TIMER_CLEAR = 133
OPCODE_LASER_CLEAR_ALL = 134
OPCODE_SPELL_TIMEOUT_FLAG = 135
ECL_OPCODE_COUNT = 136
MAX_ABSTRACT_INTEGER_RNG_BRANCHES = 64
MAX_ABSTRACT_INTEGER_RNG_EVALUATIONS = 256
ENEMY_CREATE_WORLD_REASON = "future ECL enemy creation needs a world-emitter insertion"
EFFECT_RANDOM_SPRITE_IDS = frozenset((*range(4, 12), 19))


def consume_effect_spawn_rng(rng: RngState, effect_ids) -> None:
    """Run time-zero SetRandomSprite draws from shipped effect ANM scripts."""
    for effect_id in effect_ids:
        if effect_id in EFFECT_RANDOM_SPRITE_IDS:
            rng.u16()


def _future_laser_aabb(
    hazard: LaserHazard,
    aimed: bool,
    uncertainty_x: float,
    uncertainty_y: float,
) -> tuple[float, float, float, float]:
    """Conservative AABB for one not-yet-born rotated laser hitbox."""
    half_length = hazard.size_x / 2.0
    half_width = hazard.size_y / 2.0
    if aimed:
        near = hazard.center_offset - half_length
        far = hazard.center_offset + half_length
        radius = math.hypot(max(abs(near), abs(far)), half_width)
        return (
            hazard.origin_x - radius - uncertainty_x,
            hazard.origin_y - radius - uncertainty_y,
            hazard.origin_x + radius + uncertainty_x,
            hazard.origin_y + radius + uncertainty_y,
        )
    cosine = math.cos(hazard.angle)
    sine = math.sin(hazard.angle)
    center_x = hazard.origin_x + cosine * hazard.center_offset
    center_y = hazard.origin_y + sine * hazard.center_offset
    extent_x = (
        abs(cosine) * half_length
        + abs(sine) * half_width
        + uncertainty_x
    )
    extent_y = (
        abs(sine) * half_length
        + abs(cosine) * half_width
        + uncertainty_y
    )
    return (
        center_x - extent_x,
        center_y - extent_y,
        center_x + extent_x,
        center_y + extent_y,
    )


@dataclass
class _HardLaserState:
    laser: Laser
    angle_unconstrained: bool = False
    uncertainty_x: float = 0.0
    uncertainty_y: float = 0.0
    contributes_hazards: bool = False


class HardLaserWorld:
    """Per-emitter exact phase state with conservative Hard geometry.

    ECL owns pointer writes and BulletManager owns the later timer/segment
    update.  This compact world keeps that ordering without using the current
    player position as a nominal answer for an aimed future beam.
    """

    def __init__(self, lasers: tuple[Laser, ...]):
        if (
            len({laser.slot for laser in lasers}) != len(lasers)
            or any(not 0 <= laser.slot < 64 for laser in lasers)
        ):
            raise UnsupportedBirthModel("source laser slots are incomplete")
        self._states = {
            laser.slot: _HardLaserState(replace(
                laser,
                angular_velocity=0.0,
                motion_known=True,
            ))
            for laser in lasers
        }
        self._initial_slots = frozenset(self._states)
        self._created_slots: set[int] = set()
        self.mutated_initial_slots: set[int] = set()
        self.missing_dereferences: set[int] = set()
        self.retired_created = False
        self.created_count = 0

    def spawn_laser_hard(
        self,
        laser: Laser,
        *,
        aimed: bool,
        uncertainty_x: float,
        uncertainty_y: float,
    ) -> int:
        slot = next(
            (index for index in range(64) if index not in self._states),
            None,
        )
        if slot is None:
            raise UnsupportedBirthModel("source laser pool is full")
        self._states[slot] = _HardLaserState(
            replace(
                laser,
                slot=slot,
                angular_velocity=0.0,
                motion_known=True,
            ),
            aimed,
            max(0.0, uncertainty_x),
            max(0.0, uncertainty_y),
            True,
        )
        self._created_slots.add(slot)
        self.created_count += 1
        return slot

    def spawn_laser(self, laser: Laser) -> int:
        return self.spawn_laser_hard(
            laser,
            aimed=False,
            uncertainty_x=0.0,
            uncertainty_y=0.0,
        )

    def laser_at(self, slot: int) -> Laser | None:
        state = self._states.get(slot)
        return state.laser if state is not None else None

    def observe_laser_dereference(self, slot: int, present: bool) -> None:
        if slot >= 0 and not present:
            self.missing_dereferences.add(slot)

    def replace_laser_hard(
        self,
        slot: int,
        laser: Laser,
        *,
        angle_unconstrained: bool = False,
        uncertainty_x: float | None = None,
        uncertainty_y: float | None = None,
    ) -> None:
        state = self._states.get(slot)
        if state is None:
            return
        if slot in self._initial_slots:
            self.mutated_initial_slots.add(slot)
        self._states[slot] = _HardLaserState(
            replace(
                laser,
                slot=slot,
                angular_velocity=0.0,
                motion_known=True,
            ),
            state.angle_unconstrained or angle_unconstrained,
            (
                state.uncertainty_x
                if uncertainty_x is None else max(0.0, uncertainty_x)
            ),
            (
                state.uncertainty_y
                if uncertainty_y is None else max(0.0, uncertainty_y)
            ),
            True,
        )

    def replace_laser(self, slot: int, laser: Laser) -> None:
        self.replace_laser_hard(slot, laser)

    def advance_hazards(
        self,
    ) -> tuple[
        tuple[tuple[float, float, float, float], ...],
        tuple[LaserHazard, ...],
    ]:
        boxes = []
        oriented = []
        for slot, state in tuple(sorted(self._states.items())):
            following, hazards = advance_laser(state.laser)
            if state.contributes_hazards:
                if (
                    state.angle_unconstrained
                    or state.uncertainty_x > 0.0
                    or state.uncertainty_y > 0.0
                ):
                    boxes.extend(
                        _future_laser_aabb(
                            hazard,
                            state.angle_unconstrained,
                            state.uncertainty_x,
                            state.uncertainty_y,
                        )
                        for hazard in hazards
                    )
                else:
                    oriented.extend(hazards)
            if following is None:
                if slot in self._created_slots:
                    self.retired_created = True
                del self._states[slot]
            else:
                state.laser = following
        return tuple(boxes), tuple(oriented)

# Every source opcode has one deliberate authority classification.  The
# interpreter branches below implement MODELLED_ECL_OPCODES.  Hazard-neutral
# instructions may be ignored because they only change presentation, scoring,
# drops, or remove an already modelled hazard.  The remaining instructions
# stop coverage with a source-specific reason; none silently become safe.
HAZARD_NEUTRAL_ECL_OPCODES = frozenset({
    OPCODE_BULLET_CANCEL,
    OPCODE_BULLET_SOUND,
    OPCODE_ANIMATION_MAIN,
    OPCODE_ANIMATION_POSES,
    OPCODE_ANIMATION_SLOT,
    OPCODE_EFFECT_SOUND,
    OPCODE_ANIMATION_ROTATION,
    OPCODE_STAGE_UNPAUSE,
    OPCODE_BOSS_LIFE_COUNT,
    OPCODE_DEBUG_WATCH,
    OPCODE_ANIMATION_INTERRUPT_MAIN,
    OPCODE_ANIMATION_INTERRUPT_SLOT,
    OPCODE_LASER_CLEAR_ALL,
    OPCODE_SPELL_TIMEOUT_FLAG,
})

FAIL_CLOSED_ECL_OPCODES = {
    OPCODE_SET_SELF_Z: "SETVARSELFZ needs the uncaptured source z coordinate",
    OPCODE_LASER_ROTATE: "future ECL laser mutation is not represented",
    OPCODE_LASER_ROTATE_FROM_PLAYER: "future aimed ECL laser mutation is not represented",
    OPCODE_LASER_OFFSET: "future ECL laser mutation is not represented",
    OPCODE_LASER_TEST: "future ECL laser liveness is not captured",
    OPCODE_ENEMY_CREATE: ENEMY_CREATE_WORLD_REASON,
    OPCODE_ENEMY_KILL_ALL: "ENEMYKILLALL can invoke another emitter callback",
    OPCODE_EX_CALL: "ECL external instruction can mutate world hazards",
}

MODELLED_ECL_OPCODES = frozenset(
     {OPCODE_NOP, 1, *range(OPCODE_JUMP, OPCODE_SET_SELF_Z),
     *range(OPCODE_MATH_INT_ADD, OPCODE_MOVE_BOUNDS_DISABLE + 1),
     *range(OPCODE_BULLET_FIRST, OPCODE_BULLET_EFFECTS + 1),
     OPCODE_LASER_CREATE, OPCODE_LASER_CREATE_AIMED, OPCODE_LASER_INDEX,
     OPCODE_LASER_CANCEL,
     OPCODE_SPELL_START, OPCODE_SPELL_END,
     OPCODE_HITBOX_SET, OPCODE_COLLIDABLE_FLAG,
     OPCODE_DAMAGEABLE_FLAG,
     OPCODE_DEATH_FLAG,
     OPCODE_ANIMATION_DEATH, OPCODE_SPELL_EFFECT, OPCODE_EFFECT_PARTICLE,
     OPCODE_DROP_ITEM_ID,
     OPCODE_DEATH_CALLBACK,
     OPCODE_BOSS_SET, OPCODE_INTERRUPT_SET, OPCODE_INTERRUPT,
     OPCODE_LIFE_SET, OPCODE_BOSS_TIMER_SET, OPCODE_TIMER_CALLBACK_THRESHOLD,
     OPCODE_TIMER_CALLBACK_SUB, OPCODE_LIFE_CALLBACK_THRESHOLD,
     OPCODE_LIFE_CALLBACK_SUB, OPCODE_INTERACTABLE_FLAG, OPCODE_DROP_ITEMS,
     OPCODE_EX_REPEAT, OPCODE_TIME_SET, OPCODE_CALL_STACK_DISABLED,
     OPCODE_BULLET_RANK_INFLUENCE, OPCODE_INVISIBLE_FLAG,
     OPCODE_BOSS_TIMER_CLEAR}
)

assert (
    MODELLED_ECL_OPCODES
    | HAZARD_NEUTRAL_ECL_OPCODES
    | FAIL_CLOSED_ECL_OPCODES.keys()
) == frozenset(range(ECL_OPCODE_COUNT))


@dataclass(frozen=True)
class EclItemBirth:
    """One source ECL item request before ItemManager's same-frame pass."""

    x: float
    y: float
    below_full_type: int
    full_power_type: int
    state: int = 0


@dataclass(frozen=True)
class EclForecast:
    births: tuple[tuple[Bullet, ...], ...]
    covered_frames: int
    reason: str = ""
    next_spawner: EnemySpawner | None = None
    body_hazards: tuple[tuple[tuple[float, float, float, float], ...], ...] = ()
    finished: bool = False
    unresolved_int_extent: int = 0
    created_emitters: tuple[EnemySpawner, ...] = ()
    effect_spawns: tuple[tuple[int, ...], ...] = ()
    item_spawns: tuple[int, ...] = ()
    item_births: tuple[tuple[EclItemBirth, ...], ...] = ()
    enemy_kill_all: tuple[bool, ...] = ()


@dataclass(frozen=True)
class FloatInterval:
    low: float
    high: float


@lru_cache(maxsize=256)
def _compiled_program(
    instructions: tuple[EclInstruction, ...],
) -> dict[int, EclInstruction]:
    """Compile the immutable captured ECL graph once across snapshots."""
    return {instruction.address: instruction for instruction in instructions}


@lru_cache(maxsize=None)
def _instruction_raw(instruction: EclInstruction) -> bytes:
    """Decode immutable opcode bytes once, before runtime instantiation."""
    return bytes.fromhex(instruction.raw_hex)


def _float_add(left: float | FloatInterval, right: float | FloatInterval) -> float | FloatInterval:
    if isinstance(left, FloatInterval) or isinstance(right, FloatInterval):
        left_low, left_high = (left.low, left.high) if isinstance(left, FloatInterval) else (left, left)
        right_low, right_high = (right.low, right.high) if isinstance(right, FloatInterval) else (right, right)
        return FloatInterval(left_low + right_low, left_high + right_high)
    return left + right


def _float_subtract(
    left: float | FloatInterval,
    right: float | FloatInterval,
) -> float | FloatInterval:
    if isinstance(left, FloatInterval) or isinstance(right, FloatInterval):
        left_low, left_high = (left.low, left.high) if isinstance(left, FloatInterval) else (left, left)
        right_low, right_high = (right.low, right.high) if isinstance(right, FloatInterval) else (right, right)
        return FloatInterval(left_low - right_high, left_high - right_low)
    return left - right


def _float_multiply(
    left: float | FloatInterval,
    right: float | FloatInterval,
) -> float | FloatInterval:
    if isinstance(left, FloatInterval) or isinstance(right, FloatInterval):
        left_low, left_high = (
            (left.low, left.high) if isinstance(left, FloatInterval) else (left, left)
        )
        right_low, right_high = (
            (right.low, right.high) if isinstance(right, FloatInterval) else (right, right)
        )
        products = (
            left_low * right_low,
            left_low * right_high,
            left_high * right_low,
            left_high * right_high,
        )
        return FloatInterval(min(products), max(products))
    return left * right


def _float_divide(
    left: float | FloatInterval,
    right: float | FloatInterval,
) -> float | FloatInterval:
    if not isinstance(left, FloatInterval) and not isinstance(right, FloatInterval):
        if right == 0.0:
            raise UnsupportedBirthModel("ECL float division by zero")
        return left / right
    right_low, right_high = (
        (right.low, right.high) if isinstance(right, FloatInterval) else (right, right)
    )
    if right_low <= 0.0 <= right_high:
        raise UnsupportedBirthModel("ECL float division interval contains zero")
    reciprocal = FloatInterval(1.0 / right_high, 1.0 / right_low)
    return _float_multiply(left, reciprocal)


def _maximum_magnitude(value: float | FloatInterval) -> float:
    if isinstance(value, FloatInterval):
        return max(abs(value.low), abs(value.high))
    return abs(value)


def _interval_center(
    value: float | FloatInterval,
) -> tuple[float, float]:
    """Return a conservative midpoint/half-width representation."""
    if isinstance(value, FloatInterval):
        return (value.low + value.high) / 2.0, (value.high - value.low) / 2.0
    return value, 0.0


def _copy_spawner(spawner: EnemySpawner, **changes) -> EnemySpawner:
    """Clone forecast state without reflecting over the large dataclass."""
    clone = object.__new__(EnemySpawner)
    clone.__dict__.update(spawner.__dict__)
    clone.__dict__.update(changes)
    return clone


def source_enemy_template(
    ecl_program: tuple[EclInstruction, ...],
    ecl_subroutines: tuple[int, ...],
    sub_id: int,
    x: float,
    y: float,
    life: int,
    item_drop: int = -2,
) -> EnemySpawner | None:
    """Build the hazard-relevant state copied by source ``SpawnEnemy``."""
    if not 0 <= sub_id < len(ecl_subroutines):
        return None
    start_address = ecl_subroutines[sub_id]
    first = _compiled_program(ecl_program).get(start_address)
    if first is None:
        return None
    return EnemySpawner(
        slot=-1,
        x=x,
        y=y,
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
        bullet_rank_speed_low=-0.5,
        bullet_rank_speed_high=0.5,
        bullet_rank_amount1_low=0,
        bullet_rank_amount1_high=0,
        bullet_rank_amount2_low=0,
        bullet_rank_amount2_high=0,
        life=life if life >= 0 else 1,
        shooting_disabled=False,
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
        ecl_program=ecl_program,
        hitbox_half_width=4.0,
        hitbox_half_height=4.0,
        interactable=True,
        collidable=True,
        ecl_subroutines=ecl_subroutines,
        damageable=True,
        life_callback_sub=0,
        timer_callback_sub=0,
        # SpawnEnemy stores the i16 call argument in Enemy::itemDrop (i8).
        item_drop=((item_drop + 128) % 256) - 128,
    )


def _trunc_div(numerator: int, denominator: int) -> int:
    return int(numerator / denominator)


def _rank_int(low: int, high: int, rank: int) -> int:
    return _trunc_div(rank * (high - low), 32) + low


def _rank_float(low: float, high: float, rank: int) -> float:
    return rank * (high - low) / 32.0 + low


def _int_var(value: int, integers: list[int], difficulty: int, rank: int, life: int) -> int:
    if -10004 <= value <= -10001:
        return integers[-10001 - value]
    if -10012 <= value <= -10009:
        return integers[4 + (-10009 - value)]
    if value == -10013:
        return difficulty
    if value == -10014:
        return rank
    if value == -10024:
        return life
    return value


def _float_var(
    raw: bytes,
    integers: list[int],
    floats: list[float | FloatInterval],
    difficulty: int,
    rank: int,
    life: int,
    enemy: tuple[float | FloatInterval, float | FloatInterval],
    player: tuple[float, float] | None,
) -> float | FloatInterval:
    literal = struct.unpack("<f", raw)[0]
    value = int(literal) if math.isfinite(literal) else 0
    if -10008 <= value <= -10005:
        return floats[-10005 - value]
    if value == -10015:
        return enemy[0]
    if value == -10016:
        return enemy[1]
    if value == -10018:
        if player is None:
            raise UnsupportedBirthModel("ECL reads future player x")
        return player[0]
    if value == -10019:
        if player is None:
            raise UnsupportedBirthModel("ECL reads future player y")
        return player[1]
    if value == -10021:
        if player is None:
            return FloatInterval(-math.pi, math.pi)
        if any(isinstance(axis, FloatInterval) for axis in enemy):
            return FloatInterval(-math.pi, math.pi)
        return math.atan2(player[1] - enemy[1], player[0] - enemy[0])
    if value == -10023:
        if player is None:
            raise UnsupportedBirthModel("ECL reads future player distance")
        if any(isinstance(axis, FloatInterval) for axis in enemy):
            x_values = (
                (enemy[0].low, enemy[0].high)
                if isinstance(enemy[0], FloatInterval) else (enemy[0],)
            )
            y_values = (
                (enemy[1].low, enemy[1].high)
                if isinstance(enemy[1], FloatInterval) else (enemy[1],)
            )
            return FloatInterval(0.0, max(
                math.hypot(player[0] - x, player[1] - y)
                for x in x_values for y in y_values
            ))
        return math.hypot(player[0] - enemy[0], player[1] - enemy[1])
    if value in range(-10024, -10000):
        resolved = _int_var(value, integers, difficulty, rank, life)
        return struct.unpack("<f", struct.pack("<i", resolved))[0]
    return literal


def _set_int_var(identifier: int, value: int, integers: list[int]) -> bool:
    if -10004 <= identifier <= -10001:
        integers[-10001 - identifier] = value
        return True
    if -10012 <= identifier <= -10009:
        integers[4 + (-10009 - identifier)] = value
        return True
    return False


def _set_float_var(
    identifier: int,
    value: float | FloatInterval,
    floats: list[float | FloatInterval],
) -> bool:
    if -10008 <= identifier <= -10005:
        floats[-10005 - identifier] = value
        return True
    return False


def _set_source_value(
    identifier: int,
    integers: list[int],
    floats: list[float | FloatInterval],
    difficulty: int,
    rank: int,
    life: int,
    boss_timer: int,
    enemy: tuple[float | FloatInterval, float | FloatInterval],
    player: tuple[float, float] | None,
) -> tuple[int | float | FloatInterval, bool]:
    """Resolve the raw 32-bit RHS used by source ``SetVar``.

    The boolean distinguishes a resolved float variable from integer/literal
    bits.  SETINT and SETFLOAT both call this same source function; their names
    do not constrain either operand's type.
    """
    if -10008 <= identifier <= -10005:
        return floats[-10005 - identifier], True
    if -10004 <= identifier <= -10001 or -10012 <= identifier <= -10009:
        return _int_var(identifier, integers, difficulty, rank, life), False
    if identifier == -10013:
        return difficulty, False
    if identifier == -10014:
        return rank, False
    if identifier == -10015:
        return enemy[0], True
    if identifier == -10016:
        return enemy[1], True
    if identifier in (-10017, -10020):
        raise UnsupportedBirthModel("SET reads an uncaptured z coordinate")
    if identifier == -10018:
        if player is None:
            raise UnsupportedBirthModel("SET reads future player x")
        return player[0], True
    if identifier == -10019:
        if player is None:
            raise UnsupportedBirthModel("SET reads future player y")
        return player[1], True
    if identifier == -10021:
        if player is None:
            return FloatInterval(-math.pi, math.pi), True
        if any(isinstance(axis, FloatInterval) for axis in enemy):
            return FloatInterval(-math.pi, math.pi), True
        return math.atan2(player[1] - enemy[1], player[0] - enemy[0]), True
    if identifier == -10022:
        return boss_timer, False
    if identifier == -10023:
        if player is None:
            raise UnsupportedBirthModel("SET reads future player distance")
        if any(isinstance(axis, FloatInterval) for axis in enemy):
            x_values = (
                (enemy[0].low, enemy[0].high)
                if isinstance(enemy[0], FloatInterval) else (enemy[0],)
            )
            y_values = (
                (enemy[1].low, enemy[1].high)
                if isinstance(enemy[1], FloatInterval) else (enemy[1],)
            )
            return FloatInterval(0.0, max(
                math.hypot(player[0] - x, player[1] - y)
                for x in x_values for y in y_values
            )), True
        return math.hypot(player[0] - enemy[0], player[1] - enemy[1]), True
    if identifier == -10024:
        return life, False
    if identifier == -10025:
        raise UnsupportedBirthModel("SET reads the uncaptured player shot type")
    return identifier, False


def _set_local_from_source_bits(
    target: int,
    value: int | float | FloatInterval,
    source_is_float: bool,
    integers: list[int],
    floats: list[float | FloatInterval],
) -> bool:
    if -10008 <= target <= -10005:
        if source_is_float:
            return _set_float_var(target, value, floats)
        resolved = struct.unpack("<f", struct.pack("<i", value))[0]
        return _set_float_var(target, resolved, floats)
    if -10004 <= target <= -10001 or -10012 <= target <= -10009:
        if source_is_float:
            if isinstance(value, FloatInterval):
                raise UnsupportedBirthModel(
                    "SET cannot bit-copy an uncertain float into an integer"
                )
            value = struct.unpack("<i", struct.pack("<f", value))[0]
        return _set_int_var(target, value, integers)
    return False


def _resolved_pattern(
    instruction: EclInstruction,
    spawner: EnemySpawner,
    effect_floats: tuple[float, float, float, float],
    effect_ints: tuple[int, int, int, int],
    integers: list[int],
    floats: list[float | FloatInterval],
    difficulty: int,
    rank: int,
    life: int,
    enemy: tuple[float | FloatInterval, float | FloatInterval],
    player: tuple[float, float] | None,
    bullet_sizes: tuple[tuple[float, float], ...],
    radial_births: bool,
) -> BulletPattern:
    raw = _instruction_raw(instruction)
    sprite = struct.unpack_from("<h", raw, 0x0C)[0]
    if not 0 <= sprite < len(bullet_sizes):
        raise UnsupportedBirthModel(f"ECL bullet sprite {sprite} has no size")
    half_width, half_height = bullet_sizes[sprite]
    if half_width <= 0.0 or half_height <= 0.0:
        raise UnsupportedBirthModel(f"ECL bullet sprite {sprite} is not loaded")
    count1_raw, count2_raw = struct.unpack_from("<ii", raw, 0x10)
    count1 = max(1, _int_var(count1_raw, integers, difficulty, rank, life) + _rank_int(
        spawner.bullet_rank_amount1_low,
        spawner.bullet_rank_amount1_high,
        rank,
    ))
    count2 = max(1, _int_var(count2_raw, integers, difficulty, rank, life) + _rank_int(
        spawner.bullet_rank_amount2_low,
        spawner.bullet_rank_amount2_high,
        rank,
    ))
    if count1 * count2 > 640:
        raise UnsupportedBirthModel("ECL bullet pattern exceeds the native pool")
    speed_rank = _rank_float(
        spawner.bullet_rank_speed_low,
        spawner.bullet_rank_speed_high,
        rank,
    )
    speed1 = _float_var(
        raw[0x18:0x1C], integers, floats, difficulty, rank, life, enemy, player
    )
    speed2_value = _float_var(
        raw[0x1C:0x20], integers, floats, difficulty, rank, life, enemy, player
    )
    if isinstance(speed1, FloatInterval) or isinstance(speed2_value, FloatInterval):
        if not radial_births:
            raise UnsupportedBirthModel("uncertain bullet speed needs a hard envelope")
        speed1 = max(0.3, _maximum_magnitude(_float_add(speed1, speed_rank)))
        speed2 = max(
            0.3,
            _maximum_magnitude(_float_add(speed2_value, speed_rank / 2.0)),
        )
    else:
        if not math.isfinite(speed1) or not math.isfinite(speed2_value):
            raise UnsupportedBirthModel("non-finite bullet speed")
        if speed1 != 0.0:
            speed1 = max(0.3, speed1 + speed_rank)
        speed2 = max(0.3, speed2_value + speed_rank / 2.0)
    angle1 = _float_var(
        raw[0x20:0x24], integers, floats, difficulty, rank, life, enemy, player
    )
    angle2 = _float_var(
        raw[0x24:0x28], integers, floats, difficulty, rank, life, enemy, player
    )
    if isinstance(angle1, FloatInterval) or isinstance(angle2, FloatInterval):
        if not radial_births:
            raise UnsupportedBirthModel("uncertain bullet angle needs a hard envelope")
        angle1 = angle2 = 0.0
    elif not math.isfinite(angle1) or not math.isfinite(angle2):
        raise UnsupportedBirthModel("non-finite bullet angle")
    angle1 = math.remainder(angle1, math.tau)
    flags = struct.unpack_from("<I", raw, 0x28)[0]
    return BulletPattern(
        sprite,
        angle1,
        angle2,
        speed1,
        speed2,
        effect_floats,
        effect_ints,
        count1,
        count2,
        instruction.opcode - OPCODE_BULLET_FIRST,
        flags,
        half_width,
        half_height,
    )


def _forecast_ecl_births_single(
    spawner: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    difficulty: int,
    rank: int,
    bullet_sizes: tuple[tuple[float, float], ...],
    frame_multiplier: float = 1.0,
    rng: RngState | None = None,
    allow_player_variables: bool = True,
    radial_births: bool = False,
    abstract_rng: bool = False,
    enemy_kill_all_is_noop: bool = False,
    abstract_int_choices: tuple[int, ...] = (),
    model_player_damage: bool = True,
    allow_enemy_create_audit: bool = True,
    record_enemy_kill_all: bool = False,
    laser_world=None,
    spawn_inline: bool = False,
) -> EclForecast:
    """Forecast one emitter until the first unsupported source instruction."""
    horizon = len(player_positions)
    births: list[list[Bullet]] = [[] for _ in player_positions]
    body_hazards: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    effect_spawns: list[list[int]] = [[] for _ in player_positions]
    item_spawns = [0 for _ in player_positions]
    item_births: list[list[EclItemBirth]] = [
        [] for _ in player_positions
    ]
    enemy_kill_all = [False for _ in player_positions]
    if not spawner.ecl_program or spawner.next_instruction is None:
        return EclForecast(tuple(map(tuple, births)), 0, "missing ECL instruction graph")
    if spawner.repeat_ex_index is not None:
        return EclForecast(tuple(map(tuple, births)), 0, "unsupported repeating ECL callback")
    program = _compiled_program(spawner.ecl_program)
    instruction_address = spawner.next_instruction.address
    current_time = spawner.ecl_time
    time_subframe = spawner.ecl_time_float - spawner.ecl_time
    integers = list(spawner.ecl_ints)
    floats = list(spawner.ecl_floats)
    compare_register = spawner.ecl_compare
    call_stack = list(spawner.ecl_stack)
    interactable = spawner.interactable
    collidable = spawner.collidable
    invisible = spawner.invisible
    hitbox_half_width = spawner.hitbox_half_width
    hitbox_half_height = spawner.hitbox_half_height
    call_stack_disabled = spawner.call_stack_disabled
    life = spawner.life
    life_lower_bound = spawner.life
    damageable = spawner.damageable
    death_mode = spawner.death_mode
    is_boss = spawner.is_boss
    boss_id = spawner.boss_id
    interrupts = list(spawner.interrupts)
    run_interrupt = spawner.run_interrupt
    rank_speed_low = spawner.bullet_rank_speed_low
    rank_speed_high = spawner.bullet_rank_speed_high
    rank_amount1_low = spawner.bullet_rank_amount1_low
    rank_amount1_high = spawner.bullet_rank_amount1_high
    rank_amount2_low = spawner.bullet_rank_amount2_low
    rank_amount2_high = spawner.bullet_rank_amount2_high
    boss_timer = spawner.boss_timer
    boss_timer_subframe = spawner.boss_timer_float - spawner.boss_timer
    death_callback_sub = spawner.death_callback_sub
    life_callback_threshold = spawner.life_callback_threshold
    life_callback_sub = spawner.life_callback_sub
    timer_callback_threshold = spawner.timer_callback_threshold
    timer_callback_sub = spawner.timer_callback_sub
    pattern = spawner.pattern
    effect_floats = spawner.bullet_effect_floats
    effect_ints = spawner.bullet_effect_ints
    shooting_disabled = spawner.shooting_disabled
    interval = spawner.interval
    interval_timer = spawner.timer
    interval_timer_low = interval_timer
    interval_timer_high = interval_timer
    interval_subframe = spawner.timer_float - spawner.timer
    enemy_x = spawner.x
    enemy_y = spawner.y
    velocity_x = spawner.velocity_x
    velocity_y = spawner.velocity_y
    angle = spawner.angle
    angular_velocity = spawner.angular_velocity
    speed = spawner.speed
    acceleration = spawner.acceleration
    movement_mode = spawner.movement_mode
    movement_ease = spawner.movement_ease
    move_interp_x = spawner.move_interp_x
    move_interp_y = spawner.move_interp_y
    move_start_x = spawner.move_start_x
    move_start_y = spawner.move_start_y
    move_timer = spawner.move_timer
    move_timer_float = spawner.move_timer_float
    move_start_time = spawner.move_start_time
    lower_move_x = spawner.lower_move_x
    lower_move_y = spawner.lower_move_y
    upper_move_x = spawner.upper_move_x
    upper_move_y = spawner.upper_move_y
    should_clamp_position = spawner.should_clamp_position
    if not all(
        math.isfinite(value) and value >= 0.0
        for value in (
            spawner.forecast_position_uncertainty_x,
            spawner.forecast_position_uncertainty_y,
        )
    ):
        return EclForecast(
            tuple(map(tuple, births)), 0,
            "invalid forecast position uncertainty",
        )
    position_uncertainty_x = spawner.forecast_position_uncertainty_x
    position_uncertainty_y = spawner.forecast_position_uncertainty_y
    position_uncertainty = 0.0
    velocity_uncertainty = 0.0
    uncertain_heading = False
    timed_move_radius = 0.0
    timed_move_progress = 0.0
    timed_move_next_progress = 0.0
    shoot_offset_x: float | FloatInterval = spawner.shoot_offset_x
    shoot_offset_y: float | FloatInterval = spawner.shoot_offset_y
    abstract_int_cursor = 0
    created_emitters: list[EnemySpawner] = []
    death_anm1 = spawner.death_anm1
    death_anm2 = spawner.death_anm2
    death_anm3 = spawner.death_anm3
    has_been_in_bounds = spawner.has_been_in_bounds
    laser_slots = list(spawner.laser_slots)
    laser_store = spawner.laser_store
    if (
        len(laser_slots) != 32
        or any(not -1 <= slot < 64 for slot in laser_slots)
        or not 0 <= laser_store < 32
    ):
        return EclForecast(
            tuple(map(tuple, births)), 0, "invalid source laser pointer state"
        )

    def emit(
        resolved: BulletPattern,
        origin: tuple[float | FloatInterval, float | FloatInterval],
        player: tuple[float, float],
    ) -> tuple[Bullet, ...]:
        origin_x, origin_y = origin
        origin_uncertainty_x = 0.0
        origin_uncertainty_y = 0.0
        if isinstance(origin_x, FloatInterval):
            if not radial_births:
                raise UnsupportedBirthModel("uncertain shoot-offset x needs a hard envelope")
            origin_uncertainty_x = (origin_x.high - origin_x.low) / 2.0
            origin_x = (origin_x.low + origin_x.high) / 2.0
        if isinstance(origin_y, FloatInterval):
            if not radial_births:
                raise UnsupportedBirthModel("uncertain shoot-offset y needs a hard envelope")
            origin_uncertainty_y = (origin_y.high - origin_y.low) / 2.0
            origin_y = (origin_y.low + origin_y.high) / 2.0
        if radial_births:
            return tuple(
                replace(
                    bullet,
                    half_width=(
                        bullet.half_width
                        + origin_uncertainty_x
                    ),
                    half_height=(
                        bullet.half_height
                        + origin_uncertainty_y
                    ),
                )
                for bullet in spawn_pattern_envelope(
                    resolved,
                    (origin_x, origin_y),
                )
            )
        return spawn_pattern(resolved, (origin_x, origin_y), player, rng)

    def uncertain_enemy() -> tuple[
        float | FloatInterval,
        float | FloatInterval,
    ]:
        uncertainty_x = position_uncertainty_x + position_uncertainty
        uncertainty_y = position_uncertainty_y + position_uncertainty
        return (
            (
                FloatInterval(enemy_x - uncertainty_x, enemy_x + uncertainty_x)
                if uncertainty_x > 0.0 else enemy_x
            ),
            (
                FloatInterval(enemy_y - uncertainty_y, enemy_y + uncertainty_y)
                if uncertainty_y > 0.0 else enemy_y
            ),
        )

    def clamp_position(x: float, y: float) -> tuple[float, float]:
        if not should_clamp_position:
            return x, y
        return (
            min(max(x, lower_move_x), upper_move_x),
            min(max(y, lower_move_y), upper_move_y),
        )

    for frame_index, player in enumerate(player_positions):
        variable_player = player if allow_player_variables else None
        if not spawn_inline:
            if velocity_uncertainty > 0.0:
                position_uncertainty += velocity_uncertainty
                if timed_move_radius > 0.0 and movement_mode == 2:
                    timed_move_progress = timed_move_next_progress
            else:
                enemy_x += -velocity_x if spawner.invert_x else velocity_x
                enemy_y += velocity_y
                enemy_x, enemy_y = clamp_position(enemy_x, enemy_y)
        center_in_bounds = (
            0.0 <= enemy_x <= 384.0 and 0.0 <= enemy_y <= 448.0
        )
        if not spawn_inline and not has_been_in_bounds and center_in_bounds:
            # Any source sprite rectangle contains its center, so this proves
            # IsInBounds even before its ANM-derived extent is available.
            has_been_in_bounds = True
        if (
            not spawn_inline
            and spawner.sprite_half_width > 0.0
            and spawner.sprite_half_height > 0.0
        ):
            sprite_in_bounds = not (
                enemy_x + spawner.sprite_half_width < 0.0
                or enemy_x - spawner.sprite_half_width > 384.0
                or enemy_y + spawner.sprite_half_height < 0.0
                or enemy_y - spawner.sprite_half_height > 448.0
            )
            if not has_been_in_bounds and sprite_in_bounds:
                has_been_in_bounds = True
            if has_been_in_bounds and not sprite_in_bounds:
                return EclForecast(
                    tuple(map(tuple, births)),
                    horizon,
                    "source sprite-bound retirement",
                    body_hazards=tuple(tuple(frame) for frame in body_hazards),
                    finished=True,
                    effect_spawns=tuple(
                        tuple(frame) for frame in effect_spawns
                    ),
                    item_spawns=tuple(item_spawns),
                    item_births=tuple(tuple(frame) for frame in item_births),
                    enemy_kill_all=tuple(enemy_kill_all),
                )
        if not spawn_inline and interactable and not invisible and life <= 0:
            # EnemyManager handles an exact non-positive life value after
            # RunEcl in the preceding update.  Unlike life_lower_bound, this
            # is not a possible player-damage time: ENEMYLIFESET can make the
            # transition certain.  Apply the source death mode now, before
            # the callback ECL runs on this update.
            life_callback_threshold = -1
            timer_callback_threshold = -1
            if death_mode == 0:
                return EclForecast(
                    tuple(map(tuple, births)),
                    horizon,
                    "source death mode zero despawns emitter",
                    body_hazards=tuple(tuple(frame) for frame in body_hazards),
                    finished=True,
                )
            if death_mode == 1:
                interactable = False
            elif death_mode == 3:
                life = 1
                damageable = False
                death_mode = 0

            if death_callback_sub >= 0:
                if not 0 <= death_callback_sub < len(spawner.ecl_subroutines):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"death callback subroutine {death_callback_sub} is unavailable",
                    )
                callback_address = spawner.ecl_subroutines[death_callback_sub]
                if callback_address not in program:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "death callback instruction graph is not captured",
                    )
                instruction_address = callback_address
                current_time = 0
                time_subframe = 0.0
                call_stack.clear()
                rank_speed_low = -0.5
                rank_speed_high = 0.5
                rank_amount1_low = rank_amount1_high = 0
                rank_amount2_low = rank_amount2_high = 0
                death_callback_sub = -1
            life_lower_bound = life
        if (
            not spawn_inline
            and life_callback_threshold >= 0
            and life_lower_bound < life_callback_threshold
        ):
            return EclForecast(
                tuple(map(tuple, births)),
                frame_index,
                "player damage can reach an active life callback",
            )
        if (
            not spawn_inline
            and death_callback_sub >= 0
            and life_lower_bound <= 0
        ):
            return EclForecast(
                tuple(map(tuple, births)),
                frame_index,
                "player damage can reach an active death callback",
            )
        if (
            not spawn_inline
            and timer_callback_threshold >= 0
            and boss_timer >= timer_callback_threshold
        ):
            if not 0 <= timer_callback_sub < len(spawner.ecl_subroutines):
                return EclForecast(
                    tuple(map(tuple, births)),
                    frame_index,
                    f"timer callback subroutine {timer_callback_sub} is unavailable",
                )
            callback_address = spawner.ecl_subroutines[timer_callback_sub]
            if callback_address not in program:
                return EclForecast(
                    tuple(map(tuple, births)),
                    frame_index,
                    "timer callback instruction graph is not captured",
                )
            instruction_address = callback_address
            current_time = 0
            time_subframe = 0.0
            timer_callback_threshold = -1
            timer_callback_sub = death_callback_sub
            boss_timer = 0
            boss_timer_subframe = 0.0
            rank_speed_low = -0.5
            rank_speed_high = 0.5
            rank_amount1_low = rank_amount1_high = 0
            rank_amount2_low = rank_amount2_high = 0
            call_stack.clear()
        enemy = uncertain_enemy()
        stop_after_frame = ""
        for _instruction_count in range(256):
            instruction = program.get(instruction_address)
            if instruction is None:
                return EclForecast(
                    tuple(map(tuple, births)), frame_index, "incomplete ECL instruction graph"
                )
            if run_interrupt >= 0:
                if not 0 <= run_interrupt < len(interrupts):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"invalid ECL interrupt id {run_interrupt}",
                    )
                sub_id = interrupts[run_interrupt]
                if not 0 <= sub_id < len(spawner.ecl_subroutines):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"ECL interrupt {run_interrupt} has no captured subroutine",
                    )
                if call_stack_disabled:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "ECL interrupt reaches a disabled call stack",
                    )
                if len(call_stack) > 7:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "invalid ECL interrupt stack depth",
                    )
                if len(call_stack) < 7:
                    call_stack.append(EnemyEclContext(
                        instruction.address + instruction.offset_to_next,
                        current_time,
                        current_time + time_subframe,
                        tuple(integers),
                        tuple(floats),
                        compare_register,
                        None,
                    ))
                instruction_address = spawner.ecl_subroutines[sub_id]
                current_time = 0
                time_subframe = 0.0
                run_interrupt = -1
                continue
            if instruction.time != current_time:
                break
            execute = bool(instruction.skip_for_difficulty & (1 << difficulty))
            next_address = instruction.address + instruction.offset_to_next
            raw = _instruction_raw(instruction)
            if not execute or instruction.opcode == OPCODE_NOP:
                instruction_address = next_address
                continue
            if instruction.opcode == 1:
                # RunEcl returns ZUN_ERROR; EnemyManager immediately despawns
                # this emitter before body collision or its periodic shot.
                return EclForecast(
                    tuple(map(tuple, births)),
                    horizon,
                    "source UNIMP despawns emitter",
                    body_hazards=tuple(tuple(frame) for frame in body_hazards),
                    finished=True,
                )
            if instruction.opcode in (OPCODE_JUMP, OPCODE_JUMPDEC):
                jump_time, jump_offset = struct.unpack_from("<ii", raw, 0x0C)
                take_jump = True
                if instruction.opcode == OPCODE_JUMPDEC:
                    variable = struct.unpack_from("<i", raw, 0x14)[0]
                    value = _int_var(variable, integers, difficulty, rank, life) - 1
                    if not _set_int_var(variable, value, integers):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "unsupported JUMPDEC variable"
                        )
                    take_jump = value > 0
                if take_jump:
                    current_time = jump_time
                    instruction_address = instruction.address + jump_offset
                else:
                    instruction_address = next_address
                continue
            if instruction.opcode in (OPCODE_SET_INT, OPCODE_SET_FLOAT):
                result, argument = struct.unpack_from("<ii", raw, 0x0C)
                try:
                    value, source_is_float = _set_source_value(
                        argument,
                        integers,
                        floats,
                        difficulty,
                        rank,
                        life,
                        boss_timer,
                        enemy,
                        variable_player,
                    )
                    assigned = _set_local_from_source_bits(
                        result,
                        value,
                        source_is_float,
                        integers,
                        floats,
                    )
                except UnsupportedBirthModel as error:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, str(error)
                    )
                if not assigned:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index,
                        "SET writes an unsupported world variable",
                    )
            elif instruction.opcode in (OPCODE_SET_SELF_X, OPCODE_SET_SELF_Y):
                target = struct.unpack_from("<i", raw, 0x0C)[0]
                value = enemy_x if instruction.opcode == OPCODE_SET_SELF_X else enemy_y
                if not _set_float_var(target, value, floats):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "unsupported SETVARSELF target"
                    )
            elif instruction.opcode == OPCODE_SET_SELF_Z:
                return EclForecast(
                    tuple(map(tuple, births)),
                    frame_index,
                    "SETVARSELFZ needs the uncaptured source z coordinate",
                )
            elif instruction.opcode in (
                OPCODE_SET_INT_RANDOM,
                OPCODE_SET_INT_RANDOM_MIN,
                OPCODE_SET_FLOAT_RANDOM,
                OPCODE_SET_FLOAT_RANDOM_MIN,
            ):
                result = struct.unpack_from("<i", raw, 0x0C)[0]
                if instruction.opcode in (
                    OPCODE_SET_INT_RANDOM,
                    OPCODE_SET_INT_RANDOM_MIN,
                ):
                    extent_raw = struct.unpack_from("<i", raw, 0x10)[0]
                    extent = _int_var(
                        extent_raw, integers, difficulty, rank, life
                    ) & 0xFFFFFFFF
                    if rng is None:
                        if not abstract_rng:
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "integer ECL random variable requires RNG state",
                            )
                        if extent == 0:
                            value = 0
                        elif abstract_int_cursor >= len(abstract_int_choices):
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "integer RNG needs bounded branch expansion",
                                unresolved_int_extent=extent,
                            )
                        else:
                            value = abstract_int_choices[abstract_int_cursor]
                            abstract_int_cursor += 1
                            if not 0 <= value < extent:
                                return EclForecast(
                                    tuple(map(tuple, births)),
                                    frame_index,
                                    "integer RNG branch is outside its source range",
                                )
                    else:
                        value = rng.u32_in_range(extent)
                    if instruction.opcode == OPCODE_SET_INT_RANDOM_MIN:
                        minimum_raw = struct.unpack_from("<i", raw, 0x14)[0]
                        value += _int_var(
                            minimum_raw, integers, difficulty, rank, life
                        )
                    if not _set_int_var(result, value, integers):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "unsupported random-int target"
                        )
                else:
                    extent = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    if rng is None:
                        if not abstract_rng:
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "ECL random variable requires RNG state",
                            )
                        if isinstance(extent, FloatInterval):
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "nested random-float interval is unsupported",
                            )
                        value: float | FloatInterval = FloatInterval(
                            min(0.0, extent), max(0.0, extent)
                        )
                        if instruction.opcode == OPCODE_SET_FLOAT_RANDOM_MIN:
                            value = _float_add(value, _float_var(
                                raw[0x14:0x18], integers, floats, difficulty, rank,
                                life, enemy, variable_player,
                            ))
                    else:
                        value = rng.f32_in_range(extent)
                        if instruction.opcode == OPCODE_SET_FLOAT_RANDOM_MIN:
                            value += _float_var(
                                raw[0x14:0x18], integers, floats, difficulty, rank,
                                life, enemy, variable_player,
                            )
                    if not _set_float_var(result, value, floats):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "unsupported random-float target"
                        )
            elif instruction.opcode in (
                OPCODE_MATH_INT_ADD,
                OPCODE_MATH_INT_SUBTRACT,
                OPCODE_MATH_INT_MULTIPLY,
                OPCODE_MATH_INT_DIVIDE,
                OPCODE_MATH_INT_MODULO,
            ):
                target, lhs_raw, rhs_raw = struct.unpack_from("<iii", raw, 0x0C)
                lhs = _int_var(lhs_raw, integers, difficulty, rank, life)
                rhs = _int_var(rhs_raw, integers, difficulty, rank, life)
                if instruction.opcode == OPCODE_MATH_INT_ADD:
                    value = lhs + rhs
                elif instruction.opcode == OPCODE_MATH_INT_SUBTRACT:
                    value = lhs - rhs
                elif instruction.opcode == OPCODE_MATH_INT_MULTIPLY:
                    value = lhs * rhs
                elif rhs == 0:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "ECL integer division by zero"
                    )
                elif instruction.opcode == OPCODE_MATH_INT_DIVIDE:
                    value = _trunc_div(lhs, rhs)
                else:
                    value = lhs - _trunc_div(lhs, rhs) * rhs
                if not _set_int_var(target, value, integers):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "unsupported integer-math target"
                    )
            elif instruction.opcode in (
                OPCODE_MATH_INCREMENT,
                OPCODE_MATH_DECREMENT,
            ):
                target = struct.unpack_from("<i", raw, 0x0C)[0]
                value = _int_var(
                    target, integers, difficulty, rank, life
                )
                value += 1 if instruction.opcode == OPCODE_MATH_INCREMENT else -1
                if not _set_int_var(target, value, integers):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "unsupported increment target"
                    )
            elif OPCODE_MATH_FLOAT_ADD <= instruction.opcode <= OPCODE_MATH_ATAN2:
                target = struct.unpack_from("<i", raw, 0x0C)[0]
                lhs = _float_var(
                    raw[0x10:0x14], integers, floats, difficulty, rank,
                    life, enemy, variable_player,
                )
                rhs = _float_var(
                    raw[0x14:0x18], integers, floats, difficulty, rank,
                    life, enemy, variable_player,
                )
                if instruction.opcode == OPCODE_MATH_FLOAT_ADD:
                    value = _float_add(lhs, rhs)
                elif instruction.opcode == OPCODE_MATH_FLOAT_SUBTRACT:
                    value = _float_subtract(lhs, rhs)
                elif instruction.opcode == OPCODE_MATH_FLOAT_MULTIPLY:
                    value = _float_multiply(lhs, rhs)
                elif instruction.opcode == OPCODE_MATH_FLOAT_DIVIDE:
                    try:
                        value = _float_divide(lhs, rhs)
                    except UnsupportedBirthModel as error:
                        return EclForecast(tuple(map(tuple, births)), frame_index, str(error))
                elif instruction.opcode == OPCODE_MATH_FLOAT_MODULO:
                    if isinstance(lhs, FloatInterval) or isinstance(rhs, FloatInterval):
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "float interval reaches ECL modulo",
                        )
                    if rhs == 0.0:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "ECL float modulo by zero"
                        )
                    value = math.fmod(lhs, rhs)
                else:
                    third = _float_var(
                        raw[0x18:0x1C], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    fourth = _float_var(
                        raw[0x1C:0x20], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    if any(isinstance(item, FloatInterval) for item in (
                        lhs, rhs, third, fourth
                    )):
                        if not radial_births:
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "float interval reaches ECL atan2",
                            )
                        value = FloatInterval(-math.pi, math.pi)
                    else:
                        value = math.atan2(fourth - rhs, third - lhs)
                if not _set_float_var(target, value, floats):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "unsupported float-math target"
                    )
            elif instruction.opcode == OPCODE_MATH_NORMALIZE_ANGLE:
                target = struct.unpack_from("<i", raw, 0x0C)[0]
                value = _float_var(
                    raw[0x0C:0x10], integers, floats, difficulty, rank,
                    life, enemy, variable_player,
                )
                value = (
                    FloatInterval(-math.pi, math.pi)
                    if isinstance(value, FloatInterval)
                    else math.remainder(value, math.tau)
                )
                if not _set_float_var(target, value, floats):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "unsupported angle target"
                    )
            elif instruction.opcode in (OPCODE_COMPARE_INT, OPCODE_COMPARE_FLOAT):
                if instruction.opcode == OPCODE_COMPARE_INT:
                    lhs_raw, rhs_raw = struct.unpack_from("<ii", raw, 0x0C)
                    lhs = _int_var(lhs_raw, integers, difficulty, rank, life)
                    rhs = _int_var(rhs_raw, integers, difficulty, rank, life)
                else:
                    lhs = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    rhs = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    if isinstance(lhs, FloatInterval) or isinstance(rhs, FloatInterval):
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "float interval reaches ECL comparison",
                        )
                    if not math.isfinite(lhs) or not math.isfinite(rhs):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "non-finite ECL comparison"
                        )
                compare_register = 0 if lhs == rhs else -1 if lhs < rhs else 1
            elif OPCODE_JUMP_LESS <= instruction.opcode <= OPCODE_JUMP_NOT_EQUAL:
                take_jump = (
                    compare_register < 0 if instruction.opcode == OPCODE_JUMP_LESS
                    else compare_register <= 0 if instruction.opcode == OPCODE_JUMP_LESS_EQUAL
                    else compare_register == 0 if instruction.opcode == OPCODE_JUMP_EQUAL
                    else compare_register > 0 if instruction.opcode == OPCODE_JUMP_GREATER
                    else compare_register >= 0 if instruction.opcode == OPCODE_JUMP_GREATER_EQUAL
                    else compare_register != 0
                )
                if take_jump:
                    jump_time, jump_offset = struct.unpack_from("<ii", raw, 0x0C)
                    current_time = jump_time
                    instruction_address = instruction.address + jump_offset
                    continue
            elif instruction.opcode == OPCODE_RETURN:
                if not call_stack:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "ECL return has no captured caller context",
                    )
                caller = call_stack.pop()
                if caller.repeat_ex_index is not None:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "ECL caller has a repeating callback",
                    )
                instruction_address = caller.instruction_address
                current_time = caller.time
                time_subframe = caller.time_float - caller.time
                integers = list(caller.ints)
                floats = list(caller.floats)
                compare_register = caller.compare
                continue
            elif OPCODE_CALL <= instruction.opcode <= OPCODE_CALL_NOT_EQUAL:
                take_call = instruction.opcode == OPCODE_CALL
                if instruction.opcode >= OPCODE_CALL_LESS:
                    lhs_raw, rhs = struct.unpack_from("<ii", raw, 0x18)
                    lhs = _int_var(
                        lhs_raw, integers, difficulty, rank, life
                    )
                    take_call = (
                        lhs < rhs
                        if instruction.opcode == OPCODE_CALL_LESS
                        else lhs <= rhs
                        if instruction.opcode == OPCODE_CALL_LESS_EQUAL
                        else lhs == rhs
                        if instruction.opcode == OPCODE_CALL_EQUAL
                        else lhs > rhs
                        if instruction.opcode == OPCODE_CALL_GREATER
                        else lhs >= rhs
                        if instruction.opcode == OPCODE_CALL_GREATER_EQUAL
                        else lhs != rhs
                    )
                if take_call:
                    if call_stack_disabled:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "ECL call stack is disabled",
                        )
                    if len(call_stack) > 7:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "invalid ECL call stack depth",
                        )
                    sub_id, var0 = struct.unpack_from("<ii", raw, 0x0C)
                    if not 0 <= sub_id < len(spawner.ecl_subroutines):
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            f"ECL call subroutine {sub_id} is unavailable",
                        )
                    # RunEcl still enters the callee at stackDepth == 7, but
                    # does not increment the depth.  Its write to saved slot 7
                    # is therefore not the context restored by RET: RET first
                    # decrements to 6 and restores saved slot 6.  Keeping the
                    # existing seven captured contexts models that source
                    # behavior exactly.
                    if len(call_stack) < 7:
                        call_stack.append(EnemyEclContext(
                            next_address,
                            current_time,
                            current_time + time_subframe,
                            tuple(integers),
                            tuple(floats),
                            compare_register,
                            None,
                        ))
                    integers[0] = var0
                    floats[0] = struct.unpack_from("<f", raw, 0x14)[0]
                    instruction_address = spawner.ecl_subroutines[sub_id]
                    current_time = 0
                    time_subframe = 0.0
                    continue
            elif instruction.opcode == OPCODE_COLLIDABLE_FLAG:
                collidable = bool(struct.unpack_from("<i", raw, 0x0C)[0])
            elif instruction.opcode == OPCODE_DAMAGEABLE_FLAG:
                damageable = bool(struct.unpack_from("<i", raw, 0x0C)[0])
            elif instruction.opcode == OPCODE_DEATH_FLAG:
                death_mode = struct.unpack_from("<i", raw, 0x0C)[0] & 0x07
            elif instruction.opcode == OPCODE_INTERACTABLE_FLAG:
                interactable = bool(struct.unpack_from("<i", raw, 0x0C)[0])
            elif instruction.opcode == OPCODE_INVISIBLE_FLAG:
                invisible = bool(struct.unpack_from("<i", raw, 0x0C)[0])
            elif instruction.opcode == OPCODE_HITBOX_SET:
                hitbox_x, hitbox_y = struct.unpack_from("<ff", raw, 0x0C)
                if not all(
                    math.isfinite(value) and 0.0 <= value <= 1024.0
                    for value in (hitbox_x, hitbox_y)
                ):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "invalid ECL hitbox"
                    )
                # EnemyManager uses hitboxDimensions / 1.5 as a full player
                # collision size, so the half-extent is source value / 3.
                hitbox_half_width = hitbox_x / 3.0
                hitbox_half_height = hitbox_y / 3.0
            elif OPCODE_MOVE_POSITION <= instruction.opcode <= OPCODE_MOVE_AT_PLAYER:
                timed_move_radius = 0.0
                timed_move_progress = 0.0
                timed_move_next_progress = 0.0
                if instruction.opcode == OPCODE_MOVE_POSITION:
                    next_x = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                            life, enemy, variable_player,
                    )
                    next_y = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    enemy_x, position_uncertainty_x = _interval_center(next_x)
                    enemy_y, position_uncertainty_y = _interval_center(next_y)
                    position_uncertainty = 0.0
                    enemy_x, enemy_y = clamp_position(enemy_x, enemy_y)
                    enemy = uncertain_enemy()
                elif instruction.opcode == OPCODE_MOVE_POSITION + 1:
                    velocity_x = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    velocity_y = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                            life, enemy, variable_player,
                    )
                    movement_mode = 0
                    uncertain_heading = False
                    velocity_uncertainty = 0.0
                elif instruction.opcode == OPCODE_MOVE_POSITION + 2:
                    angle = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    speed = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    movement_mode = 1
                    if isinstance(angle, FloatInterval) and radial_births:
                        angle = 0.0
                        uncertain_heading = True
                    else:
                        uncertain_heading = False
                        velocity_uncertainty = 0.0
                elif instruction.opcode == OPCODE_MOVE_POSITION + 3:
                    angular_velocity = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    movement_mode = 1
                elif instruction.opcode == OPCODE_MOVE_POSITION + 4:
                    speed = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    movement_mode = 1
                elif instruction.opcode == OPCODE_MOVE_POSITION + 5:
                    acceleration = _float_var(
                        raw[0x0C:0x10], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    movement_mode = 1
                elif instruction.opcode in (OPCODE_MOVE_RANDOM, OPCODE_MOVE_RANDOM_IN_BOUNDS):
                    if rng is None:
                        if not abstract_rng:
                            return EclForecast(
                                tuple(map(tuple, births)), frame_index, "random movement requires RNG state"
                            )
                        angle = 0.0
                        uncertain_heading = True
                    else:
                        low, high = struct.unpack_from("<ff", raw, 0x0C)
                        angle = rng.f32_in_range(high - low) + low
                        uncertain_heading = False
                        velocity_uncertainty = 0.0
                    if instruction.opcode == OPCODE_MOVE_RANDOM_IN_BOUNDS:
                        if not should_clamp_position:
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "MOVERANDINBOUND has no active source bounds",
                            )
                        if not uncertain_heading:
                            if enemy_x < lower_move_x + 96.0 and (
                                angle > math.pi / 2.0 or angle < -math.pi / 2.0
                            ):
                                angle = (
                                    math.pi - angle
                                    if angle > math.pi / 2.0
                                    else -math.pi - angle
                                )
                            if enemy_x > upper_move_x - 96.0 and (
                                0.0 <= angle < math.pi / 2.0
                                or -math.pi / 2.0 < angle <= 0.0
                            ):
                                angle = (
                                    math.pi - angle if angle >= 0.0 else -math.pi - angle
                                )
                            if enemy_y < lower_move_y + 48.0 and angle < 0.0:
                                angle = -angle
                            if enemy_y > upper_move_y - 48.0 and angle > 0.0:
                                angle = -angle
                else:
                    if not allow_player_variables:
                        angle = 0.0
                        uncertain_heading = True
                    else:
                        angle_offset = struct.unpack_from("<f", raw, 0x0C)[0]
                        angle = math.atan2(
                            player[1] - enemy_y, player[0] - enemy_x
                        ) + angle_offset
                        uncertain_heading = False
                        velocity_uncertainty = 0.0
                    speed = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    movement_mode = 1
                if isinstance(enemy_x, FloatInterval) or isinstance(enemy_y, FloatInterval):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "position interval needs clamp bounds"
                    )
                if isinstance(velocity_x, FloatInterval) or isinstance(velocity_y, FloatInterval):
                    if not radial_births:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "uncertain axis velocity"
                        )
                    velocity_uncertainty = math.hypot(
                        _maximum_magnitude(velocity_x),
                        _maximum_magnitude(velocity_y),
                    )
                    velocity_x = velocity_y = 0.0
                    uncertain_heading = True
                if isinstance(angle, FloatInterval):
                    if not radial_births:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "uncertain movement angle"
                        )
                    angle = 0.0
                    uncertain_heading = True
                if isinstance(speed, FloatInterval):
                    if not radial_births:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "uncertain movement speed"
                        )
                    speed = _maximum_magnitude(speed)
                    uncertain_heading = True
                if isinstance(angular_velocity, FloatInterval):
                    if not radial_births:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "uncertain angular velocity"
                        )
                    angular_velocity = _maximum_magnitude(angular_velocity)
                    uncertain_heading = True
                if isinstance(acceleration, FloatInterval):
                    if not radial_births:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, "uncertain acceleration"
                        )
                    acceleration = _maximum_magnitude(acceleration)
                    uncertain_heading = True
                if not all(math.isfinite(value) for value in (
                    enemy_x,
                    enemy_y,
                    velocity_x,
                    velocity_y,
                    angle,
                    angular_velocity,
                    speed,
                    acceleration,
                )):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "future player dependency reaches emitter motion",
                    )
            elif OPCODE_MOVE_DIR_TIME_FIRST <= instruction.opcode <= OPCODE_MOVE_TIME_LAST:
                duration = struct.unpack_from("<i", raw, 0x0C)[0]
                if duration <= 0:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "timed ECL movement has a non-positive duration",
                    )
                timed_move_radius = 0.0
                timed_move_progress = 0.0
                timed_move_next_progress = 0.0
                timed_heading_uncertain = False
                if instruction.opcode < OPCODE_MOVE_POSITION_TIME_FIRST:
                    timed_angle = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    timed_speed = struct.unpack_from("<f", raw, 0x14)[0]
                    if isinstance(timed_angle, FloatInterval):
                        if not radial_births:
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "uncertain timed-movement angle",
                            )
                        timed_move_radius = abs(timed_speed) * duration / 2.0
                        move_interp_x = move_interp_y = 0.0
                        timed_heading_uncertain = True
                    else:
                        move_interp_x = math.cos(timed_angle) * timed_speed * duration / 2.0
                        move_interp_y = math.sin(timed_angle) * timed_speed * duration / 2.0
                    movement_ease = instruction.opcode - OPCODE_MOVE_DIR_TIME_FIRST + 1
                elif instruction.opcode < OPCODE_MOVE_TIME_FIRST:
                    target_x = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    target_y = _float_var(
                        raw[0x14:0x18], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    if isinstance(target_x, FloatInterval) or isinstance(
                        target_y, FloatInterval
                    ):
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "uncertain timed-movement target",
                        )
                    move_interp_x = target_x - enemy_x
                    move_interp_y = target_y - enemy_y
                    movement_ease = instruction.opcode - OPCODE_MOVE_POSITION_TIME_FIRST
                    velocity_x = velocity_y = 0.0
                else:
                    if uncertain_heading:
                        if not radial_births:
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index,
                                "uncertain heading reaches timed movement",
                            )
                        timed_move_radius = abs(speed) * duration / 2.0
                        move_interp_x = move_interp_y = 0.0
                        timed_heading_uncertain = True
                    else:
                        move_interp_x = math.cos(angle) * speed * duration / 2.0
                        move_interp_y = math.sin(angle) * speed * duration / 2.0
                    movement_ease = instruction.opcode - OPCODE_MOVE_TIME_FIRST + 1
                move_start_x = enemy_x
                move_start_y = enemy_y
                move_start_time = duration
                move_timer = duration
                move_timer_float = float(duration)
                movement_mode = 2
                uncertain_heading = timed_heading_uncertain
                velocity_uncertainty = 0.0
            elif instruction.opcode == OPCODE_MOVE_BOUNDS_SET:
                (
                    lower_move_x,
                    lower_move_y,
                    upper_move_x,
                    upper_move_y,
                ) = struct.unpack_from("<ffff", raw, 0x0C)
                if (
                    not all(math.isfinite(value) for value in (
                        lower_move_x, lower_move_y, upper_move_x, upper_move_y
                    ))
                    or lower_move_x > upper_move_x
                    or lower_move_y > upper_move_y
                ):
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "invalid ECL move bounds"
                    )
                should_clamp_position = True
                enemy_x, enemy_y = clamp_position(enemy_x, enemy_y)
                enemy = uncertain_enemy()
            elif instruction.opcode == OPCODE_MOVE_BOUNDS_DISABLE:
                should_clamp_position = False
            elif OPCODE_BULLET_FIRST <= instruction.opcode <= OPCODE_BULLET_LAST:
                try:
                    pattern = _resolved_pattern(
                        instruction,
                        _copy_spawner(
                            spawner,
                            bullet_rank_speed_low=rank_speed_low,
                            bullet_rank_speed_high=rank_speed_high,
                            bullet_rank_amount1_low=rank_amount1_low,
                            bullet_rank_amount1_high=rank_amount1_high,
                            bullet_rank_amount2_low=rank_amount2_low,
                            bullet_rank_amount2_high=rank_amount2_high,
                        ),
                        effect_floats,
                        effect_ints,
                        integers,
                        floats,
                        difficulty,
                        rank,
                        life,
                        enemy,
                        variable_player,
                        bullet_sizes,
                        radial_births,
                    )
                    if not shooting_disabled:
                        births[frame_index].extend(emit(
                            pattern,
                            (
                                _float_add(enemy[0], shoot_offset_x),
                                _float_add(enemy[1], shoot_offset_y),
                            ),
                            player,
                        ))
                except UnsupportedBirthModel as error:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, str(error)
                    )
            elif instruction.opcode == OPCODE_SHOOT_INTERVAL:
                base_interval = struct.unpack_from("<i", raw, 0x0C)[0]
                low = _trunc_div(base_interval, 5)
                high = _trunc_div(-base_interval, 5)
                interval = base_interval + _rank_int(low, high, rank)
                interval_timer = 0
                interval_timer_low = 0
                interval_timer_high = 0
                interval_subframe = 0.0
            elif instruction.opcode == OPCODE_SHOOT_INTERVAL_DELAYED:
                base_interval = struct.unpack_from("<i", raw, 0x0C)[0]
                low = _trunc_div(base_interval, 5)
                high = _trunc_div(-base_interval, 5)
                interval = base_interval + _rank_int(low, high, rank)
                if rng is not None:
                    interval_timer = rng.u32_in_range(interval & 0xFFFFFFFF)
                    interval_timer_low = interval_timer
                    interval_timer_high = interval_timer
                elif abstract_rng and interval > 0:
                    # Randomness selects only the phase of this known periodic
                    # source. Keep every possible phase instead of sampling.
                    interval_timer = 0
                    interval_timer_low = 0
                    interval_timer_high = interval - 1
                elif not abstract_rng:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "delayed interval requires RNG state",
                    )
                interval_subframe = 0.0
            elif instruction.opcode == OPCODE_SHOOT_DISABLED:
                shooting_disabled = True
            elif instruction.opcode == OPCODE_SHOOT_ENABLED:
                shooting_disabled = False
            elif instruction.opcode == OPCODE_SHOOT_NOW:
                if pattern is None:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "SHOOTNOW has no resolved pattern"
                    )
                try:
                    births[frame_index].extend(emit(
                        pattern,
                        (
                            _float_add(enemy[0], shoot_offset_x),
                            _float_add(enemy[1], shoot_offset_y),
                        ),
                        player,
                    ))
                except UnsupportedBirthModel as error:
                    return EclForecast(tuple(map(tuple, births)), frame_index, str(error))
            elif instruction.opcode == OPCODE_SHOOT_OFFSET:
                shoot_offset_x = _float_var(
                    raw[0x0C:0x10], integers, floats, difficulty, rank,
                    life, enemy, variable_player,
                )
                shoot_offset_y = _float_var(
                    raw[0x10:0x14], integers, floats, difficulty, rank,
                    life, enemy, variable_player,
                )
                # The source also stores z. It does not affect TH06's 2D
                # collision geometry, but resolve it so unknown variables
                # still fail closed consistently.
                _float_var(
                    raw[0x14:0x18], integers, floats, difficulty, rank,
                    life, enemy, variable_player,
                )
            elif instruction.opcode == OPCODE_BULLET_EFFECTS:
                effect_ints = tuple(
                    _int_var(value, integers, difficulty, rank, life)
                    for value in struct.unpack_from("<iiii", raw, 0x0C)
                )
                resolved_effect_floats = tuple(
                    _float_var(
                        raw[offset:offset + 4], integers, floats, difficulty,
                        rank, life, enemy, variable_player,
                    )
                    for offset in range(0x1C, 0x2C, 4)
                )
                if any(
                    isinstance(value, FloatInterval)
                    for value in resolved_effect_floats
                ):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "uncertain bullet effects need a hard envelope",
                    )
                effect_floats = resolved_effect_floats
                if pattern is not None:
                    pattern = replace(
                        pattern,
                        ex_ints=effect_ints,
                        ex_floats=effect_floats,
                    )
            elif instruction.opcode == OPCODE_ANIMATION_DEATH:
                death_anm1, death_anm2, death_anm3 = struct.unpack_from(
                    "<BBB", raw, 0x0C
                )
            elif instruction.opcode == OPCODE_SPELL_EFFECT:
                effect_spawns[frame_index].append(13)
            elif instruction.opcode == OPCODE_EFFECT_PARTICLE:
                effect_id, count = struct.unpack_from("<ii", raw, 0x0C)
                if not 0 <= effect_id < 20 or not 0 <= count <= 512:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index,
                        "invalid ECL effect-particle request",
                    )
                if rng is not None:
                    consume_effect_spawn_rng(rng, (effect_id,) * count)
                effect_spawns[frame_index].extend((effect_id,) * count)
            elif instruction.opcode == OPCODE_SPELL_START:
                # SpellcardStart source defaults. Bullet cancellation is a
                # hazard removal, so Hard worlds retain existing bullets
                # conservatively. The exact nominal combat world additionally
                # models RemoveAllBullets(true), point-item allocation, and
                # source laser retirement at this exact ECL instruction.
                spell_start = getattr(laser_world, "spell_start", None)
                if spell_start is not None:
                    if rng is None:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "nominal spell start needs exact RNG/allocation state",
                        )
                    spell_start(rng)
                rank_speed_low = -0.5
                rank_speed_high = 0.5
                rank_amount1_low = rank_amount1_high = 0
                rank_amount2_low = rank_amount2_high = 0
            elif instruction.opcode == OPCODE_SPELL_END:
                # Hard worlds retain current hazards conservatively.  The
                # exact nominal combat world applies DespawnBullets, including
                # point-item allocation and the same-frame bullet/laser pass.
                spell_end = getattr(laser_world, "spell_end", None)
                if spell_end is not None:
                    if rng is None:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "nominal spell end needs exact RNG/allocation state",
                        )
                    spell_end(rng)
            elif instruction.opcode == OPCODE_LIFE_SET:
                life = struct.unpack_from("<i", raw, 0x0C)[0]
                life_lower_bound = life
            elif instruction.opcode == OPCODE_DEATH_CALLBACK:
                death_callback_sub = struct.unpack_from("<i", raw, 0x0C)[0]
            elif instruction.opcode == OPCODE_BOSS_TIMER_SET:
                boss_timer = struct.unpack_from("<i", raw, 0x0C)[0]
                boss_timer_subframe = 0.0
            elif instruction.opcode == OPCODE_LIFE_CALLBACK_THRESHOLD:
                life_callback_threshold = struct.unpack_from("<i", raw, 0x0C)[0]
            elif instruction.opcode == OPCODE_LIFE_CALLBACK_SUB:
                life_callback_sub = struct.unpack_from("<i", raw, 0x0C)[0]
            elif instruction.opcode == OPCODE_TIMER_CALLBACK_THRESHOLD:
                timer_callback_threshold = struct.unpack_from("<i", raw, 0x0C)[0]
                boss_timer = 0
                boss_timer_subframe = 0.0
            elif instruction.opcode == OPCODE_TIMER_CALLBACK_SUB:
                timer_callback_sub = struct.unpack_from("<i", raw, 0x0C)[0]
            elif instruction.opcode == OPCODE_DROP_ITEMS:
                count = struct.unpack_from("<i", raw, 0x0C)[0]
                if count < 0:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "negative DROPITEMS count"
                    )
                if rng is not None:
                    for index in range(count):
                        offset_x = _f32(rng.f32_in_range(144.0) - 72.0)
                        offset_y = _f32(rng.f32_in_range(144.0) - 72.0)
                        item_births[frame_index].append(EclItemBirth(
                            _f32(enemy_x + offset_x),
                            _f32(enemy_y + offset_y),
                            2 if index == 0 else 0,
                            1,
                        ))
                item_spawns[frame_index] += count
            elif instruction.opcode == OPCODE_DROP_ITEM_ID:
                item_type = struct.unpack_from("<i", raw, 0x0C)[0]
                if item_type not in range(7):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "invalid DROPITEMID type",
                    )
                item_births[frame_index].append(EclItemBirth(
                    _f32(enemy_x),
                    _f32(enemy_y),
                    item_type,
                    item_type,
                ))
                item_spawns[frame_index] += 1
            elif instruction.opcode == OPCODE_EX_REPEAT:
                index = struct.unpack_from("<i", raw, 0x0C)[0]
                if index >= 0:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"repeating ECL external instruction {index} can mutate hazards",
                    )
            elif instruction.opcode == OPCODE_TIME_SET:
                variable = struct.unpack_from("<i", raw, 0x0C)[0]
                current_time += _int_var(variable, integers, difficulty, rank, life)
            elif instruction.opcode == OPCODE_CALL_STACK_DISABLED:
                call_stack_disabled = bool(struct.unpack_from("<i", raw, 0x0C)[0])
            elif instruction.opcode == OPCODE_BULLET_RANK_INFLUENCE:
                rank_speed_low, rank_speed_high = struct.unpack_from("<ff", raw, 0x0C)
                (
                    rank_amount1_low,
                    rank_amount1_high,
                    rank_amount2_low,
                    rank_amount2_high,
                ) = struct.unpack_from("<iiii", raw, 0x14)
            elif instruction.opcode == OPCODE_BOSS_TIMER_CLEAR:
                timer_callback_sub = death_callback_sub
                boss_timer = 0
                boss_timer_subframe = 0.0
            elif instruction.opcode == OPCODE_BOSS_SET:
                new_boss_id = struct.unpack_from("<i", raw, 0x0C)[0]
                if new_boss_id >= 0:
                    if new_boss_id >= 8:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            f"invalid source boss id {new_boss_id}",
                        )
                    is_boss = True
                    boss_id = new_boss_id
                else:
                    is_boss = False
                    boss_id = -1
            elif instruction.opcode == OPCODE_INTERRUPT_SET:
                interrupt_sub, interrupt_id = struct.unpack_from(
                    "<ii", raw, 0x0C
                )
                if not 0 <= interrupt_id < 8:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"invalid source interrupt id {interrupt_id}",
                    )
                if not 0 <= interrupt_sub < len(spawner.ecl_subroutines):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"invalid source interrupt subroutine {interrupt_sub}",
                    )
                interrupts[interrupt_id] = interrupt_sub
            elif instruction.opcode == OPCODE_INTERRUPT:
                run_interrupt = struct.unpack_from("<i", raw, 0x0C)[0]
                continue
            elif instruction.opcode == OPCODE_ENEMY_CREATE:
                # SpawnEnemy first runs the newborn's time-zero ECL inline.
                # A later free slot may then receive its ordinary update in
                # this same EnemyManager pass.  Fold both slot-order outcomes
                # into the remaining Hard window: their union is conservative,
                # and the next live snapshot captures whichever persistent
                # child source actually inserted.  Nominal forecasting still
                # needs exact slot/RNG insertion and therefore fails closed.
                hard_audit = radial_births and allow_enemy_create_audit
                nominal_insertion = (
                    not radial_births
                    and not abstract_rng
                    and rng is not None
                )
                if not hard_audit and not nominal_insertion:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        ENEMY_CREATE_WORLD_REASON,
                    )
                if record_enemy_kill_all and enemy_kill_all[frame_index]:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "same-frame ENEMYKILLALL/ENEMYCREATE order needs "
                        "an exact world event stream",
                    )
                sub_id = struct.unpack_from("<i", raw, 0x0C)[0]
                try:
                    child_x = _float_var(
                        raw[0x10:0x14], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                    child_y = _float_var(
                        raw[0x14:0x18], integers, floats, difficulty, rank,
                        life, enemy, variable_player,
                    )
                except UnsupportedBirthModel as error:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, str(error)
                    )
                if isinstance(child_x, FloatInterval) or isinstance(
                    child_y, FloatInterval
                ):
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        "uncertain ECL enemy position needs a world envelope",
                    )
                child_life = struct.unpack_from("<h", raw, 0x1C)[0]
                child_item_drop = struct.unpack_from("<h", raw, 0x1E)[0]
                child = source_enemy_template(
                    spawner.ecl_program,
                    spawner.ecl_subroutines,
                    sub_id,
                    child_x,
                    child_y,
                    child_life,
                    child_item_drop,
                )
                if child is None:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        ENEMY_CREATE_WORLD_REASON,
                    )
                newborn = _forecast_ecl_births_single(
                    child,
                    (player,),
                    difficulty,
                    rank,
                    bullet_sizes,
                    frame_multiplier,
                    rng if nominal_insertion else None,
                    allow_player_variables,
                    radial_births,
                    abstract_rng,
                    False,
                    model_player_damage=False,
                    allow_enemy_create_audit=False,
                    record_enemy_kill_all=record_enemy_kill_all,
                    laser_world=laser_world,
                )
                if newborn.covered_frames < 1:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        f"spawned emitter {sub_id}: {newborn.reason}",
                    )
                births[frame_index].extend(newborn.births[0])
                if newborn.effect_spawns:
                    effect_spawns[frame_index].extend(
                        newborn.effect_spawns[0]
                    )
                if newborn.item_spawns:
                    item_spawns[frame_index] += newborn.item_spawns[0]
                if newborn.item_births:
                    item_births[frame_index].extend(newborn.item_births[0])
                if nominal_insertion:
                    if newborn.created_emitters:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "nested inline ECL enemy creation needs exact "
                            "slot insertion",
                        )
                    if newborn.next_spawner is not None:
                        created_emitters.append(newborn.next_spawner)
                    instruction_address = next_address
                    continue
                child_states = []
                if newborn.next_spawner is not None:
                    # If SpawnEnemy chose an already-passed slot, this inline
                    # state is the one carried into the next physical frame.
                    child_states.append(newborn.next_spawner)
                    updated = _forecast_ecl_births_single(
                        newborn.next_spawner,
                        (player,),
                        difficulty,
                        rank,
                        bullet_sizes,
                        frame_multiplier,
                        None,
                        allow_player_variables,
                        radial_births,
                        abstract_rng,
                        False,
                        model_player_damage=False,
                        allow_enemy_create_audit=False,
                        record_enemy_kill_all=record_enemy_kill_all,
                        laser_world=laser_world,
                    )
                    if updated.covered_frames < 1:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            f"spawned emitter {sub_id}: {updated.reason}",
                        )
                    births[frame_index].extend(updated.births[0])
                    if updated.effect_spawns:
                        effect_spawns[frame_index].extend(
                            updated.effect_spawns[0]
                        )
                    if updated.item_spawns:
                        item_spawns[frame_index] += updated.item_spawns[0]
                    if updated.item_births:
                        item_births[frame_index].extend(updated.item_births[0])
                    if updated.body_hazards:
                        body_hazards[frame_index].extend(
                            updated.body_hazards[0]
                        )
                    if updated.next_spawner is not None:
                        # If the allocated slot is later than the parent, the
                        # manager reaches it and carries this once-updated
                        # state into the next physical frame.
                        child_states.append(updated.next_spawner)
                remaining_positions = player_positions[frame_index + 1:]
                if remaining_positions:
                    # Slot order is observable in a live snapshot but the
                    # compact ECL forecast owns only one emitter.  Carry both
                    # physically possible states and union their hazards.  A
                    # child world mutation that this local audit cannot fold
                    # remains an ordinary fail-closed boundary.
                    for child_state in child_states:
                        future = _forecast_ecl_births_single(
                            child_state,
                            remaining_positions,
                            difficulty,
                            rank,
                            bullet_sizes,
                            frame_multiplier,
                            None,
                            allow_player_variables,
                            radial_births,
                            abstract_rng,
                            False,
                            model_player_damage=True,
                            allow_enemy_create_audit=False,
                            record_enemy_kill_all=record_enemy_kill_all,
                        )
                        for offset, frame_births in enumerate(
                            future.births,
                            frame_index + 1,
                        ):
                            births[offset].extend(frame_births)
                        for offset, frame_bodies in enumerate(
                            future.body_hazards,
                            frame_index + 1,
                        ):
                            body_hazards[offset].extend(frame_bodies)
                        if future.covered_frames < len(remaining_positions):
                            return EclForecast(
                                tuple(map(tuple, births)),
                                frame_index + 1 + future.covered_frames,
                                f"spawned emitter {sub_id}: {future.reason}",
                                body_hazards=tuple(
                                    tuple(frame) for frame in body_hazards
                                ),
                            )
            elif instruction.opcode in (
                OPCODE_LASER_CREATE,
                OPCODE_LASER_CREATE_AIMED,
            ):
                (
                    start_time,
                    _duration,
                    _despawn_duration,
                    hitbox_start_time,
                    _hitbox_end_delay,
                    _flags,
                ) = struct.unpack_from("<iiiiii", raw, 0x28)
                if laser_world is None:
                    try:
                        values = tuple(
                            _float_var(
                                raw[offset:offset + 4],
                                integers,
                                floats,
                                difficulty,
                                rank,
                                life,
                                enemy,
                                variable_player,
                            )
                            for offset in (0x10, 0x14, 0x18, 0x1C, 0x20)
                        )
                    except UnsupportedBirthModel as error:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, str(error)
                        )
                    if any(isinstance(value, FloatInterval) for value in values):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index,
                            "uncertain ECL laser parameters need a hard envelope",
                        )
                    (
                        laser_angle,
                        laser_speed,
                        start_offset,
                        end_offset,
                        start_length,
                    ) = values
                    width = struct.unpack_from("<f", raw, 0x24)[0]
                    origin_x = _float_add(enemy_x, shoot_offset_x)
                    origin_y = _float_add(enemy_y, shoot_offset_y)
                    uncertainty_x = (
                        position_uncertainty_x + position_uncertainty
                    )
                    uncertainty_y = (
                        position_uncertainty_y + position_uncertainty
                    )
                    if isinstance(origin_x, FloatInterval):
                        uncertainty_x += (origin_x.high - origin_x.low) / 2.0
                        origin_x = (origin_x.low + origin_x.high) / 2.0
                    if isinstance(origin_y, FloatInterval):
                        uncertainty_y += (origin_y.high - origin_y.low) / 2.0
                        origin_y = (origin_y.low + origin_y.high) / 2.0
                    numbers = (
                        laser_angle,
                        laser_speed,
                        start_offset,
                        end_offset,
                        start_length,
                        width,
                        origin_x,
                        origin_y,
                    )
                    if (
                        not all(math.isfinite(value) for value in numbers)
                        or width <= 0.0
                        or start_length < 0.0
                        or any(value < 0 for value in (
                            start_time,
                            _duration,
                            _despawn_duration,
                            hitbox_start_time,
                            _hitbox_end_delay,
                        ))
                    ):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index,
                            "invalid future ECL laser creation request",
                        )
                    aimed = instruction.opcode == OPCODE_LASER_CREATE_AIMED
                    laser_angle = _f32(laser_angle)
                    if aimed:
                        # Hard source forecasting is shared across candidates.
                        # Use the union over every possible aimed angle below,
                        # not the nominal root-player angle stored here.
                        laser_angle = 0.0
                    future_laser = Laser(
                        x=_f32(origin_x),
                        y=_f32(origin_y),
                        angle=laser_angle,
                        start_offset=_f32(start_offset),
                        end_offset=_f32(end_offset),
                        start_length=_f32(start_length),
                        width=_f32(width),
                        speed=_f32(laser_speed),
                        start_time=start_time,
                        hitbox_start_time=hitbox_start_time,
                        duration=_duration,
                        despawn_duration=_despawn_duration,
                        hitbox_end_delay=_hitbox_end_delay,
                        timer=0,
                        timer_float=0.0,
                        flags=_flags & 0xFFFF,
                        state=1 if start_time == 0 else 0,
                        motion_known=True,
                    )
                    for laser_frame, frame_hazards in enumerate(
                        future_hazards(
                            future_laser,
                            horizon - frame_index,
                        ),
                        frame_index,
                    ):
                        body_hazards[laser_frame].extend(
                            _future_laser_aabb(
                                hazard,
                                aimed,
                                uncertainty_x,
                                uncertainty_y,
                            )
                            for hazard in frame_hazards
                        )
                else:
                    hard_spawn = getattr(
                        laser_world, "spawn_laser_hard", None
                    )
                    try:
                        values = tuple(
                            _float_var(
                                raw[offset:offset + 4],
                                integers,
                                floats,
                                difficulty,
                                rank,
                                life,
                                enemy,
                                variable_player,
                            )
                            for offset in (0x10, 0x14, 0x18, 0x1C, 0x20)
                        )
                    except UnsupportedBirthModel as error:
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index, str(error)
                        )
                    if any(isinstance(value, FloatInterval) for value in values):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index,
                            "uncertain ECL laser parameters need a hard envelope",
                        )
                    (
                        laser_angle,
                        laser_speed,
                        start_offset,
                        end_offset,
                        start_length,
                    ) = values
                    width = struct.unpack_from("<f", raw, 0x24)[0]
                    (
                        start_time,
                        duration,
                        despawn_duration,
                        hitbox_start_time,
                        hitbox_end_delay,
                        flags,
                    ) = struct.unpack_from("<iiiiii", raw, 0x28)
                    origin_x = _float_add(enemy_x, shoot_offset_x)
                    origin_y = _float_add(enemy_y, shoot_offset_y)
                    uncertainty_x = (
                        position_uncertainty_x + position_uncertainty
                    )
                    uncertainty_y = (
                        position_uncertainty_y + position_uncertainty
                    )
                    if hard_spawn is not None:
                        if isinstance(origin_x, FloatInterval):
                            uncertainty_x += (
                                origin_x.high - origin_x.low
                            ) / 2.0
                            origin_x = (origin_x.low + origin_x.high) / 2.0
                        if isinstance(origin_y, FloatInterval):
                            uncertainty_y += (
                                origin_y.high - origin_y.low
                            ) / 2.0
                            origin_y = (origin_y.low + origin_y.high) / 2.0
                    numbers = (
                        laser_angle,
                        laser_speed,
                        start_offset,
                        end_offset,
                        start_length,
                        width,
                        origin_x,
                        origin_y,
                    )
                    if (
                        not all(math.isfinite(value) for value in numbers)
                        or width <= 0.0
                        or start_length < 0.0
                        or any(value < 0 for value in (
                            start_time,
                            duration,
                            despawn_duration,
                            hitbox_start_time,
                            hitbox_end_delay,
                        ))
                    ):
                        return EclForecast(
                            tuple(map(tuple, births)), frame_index,
                            "invalid ECL laser request",
                        )
                    laser_angle = _f32(laser_angle)
                    aimed = instruction.opcode == OPCODE_LASER_CREATE_AIMED
                    if aimed and hard_spawn is None:
                        laser_angle = _f32(
                            laser_angle
                            + _f32(math.atan2(
                                player[1] - origin_y,
                                player[0] - origin_x,
                            ))
                        )
                    elif aimed:
                        laser_angle = 0.0
                    created_laser = Laser(
                        x=_f32(origin_x),
                        y=_f32(origin_y),
                        angle=laser_angle,
                        start_offset=_f32(start_offset),
                        end_offset=_f32(end_offset),
                        start_length=_f32(start_length),
                        width=_f32(width),
                        speed=_f32(laser_speed),
                        start_time=start_time,
                        hitbox_start_time=hitbox_start_time,
                        duration=duration,
                        despawn_duration=despawn_duration,
                        hitbox_end_delay=hitbox_end_delay,
                        timer=0,
                        timer_float=0.0,
                        flags=flags & 0xFFFF,
                        state=1 if start_time == 0 else 0,
                        motion_known=True,
                    )
                    if hard_spawn is None:
                        laser_slot = laser_world.spawn_laser(created_laser)
                    else:
                        laser_slot = hard_spawn(
                            created_laser,
                            aimed=aimed,
                            uncertainty_x=uncertainty_x,
                            uncertainty_y=uncertainty_y,
                        )
                    laser_slots[laser_store] = laser_slot
            elif instruction.opcode == OPCODE_LASER_INDEX:
                laser_store = _int_var(
                    struct.unpack_from("<i", raw, 0x0C)[0],
                    integers,
                    difficulty,
                    rank,
                    life,
                )
                if not 0 <= laser_store < 32:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index,
                        "invalid ECL laser store index",
                    )
            elif instruction.opcode in (
                OPCODE_LASER_ROTATE,
                OPCODE_LASER_ROTATE_FROM_PLAYER,
                OPCODE_LASER_OFFSET,
                OPCODE_LASER_TEST,
                OPCODE_LASER_CANCEL,
            ):
                if laser_world is None:
                    if instruction.opcode == OPCODE_LASER_CANCEL:
                        # Retaining the projected active beam after a cancel
                        # is conservative. Any later liveness-dependent ECL
                        # still fails at LASERTEST.
                        instruction_address = next_address
                        continue
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index,
                        FAIL_CLOSED_ECL_OPCODES[instruction.opcode],
                    )
                laser_index = struct.unpack_from("<i", raw, 0x0C)[0]
                if not 0 <= laser_index < 32:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index,
                        "invalid ECL laser pointer index",
                    )
                pointer_slot = laser_slots[laser_index]
                pointed = (
                    laser_world.laser_at(pointer_slot)
                    if pointer_slot >= 0 else None
                )
                observe_dereference = getattr(
                    laser_world, "observe_laser_dereference", None
                )
                if observe_dereference is not None:
                    observe_dereference(pointer_slot, pointed is not None)
                hard_replace = getattr(
                    laser_world, "replace_laser_hard", None
                )
                if instruction.opcode == OPCODE_LASER_TEST:
                    compare_register = int(pointed is None)
                elif pointed is not None:
                    if instruction.opcode in (
                        OPCODE_LASER_ROTATE,
                        OPCODE_LASER_ROTATE_FROM_PLAYER,
                    ):
                        try:
                            delta = _float_var(
                                raw[0x10:0x14],
                                integers,
                                floats,
                                difficulty,
                                rank,
                                life,
                                enemy,
                                variable_player,
                            )
                        except UnsupportedBirthModel as error:
                            return EclForecast(
                                tuple(map(tuple, births)), frame_index, str(error)
                            )
                        if (
                            isinstance(delta, FloatInterval)
                            and hard_replace is None
                        ):
                            return EclForecast(
                                tuple(map(tuple, births)), frame_index,
                                "uncertain ECL laser rotation",
                            )
                        unconstrained = (
                            isinstance(delta, FloatInterval)
                            or instruction.opcode
                                == OPCODE_LASER_ROTATE_FROM_PLAYER
                        )
                        if unconstrained and hard_replace is not None:
                            next_angle = 0.0
                        else:
                            next_angle = (
                                pointed.angle + delta
                                if instruction.opcode == OPCODE_LASER_ROTATE
                                else _f32(math.atan2(
                                    player[1] - pointed.y,
                                    player[0] - pointed.x,
                                )) + delta
                            )
                        updated = replace(
                            pointed, angle=_f32(next_angle)
                        )
                        if hard_replace is None:
                            laser_world.replace_laser(pointer_slot, updated)
                        else:
                            hard_replace(
                                pointer_slot,
                                updated,
                                angle_unconstrained=unconstrained,
                            )
                    elif instruction.opcode == OPCODE_LASER_OFFSET:
                        offset_x, offset_y = struct.unpack_from(
                            "<ff", raw, 0x10
                        )
                        next_x = _float_add(enemy_x, offset_x)
                        next_y = _float_add(enemy_y, offset_y)
                        uncertainty_x = (
                            position_uncertainty_x + position_uncertainty
                        )
                        uncertainty_y = (
                            position_uncertainty_y + position_uncertainty
                        )
                        if isinstance(next_x, FloatInterval):
                            if hard_replace is None:
                                return EclForecast(
                                    tuple(map(tuple, births)), frame_index,
                                    "uncertain exact ECL laser offset",
                                )
                            uncertainty_x += (
                                next_x.high - next_x.low
                            ) / 2.0
                            next_x = (next_x.low + next_x.high) / 2.0
                        if isinstance(next_y, FloatInterval):
                            if hard_replace is None:
                                return EclForecast(
                                    tuple(map(tuple, births)), frame_index,
                                    "uncertain exact ECL laser offset",
                                )
                            uncertainty_y += (
                                next_y.high - next_y.low
                            ) / 2.0
                            next_y = (next_y.low + next_y.high) / 2.0
                        updated = replace(
                            pointed,
                            x=_f32(next_x),
                            y=_f32(next_y),
                        )
                        if hard_replace is None:
                            laser_world.replace_laser(pointer_slot, updated)
                        else:
                            hard_replace(
                                pointer_slot,
                                updated,
                                uncertainty_x=uncertainty_x,
                                uncertainty_y=uncertainty_y,
                            )
                    elif (
                        instruction.opcode == OPCODE_LASER_CANCEL
                        and pointed.state < 2
                    ):
                        laser_world.replace_laser(
                            pointer_slot,
                            replace(
                                pointed,
                                state=2,
                                timer=0,
                                timer_float=0.0,
                            ),
                        )
            elif instruction.opcode == OPCODE_LASER_CLEAR_ALL:
                if laser_world is not None:
                    laser_slots = [-1] * 32
            elif instruction.opcode == OPCODE_ENEMY_KILL_ALL:
                if enemy_kill_all_is_noop:
                    pass
                elif record_enemy_kill_all:
                    if not is_boss:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "nonboss ENEMYKILLALL self-transition needs "
                            "inline ECL context replay",
                        )
                    if created_emitters:
                        return EclForecast(
                            tuple(map(tuple, births)),
                            frame_index,
                            "same-frame ENEMYCREATE/ENEMYKILLALL order needs "
                            "an exact world event stream",
                        )
                    enemy_kill_all[frame_index] = True
                else:
                    return EclForecast(
                        tuple(map(tuple, births)),
                        frame_index,
                        FAIL_CLOSED_ECL_OPCODES[instruction.opcode],
                    )
            elif instruction.opcode in HAZARD_NEUTRAL_ECL_OPCODES:
                pass
            elif instruction.opcode in FAIL_CLOSED_ECL_OPCODES:
                return EclForecast(
                    tuple(map(tuple, births)),
                    frame_index,
                    FAIL_CLOSED_ECL_OPCODES[instruction.opcode],
                )
            else:
                return EclForecast(
                    tuple(map(tuple, births)),
                    frame_index,
                    f"unclassified ECL opcode {instruction.opcode}",
                )
            instruction_address = next_address
        else:
            return EclForecast(
                tuple(map(tuple, births)), frame_index, "ECL instruction budget exhausted"
            )

        motion = finish_motion_values(
            enemy_x,
            enemy_y,
            velocity_x,
            velocity_y,
            angle,
            speed,
            angular_velocity,
            acceleration,
            movement_mode,
            movement_ease,
            move_interp_x,
            move_interp_y,
            move_start_x,
            move_start_y,
            move_timer,
            move_timer_float,
            move_start_time,
        )
        enemy_x, enemy_y = motion.x, motion.y
        enemy_x, enemy_y = clamp_position(enemy_x, enemy_y)
        velocity_x, velocity_y = motion.velocity_x, motion.velocity_y
        angle, speed = motion.angle, motion.speed
        movement_mode = motion.movement_mode
        move_timer, move_timer_float = motion.move_timer, motion.move_timer_float
        if timed_move_radius > 0.0:
            if movement_mode == 0:
                position_uncertainty += (
                    1.0 - timed_move_progress
                ) * timed_move_radius
                timed_move_progress = 1.0
                timed_move_next_progress = 1.0
                timed_move_radius = 0.0
                velocity_uncertainty = 0.0
            else:
                remaining = min(1.0, move_timer_float / move_start_time)
                timed_move_next_progress = interpolation_progress(
                    remaining,
                    movement_ease,
                )
                velocity_uncertainty = max(
                    0.0,
                    timed_move_next_progress - timed_move_progress,
                ) * timed_move_radius
            velocity_x = 0.0
            velocity_y = 0.0
        elif uncertain_heading and movement_mode == 1:
            velocity_uncertainty = abs(speed)
            velocity_x = 0.0
            velocity_y = 0.0
        else:
            velocity_uncertainty = 0.0
        enemy = uncertain_enemy()

        if (
            not spawn_inline
            and interactable
            and collidable
            and not invisible
            and hitbox_half_width > 0.0
            and hitbox_half_height > 0.0
        ):
            total_uncertainty_x = (
                position_uncertainty_x + position_uncertainty
            )
            total_uncertainty_y = (
                position_uncertainty_y + position_uncertainty
            )
            body_hazards[frame_index].append((
                enemy_x - hitbox_half_width - total_uncertainty_x,
                enemy_y - hitbox_half_height - total_uncertainty_y,
                enemy_x + hitbox_half_width + total_uncertainty_x,
                enemy_y + hitbox_half_height + total_uncertainty_y,
            ))

        if life > 0 and interval > 0:
            interval_subframe += frame_multiplier
            while interval_subframe >= 1.0:
                interval_timer += 1
                interval_timer_low += 1
                interval_timer_high += 1
                interval_subframe -= 1.0
            if interval_timer_high >= interval:
                if pattern is None:
                    return EclForecast(
                        tuple(map(tuple, births)), frame_index, "periodic shooter has no resolved pattern"
                    )
                try:
                    births[frame_index].extend(emit(
                        pattern,
                        (
                            _float_add(enemy[0], shoot_offset_x),
                            _float_add(enemy[1], shoot_offset_y),
                        ),
                        player,
                    ))
                except UnsupportedBirthModel as error:
                    return EclForecast(tuple(map(tuple, births)), frame_index, str(error))
                if interval_timer_low >= interval:
                    interval_timer_low = 0
                    interval_timer_high = 0
                else:
                    # Union the fired phase (timer 0) with every phase that
                    # has not fired. This compact interval remains sound.
                    interval_timer_low = 0
                    interval_timer_high = min(interval_timer_high, interval - 1)
                interval_timer = interval_timer_low
                interval_subframe = 0.0
        if not spawn_inline and interactable and model_player_damage:
            # EnemyManager caps player-shot damage at 70 per update. A
            # collidable non-boss can additionally lose 10 from kill-box
            # contact. This lower bound decides only whether an asynchronous
            # life callback is reachable; it never removes a hazard.
            life_lower_bound -= (70 if damageable else 0) + (
                10 if collidable and not is_boss else 0
            )
        time_subframe += frame_multiplier
        while time_subframe >= 1.0:
            current_time += 1
            time_subframe -= 1.0
        if not spawn_inline:
            boss_timer_subframe += frame_multiplier
            while boss_timer_subframe >= 1.0:
                boss_timer += 1
                boss_timer_subframe -= 1.0
        if stop_after_frame:
            return EclForecast(
                tuple(map(tuple, births)), frame_index + 1, stop_after_frame
            )
    next_instruction = program.get(instruction_address)
    return EclForecast(
        tuple(map(tuple, births)),
        horizon,
        next_spawner=_copy_spawner(
            spawner,
            x=enemy_x,
            y=enemy_y,
            velocity_x=velocity_x,
            velocity_y=velocity_y,
            angle=angle,
            angular_velocity=angular_velocity,
            speed=speed,
            acceleration=acceleration,
            movement_mode=movement_mode,
            movement_ease=movement_ease,
            move_interp_x=move_interp_x,
            move_interp_y=move_interp_y,
            move_start_x=move_start_x,
            move_start_y=move_start_y,
            move_timer=move_timer,
            move_timer_float=move_timer_float,
            move_start_time=move_start_time,
            shooting_disabled=shooting_disabled,
            shoot_offset_x=shoot_offset_x,
            shoot_offset_y=shoot_offset_y,
            interval=interval,
            timer=interval_timer,
            timer_float=interval_timer + interval_subframe,
            pattern=pattern,
            ecl_time=current_time,
            ecl_time_float=current_time + time_subframe,
            ecl_ints=tuple(integers),
            ecl_floats=tuple(floats),
            ecl_compare=compare_register,
            next_instruction=next_instruction,
            ecl_stack=tuple(call_stack),
            interactable=interactable,
            collidable=collidable,
            invisible=invisible,
            hitbox_half_width=hitbox_half_width,
            hitbox_half_height=hitbox_half_height,
            call_stack_disabled=call_stack_disabled,
            life=life,
            bullet_rank_speed_low=rank_speed_low,
            bullet_rank_speed_high=rank_speed_high,
            bullet_rank_amount1_low=rank_amount1_low,
            bullet_rank_amount1_high=rank_amount1_high,
            bullet_rank_amount2_low=rank_amount2_low,
            bullet_rank_amount2_high=rank_amount2_high,
            lower_move_x=lower_move_x,
            lower_move_y=lower_move_y,
            upper_move_x=upper_move_x,
            upper_move_y=upper_move_y,
            should_clamp_position=should_clamp_position,
            boss_timer=boss_timer,
            boss_timer_float=boss_timer + boss_timer_subframe,
            death_callback_sub=death_callback_sub,
            life_callback_threshold=life_callback_threshold,
            life_callback_sub=life_callback_sub,
            timer_callback_threshold=timer_callback_threshold,
            timer_callback_sub=timer_callback_sub,
            is_boss=is_boss,
            boss_id=boss_id,
            interrupts=tuple(interrupts),
            run_interrupt=run_interrupt,
            damageable=damageable,
            death_mode=death_mode,
            death_anm1=death_anm1,
            death_anm2=death_anm2,
            death_anm3=death_anm3,
            has_been_in_bounds=has_been_in_bounds,
            bullet_effect_floats=effect_floats,
            bullet_effect_ints=effect_ints,
            laser_slots=tuple(laser_slots),
            laser_store=laser_store,
            forecast_position_uncertainty_x=(
                position_uncertainty_x + position_uncertainty
            ),
            forecast_position_uncertainty_y=(
                position_uncertainty_y + position_uncertainty
            ),
        ),
        body_hazards=tuple(tuple(frame) for frame in body_hazards),
        created_emitters=tuple(created_emitters),
        effect_spawns=tuple(tuple(frame) for frame in effect_spawns),
        item_spawns=tuple(item_spawns),
        item_births=tuple(tuple(frame) for frame in item_births),
        enemy_kill_all=tuple(enemy_kill_all),
    )


def _forecast_ecl_births_with_death_callbacks(
    spawner: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    difficulty: int,
    rank: int,
    bullet_sizes: tuple[tuple[float, float], ...],
    frame_multiplier: float = 1.0,
    rng: RngState | None = None,
    allow_player_variables: bool = True,
    radial_births: bool = False,
    abstract_rng: bool = False,
    enemy_kill_all_is_noop: bool = False,
    model_player_damage: bool = True,
) -> EclForecast:
    """Union every reachable source death-callback pickup frame."""
    horizon = len(player_positions)
    callback_damage = (70 if spawner.damageable else 0) + (
        10 if spawner.collidable and not spawner.is_boss else 0
    )
    earliest_callback = (
        max(0, (spawner.life + callback_damage - 1) // callback_damage)
        if callback_damage > 0 and spawner.life > 0
        else (0 if spawner.life <= 0 else horizon)
    )
    should_branch = (
        model_player_damage
        and abstract_rng
        and spawner.interactable
        and spawner.death_callback_sub >= 0
        and 0 <= spawner.death_callback_sub < len(spawner.ecl_subroutines)
        and earliest_callback < horizon
    )
    if not should_branch:
        return _forecast_ecl_births_single(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            rng,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage=model_player_damage,
        )

    program = _compiled_program(spawner.ecl_program)
    callback_address = spawner.ecl_subroutines[spawner.death_callback_sub]
    callback_instruction = program.get(callback_address)
    if callback_instruction is None:
        return EclForecast(
            tuple(() for _ in player_positions),
            0,
            "death callback instruction graph is not captured",
        )

    # Zero damage remains physically possible.  Keeping this branch alive is
    # conservative; it also provides the state prefix for each death frame.
    no_callback_spawner = _copy_spawner(spawner, death_callback_sub=-1)
    no_callback = _forecast_ecl_births_single(
        no_callback_spawner,
        player_positions,
        difficulty,
        rank,
        bullet_sizes,
        frame_multiplier,
        None,
        allow_player_variables,
        radial_births,
        abstract_rng,
        enemy_kill_all_is_noop,
        model_player_damage=model_player_damage,
    )
    births = [list(frame) for frame in no_callback.births]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    for index, frame_bodies in enumerate(no_callback.body_hazards):
        bodies[index].extend(frame_bodies)
    covered_frames = no_callback.covered_frames
    reason = no_callback.reason

    for callback_frame in range(earliest_callback, horizon):
        if callback_frame:
            prefix = _forecast_ecl_births_single(
                no_callback_spawner,
                player_positions[:callback_frame],
                difficulty,
                rank,
                bullet_sizes,
                frame_multiplier,
                None,
                allow_player_variables,
                radial_births,
                abstract_rng,
                enemy_kill_all_is_noop,
                model_player_damage=model_player_damage,
            )
            if prefix.covered_frames < callback_frame:
                if prefix.covered_frames < covered_frames:
                    covered_frames = prefix.covered_frames
                    reason = prefix.reason
                continue
            callback_source = prefix.next_spawner
            if callback_source is None:
                # The main ECL already despawned this branch before damage
                # could invoke its callback.
                continue
        else:
            callback_source = no_callback_spawner

        # EnemyManager applies the death-mode state transition before
        # CallEclSub.  Mode zero removes the slot, so the newly assigned
        # context is never executed on a later update.
        if callback_source.death_mode == 0:
            continue
        if callback_source.death_mode == 1:
            callback_source = _copy_spawner(
                callback_source,
                interactable=False,
                life=0,
            )
        elif callback_source.death_mode == 3:
            callback_source = _copy_spawner(
                callback_source,
                life=1,
                damageable=False,
                death_mode=0,
            )
        else:
            # Mode two and the source switch's unused values retain the slot
            # and interactability.  Keeping both is the conservative hazard
            # state while the callback ECL decides what happens next.
            callback_source = _copy_spawner(callback_source, life=0)

        callback_source = _copy_spawner(
            callback_source,
            death_callback_sub=-1,
            life_callback_threshold=-1,
            timer_callback_threshold=-1,
            next_instruction=callback_instruction,
            ecl_time=0,
            ecl_time_float=0.0,
            ecl_stack=(),
            bullet_rank_speed_low=-0.5,
            bullet_rank_speed_high=0.5,
            bullet_rank_amount1_low=0,
            bullet_rank_amount1_high=0,
            bullet_rank_amount2_low=0,
            bullet_rank_amount2_high=0,
        )
        callback = _forecast_ecl_births_single(
            callback_source,
            player_positions[callback_frame:],
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            None,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage=model_player_damage,
        )
        for index, frame_births in enumerate(callback.births, callback_frame):
            births[index].extend(frame_births)
        for index, frame_bodies in enumerate(
            callback.body_hazards, callback_frame
        ):
            bodies[index].extend(frame_bodies)
        branch_coverage = callback_frame + callback.covered_frames
        if branch_coverage < covered_frames:
            covered_frames = branch_coverage
            reason = callback.reason

    return EclForecast(
        tuple(tuple(frame) for frame in births),
        covered_frames,
        reason if covered_frames < horizon else "",
        body_hazards=tuple(tuple(frame) for frame in bodies),
    )


def _forecast_ecl_births_with_life_callbacks(
    spawner: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    difficulty: int,
    rank: int,
    bullet_sizes: tuple[tuple[float, float], ...],
    frame_multiplier: float = 1.0,
    rng: RngState | None = None,
    allow_player_variables: bool = True,
    radial_births: bool = False,
    abstract_rng: bool = False,
    enemy_kill_all_is_noop: bool = False,
    model_player_damage: bool = True,
) -> EclForecast:
    """Forecast an emitter, branching over reachable hard life callbacks."""
    horizon = len(player_positions)
    callback_damage = (70 if spawner.damageable else 0) + (
        10 if spawner.collidable and not spawner.is_boss else 0
    )
    callback_gap = spawner.life - spawner.life_callback_threshold
    earliest_callback = (
        max(0, callback_gap // callback_damage + 1)
        if callback_damage > 0 and spawner.life_callback_threshold >= 0
        else horizon
    )
    should_branch = (
        model_player_damage
        and abstract_rng
        and spawner.interactable
        and spawner.life_callback_threshold >= 0
        and 0 <= spawner.life_callback_sub < len(spawner.ecl_subroutines)
        and earliest_callback < horizon
    )
    if not should_branch:
        return _forecast_ecl_births_with_death_callbacks(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            rng,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage,
        )

    program = _compiled_program(spawner.ecl_program)
    callback_address = spawner.ecl_subroutines[spawner.life_callback_sub]
    callback_instruction = program.get(callback_address)
    if callback_instruction is None:
        return EclForecast(
            tuple(() for _ in player_positions),
            0,
            "life callback instruction graph is not captured",
        )

    no_callback_spawner = _copy_spawner(
        spawner,
        life_callback_threshold=-1,
    )
    no_callback = _forecast_ecl_births_with_death_callbacks(
        no_callback_spawner,
        player_positions,
        difficulty,
        rank,
        bullet_sizes,
        frame_multiplier,
        None,
        allow_player_variables,
        radial_births,
        abstract_rng,
        enemy_kill_all_is_noop,
        model_player_damage,
    )
    births = [list(frame) for frame in no_callback.births]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    for index, frame_bodies in enumerate(no_callback.body_hazards):
        bodies[index].extend(frame_bodies)
    covered_frames = no_callback.covered_frames
    reason = no_callback.reason

    for callback_frame in range(earliest_callback, horizon):
        if callback_frame:
            prefix = _forecast_ecl_births_with_death_callbacks(
                no_callback_spawner,
                player_positions[:callback_frame],
                difficulty,
                rank,
                bullet_sizes,
                frame_multiplier,
                None,
                allow_player_variables,
                radial_births,
                abstract_rng,
                enemy_kill_all_is_noop,
                model_player_damage,
            )
            if prefix.covered_frames < callback_frame or prefix.next_spawner is None:
                branch_coverage = prefix.covered_frames
                if branch_coverage < covered_frames:
                    covered_frames = branch_coverage
                    reason = prefix.reason
                continue
            callback_source = prefix.next_spawner
        else:
            callback_source = no_callback_spawner
        callback_source = _copy_spawner(
            callback_source,
            life=spawner.life_callback_threshold,
            life_callback_threshold=-1,
            next_instruction=callback_instruction,
            ecl_time=0,
            ecl_time_float=0.0,
            ecl_stack=(),
            timer_callback_sub=callback_source.death_callback_sub,
            bullet_rank_speed_low=-0.5,
            bullet_rank_speed_high=0.5,
            bullet_rank_amount1_low=0,
            bullet_rank_amount1_high=0,
            bullet_rank_amount2_low=0,
            bullet_rank_amount2_high=0,
        )
        callback = _forecast_ecl_births_with_death_callbacks(
            callback_source,
            player_positions[callback_frame:],
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            None,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage,
        )
        for index, frame_births in enumerate(callback.births, callback_frame):
            births[index].extend(frame_births)
        for index, frame_bodies in enumerate(callback.body_hazards, callback_frame):
            bodies[index].extend(frame_bodies)
        branch_coverage = callback_frame + callback.covered_frames
        if branch_coverage < covered_frames:
            covered_frames = branch_coverage
            reason = callback.reason

    return EclForecast(
        tuple(tuple(frame) for frame in births),
        covered_frames,
        reason if covered_frames < horizon else "",
        body_hazards=tuple(tuple(frame) for frame in bodies),
    )


def _life_callback_can_branch(
    spawner: EnemySpawner,
    horizon: int,
    abstract_rng: bool,
) -> bool:
    callback_damage = (70 if spawner.damageable else 0) + (
        10 if spawner.collidable and not spawner.is_boss else 0
    )
    callback_gap = spawner.life - spawner.life_callback_threshold
    earliest_callback = (
        max(0, callback_gap // callback_damage + 1)
        if callback_damage > 0 and spawner.life_callback_threshold >= 0
        else horizon
    )
    return (
        abstract_rng
        and spawner.interactable
        and spawner.life_callback_threshold >= 0
        and 0 <= spawner.life_callback_sub < len(spawner.ecl_subroutines)
        and earliest_callback < horizon
    )


def _death_callback_can_branch(
    spawner: EnemySpawner,
    horizon: int,
    abstract_rng: bool,
) -> bool:
    callback_damage = (70 if spawner.damageable else 0) + (
        10 if spawner.collidable and not spawner.is_boss else 0
    )
    earliest_callback = (
        max(0, (spawner.life + callback_damage - 1) // callback_damage)
        if callback_damage > 0 and spawner.life > 0
        else (0 if spawner.life <= 0 else horizon)
    )
    return (
        abstract_rng
        and spawner.interactable
        and 0 <= spawner.death_callback_sub < len(spawner.ecl_subroutines)
        and earliest_callback < horizon
    )


def _forecast_abstract_integer_domains(
    spawner: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    difficulty: int,
    rank: int,
    bullet_sizes: tuple[tuple[float, float], ...],
    frame_multiplier: float,
    allow_player_variables: bool,
    radial_births: bool,
    enemy_kill_all_is_noop: bool,
) -> EclForecast:
    """Union every bounded source integer-RNG control-flow outcome."""
    pending: list[tuple[int, ...]] = [()]
    leaves: list[EclForecast] = []
    evaluated = 0
    while pending:
        choices = pending.pop()
        evaluated += 1
        if evaluated > MAX_ABSTRACT_INTEGER_RNG_EVALUATIONS:
            return EclForecast(
                tuple(() for _ in player_positions),
                0,
                "integer RNG branch budget exhausted",
            )
        forecast = _forecast_ecl_births_single(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            None,
            allow_player_variables,
            radial_births,
            True,
            enemy_kill_all_is_noop,
            choices,
        )
        extent = forecast.unresolved_int_extent
        if extent:
            future_branch_count = len(pending) + len(leaves) + extent
            if future_branch_count > MAX_ABSTRACT_INTEGER_RNG_BRANCHES:
                return EclForecast(
                    tuple(() for _ in player_positions),
                    0,
                    f"integer RNG domain {extent} exceeds branch budget",
                )
            pending.extend(choices + (value,) for value in range(extent))
        else:
            leaves.append(forecast)

    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    body_seen = [set() for _ in player_positions]
    for index in range(len(player_positions)):
        maximum_counts: Counter[Bullet] = Counter()
        for forecast in leaves:
            branch_counts = Counter(forecast.births[index])
            maximum_counts |= branch_counts
        for bullet, count in maximum_counts.items():
            births[index].extend((bullet,) * count)
    for forecast in leaves:
        for index, frame_bodies in enumerate(forecast.body_hazards):
            for body in frame_bodies:
                if body not in body_seen[index]:
                    body_seen[index].add(body)
                    bodies[index].append(body)
    covered_frames = min(
        (forecast.covered_frames for forecast in leaves),
        default=0,
    )
    reason = next(
        (
            forecast.reason for forecast in leaves
            if forecast.covered_frames == covered_frames
            and covered_frames < len(player_positions)
        ),
        "",
    )
    first_next = leaves[0].next_spawner if leaves else None
    common_next = (
        first_next
        if all(forecast.next_spawner == first_next for forecast in leaves)
        else None
    )
    return EclForecast(
        tuple(tuple(frame) for frame in births),
        covered_frames,
        reason,
        next_spawner=common_next,
        body_hazards=tuple(tuple(frame) for frame in bodies),
        finished=bool(leaves) and all(forecast.finished for forecast in leaves),
    )


def forecast_ecl_births(
    spawner: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    difficulty: int,
    rank: int,
    bullet_sizes: tuple[tuple[float, float], ...],
    frame_multiplier: float = 1.0,
    rng: RngState | None = None,
    allow_player_variables: bool = True,
    radial_births: bool = False,
    abstract_rng: bool = False,
    enemy_kill_all_is_noop: bool = False,
    model_player_damage: bool = True,
    record_enemy_kill_all: bool = False,
    laser_world=None,
    spawn_inline: bool = False,
) -> EclForecast:
    """Forecast one emitter and preserve every bounded hard uncertainty."""
    if spawn_inline:
        if len(player_positions) != 1:
            raise ValueError("SpawnEnemy inline ECL is exactly one source call")
        return _forecast_ecl_births_single(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            rng,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage=model_player_damage,
            record_enemy_kill_all=record_enemy_kill_all,
            laser_world=laser_world,
            spawn_inline=True,
        )
    if record_enemy_kill_all:
        if abstract_rng or model_player_damage or rng is None:
            raise ValueError(
                "world ENEMYKILLALL recording requires exact nominal state"
            )
        if laser_world is not None and len(player_positions) != 1:
            raise ValueError("exact nominal laser world advances one frame")
        return _forecast_ecl_births_single(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            rng,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage=model_player_damage,
            record_enemy_kill_all=True,
            laser_world=laser_world,
        )
    if laser_world is not None:
        if len(player_positions) != 1:
            raise ValueError("a mutable laser world advances one source frame")
        if (
            _life_callback_can_branch(spawner, 1, abstract_rng)
            or _death_callback_can_branch(spawner, 1, abstract_rng)
        ):
            return EclForecast(
                ((),),
                0,
                "future laser world branches on candidate player damage",
            )
        # A mutable pool cannot be unioned after the fact. Run the ordinary
        # one-frame interpreter directly; an unresolved integer/RNG branch
        # remains an explicit fail-closed boundary.
        return _forecast_ecl_births_single(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            rng,
            allow_player_variables,
            radial_births,
            abstract_rng,
            enemy_kill_all_is_noop,
            model_player_damage=model_player_damage,
            laser_world=laser_world,
        )
    if (
        abstract_rng
        and rng is None
        and not (
            _life_callback_can_branch(
                spawner,
                len(player_positions),
                abstract_rng,
            )
            or _death_callback_can_branch(
                spawner,
                len(player_positions),
                abstract_rng,
            )
        )
    ):
        return _forecast_abstract_integer_domains(
            spawner,
            player_positions,
            difficulty,
            rank,
            bullet_sizes,
            frame_multiplier,
            allow_player_variables,
            radial_births,
            enemy_kill_all_is_noop,
        )
    return _forecast_ecl_births_with_life_callbacks(
        spawner,
        player_positions,
        difficulty,
        rank,
        bullet_sizes,
        frame_multiplier,
        rng,
        allow_player_variables,
        radial_births,
        abstract_rng,
        enemy_kill_all_is_noop,
        model_player_damage,
    )
