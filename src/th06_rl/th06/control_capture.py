"""Small frame-coherent TH06 capture for the resident safety loop.

The authoritative retail snapshot is deliberately exhaustive: it retains
player attacks, items, effects, immutable ECL graphs, sprite geometry, and
callback state for source replay.  Decoding all of that before every input
publication made capture, rather than the native Hard gate, the hot path.

This module keeps a compact collision decode and, in the same paused epoch,
pairs it with the exhaustive source root. Hazard authority consumes only the
bounded source projection; offline-only attacks, items, effects, and resource
counters are attached to the dense root without entering the Hard gate.
"""

from __future__ import annotations

from contextlib import nullcontext
import ctypes
from dataclasses import dataclass, replace
import math
import struct
import time

from ..retail import native
from ..retail.model import (
    Bullet,
    EnemyBody,
    ItemState,
    Laser,
    PlayerAttackState,
    PlayerShot,
    StageTimelineInstruction,
)


CONTROL_CAPTURE_TIER = "control-v4"
SOURCE_RECORD_SCHEMA = "th06-1.02h-source-records-v2"
OFFLINE_FACT_SCHEMA = "th06-1.02h-offline-facts-v2"
MAX_CAPTURE_ATTEMPTS = 8
DYNAMIC_BULLET_FLAGS = 0xDF1


def _read_bulk_view(process, address: int, size: int) -> memoryview:
    """Read the hot manager interval into one process-owned reusable buffer.

    The retail adapter's general-purpose ``read`` API returns an owning
    ``bytes`` object, but that means allocating and zeroing a new ~2 MiB
    ctypes buffer and then copying it into another ~2 MiB Python object every
    controlled frame.  This capture consumes the interval synchronously, so
    one exact-PID buffer can safely be reused until the next snapshot.

    Tests and non-native readers retain the ordinary API as a strict fallback.
    """
    kernel32 = getattr(process, "kernel32", None)
    handle = getattr(process, "handle", None)
    if kernel32 is None or handle is None:
        return memoryview(process.read(address, size))
    buffer = getattr(process, "_th06_rl_control_pool_buffer", None)
    if buffer is None or ctypes.sizeof(buffer) != size:
        buffer = ctypes.create_string_buffer(size)
        process._th06_rl_control_pool_buffer = buffer
    count = ctypes.c_size_t()
    if not kernel32.ReadProcessMemory(
        handle,
        ctypes.c_void_p(address),
        buffer,
        size,
        ctypes.byref(count),
    ) or count.value != size:
        raise ctypes.WinError(ctypes.get_last_error())
    return memoryview(buffer).cast("B")[:size]


def _completed_calc_lag(
    known_stage: int | None,
    known_lag: int | None,
    *,
    stage: int,
    game_frame: int,
    bullet_time: int,
    passive: bool,
) -> tuple[int, int | None, bool]:
    """Track the source-defined completed-chain clock relationship.

    GameManager::OnUpdate advances ``gameFrames`` at priority 4.  The
    BulletManager advances its timer at priority 11, except that it returns
    without ticking while ``isTimeStopped`` is set.  Consequently equality is
    a valid initial-stage witness, but not a valid invariant after a pause.
    The completed-chain lag is stable during active play; the priority-4..10
    tear is exactly one larger.  A passive sample is safe to use as the new
    baseline because battle movement is not published from it.
    """
    if game_frame < 0 or bullet_time < 0 or bullet_time > game_frame:
        raise ValueError(
            f"invalid control clocks: game={game_frame}, bullet={bullet_time}"
        )
    lag = game_frame - bullet_time
    if known_stage != stage:
        known_lag = None
    if passive:
        return stage, lag, True
    if known_lag is None:
        # Both source timers are initialized to zero for a fresh stage.  Do
        # not guess a non-zero baseline from an arbitrary active-frame attach.
        if lag != 0:
            return stage, None, False
        return stage, 0, True
    return stage, known_lag, lag == known_lag


def _accept_completed_calc_phase(
    process,
    native,
    *,
    stage: int,
    game_frame: int,
    bullet_time: int,
    passive: bool,
) -> None:
    known_stage = getattr(process, "_th06_rl_calc_stage", None)
    known_lag = getattr(process, "_th06_rl_completed_calc_lag", None)
    next_stage, next_lag, complete = _completed_calc_lag(
        known_stage,
        known_lag,
        stage=stage,
        game_frame=game_frame,
        bullet_time=bullet_time,
        passive=passive,
    )
    if not complete:
        raise native._SnapshotPhaseIncomplete(game_frame, bullet_time)
    process._th06_rl_calc_stage = next_stage
    process._th06_rl_completed_calc_lag = next_lag


@dataclass(frozen=True)
class ControlSnapshot:
    capture_tier: str
    frame: int
    stage: int
    player_state: int
    x: float
    y: float
    half_width: float
    half_height: float
    normal_speed: float
    focus_speed: float
    normal_diagonal_speed: float
    focus_diagonal_speed: float
    frame_multiplier: float
    input_mask: int
    # Resident roots retain only the conservative reachable subset as Python
    # objects. ``raw_bullet_tails`` retains every occupied slot's collision
    # and motion tail; source records appended below retain the dynamic state
    # intentionally omitted from the low-latency Python decode.
    bullets: tuple[object, ...]
    live_bullet_count: int
    raw_bullet_tails: bytes
    # (AnmLoadedSprite pointer, width, height).  Boundary reflection uses
    # visual sprite geometry rather than the much smaller collision box, so
    # the pointer alone is not sufficient for source-exact offline decode.
    bullet_sprite_dimensions: tuple[tuple[int, float, float], ...]
    bullets_are_reachable_subset: bool
    laser_count: int
    in_menu: bool
    time_stopped: bool
    replay_or_demo: bool
    lasers: tuple[object, ...]
    enemies: tuple[object, ...]
    difficulty: int
    character: int
    shot_type: int
    bomb_active: bool
    spell_active: bool
    rank: int
    subrank: int
    max_rank: int
    min_rank: int
    rng_seed: int
    rng_generation: int
    current_power: int
    lives_remaining: int
    source_context: str
    boss_life: int | None
    timeline_time: int
    timeline_time_float: float
    capture_attempts: int
    bullet_read_retries: int
    # Cheap factual linkage for auditing the resident prefilter.  The corpus
    # intentionally avoids duplicating full Bullet rows.
    reachable_bullet_slots: tuple[int, ...] = ()
    # Packed records are ``u16 slot`` followed by the exact retail struct.
    # Spawn bullets retain their complete ANM VM because its completion tick
    # controls the state-2/3/4 -> fired fallthrough. Every occupied Enemy and
    # Laser is retained so offline source analysis has ECL stacks, callbacks,
    # shooter state, laser ownership, bounds, and raw animation state.
    raw_spawn_bullet_records: bytes = b""
    raw_enemy_records: bytes = b""
    raw_laser_records: bytes = b""
    # Exact EnemyManager bytes after the 256-slot pool: timeline instruction,
    # timers, boss pointer table, random-item cursors, and spell state.
    raw_enemy_manager_tail: bytes = b""
    source_record_schema: str = ""
    # Offline-only factual state hydrated from the exhaustive root captured in
    # the same pause. None of these fields participates in online Hard.
    factual_state_schema: str = ""
    player_attack: PlayerAttackState | None = None
    item_states: tuple[ItemState, ...] = ()
    item_next_index: int = 0
    effect_active_upper_bound: int = -1
    item_active_upper_bound: int = -1
    pending_effect_rng_ids: tuple[int, ...] = ()
    random_item_spawn_index: int = 0
    random_item_table_index: int = 0
    gui_score: int = 0
    score: int = 0
    next_score_increment: int = 0
    high_score: int = 0
    graze_in_stage: int = 0
    graze_total: int = 0
    deaths: int = 0
    bombs_used: int = 0
    spellcards_captured: int = 0
    point_items_collected_in_stage: int = 0
    point_items_collected: int = 0
    retries: int = 0
    power_item_count_for_score: int = 0
    bombs_remaining: int = 0
    extra_lives: int = 0
    # Per-frame visual geometry for every occupied Enemy ANM VM. Enemy bounds
    # retirement uses this geometry, which cannot be recovered from the raw
    # record's pointer after Wine exits.
    enemy_sprite_dimensions: tuple[tuple[int, float, float], ...] = ()
    # Offline source replay facts copied from the exhaustive root captured in
    # the exact same process pause. They are not consumed by online Hard.
    ecl_ex_function_addresses: tuple[int, ...] = ()
    timeline_current_message_waits: int = 0
    message_active: bool = False
    timeline_boss_slots: tuple[int, ...] = ()
    timeline_time_previous: int | None = None
    boss_present: bool | None = None


def _finite(*values: float) -> bool:
    return all(math.isfinite(value) for value in values)


def _tail_may_reach_player(
    tail: bytes,
    native,
    *,
    player_x: float,
    player_y: float,
    player_half_width: float,
    player_half_height: float,
    player_speed: float,
    horizon: int,
    collision_margin: float,
) -> bool:
    """Cheap scalar equivalent of the retained conservative sweep reject."""
    relative = lambda absolute: absolute - native.BULLET_SIZE_OFFSET
    size_x, size_y = struct.unpack_from("<ff", tail, 0)
    x, y = struct.unpack_from(
        "<ff", tail, relative(native.BULLET_POSITION_OFFSET)
    )
    vx, vy = struct.unpack_from(
        "<ff", tail, relative(native.BULLET_VELOCITY_OFFSET)
    )
    if (
        not _finite(size_x, size_y, x, y, vx, vy)
        or not 0.0 < size_x <= 256.0
        or not 0.0 < size_y <= 256.0
    ):
        # Force the exact decoder to reject incoherent state rather than
        # allowing NaN comparison semantics to turn it into a far-away skip.
        return True
    flags = struct.unpack_from(
        "<H", tail, relative(native.BULLET_EX_FLAGS_OFFSET)
    )[0]
    state = struct.unpack_from(
        "<H", tail, relative(native.BULLET_STATE_OFFSET)
    )[0]
    if flags & DYNAMIC_BULLET_FLAGS:
        ax, ay = struct.unpack_from(
            "<ff", tail, relative(native.BULLET_ACCELERATION_OFFSET)
        )
        speed, curve_accel, turn_speed = struct.unpack_from(
            "<fff", tail, relative(native.BULLET_SPEED_OFFSET)
        )
        base_speed = max(math.hypot(vx, vy), abs(speed), abs(turn_speed))
        acceleration = math.hypot(ax, ay) + abs(curve_accel)
        if not _finite(ax, ay, speed, curve_accel, turn_speed, acceleration):
            return True
        reach_x = reach_y = (
            (base_speed + 5.0) * horizon
            + acceleration * horizon * (horizon + 1) / 2.0
        )
    else:
        reach_frames = horizon + (state in (2, 3, 4))
        reach_x = abs(vx) * reach_frames
        reach_y = abs(vy) * reach_frames
    margin = max(0.0, collision_margin)
    player_left = (
        max(8.0, player_x - player_speed * horizon)
        - player_half_width
        - margin
    )
    player_right = (
        min(376.0, player_x + player_speed * horizon)
        + player_half_width
        + margin
    )
    player_top = (
        max(16.0, player_y - player_speed * horizon)
        - player_half_height
        - margin
    )
    player_bottom = (
        min(432.0, player_y + player_speed * horizon)
        + player_half_height
        + margin
    )
    return not (
        x + size_x / 2.0 + reach_x < player_left
        or x - size_x / 2.0 - reach_x > player_right
        or y + size_y / 2.0 + reach_y < player_top
        or y - size_y / 2.0 - reach_y > player_bottom
    )


def _ensure_subroutines(process, stage: int, native) -> tuple[int, ...]:
    """Load immutable stage subroutine addresses without decoding programs."""
    if getattr(process, "_th06_rl_control_ecl_stage", None) != stage:
        process.ecl_subroutines = native._read_ecl_subroutines(process)
        process._th06_rl_control_ecl_stage = stage
    return process.ecl_subroutines


def _control_sprite_dimensions(
    process,
    native,
    stage: int,
    pointers: set[int],
) -> dict[int, tuple[float, float]]:
    """Resolve immutable visual geometry once per loaded stage resource."""
    if getattr(process, "_th06_rl_control_sprite_stage", None) != stage:
        process._th06_rl_control_sprite_stage = stage
        process._th06_rl_control_sprite_dimensions = {}
    cached = process._th06_rl_control_sprite_dimensions
    missing = pointers - set(cached)
    if missing:
        cached.update(native._read_sprite_dimensions(process, missing))
    return {pointer: cached[pointer] for pointer in pointers}


def _source_context(
    process,
    native,
    enemy_pool: bytes,
    manager_bytes: bytes,
    manager_relative,
    stage: int,
    spell_active: bool,
) -> tuple[str, int | None]:
    subroutines = _ensure_subroutines(process, stage, native)
    bosses: list[tuple[int, int, int, int, int, int]] = []
    boss_life: int | None = None
    for slot in range(native.ENEMY_COUNT):
        base = slot * native.ENEMY_STRIDE
        flags0, flags1 = struct.unpack_from(
            "<BB", enemy_pool, base + native.ENEMY_FLAGS_OFFSET
        )
        if not flags0 & 0x80 or not flags1 & 0x08:
            continue
        boss_id = enemy_pool[base + native.ENEMY_BOSS_ID_OFFSET]
        instruction = struct.unpack_from(
            "<I", enemy_pool, base + native.ENEMY_ECL_CONTEXT_OFFSET
        )[0]
        life = struct.unpack_from(
            "<i", enemy_pool, base + native.ENEMY_LIFE_OFFSET
        )[0]
        _life_threshold, life_callback, _timer_threshold, timer_callback = (
            struct.unpack_from(
                "<iiii",
                enemy_pool,
                base + native.ENEMY_LIFE_CALLBACK_THRESHOLD_OFFSET,
            )
        )
        bosses.append((
            boss_id,
            slot,
            instruction,
            life_callback,
            timer_callback,
            life,
        ))
    if bosses:
        boss_id, slot, instruction, life_callback, timer_callback, life = min(
            bosses, key=lambda item: (item[0], item[1])
        )
        boss_life = life
        containing = [
            (address, index)
            for index, address in enumerate(subroutines)
            if instruction and address <= instruction
        ]
        subroutine = str(max(containing)[1]) if containing else "unknown"
        return ":".join((
            "boss",
            str(boss_id),
            f"sub{subroutine}",
            f"life_cb{life_callback}",
            f"timer_cb{timer_callback}",
            "spell" if spell_active else "nonspell",
        )), boss_life

    address = struct.unpack_from(
        "<I",
        manager_bytes,
        manager_relative(native.ENEMY_TIMELINE_INSTRUCTION_OFFSET),
    )[0]
    if not address:
        return "timeline-unknown", None
    if not 0x10000 <= address < 0x80000000:
        raise RuntimeError(f"invalid ECL timeline pointer 0x{address:08X}")
    cache = process.ecl_timeline_instruction_cache
    instruction = cache.get(address)
    if instruction is None:
        header = process.read(address, 8)
        time_value, arg0, opcode, size = struct.unpack("<hhhh", header)
        if time_value >= 0 and not 0x08 <= size <= 0x100:
            raise RuntimeError(
                f"invalid ECL timeline instruction size {size} "
                f"at 0x{address:08X}"
            )
        instruction = StageTimelineInstruction(
            address, time_value, arg0, opcode, size, header.hex()
        )
        cache[address] = instruction
    if instruction.time < 0:
        return "timeline-complete", None
    return (
        f"timeline:before-t{instruction.time}:"
        f"op{instruction.opcode}:arg{instruction.arg0}"
    ), None


def _decode_control_once(
    process,
    native,
    attempt: int,
    horizon: int,
    collision_margin: float,
) -> ControlSnapshot:
    # EnemyManager, BulletManager, the 640 bullet slots, and 64 laser slots
    # occupy one mapped interval in TH06 1.02h.  One bulk copy plus the source
    # timer witness is both faster and more coherent than hundreds of RPMs.
    bullet_time_before = struct.unpack(
        "<i",
        process.read(
            native.ADDR_BULLET_MANAGER + native.BULLET_MANAGER_TIME_OFFSET + 8,
            4,
        ),
    )[0]
    pool_start = native.ADDR_ENEMY_MANAGER + native.ENEMY_ARRAY_OFFSET
    pool_end = native.ADDR_BULLET_MANAGER + native.BULLET_MANAGER_SIZE
    manager_bytes = _read_bulk_view(process, pool_start, pool_end - pool_start)
    enemy_pool = manager_bytes[: native.ENEMY_COUNT * native.ENEMY_STRIDE]
    bullet_offset = native.ADDR_BULLET_ARRAY - pool_start
    bullet_pool = manager_bytes[bullet_offset:]
    manager_relative = lambda absolute: native.ADDR_ENEMY_MANAGER + absolute - pool_start
    bullet_manager_relative = native.ADDR_BULLET_MANAGER - pool_start
    _previous, bullet_subframe, bullet_time = struct.unpack_from(
        "<ifi",
        manager_bytes,
        bullet_manager_relative + native.BULLET_MANAGER_TIME_OFFSET,
    )
    if (
        bullet_time_before != bullet_time
        or not math.isfinite(bullet_subframe)
        or not 0.0 <= bullet_subframe < 1.0
    ):
        raise native._SnapshotReadTorn(
            "bullet-manager phase changed during control pool copy"
        )

    game_start = native.GAME_GUI_SCORE_OFFSET
    game_end = native.GAME_SUBRANK_OFFSET + 4
    game = process.read(
        native.ADDR_GAME_MANAGER + game_start,
        game_end - game_start,
    )
    relative_game = lambda absolute: absolute - game_start
    frame = struct.unpack_from("<I", game, relative_game(native.GAME_FRAMES_OFFSET))[0]
    stage = struct.unpack_from("<i", game, relative_game(native.GAME_STAGE_OFFSET))[0]
    difficulty = struct.unpack_from(
        "<i", game, relative_game(native.GAME_DIFFICULTY_OFFSET)
    )[0]
    rank, max_rank, min_rank, subrank = struct.unpack_from(
        "<iiii", game, relative_game(native.GAME_RANK_OFFSET)
    )
    current_power = struct.unpack_from(
        "<H", game, relative_game(native.GAME_CURRENT_POWER_OFFSET)
    )[0]
    lives_remaining = struct.unpack_from(
        "<b", game, relative_game(native.GAME_LIVES_REMAINING_OFFSET)
    )[0]
    character = game[relative_game(native.GAME_CHARACTER_OFFSET)]
    shot_type = game[relative_game(native.GAME_SHOT_TYPE_OFFSET)]
    gui_score, score, next_score_increment, high_score = struct.unpack_from(
        "<IIII", game, relative_game(native.GAME_GUI_SCORE_OFFSET)
    )
    graze_in_stage, graze_total = struct.unpack_from(
        "<ii", game, relative_game(native.GAME_GRAZE_IN_STAGE_OFFSET)
    )
    deaths, bombs_used, spellcards_captured = struct.unpack_from(
        "<iii", game, relative_game(native.GAME_DEATHS_OFFSET)
    )
    point_items_collected_in_stage, point_items_collected = struct.unpack_from(
        "<HH", game, relative_game(native.GAME_POINT_ITEMS_STAGE_OFFSET)
    )
    retries, power_item_count_for_score, bombs_remaining, extra_lives = (
        struct.unpack_from(
            "<Bbbb", game, relative_game(native.GAME_RETRIES_OFFSET)
        )
    )
    flags = relative_game(native.GAME_FLAGS_OFFSET)
    game_menu, retry_menu, gameplay_active, _complete, _practice, demo_mode = (
        game[flags : flags + 6]
    )
    in_menu = bool(game_menu or retry_menu or not gameplay_active or demo_mode)
    time_stopped = bool(game[relative_game(native.GAME_TIME_STOPPED_OFFSET)])
    replay = bool(struct.unpack_from("<I", game, relative_game(0x1C))[0])
    if not (
        difficulty in (0, 1, 2, 3, 4)
        and character in (0, 1)
        and shot_type in (0, 1)
        and 0 <= lives_remaining <= 8
        and min_rank <= rank <= max_rank
        and 0 <= subrank < 100
    ):
        raise RuntimeError("invalid compact GameManager state")

    rng_seed, rng_generation = struct.unpack("<HxxI", process.read(native.ADDR_RNG, 8))
    frame_multiplier = struct.unpack(
        "<f", process.read(native.ADDR_FRAME_MULTIPLIER, 4)
    )[0]
    input_mask = struct.unpack("<H", process.read(native.ADDR_CURRENT_INPUT, 2))[0]
    player_start = native.PLAYER_POSITION_OFFSET
    player = process.read(
        native.ADDR_PLAYER + player_start,
        native.PLAYER_BOMB_ACTIVE_OFFSET + 4 - player_start,
    )
    relative_player = lambda absolute: absolute - player_start
    x, y = struct.unpack_from(
        "<ff", player, relative_player(native.PLAYER_POSITION_OFFSET)
    )
    hit_left, hit_top = struct.unpack_from(
        "<ff", player, relative_player(native.PLAYER_HITBOX_TOP_LEFT_OFFSET)
    )
    hit_right, hit_bottom = struct.unpack_from(
        "<ff", player, relative_player(native.PLAYER_HITBOX_BOTTOM_RIGHT_OFFSET)
    )
    player_state = player[relative_player(native.PLAYER_STATE_OFFSET)]
    normal_speed, focus_speed, normal_diagonal, focus_diagonal = struct.unpack_from(
        "<ffff", player, relative_player(native.PLAYER_SPEEDS_OFFSET)
    )
    bomb_active = bool(struct.unpack_from(
        "<I", player, relative_player(native.PLAYER_BOMB_ACTIVE_OFFSET)
    )[0])
    half_width = max(0.0, (hit_right - hit_left) / 2.0)
    half_height = max(0.0, (hit_bottom - hit_top) / 2.0)
    if not _finite(
        x, y, hit_left, hit_top, hit_right, hit_bottom,
        normal_speed, focus_speed, normal_diagonal, focus_diagonal,
        frame_multiplier,
    ):
        raise RuntimeError("non-finite compact player state")
    active_geometry = (
        0.5 <= half_width <= 8.0
        and 0.5 <= half_height <= 8.0
        and 0.5 <= focus_speed <= 8.0
        and 0.5 <= normal_speed <= 8.0
        and -64.0 <= x <= 448.0
        and -64.0 <= y <= 512.0
    )
    if player_state in (0, 3) and not active_geometry:
        in_menu = True

    spell_active = bool(struct.unpack_from(
        "<i",
        manager_bytes,
        manager_relative(native.ENEMY_SPELL_ACTIVE_OFFSET),
    )[0])
    timeline_previous, timeline_subframe, timeline_time = struct.unpack_from(
        "<ifi",
        manager_bytes,
        manager_relative(native.ENEMY_TIMELINE_TIMER_OFFSET),
    )
    if not _finite(timeline_subframe) or not 0.0 <= timeline_subframe < 1.0:
        raise RuntimeError("invalid compact timeline timer")

    bullets = []
    live_bullet_count = 0
    raw_bullet_tails = bytearray()
    raw_spawn_bullet_records = bytearray()
    occupied_bullets = []
    player_speed = max(
        abs(normal_speed),
        abs(focus_speed),
        abs(normal_diagonal),
        abs(focus_diagonal),
    )
    for slot in range(native.BULLET_COUNT):
        base = slot * native.BULLET_STRIDE
        tail = bytes(
            bullet_pool[
                base + native.BULLET_SIZE_OFFSET : base + native.BULLET_STRIDE
            ]
        )
        state = struct.unpack_from(
            "<H",
            tail,
            native.BULLET_STATE_OFFSET - native.BULLET_SIZE_OFFSET,
        )[0]
        if state == 0:
            continue
        live_bullet_count += state != 5
        sprite_pointer = struct.unpack_from(
            "<I", bullet_pool, base + native.ANM_VM_SPRITE_OFFSET
        )[0]
        raw_bullet_tails.extend(struct.pack("<HI", slot, sprite_pointer))
        raw_bullet_tails.extend(tail)
        if state in (2, 3, 4):
            raw_spawn_bullet_records.extend(struct.pack("<H", slot))
            raw_spawn_bullet_records.extend(
                bullet_pool[base : base + native.BULLET_STRIDE]
            )
        occupied_bullets.append((slot, state, sprite_pointer, tail))

    # BulletManager::SpawnBullet sets state=FIRED before copying the template
    # AnmVm and before SetActiveSprite (authoritative BulletManager.cpp
    # lines 171-180, 286-287). Suspending inside that short function can
    # therefore expose an active slot with a null/transient sprite pointer.
    # It is not a decodable hazard root; resume and retry the entire snapshot.
    sprite_pointers = {item[2] for item in occupied_bullets}
    enemy_sprite_pointers = {
        struct.unpack_from(
            "<I",
            enemy_pool,
            slot * native.ENEMY_STRIDE + native.ANM_VM_SPRITE_OFFSET,
        )[0]
        for slot in range(native.ENEMY_COUNT)
        if enemy_pool[slot * native.ENEMY_STRIDE + native.ENEMY_FLAGS_OFFSET]
        & 0x80
    }
    if 0 in sprite_pointers:
        raise native._SnapshotReadTorn(
            "active bullet is between state publication and sprite setup"
        )
    try:
        source_sprite_dimensions = _control_sprite_dimensions(
            process,
            native,
            stage,
            sprite_pointers | enemy_sprite_pointers,
        )
    except RuntimeError as error:
        raise native._SnapshotReadTorn(str(error)) from error
    for slot, state, sprite_pointer, tail in occupied_bullets:
        if state == 5 or not _tail_may_reach_player(
            tail,
            native,
            player_x=x,
            player_y=y,
            player_half_width=half_width,
            player_half_height=half_height,
            player_speed=player_speed,
            horizon=horizon,
            collision_margin=collision_margin,
        ):
            continue
        bullet = native._decode_bullet_tail(
            tail,
            slot,
            source_sprite_dimensions[sprite_pointer],
        )
        if bullet is not None:
            bullets.append(bullet)

    lasers = []
    raw_laser_records = bytearray()
    laser_base = native.ADDR_LASER_ARRAY - native.ADDR_BULLET_ARRAY
    for slot in range(native.LASER_COUNT):
        base = laser_base + slot * native.LASER_STRIDE
        if not struct.unpack_from(
            "<i", bullet_pool, base + native.LASER_IN_USE_OFFSET
        )[0]:
            continue
        raw_laser_records.extend(struct.pack("<H", slot))
        raw_laser_records.extend(
            bullet_pool[base : base + native.LASER_STRIDE]
        )
        lx, ly = struct.unpack_from(
            "<ff", bullet_pool, base + native.LASER_POSITION_OFFSET
        )
        angle, start, end, start_length, width, speed = struct.unpack_from(
            "<ffffff", bullet_pool, base + native.LASER_ANGLE_OFFSET
        )
        start_time, hitbox_start, duration, despawn, hitbox_end = struct.unpack_from(
            "<iiiii", bullet_pool, base + native.LASER_START_TIME_OFFSET
        )
        timer_subframe = struct.unpack_from(
            "<f", bullet_pool, base + native.LASER_TIMER_SUBFRAME_OFFSET
        )[0]
        timer = struct.unpack_from(
            "<i", bullet_pool, base + native.LASER_TIMER_OFFSET
        )[0]
        laser_flags = struct.unpack_from(
            "<H", bullet_pool, base + native.LASER_FLAGS_OFFSET
        )[0]
        state = bullet_pool[base + native.LASER_STATE_OFFSET]
        numbers = (lx, ly, angle, start, end, start_length, width, speed, timer_subframe)
        times = (start_time, hitbox_start, duration, despawn, hitbox_end, timer)
        if (
            not _finite(*numbers)
            or not 0.0 < width <= 1024.0
            or not 0.0 <= start_length <= 4096.0
            or not all(0 <= value < 10_000_000 for value in times)
            or state not in (0, 1, 2)
        ):
            raise RuntimeError(f"invalid compact laser state at slot {slot}")
        lasers.append(Laser(
            lx, ly, angle, start, end, start_length, width, speed,
            start_time, hitbox_start, duration, despawn, hitbox_end,
            timer, timer + timer_subframe, laser_flags, state, slot=slot,
        ))

    enemies = []
    raw_enemy_records = bytearray()
    for slot in range(native.ENEMY_COUNT):
        base = slot * native.ENEMY_STRIDE
        flags0, flags1, flags2 = struct.unpack_from(
            "<BBB", enemy_pool, base + native.ENEMY_FLAGS_OFFSET
        )
        if flags0 & 0x80:
            raw_enemy_records.extend(struct.pack("<H", slot))
            raw_enemy_records.extend(
                enemy_pool[base : base + native.ENEMY_STRIDE]
            )
        lethal = (
            flags0 & 0x80
            and flags1 & 0x01
            and flags1 & 0x02
            and flags1 & 0x04
            and not flags2 & 0x08
        )
        if not lethal:
            continue
        ex, ey = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_POSITION_OFFSET
        )
        vx, vy = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_AXIS_SPEED_OFFSET
        )
        angle, angular_velocity, speed, acceleration = struct.unpack_from(
            "<ffff", enemy_pool, base + native.ENEMY_ANGLE_OFFSET
        )
        interp_x, interp_y = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_MOVE_INTERP_OFFSET
        )
        start_x, start_y = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_MOVE_START_OFFSET
        )
        move_subframe = struct.unpack_from(
            "<f", enemy_pool, base + native.ENEMY_MOVE_TIMER_SUBFRAME_OFFSET
        )[0]
        move_timer = struct.unpack_from(
            "<i", enemy_pool, base + native.ENEMY_MOVE_TIMER_OFFSET
        )[0]
        move_start_time = struct.unpack_from(
            "<i", enemy_pool, base + native.ENEMY_MOVE_START_TIME_OFFSET
        )[0]
        hitbox_x, hitbox_y = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_HITBOX_OFFSET
        )
        lower_move_x, lower_move_y = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_LOWER_MOVE_LIMIT_OFFSET
        )
        upper_move_x, upper_move_y = struct.unpack_from(
            "<ff", enemy_pool, base + native.ENEMY_UPPER_MOVE_LIMIT_OFFSET
        )
        should_clamp_position = bool(flags2 & 0x01)
        movement_mode = flags0 & 0x03
        movement_ease = (flags0 >> 2) & 0x07
        if (
            not _finite(
                ex, ey, vx, vy, angle, angular_velocity, speed, acceleration,
                interp_x, interp_y, start_x, start_y, move_subframe,
                hitbox_x, hitbox_y, lower_move_x, lower_move_y,
                upper_move_x, upper_move_y,
            )
            or not 0.0 <= hitbox_x <= 1024.0
            or not 0.0 <= hitbox_y <= 1024.0
            or movement_mode == 2
            and (movement_ease > 4 or move_start_time <= 0 or move_timer < 0)
            or should_clamp_position and (
                lower_move_x > upper_move_x or lower_move_y > upper_move_y
            )
        ):
            raise RuntimeError(f"invalid compact enemy state at slot {slot}")
        enemies.append(EnemyBody(
            ex, ey, hitbox_x / 3.0, hitbox_y / 3.0,
            vx, vy, angle, angular_velocity, speed, acceleration,
            movement_mode, movement_ease, bool(flags0 & 0x40),
            interp_x, interp_y, start_x, start_y,
            move_timer, move_timer + move_subframe, move_start_time,
            should_clamp_position,
            lower_move_x, lower_move_y, upper_move_x, upper_move_y,
        ))

    source_context, boss_life = _source_context(
        process,
        native,
        enemy_pool,
        manager_bytes,
        manager_relative,
        stage,
        spell_active,
    )
    after = native.read_game_frame(process)
    if after != frame:
        raise native._SnapshotEpochChanged
    if frame_multiplier != 1.0:
        raise RuntimeError(
            "online safety requires exact normal-speed frame multiplier 1.0; "
            f"observed {frame_multiplier!r}"
        )
    _accept_completed_calc_phase(
        process,
        native,
        stage=stage,
        game_frame=frame,
        bullet_time=bullet_time,
        passive=bool(in_menu or time_stopped),
    )
    enemy_manager_tail_start = native.ENEMY_COUNT * native.ENEMY_STRIDE
    enemy_manager_tail_end = native.ENEMY_MANAGER_SIZE - native.ENEMY_ARRAY_OFFSET
    raw_enemy_manager_tail = bytes(
        manager_bytes[enemy_manager_tail_start:enemy_manager_tail_end]
    )
    if len(raw_enemy_manager_tail) != enemy_manager_tail_end - enemy_manager_tail_start:
        raise native._SnapshotReadTorn("truncated EnemyManager source tail")
    control = ControlSnapshot(
        CONTROL_CAPTURE_TIER,
        frame, stage, player_state, x, y, half_width, half_height,
        normal_speed, focus_speed, normal_diagonal, focus_diagonal,
        frame_multiplier, input_mask, tuple(bullets), live_bullet_count,
        bytes(raw_bullet_tails), tuple(
            (pointer, *source_sprite_dimensions[pointer])
            for pointer in sorted(sprite_pointers)
        ), True, len(lasers), in_menu,
        time_stopped, bool(replay or demo_mode), tuple(lasers), tuple(enemies),
        difficulty, character, shot_type, bomb_active, spell_active,
        rank, subrank, max_rank, min_rank, rng_seed, rng_generation,
        current_power, lives_remaining, source_context, boss_life,
        timeline_time, timeline_time + timeline_subframe,
        attempt, 0,
        tuple(bullet.slot for bullet in bullets),
        bytes(raw_spawn_bullet_records),
        bytes(raw_enemy_records),
        bytes(raw_laser_records),
        raw_enemy_manager_tail,
        SOURCE_RECORD_SCHEMA,
        "",
        None,
        (),
        0,
        -1,
        -1,
        (),
        0,
        0,
        gui_score,
        score,
        next_score_increment,
        high_score,
        graze_in_stage,
        graze_total,
        deaths,
        bombs_used,
        spellcards_captured,
        point_items_collected_in_stage,
        point_items_collected,
        retries,
        power_item_count_for_score,
        bombs_remaining,
        extra_lives,
    )
    return replace(
        control,
        enemy_sprite_dimensions=tuple(
            (pointer, *source_sprite_dimensions[pointer])
            for pointer in sorted(enemy_sprite_pointers)
        ),
    )


def read_control_snapshot(
    process,
    *,
    horizon: int = 12,
    collision_margin: float = 0.35,
    suspend=None,
    max_attempts: int = MAX_CAPTURE_ATTEMPTS,
) -> ControlSnapshot:
    """Return one coherent observed-hazard root or fail closed.

    Armed Windows control supplies a narrow exact-PID suspend context for
    each attempt.  This makes the multi-region copy physically atomic while
    the existing source timer witnesses still reject a process suspended in
    the middle of its update chain.
    """
    if max_attempts <= 0:
        raise ValueError("compact capture attempts must be positive")
    observed_epochs: list[int] = []
    last_error: BaseException | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            with suspend() if suspend is not None else nullcontext():
                return _decode_control_once(
                    process,
                    native,
                    attempt,
                    horizon,
                    collision_margin,
                )
        except (
            native._SnapshotEpochChanged,
            native._SnapshotPhaseIncomplete,
            native._SnapshotReadTorn,
            native.NativeDecodeError,
        ) as error:
            observed_epochs.append(native.read_game_frame(process))
            last_error = error
            if suspend is not None:
                # Let the exact process advance out of a source-defined
                # priority-4..10 partial update before suspending the retry.
                # Wine needs a real host scheduling window; 0.5 ms can
                # repeatedly suspend the exact same source instruction. Eight
                # 2 ms windows remain a fixed 16 ms worst-case retry budget.
                time.sleep(0.002)
    raise RuntimeError(
        "compact coherent capture exhausted retries; "
        f"epochs={observed_epochs}; last={last_error}"
    ) from last_error


def read_safety_snapshot_pair(
    process,
    *,
    horizon: int = 12,
    collision_margin: float = 0.35,
    suspend=None,
    compact_attempts: int = MAX_CAPTURE_ATTEMPTS,
):
    """Capture compact data and exhaustive source authority in one pause.

    The compact root keeps the resident/data-plane representation. The
    exhaustive root supplies immutable ECL graphs and every dynamic source
    field consumed by the bounded Hard forecast. Both reads occur while the
    exact process is suspended and are cross-checked before either is used.
    """
    with suspend() if suspend is not None else nullcontext():
        control = read_control_snapshot(
            process,
            horizon=horizon,
            collision_margin=collision_margin,
            suspend=None,
            max_attempts=compact_attempts,
        )
        authority = native.read_snapshot(process)
    mismatches = []
    for name in (
        "frame",
        "stage",
        "player_state",
        "input_mask",
        "difficulty",
        "character",
        "rank",
        "subrank",
        "rng_seed",
        "rng_generation",
        "current_power",
        "lives_remaining",
        "gui_score",
        "score",
        "next_score_increment",
        "high_score",
        "graze_in_stage",
        "graze_total",
        "deaths",
        "bombs_used",
        "spellcards_captured",
        "point_items_collected_in_stage",
        "point_items_collected",
        "retries",
        "power_item_count_for_score",
        "bombs_remaining",
        "extra_lives",
    ):
        if getattr(control, name) != getattr(authority, name):
            mismatches.append(name)
    for name in ("x", "y", "half_width", "half_height", "frame_multiplier"):
        if getattr(control, name) != getattr(authority, name):
            mismatches.append(name)
    if control.live_bullet_count != len(authority.bullets):
        mismatches.append("live_bullet_count")
    if control.laser_count != authority.laser_count:
        mismatches.append("laser_count")
    authority_spawners = {spawner.slot: spawner for spawner in authority.spawners}
    enemy_dimensions = {
        pointer: (width, height)
        for pointer, width, height in control.enemy_sprite_dimensions
    }
    enemy_record_size = 2 + native.ENEMY_STRIDE
    for offset in range(0, len(control.raw_enemy_records), enemy_record_size):
        slot = struct.unpack_from("<H", control.raw_enemy_records, offset)[0]
        pointer = struct.unpack_from(
            "<I",
            control.raw_enemy_records,
            offset + 2 + native.ANM_VM_SPRITE_OFFSET,
        )[0]
        spawner = authority_spawners.get(slot)
        if spawner is None or enemy_dimensions.get(pointer) != (
            spawner.sprite_half_width * 2.0,
            spawner.sprite_half_height * 2.0,
        ):
            mismatches.append(f"enemy_sprite_geometry:{slot}")
    if mismatches:
        raise RuntimeError(
            "compact/source safety roots disagree: " + ", ".join(mismatches)
        )
    control = replace(
        control,
        factual_state_schema=OFFLINE_FACT_SCHEMA,
        player_attack=authority.player_attack,
        item_states=authority.item_states,
        item_next_index=authority.item_next_index,
        effect_active_upper_bound=authority.effect_active_upper_bound,
        item_active_upper_bound=authority.item_active_upper_bound,
        pending_effect_rng_ids=authority.pending_effect_rng_ids,
        random_item_spawn_index=authority.random_item_spawn_index,
        random_item_table_index=authority.random_item_table_index,
        ecl_ex_function_addresses=authority.ecl_ex_function_addresses,
        timeline_current_message_waits=authority.timeline_current_message_waits,
        message_active=authority.message_active,
        timeline_boss_slots=authority.timeline_boss_slots,
        timeline_time_previous=authority.timeline_time_previous,
        boss_present=authority.boss_present,
    )
    return control, authority


def observe_passive_control_clock(process) -> bool:
    """Remember a coherent time-stop clock without copying hazard pools.

    Dialogue control intentionally bypasses the battle snapshot.  Sampling
    this tiny source state keeps the calc-phase witness synchronized across
    those skipped frames, without admitting dialogue state into movement or
    learning.
    """
    def read_clock() -> tuple[bool, int, int]:
        time_stopped = bool(process.read(
            native.ADDR_GAME_MANAGER + native.GAME_TIME_STOPPED_OFFSET,
            1,
        )[0])
        frame, stage = struct.unpack(
            "<Ii",
            process.read(
                native.ADDR_GAME_MANAGER + native.GAME_FRAMES_OFFSET,
                8,
            ),
        )
        return time_stopped, frame, stage

    before = read_clock()
    bullet_time = struct.unpack(
        "<i",
        process.read(
            native.ADDR_BULLET_MANAGER + native.BULLET_MANAGER_TIME_OFFSET + 8,
            4,
        ),
    )[0]
    after = read_clock()
    if before != after or not before[0]:
        return False
    _accept_completed_calc_phase(
        process,
        native,
        stage=before[2],
        game_frame=before[1],
        bullet_time=bullet_time,
        passive=True,
    )
    return True


def read_passive_input_delivery(process) -> tuple[int, int, int, int, int, int]:
    """Read one coherent retail input sample without copying battle hazards."""
    for _attempt in range(MAX_CAPTURE_ATTEMPTS):
        before = native.read_game_clock(process)
        block = process.read(native.ADDR_CURRENT_INPUT, 14)
        current, previous, held_repeat, held_frames = (
            struct.unpack_from("<H", block, offset)[0]
            for offset in (0, 4, 8, 12)
        )
        after = native.read_game_clock(process)
        if before == after:
            frame, stage = before
            return stage, frame, current, previous, held_repeat, held_frames
    raise RuntimeError("passive input delivery sample crossed game clocks")


def with_tracked_lasers(snapshot: ControlSnapshot, lasers) -> ControlSnapshot:
    return replace(snapshot, lasers=tuple(lasers))


def decode_control_snapshot(raw: dict[str, object]) -> ControlSnapshot:
    """Hydrate a serialized control root for offline replay/audit."""
    values = dict(raw)
    values.setdefault("live_bullet_count", len(values.get("bullets", ())))
    values.setdefault("raw_bullet_tails", b"")
    values.setdefault("bullet_sprite_dimensions", ())
    values.setdefault("bullets_are_reachable_subset", False)
    values.setdefault("reachable_bullet_slots", ())
    values.setdefault("raw_spawn_bullet_records", b"")
    values.setdefault("raw_enemy_records", b"")
    values.setdefault("raw_laser_records", b"")
    values.setdefault("raw_enemy_manager_tail", b"")
    values.setdefault("source_record_schema", "")
    values.setdefault("factual_state_schema", "")
    values.setdefault("player_attack", None)
    values.setdefault("item_states", ())
    values.setdefault("item_next_index", 0)
    values.setdefault("effect_active_upper_bound", -1)
    values.setdefault("item_active_upper_bound", -1)
    values.setdefault("pending_effect_rng_ids", ())
    values.setdefault("random_item_spawn_index", 0)
    values.setdefault("random_item_table_index", 0)
    values.setdefault("enemy_sprite_dimensions", ())
    values.setdefault("ecl_ex_function_addresses", ())
    values.setdefault("timeline_current_message_waits", 0)
    values.setdefault("message_active", False)
    values.setdefault("timeline_boss_slots", ())
    values.setdefault("timeline_time_previous", None)
    values.setdefault("boss_present", None)

    def decode_dimensions(name: str) -> dict[int, tuple[float, float]]:
        dimensions = {}
        for row in values[name]:
            if not isinstance(row, (tuple, list)) or len(row) != 3:
                raise ValueError("invalid packed control sprite dimensions")
            pointer, width, height = int(row[0]), float(row[1]), float(row[2])
            if (
                pointer in dimensions
                or not 0x10000 <= pointer < 0x80000000
                or not math.isfinite(width)
                or not math.isfinite(height)
                or not 0.0 < width <= 4096.0
                or not 0.0 < height <= 4096.0
            ):
                raise ValueError("invalid packed control sprite dimensions")
            dimensions[pointer] = (width, height)
        values[name] = tuple(
            (pointer, *dimensions[pointer]) for pointer in sorted(dimensions)
        )
        return dimensions

    dimensions = decode_dimensions("bullet_sprite_dimensions")
    enemy_dimensions = decode_dimensions("enemy_sprite_dimensions")
    raw_tails = values.get("raw_bullet_tails", b"")
    if raw_tails:
        tail_size = native.BULLET_STRIDE - native.BULLET_SIZE_OFFSET
        record_size = 6 + tail_size
        if len(raw_tails) % record_size:
            raise ValueError("invalid packed control bullet tails")
        bullets = []
        for offset in range(0, len(raw_tails), record_size):
            slot, sprite_pointer = struct.unpack_from("<HI", raw_tails, offset)
            tail = raw_tails[offset + 6 : offset + record_size]
            sprite_size = dimensions.get(sprite_pointer)
            if str(values.get("capture_tier", "")).startswith("control-v") and sprite_size is None:
                raise ValueError(
                    f"control bullet slot {slot} lacks visual sprite geometry"
                )
            bullet = native._decode_bullet_tail(tail, slot, sprite_size)
            if bullet is not None and bullet.state != 5:
                bullets.append(bullet)
        values["bullets"] = tuple(bullets)
        values["bullets_are_reachable_subset"] = False
    else:
        values["bullets"] = tuple(Bullet(**item) for item in values["bullets"])
    values["lasers"] = tuple(Laser(**item) for item in values["lasers"])
    values["enemies"] = tuple(EnemyBody(**item) for item in values["enemies"])
    player_attack = values["player_attack"]
    if isinstance(player_attack, dict):
        player_attack = dict(player_attack)
        player_attack["shots"] = tuple(
            PlayerShot(**shot) if isinstance(shot, dict) else shot
            for shot in player_attack.get("shots", ())
        )
        values["player_attack"] = PlayerAttackState(**player_attack)
    values["item_states"] = tuple(
        ItemState(**item) if isinstance(item, dict) else item
        for item in values["item_states"]
    )
    values["pending_effect_rng_ids"] = tuple(
        int(effect_id) for effect_id in values["pending_effect_rng_ids"]
    )
    values["reachable_bullet_slots"] = tuple(
        int(slot) for slot in values["reachable_bullet_slots"]
    )

    def validate_records(name: str, stride: int, count: int) -> None:
        payload = values[name]
        if not isinstance(payload, bytes) or len(payload) % (2 + stride):
            raise ValueError(f"invalid packed {name}")
        slots = [
            struct.unpack_from("<H", payload, offset)[0]
            for offset in range(0, len(payload), 2 + stride)
        ]
        if len(slots) != len(set(slots)) or any(slot >= count for slot in slots):
            raise ValueError(f"invalid packed {name} slots")

    validate_records(
        "raw_spawn_bullet_records", native.BULLET_STRIDE, native.BULLET_COUNT
    )
    validate_records("raw_enemy_records", native.ENEMY_STRIDE, native.ENEMY_COUNT)
    validate_records("raw_laser_records", native.LASER_STRIDE, native.LASER_COUNT)
    if values["capture_tier"] == CONTROL_CAPTURE_TIER:
        expected_tail = (
            native.ENEMY_MANAGER_SIZE
            - native.ENEMY_ARRAY_OFFSET
            - native.ENEMY_COUNT * native.ENEMY_STRIDE
        )
        enemy_record_size = 2 + native.ENEMY_STRIDE
        enemy_sprite_pointers = {
            struct.unpack_from(
                "<I",
                values["raw_enemy_records"],
                offset + 2 + native.ANM_VM_SPRITE_OFFSET,
            )[0]
            for offset in range(
                0, len(values["raw_enemy_records"]), enemy_record_size
            )
        }
        ex_addresses = tuple(
            int(address) for address in values["ecl_ex_function_addresses"]
        )
        timeline_boss_slots = tuple(
            int(slot) for slot in values["timeline_boss_slots"]
        )
        values["ecl_ex_function_addresses"] = ex_addresses
        values["timeline_boss_slots"] = timeline_boss_slots
        if (
            values["source_record_schema"] != SOURCE_RECORD_SCHEMA
            or values["factual_state_schema"] != OFFLINE_FACT_SCHEMA
            or len(values["raw_enemy_manager_tail"]) != expected_tail
            or enemy_sprite_pointers - set(enemy_dimensions)
            or len(ex_addresses) != native.ECL_EX_COUNT
            or any(not 0x10000 <= address < 0x80000000 for address in ex_addresses)
            or len(timeline_boss_slots) != 8
            or any(not -1 <= slot < native.ENEMY_COUNT for slot in timeline_boss_slots)
            or int(values["timeline_current_message_waits"]) < 0
            or values["timeline_time_previous"] is None
            or values["boss_present"] is None
        ):
            raise ValueError("control-v4 source/factual records are incomplete")
    return ControlSnapshot(**values)
