from __future__ import annotations

from dataclasses import dataclass


BUTTON_SHOOT = 0x01
BUTTON_BOMB = 0x02
BUTTON_FOCUS = 0x04
BUTTON_UP = 0x10
BUTTON_DOWN = 0x20
BUTTON_LEFT = 0x40
BUTTON_RIGHT = 0x80
BUTTON_SKIP = 0x100

PLAYER_ALIVE = 0
PLAYER_SPAWNING = 1
PLAYER_DEAD = 2
PLAYER_INVULNERABLE = 3


@dataclass(frozen=True)
class Action:
    name: str
    dx: int
    dy: int
    focused: bool = True


ACTIONS = (
    Action("stay", 0, 0),
    Action("up", 0, -1),
    Action("down", 0, 1),
    Action("left", -1, 0),
    Action("right", 1, 0),
    Action("up_left", -1, -1),
    Action("up_right", 1, -1),
    Action("down_left", -1, 1),
    Action("down_right", 1, 1),
)
ACTION_BY_VECTOR = {(action.dx, action.dy): action for action in ACTIONS}
FAST_ACTIONS = tuple(
    Action(f"{action.name}_fast", action.dx, action.dy, False)
    for action in ACTIONS
)
FAST_ACTION_BY_VECTOR = {
    (action.dx, action.dy): action for action in FAST_ACTIONS
}
CONTROL_ACTIONS = ACTIONS + FAST_ACTIONS


@dataclass(frozen=True)
class Bullet:
    x: float
    y: float
    vx: float
    vy: float
    half_width: float
    half_height: float
    state: int
    ex_flags: int = 0
    acceleration: float = 0.0
    speed: float = 0.0
    turn_speed: float = 0.0
    acceleration_x: float = 0.0
    acceleration_y: float = 0.0
    angle: float = 0.0
    direction_rotation: float = 0.0
    timer: int = 0
    timer_float: float = 0.0
    acceleration_duration: int = 0
    direction_interval: int = 0
    direction_num_times: int = 0
    direction_max_times: int = 0
    curve_speed_acceleration: float = 0.0
    curve_angular_velocity: float = 0.0
    slot: int = -1
    # BulletManager permits exactly one graze score/effect per bullet.  This
    # byte is physical future state: omitting it lets candidate continuations
    # manufacture repeated graze RNG and rank changes.
    is_grazed: bool = False
    # BulletManager retires a fired bullet only after testing the full visual
    # sprite against the 384x448 playfield.  This is distinct from the much
    # smaller graze/kill box above.
    sprite_half_width: float = 0.0
    sprite_half_height: float = 0.0
    # Source ``Bullet::unk_5c0`` counts consecutive out-of-bounds updates for
    # direction-changing bullets.  It is reset on re-entry and caps at 0x100.
    out_of_bounds_frames: int = 0
    # Simulator births retain the source bullet-template index so their
    # immutable visual geometry can be recovered from the shipped ANM data.
    sprite: int = -1


@dataclass(frozen=True)
class Laser:
    x: float
    y: float
    angle: float
    start_offset: float
    end_offset: float
    start_length: float
    width: float
    speed: float
    start_time: int
    hitbox_start_time: int
    duration: int
    despawn_duration: int
    hitbox_end_delay: int
    timer: int
    timer_float: float
    flags: int
    state: int
    slot: int = -1
    angular_velocity: float = 0.0
    motion_known: bool = False


@dataclass(frozen=True)
class PlayerShot:
    """One occupied source ``PlayerBullet`` slot at the frame boundary."""

    slot: int
    x: float
    y: float
    half_width: float
    half_height: float
    vx: float
    vy: float
    homing_speed: float
    timer_previous: int
    timer: int
    timer_float: float
    damage: int
    state: int
    bullet_type: int
    anm_script: int
    anm_timer: int
    anm_timer_float: float
    sprite_half_width: float
    sprite_half_height: float
    sideways_motion: float = 0.0
    laser_index: int = 0
    spawn_position_index: int = 0


@dataclass(frozen=True)
class ItemState:
    """One occupied source ``ItemManager`` slot at the frame boundary."""

    slot: int
    x: float
    y: float
    start_x: float
    start_y: float
    target_x: float
    target_y: float
    timer_previous: int
    timer: int
    timer_float: float
    item_type: int
    state: int


@dataclass(frozen=True)
class PlayerAttackState:
    """Player-owned attack state needed for exact future damage."""

    shots: tuple[PlayerShot, ...]
    last_enemy_hit_x: float
    last_enemy_hit_y: float
    orb_state: int
    is_focus: bool
    focus_timer_previous: int
    focus_timer: int
    focus_timer_float: float
    fire_timer_previous: int
    fire_timer: int
    fire_timer_float: float
    orb_positions: tuple[tuple[float, float], tuple[float, float]]
    shot_type: int
    bomb_active: bool
    spell_active: bool


@dataclass(frozen=True)
class EnemyBody:
    x: float
    y: float
    half_width: float
    half_height: float
    velocity_x: float
    velocity_y: float
    angle: float
    angular_velocity: float
    speed: float
    acceleration: float
    movement_mode: int
    movement_ease: int
    invert_x: bool
    move_interp_x: float
    move_interp_y: float
    move_start_x: float
    move_start_y: float
    move_timer: int
    move_timer_float: float
    move_start_time: int


@dataclass(frozen=True)
class BulletPattern:
    """Runtime-resolved EnemyBulletShooter plus its copied collision size."""

    sprite: int
    angle1: float
    angle2: float
    speed1: float
    speed2: float
    ex_floats: tuple[float, float, float, float]
    ex_ints: tuple[int, int, int, int]
    count1: int
    count2: int
    aim_mode: int
    flags: int
    half_width: float
    half_height: float


@dataclass(frozen=True)
class EclInstruction:
    address: int
    time: int
    opcode: int
    offset_to_next: int
    skip_for_difficulty: int
    raw_hex: str


@dataclass(frozen=True)
class StageTimelineInstruction:
    """One immutable source ``EclTimelineInstr`` from the loaded stage."""

    address: int
    time: int
    arg0: int
    opcode: int
    size: int
    raw_hex: str


@dataclass(frozen=True)
class MessageInstruction:
    """One immutable instruction from the stage's loaded ``.msg`` file."""

    address: int
    time: int
    opcode: int
    arg_size: int
    raw_hex: str


@dataclass(frozen=True)
class EnemyEclContext:
    instruction_address: int
    time: int
    time_float: float
    ints: tuple[int, int, int, int, int, int, int, int]
    floats: tuple[float, float, float, float]
    compare: int
    repeat_ex_index: int | None


@dataclass(frozen=True)
class EnemySpawner:
    """An occupied enemy's observable periodic and ECL emission state."""

    slot: int
    x: float
    y: float
    velocity_x: float
    velocity_y: float
    angle: float
    angular_velocity: float
    speed: float
    acceleration: float
    movement_mode: int
    movement_ease: int
    invert_x: bool
    move_interp_x: float
    move_interp_y: float
    move_start_x: float
    move_start_y: float
    move_timer: int
    move_timer_float: float
    move_start_time: int
    shoot_offset_x: float
    shoot_offset_y: float
    bullet_rank_speed_low: float
    bullet_rank_speed_high: float
    bullet_rank_amount1_low: int
    bullet_rank_amount1_high: int
    bullet_rank_amount2_low: int
    bullet_rank_amount2_high: int
    life: int
    shooting_disabled: bool
    interval: int
    timer: int
    timer_float: float
    pattern: BulletPattern | None
    ecl_time: int
    ecl_time_float: float
    ecl_ints: tuple[int, int, int, int, int, int, int, int]
    ecl_floats: tuple[float, float, float, float]
    ecl_compare: int
    repeat_ex_index: int | None
    next_instruction: EclInstruction | None
    ecl_program: tuple[EclInstruction, ...]
    ecl_stack: tuple[EnemyEclContext, ...] = ()
    hitbox_half_width: float = 0.0
    hitbox_half_height: float = 0.0
    interactable: bool = False
    collidable: bool = False
    invisible: bool = False
    call_stack_disabled: bool = False
    ecl_subroutines: tuple[int, ...] = ()
    lower_move_x: float = 0.0
    lower_move_y: float = 0.0
    upper_move_x: float = 0.0
    upper_move_y: float = 0.0
    should_clamp_position: bool = False
    boss_timer: int = 0
    boss_timer_float: float = 0.0
    death_callback_sub: int = -1
    life_callback_threshold: int = -1
    life_callback_sub: int = -1
    timer_callback_threshold: int = -1
    timer_callback_sub: int = -1
    is_boss: bool = False
    timeout_spell: bool = False
    damageable: bool = True
    bullet_effect_floats: tuple[float, float, float, float] = (0.0,) * 4
    bullet_effect_ints: tuple[int, int, int, int] = (0,) * 4
    death_mode: int = 0
    boss_id: int = -1
    interrupts: tuple[int, ...] = (-1,) * 8
    run_interrupt: int = -1
    has_been_in_bounds: bool = False
    sprite_half_width: float = 0.0
    sprite_half_height: float = 0.0
    death_anm1: int = 0
    death_anm2: int = 0
    death_anm3: int = 0
    item_drop: int = -2
    # Enemy::lasers keeps raw pointers into BulletManager's 64-slot pool.
    # A pointer may remain after its laser becomes inactive and later alias a
    # reused slot, so pointer identity is physical ECL state, not presentation.
    laser_slots: tuple[int, ...] = (-1,) * 32
    laser_store: int = 0
    # Forecast-only uncertainty carried by not-yet-born enemies whose source
    # timeline coordinates come from the shared RNG. Physical captures leave
    # both values at zero; Hard forecasts use them to preserve a bounded
    # position world without pretending that a future RNG value is known.
    forecast_position_uncertainty_x: float = 0.0
    forecast_position_uncertainty_y: float = 0.0


@dataclass(frozen=True)
class Snapshot:
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
    bullets: tuple[Bullet, ...]
    laser_count: int
    in_menu: bool
    time_stopped: bool
    replay_or_demo: bool
    lasers: tuple[Laser, ...] = ()
    enemies: tuple[EnemyBody, ...] = ()
    despawning_bullets: tuple[Bullet, ...] = ()
    bullet_read_retries: int = 0
    spawners: tuple[EnemySpawner, ...] = ()
    difficulty: int = 2
    rank: int = 0
    bullet_sizes: tuple[tuple[float, float], ...] = ()
    rng_seed: int = 0
    rng_generation: int = 0
    current_power: int = 0
    timeline_time: int = 0
    timeline_time_float: float = 0.0
    timeline_instructions: tuple[StageTimelineInstruction, ...] = ()
    timeline_complete: bool = False
    timeline_emitter_subs: tuple[int, ...] = ()
    timeline_boss_subs: tuple[int, ...] = ()
    ecl_subroutines: tuple[int, ...] = ()
    timeline_ecl_program: tuple[EclInstruction, ...] = ()
    character: int = 0
    timeline_message_delays: tuple[tuple[int, int], ...] = ()
    timeline_current_message_waits: int = 0
    player_attack: PlayerAttackState | None = None
    effect_active_upper_bound: int = -1
    item_active_upper_bound: int = -1
    random_item_spawn_index: int = 0
    random_item_table_index: int = 0
    message_active: bool = False
    # GameManager::IncreaseSubrank carries this remainder across updates.
    # Graze happens after EnemyManager, so the updated rank first affects ECL
    # on the following frame.
    subrank: int = 0
    max_rank: int = 32
    min_rank: int = 0
    # Effects born after priority-10 EffectManager retain timer (-999, 0) at
    # the frame boundary.  Their callbacks consume RNG on the next update.
    # Only the source effect IDs are needed; visual state is hazard-neutral.
    pending_effect_rng_ids: tuple[int, ...] = ()
    item_states: tuple[ItemState, ...] = ()
    item_next_index: int = 0
    # EnemyManager::bosses is a raw stage-timeline target table. It can
    # retain a pointer after that slot stops being a boss, so it is distinct
    # from Enemy::flags.isBoss / bossId captured in ``spawners``.
    timeline_boss_slots: tuple[int, ...] = ()
    # EnemyManager::RunEclTimeline uses the remaining-life count to schedule
    # its periodic +100 subrank transition. A no-miss route starts at two,
    # but source-valid offline battle worlds must carry the live value.
    lives_remaining: int = 2
    # ZunTimer::HasTicked compares these values before the current timeline
    # update. ``None`` keeps older corpus fixtures backward compatible; a
    # positive legacy timer is known to have ticked in the supported no-wait
    # battle replay.
    timeline_time_previous: int | None = None
    # EnemyManager::RunEclTimeline gates enemy records on Gui::bossPresent,
    # not on an occupied Enemy slot's isBoss flag.  A death-mode-1 boss can
    # remain in the pool after Enemy::Despawn has cleared this byte. ``None``
    # keeps older retained artifacts explicit rather than inventing its value.
    boss_present: bool | None = None
    # Offline battle replay owns these countdowns.  Each entry is one
    # simulator-born, source-proven finite effect and records how many future
    # EffectManager passes still include its occupied slot.  Physical capture
    # leaves this empty; its existing pool remains in the conservative upper
    # bound because the compact snapshot does not retain every ANM VM.
    simulated_effect_expiry_updates: tuple[int, ...] = ()


@dataclass(frozen=True)
class SafeAction:
    action: Action
    clearance: float
    final_x: float
    final_y: float


@dataclass(frozen=True)
class Decision:
    action: Action | None
    safe_actions: tuple[SafeAction, ...]
    clearance: float
    horizon: int
    reason: str
    effort_horizon: int = 0
    effort_safe_count: int = 0
    repairable_count: int = 0
    held_horizon: int = 0
    suppression_target_x: float | None = None
    suppression_deadline: int | None = None
    suppression_life: int = 0
    suppression_source: str = ""
    route_id: str = ""
    phase_id: str = ""
    policy_state: str = ""
    proposal_source: str = ""


def action_from_input(mask: int) -> Action:
    """Match Player::HandlePlayerInputs directional precedence."""
    dx = 0
    dy = 0
    if mask & BUTTON_UP:
        dy = -1
        if mask & BUTTON_LEFT:
            dx = -1
        if mask & BUTTON_RIGHT:
            dx = 1
    elif mask & BUTTON_DOWN:
        dy = 1
        if mask & BUTTON_LEFT:
            dx = -1
        if mask & BUTTON_RIGHT:
            dx = 1
    else:
        if mask & BUTTON_LEFT:
            dx = -1
        if mask & BUTTON_RIGHT:
            dx = 1
    actions = ACTION_BY_VECTOR if mask & BUTTON_FOCUS else FAST_ACTION_BY_VECTOR
    return actions[(dx, dy)]
