"""One compact world forecast for source-defined bullet births."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from ..model import (
    Bullet,
    EnemySpawner,
    RepeatStarState,
    Snapshot,
    StageTimelineInstruction,
)
from .births import UnsupportedBirthModel
from .bullets import (
    RAINBOW_ACCELERATION_AXIS_BOUND,
    hazard_boxes,
    radial_hazard_box,
)
from .ecl import (
    HardLaserWorld,
    forecast_ecl_births,
    forecast_ecl_forced_kill_all_update,
    source_enemy_template,
)
from .lasers import LaserHazard
from .rng import RngState
from .timeline import (
    TimelineBossInterrupt,
    TimelineEnemySpawn,
    decode_boss_interrupt,
    decode_enemy_spawn,
    scheduled_timeline,
)


SOURCE_ENEMY_SLOT_COUNT = 255
# GameManager::AddedCallback initializes the exact ranges consumed by
# RunEclTimeline's random-coordinate opcodes. Unlike player clamping, the
# timeline draw does not add playerMovementAreaTopLeftPos.
SOURCE_RANDOM_TIMELINE_WIDTH = 368.0
SOURCE_RANDOM_TIMELINE_HEIGHT = 416.0


def _program_can_create_enemy(emitter: EnemySpawner) -> bool:
    return any(instruction.opcode == 95 for instruction in emitter.ecl_program)


def _program_can_repeat_star(emitter: EnemySpawner) -> bool:
    if emitter.repeat_ex_index == 2:
        return True
    for instruction in emitter.ecl_program:
        if instruction.opcode != 122:
            continue
        raw = bytes.fromhex(instruction.raw_hex)
        if len(raw) >= 16 and int.from_bytes(
            raw[12:16], "little", signed=True
        ) == 2:
            return True
    return False


@dataclass(frozen=True)
class WorldBirthForecast:
    births: tuple[tuple[Bullet, ...], ...]
    hazards: tuple[tuple[tuple[float, float, float, float], ...], ...]
    covered_frames: int
    reason: str = ""
    body_hazards: tuple[tuple[tuple[float, float, float, float], ...], ...] = ()
    continuation: "WorldForecastContinuation | None" = None
    laser_births: int = 0
    mutated_initial_lasers: tuple[int, ...] = ()
    missing_laser_dereferences: tuple[int, ...] = ()
    retired_future_laser: bool = False
    laser_hazards: tuple[tuple[LaserHazard, ...], ...] = ()
    # Leads at which a reachable source branch executes deterministic
    # EX_CALL 0,param 0 before BulletManager's update.  Consumers retain the
    # ordinary trajectory too: hard damage branches are unioned, so a stop
    # observed in one branch is not assumed to occur in every branch.
    bullet_stop_frames: tuple[int, ...] = ()
    bullet_release_frames: tuple[int, ...] = ()
    # A source-exact timeline transition can replace every pre-existing live
    # slot from this lead onward. Independent live envelopes still contribute
    # their prefix and conservative hazards, but not suffix coverage failures
    # from a physically impossible no-transition branch.
    replaces_live_from_lead: int | None = None
    # Leads at which this live-emitter envelope reached ENEMYKILLALL.  The
    # world layer owns the cross-slot effect; individual emitters may only
    # report the event when their slot order makes that effect replayable.
    enemy_kill_all_frames: tuple[int, ...] = ()
    # Number of independently replayed mutable laser worlds that observed a
    # pool or pointer effect.  Pointer reuse inside one world is source-ordered
    # by HardLaserWorld; only effects split across worlds can be aliased in an
    # order that the independent envelopes do not represent.
    laser_effect_worlds: int = 0


@dataclass(frozen=True)
class WorldForecastContinuation:
    emitters: tuple[EnemySpawner, ...]
    rng_seed: int
    rng_generation: int
    framewise: bool
    elapsed_frames: int = 0
    boss_present: bool | None = None
    repeat_star_state: RepeatStarState | None = None


def _boss_present_after_emitter(
    current: bool | None,
    before: EnemySpawner,
    after: EnemySpawner | None,
) -> bool | None:
    """Apply source-visible BOSSSET/Despawn writes represented by ECL state."""
    if after is None:
        return False if before.is_boss else current
    if before.is_boss != after.is_boss:
        return after.is_boss
    return current


class _NominalRngConsumed(Exception):
    pass


class _NoRngState(RngState):
    """Abort a speculative batch at its first shared-RNG dependency."""

    def u16(self) -> int:
        raise _NominalRngConsumed


def _laser_world_changed(world: HardLaserWorld) -> bool:
    return bool(
        world.created_count
        or world.mutated_initial_slots
        or world.missing_dereferences
        or world.retired_created
    )


def _project_hazards(
    births: list[list[Bullet]],
    radial: bool,
    bullet_release_frames: tuple[int, ...] = (),
) -> tuple[tuple[tuple[float, float, float, float], ...], ...]:
    frames: list[list[tuple[float, float, float, float]]] = [
        [] for _ in births
    ]
    for birth_frame, bullets in enumerate(births):
        remaining = len(frames) - birth_frame
        for bullet in bullets:
            hazards = (
                (
                    radial_hazard_box(bullet, age)
                    for age in range(1, remaining + 1)
                )
                if radial
                else hazard_boxes(bullet, remaining)
            )
            for frame_index, hazard in enumerate(hazards, birth_frame):
                release_radius = sum(
                    RAINBOW_ACCELERATION_AXIS_BOUND
                    * (frame_index - release_frame + 1)
                    * (frame_index - release_frame + 2)
                    / 2.0
                    for release_frame in bullet_release_frames
                    if birth_frame <= release_frame <= frame_index
                )
                if release_radius:
                    left, top, right, bottom = hazard
                    hazard = (
                        left - release_radius,
                        top - release_radius,
                        right + release_radius,
                        bottom + release_radius,
                    )
                frames[frame_index].append(hazard)
    return tuple(tuple(frame) for frame in frames)


def _scheduled_boss_interrupts(
    snapshot: Snapshot,
    horizon: int,
) -> tuple[tuple[int, TimelineBossInterrupt | None], ...]:
    result = []
    for lead, instruction in scheduled_timeline(
        snapshot.timeline_instructions,
        snapshot.timeline_time,
        stage=snapshot.stage,
        difficulty=snapshot.difficulty,
        character=snapshot.character,
        message_delays=snapshot.timeline_message_delays,
        current_message_waits=snapshot.timeline_current_message_waits,
    ):
        if lead >= horizon:
            break
        if instruction.opcode == 10:
            result.append((lead, decode_boss_interrupt(instruction)))
    return tuple(result)


def _live_kill_all_has_no_external_target(
    snapshot: Snapshot,
    horizon: int,
) -> bool:
    """Prove that a live boss cannot see a non-boss slot in this window.

    RunEclTimeline precedes the EnemyManager slot loop, so even a same-lead
    timeline child would be a target. ECL-created children are checked at the
    interpreter boundary by callers using ``enemy_kill_all_is_noop``.
    """
    if (
        horizon <= 0
        or not snapshot.timeline_complete
        or len(snapshot.spawners) != 1
        or not snapshot.spawners[0].is_boss
    ):
        return False
    for lead, instruction in scheduled_timeline(
        snapshot.timeline_instructions,
        snapshot.timeline_time,
        stage=snapshot.stage,
        difficulty=snapshot.difficulty,
        character=snapshot.character,
        message_delays=snapshot.timeline_message_delays,
        current_message_waits=snapshot.timeline_current_message_waits,
    ):
        if lead >= horizon:
            break
        if 0 <= instruction.opcode <= 7:
            return False
    return True


def _timeline_interrupt_targets(
    snapshot: Snapshot,
    emitter: EnemySpawner,
    event: TimelineBossInterrupt,
) -> bool:
    """Match opcode 10 through the source raw pointer table when captured."""
    if emitter.slot < 0:
        # A timeline child forecast before slot insertion can still execute
        # BOSSSET inline. Its predicted own boss id is the only available
        # binding inside this compact Hard child world.
        return event.boss_id == emitter.boss_id
    if len(snapshot.timeline_boss_slots) == 8:
        if not 0 <= event.boss_id < len(snapshot.timeline_boss_slots):
            return False
        return snapshot.timeline_boss_slots[event.boss_id] == emitter.slot
    # Older corpus records predate capture of EnemyManager::bosses.
    return event.boss_id == emitter.boss_id


def _timeline_interrupt_has_resolved_target(
    snapshot: Snapshot,
    event: TimelineBossInterrupt,
    known_boss_ids: set[int],
) -> bool:
    if len(snapshot.timeline_boss_slots) == 8:
        if not 0 <= event.boss_id < len(snapshot.timeline_boss_slots):
            return False
        target_slot = snapshot.timeline_boss_slots[event.boss_id]
        if 0 <= target_slot < SOURCE_ENEMY_SLOT_COUNT + 1:
            return True
        # A deterministic timeline child can execute BOSSSET between the
        # capture and this future opcode 10. The child world records that
        # binding in known_boss_ids even though no physical pointer exists yet.
        return target_slot == -1 and event.boss_id in known_boss_ids
    return event.boss_id in known_boss_ids


def _forecast_hard_emitter_batched(
    snapshot: Snapshot,
    emitter: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    *,
    start_lead: int = 0,
    enemy_kill_all_is_noop: bool,
    allow_repeat_star: bool,
    record_laser_create=None,
) -> WorldBirthForecast:
    """Advance one emitter across source timeline interrupt boundaries."""
    horizon = len(player_positions)
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    bullet_stop_frames: set[int] = set()
    bullet_release_frames: set[int] = set()
    enemy_kill_all_frames: set[int] = set()
    events = tuple(
        (lead, event)
        for lead, event in _scheduled_boss_interrupts(snapshot, horizon)
        if event is not None
        and _timeline_interrupt_targets(snapshot, emitter, event)
        and lead >= start_lead
    )
    cursor = start_lead
    state: EnemySpawner | None = emitter
    repeat_star_state = (
        snapshot.repeat_star_state if allow_repeat_star else None
    )
    life_uncertain = False
    for boundary, event in (*events, (horizon, None)):
        if state is None:
            if event is not None:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    boundary,
                    f"timeline interrupt targets a finished boss {emitter.boss_id}",
                    tuple(map(tuple, bodies)),
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            break
        if boundary > cursor:
            try:
                forecast = forecast_ecl_births(
                    state,
                    player_positions[cursor:boundary],
                    snapshot.difficulty,
                    snapshot.rank,
                    snapshot.bullet_sizes,
                    snapshot.frame_multiplier,
                    allow_player_variables=False,
                    radial_births=True,
                    abstract_rng=True,
                    enemy_kill_all_is_noop=enemy_kill_all_is_noop,
                    record_bullet_stop=(
                        lambda frame, offset=cursor: bullet_stop_frames.add(
                            offset + frame
                        )
                    ),
                    record_bullet_release=(
                        lambda frame, offset=cursor: bullet_release_frames.add(
                            offset + frame
                        )
                    ),
                    record_laser_create=(
                        None
                        if record_laser_create is None
                        else lambda frame, offset=cursor: record_laser_create(
                            offset + frame
                        )
                    ),
                    repeat_star_state=repeat_star_state,
                    initial_life_uncertain=life_uncertain,
                )
            except UnsupportedBirthModel as error:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    cursor,
                    str(error),
                    tuple(map(tuple, bodies)),
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            for offset, frame_births in enumerate(forecast.births, cursor):
                births[offset].extend(frame_births)
            for offset, frame_bodies in enumerate(
                forecast.body_hazards, cursor
            ):
                bodies[offset].extend(frame_bodies)
            enemy_kill_all_frames.update(
                cursor + offset
                for offset, reached in enumerate(forecast.enemy_kill_all)
                if reached
            )
            if forecast.covered_frames < boundary - cursor:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    cursor + forecast.covered_frames,
                    forecast.reason,
                    tuple(map(tuple, bodies)),
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            if enemy_kill_all_is_noop and forecast.created_emitters:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    cursor,
                    "live kill-all no-target proof ended at ENEMYCREATE",
                    tuple(map(tuple, bodies)),
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            state = forecast.next_spawner
            repeat_star_state = forecast.repeat_star_state
            life_uncertain = forecast.life_uncertain
            if state is None and not forecast.finished:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    boundary,
                    forecast.reason or "emitter continuation is unresolved",
                    tuple(map(tuple, bodies)),
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            cursor = boundary
        if event is not None:
            if state is None:
                continue
            state = replace(state, run_interrupt=event.interrupt_id)
    return WorldBirthForecast(
        tuple(map(tuple, births)),
        _project_hazards(births, True),
        horizon,
        body_hazards=tuple(map(tuple, bodies)),
        bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
        bullet_release_frames=tuple(sorted(bullet_release_frames)),
        enemy_kill_all_frames=tuple(sorted(enemy_kill_all_frames)),
    )


def _forecast_hard_emitter_with_lasers(
    snapshot: Snapshot,
    emitter: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    *,
    start_lead: int,
    enemy_kill_all_is_noop: bool,
    laser_world: HardLaserWorld,
    allow_repeat_star: bool,
) -> WorldBirthForecast:
    """Advance ECL then the shared BulletManager phase one frame at a time."""
    horizon = len(player_positions)
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    lasers: list[list[LaserHazard]] = [[] for _ in player_positions]
    bullet_stop_frames: set[int] = set()
    bullet_release_frames: set[int] = set()
    enemy_kill_all_frames: set[int] = set()
    events_by_lead = {
        lead: event
        for lead, event in _scheduled_boss_interrupts(snapshot, horizon)
        if event is not None
        and _timeline_interrupt_targets(snapshot, emitter, event)
        and lead >= start_lead
    }
    state: EnemySpawner | None = emitter
    repeat_star_state = (
        snapshot.repeat_star_state if allow_repeat_star else None
    )
    life_uncertain = False

    def result(covered: int, reason: str = "") -> WorldBirthForecast:
        return WorldBirthForecast(
            tuple(map(tuple, births)),
            _project_hazards(
                births,
                True,
                tuple(sorted(set(bullet_release_frames))),
            ),
            covered,
            reason,
            tuple(map(tuple, bodies)),
            laser_births=laser_world.created_count,
            mutated_initial_lasers=tuple(sorted(
                laser_world.mutated_initial_slots
            )),
            missing_laser_dereferences=tuple(sorted(
                laser_world.missing_dereferences
            )),
            retired_future_laser=laser_world.retired_created,
            laser_effect_worlds=int(_laser_world_changed(laser_world)),
            laser_hazards=tuple(map(tuple, lasers)),
            bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
            bullet_release_frames=tuple(sorted(bullet_release_frames)),
            enemy_kill_all_frames=tuple(sorted(enemy_kill_all_frames)),
        )

    for frame_index in range(start_lead, horizon):
        event = events_by_lead.get(frame_index)
        if event is not None:
            if state is None:
                return result(
                    frame_index,
                    f"timeline interrupt targets a finished boss {emitter.boss_id}",
                )
            state = replace(state, run_interrupt=event.interrupt_id)
        if state is not None:
            try:
                forecast = forecast_ecl_births(
                    state,
                    (player_positions[frame_index],),
                    snapshot.difficulty,
                    snapshot.rank,
                    snapshot.bullet_sizes,
                    snapshot.frame_multiplier,
                    allow_player_variables=False,
                    radial_births=True,
                    abstract_rng=True,
                    enemy_kill_all_is_noop=enemy_kill_all_is_noop,
                    laser_world=laser_world,
                    record_bullet_stop=(
                        lambda frame, offset=frame_index: bullet_stop_frames.add(
                            offset + frame
                        )
                    ),
                    record_bullet_release=(
                        lambda frame, offset=frame_index: bullet_release_frames.add(
                            offset + frame
                        )
                    ),
                    repeat_star_state=repeat_star_state,
                    initial_life_uncertain=life_uncertain,
                )
            except UnsupportedBirthModel as error:
                return result(frame_index, str(error))
            births[frame_index].extend(forecast.births[0])
            if forecast.body_hazards:
                bodies[frame_index].extend(forecast.body_hazards[0])
            if forecast.enemy_kill_all and forecast.enemy_kill_all[0]:
                enemy_kill_all_frames.add(frame_index)
            if forecast.covered_frames < 1:
                return result(frame_index, forecast.reason)
            if enemy_kill_all_is_noop and forecast.created_emitters:
                return result(
                    frame_index,
                    "live kill-all no-target proof ended at ENEMYCREATE",
                )
            state = forecast.next_spawner
            repeat_star_state = forecast.repeat_star_state
            life_uncertain = forecast.life_uncertain
            if state is None and not forecast.finished:
                return result(
                    frame_index + 1,
                    forecast.reason or "emitter continuation is unresolved",
                )
        # EnemyManager/ECL priority 9 precedes BulletManager priority 11.
        laser_boxes, oriented_lasers = laser_world.advance_hazards()
        bodies[frame_index].extend(laser_boxes)
        lasers[frame_index].extend(oriented_lasers)
    return result(horizon)


def _forecast_hard_emitter(
    snapshot: Snapshot,
    emitter: EnemySpawner,
    player_positions: tuple[tuple[float, float], ...],
    *,
    start_lead: int = 0,
    enemy_kill_all_is_noop: bool,
    laser_world: HardLaserWorld | None = None,
    allow_repeat_star: bool = True,
    force_mutable_laser_pool: bool = False,
    record_laser_create=None,
) -> WorldBirthForecast:
    """Use the compact mutable world only when a reachable laser op needs it."""
    if laser_world is None:
        laser_create_frames: set[int] = set()
        batched = _forecast_hard_emitter_batched(
            snapshot,
            emitter,
            player_positions,
            start_lead=start_lead,
            enemy_kill_all_is_noop=enemy_kill_all_is_noop,
            allow_repeat_star=allow_repeat_star,
            record_laser_create=laser_create_frames.add,
        )
        if record_laser_create is not None:
            for frame in sorted(laser_create_frames):
                record_laser_create(frame)
        if laser_create_frames and force_mutable_laser_pool:
            # Static Hard projection already enclosed each beam's geometry,
            # but a reachable allocation also changes the global 64-slot pool
            # and may alias another emitter's stale Laser*. Re-run only this
            # source-reachable window in the mutable world when another
            # independent laser-effect owner makes pool ordering observable.
            laser_world = HardLaserWorld(snapshot.lasers)
        elif (
            batched.covered_frames == len(player_positions)
            or "laser" not in batched.reason.lower()
        ):
            return batched
        else:
            laser_world = HardLaserWorld(snapshot.lasers)
    return _forecast_hard_emitter_with_lasers(
        snapshot,
        emitter,
        player_positions,
        start_lead=start_lead,
        enemy_kill_all_is_noop=enemy_kill_all_is_noop,
        laser_world=laser_world,
        allow_repeat_star=allow_repeat_star,
    )


def _forecast_live_slots_after_kill_all(
    snapshot: Snapshot,
    player_positions: tuple[tuple[float, float], ...],
    kill_lead: int,
) -> WorldBirthForecast:
    """Replay every pre-existing slot through one pre-slot kill-all event.

    The ordinary hard emitter worlds deliberately retain their no-kill
    branches.  This second bounded world adds the exact life-zero ECL branch
    and any later callback hazards.  Unioning both is conservative while
    keeping the shared timeline/slot transition explicit and fail-closed.
    """
    horizon = len(player_positions)
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    bullet_stop_frames: set[int] = set()
    bullet_release_frames: set[int] = set()

    def result(covered: int, reason: str = "") -> WorldBirthForecast:
        return WorldBirthForecast(
            tuple(map(tuple, births)),
            _project_hazards(
                births,
                True,
                tuple(sorted(bullet_release_frames)),
            ),
            covered,
            reason,
            tuple(map(tuple, bodies)),
            bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
            bullet_release_frames=tuple(sorted(bullet_release_frames)),
        )

    if not 0 <= kill_lead < horizon:
        return result(0, "invalid timeline kill-all lead")
    emitters = tuple(sorted(snapshot.spawners, key=lambda item: item.slot))
    for emitter in emitters:
        # ENEMYKILLALL explicitly skips bosses.  Their independent envelope
        # remains authoritative and must neither be replayed nor replaced.
        if emitter.is_boss:
            continue
        state: EnemySpawner | None = emitter
        repeat_star_state = snapshot.repeat_star_state
        laser_world = HardLaserWorld(snapshot.lasers)
        enemy_create_frames: set[int] = set()
        if kill_lead:
            damage_proof = forecast_ecl_births(
                emitter,
                player_positions[:kill_lead],
                snapshot.difficulty,
                snapshot.rank,
                snapshot.bullet_sizes,
                snapshot.frame_multiplier,
                allow_player_variables=False,
                radial_births=True,
                abstract_rng=True,
                enemy_kill_all_is_noop=False,
                # A unique target state is valid only while no candidate
                # player-damage branch can fire a life/death callback before
                # the timeline opcode.  The ECL layer unions such branches
                # and withholds a continuation, making this replay fail
                # closed instead of selecting the zero-damage state.
                model_player_damage=True,
                repeat_star_state=repeat_star_state,
            )
            if damage_proof.covered_frames < kill_lead:
                return result(
                    damage_proof.covered_frames,
                    f"slot {emitter.slot} kill-all damage proof: "
                    f"{damage_proof.reason}",
                )
            if damage_proof.created_emitters:
                return result(
                    kill_lead,
                    f"slot {emitter.slot} kill-all prefix creates an enemy",
                )
            if any(damage_proof.enemy_kill_all):
                return result(
                    kill_lead,
                    f"slot {emitter.slot} kill-all prefix mutates all slots",
                )
            if damage_proof.next_spawner is None:
                if damage_proof.finished:
                    continue
                return result(
                    kill_lead,
                    f"slot {emitter.slot} kill-all damage proof has no "
                    "unique continuation",
                )

            events_by_lead = {
                lead: event
                for lead, event in _scheduled_boss_interrupts(
                    snapshot, kill_lead
                )
                if event is not None
                and _timeline_interrupt_targets(snapshot, emitter, event)
            }
            for prefix_lead in range(kill_lead):
                event = events_by_lead.get(prefix_lead)
                if event is not None:
                    if state is None:
                        return result(
                            prefix_lead,
                            f"timeline interrupt targets finished slot "
                            f"{emitter.slot}",
                        )
                    state = replace(state, run_interrupt=event.interrupt_id)
                if state is not None:
                    prefix = forecast_ecl_births(
                        state,
                        (player_positions[prefix_lead],),
                        snapshot.difficulty,
                        snapshot.rank,
                        snapshot.bullet_sizes,
                        snapshot.frame_multiplier,
                        allow_player_variables=False,
                        radial_births=True,
                        abstract_rng=True,
                        enemy_kill_all_is_noop=False,
                        model_player_damage=False,
                        laser_world=laser_world,
                        repeat_star_state=repeat_star_state,
                        record_enemy_create=(
                            lambda frame, offset=prefix_lead: (
                                enemy_create_frames.add(offset + frame)
                            )
                        ),
                    )
                    if prefix.covered_frames < 1:
                        return result(
                            prefix_lead,
                            f"slot {emitter.slot} kill-all prefix: "
                            f"{prefix.reason}",
                        )
                    if prefix.created_emitters:
                        return result(
                            prefix_lead,
                            f"slot {emitter.slot} kill-all prefix creates "
                            "an enemy",
                        )
                    if enemy_create_frames:
                        return result(
                            prefix_lead,
                            f"slot {emitter.slot} kill-all prefix inserts "
                            "a live slot",
                        )
                    if any(prefix.enemy_kill_all):
                        return result(
                            prefix_lead,
                            f"slot {emitter.slot} kill-all prefix mutates "
                            "all slots",
                        )
                    state = prefix.next_spawner
                    repeat_star_state = prefix.repeat_star_state
                    if state is None and not prefix.finished:
                        return result(
                            prefix_lead + 1,
                            f"slot {emitter.slot} kill-all prefix has no "
                            "continuation",
                        )
                laser_world.advance_hazards()
                if _laser_world_changed(laser_world):
                    return result(
                        prefix_lead,
                        f"slot {emitter.slot} kill-all prefix changes the "
                        "shared laser world",
                    )
            if state is None:
                continue
        try:
            forced = forecast_ecl_forced_kill_all_update(
                state,
                player_positions[kill_lead],
                snapshot.difficulty,
                snapshot.rank,
                snapshot.bullet_sizes,
                snapshot.frame_multiplier,
                record_bullet_stop=(
                    lambda frame, offset=kill_lead: bullet_stop_frames.add(
                        offset + frame
                    )
                ),
                record_bullet_release=(
                    lambda frame, offset=kill_lead: bullet_release_frames.add(
                        offset + frame
                    )
                ),
                repeat_star_state=repeat_star_state,
                laser_world=laser_world,
                record_enemy_create=(
                    lambda frame, offset=kill_lead: enemy_create_frames.add(
                        offset + frame
                    )
                ),
            )
        except UnsupportedBirthModel as error:
            return result(
                kill_lead,
                f"slot {emitter.slot} forced kill-all: {error}",
            )
        births[kill_lead].extend(forced.births[0])
        if forced.body_hazards:
            bodies[kill_lead].extend(forced.body_hazards[0])
        if forced.covered_frames < 1:
            return result(
                kill_lead,
                f"slot {emitter.slot} forced kill-all: {forced.reason}",
            )
        if forced.created_emitters:
            return result(
                kill_lead,
                f"slot {emitter.slot} forced kill-all creates an enemy",
            )
        if enemy_create_frames:
            return result(
                kill_lead,
                f"slot {emitter.slot} forced kill-all inserts a live slot",
            )
        if any(forced.enemy_kill_all):
            return result(
                kill_lead,
                f"slot {emitter.slot} forced ECL mutates all slots",
            )
        laser_world.advance_hazards()
        if _laser_world_changed(laser_world):
            return result(
                kill_lead,
                f"slot {emitter.slot} forced kill-all changes the shared "
                "laser world",
            )
        state = forced.next_spawner
        repeat_star_state = forced.repeat_star_state
        if state is None:
            if forced.finished:
                continue
            return result(
                kill_lead + 1,
                f"slot {emitter.slot} forced kill-all has no continuation",
            )
        if kill_lead + 1 >= horizon:
            continue

        future_offset = kill_lead + 1
        for future_lead in range(future_offset, horizon):
            if state.interactable:
                return result(
                    future_lead,
                    f"slot {emitter.slot} post-kill continuation becomes "
                    "interactable",
                )
            future = forecast_ecl_births(
                state,
                (player_positions[future_lead],),
                snapshot.difficulty,
                snapshot.rank,
                snapshot.bullet_sizes,
                snapshot.frame_multiplier,
                allow_player_variables=False,
                radial_births=True,
                abstract_rng=True,
                enemy_kill_all_is_noop=False,
                model_player_damage=False,
                laser_world=laser_world,
                record_enemy_create=(
                    lambda frame, offset=future_lead: enemy_create_frames.add(
                        offset + frame
                    )
                ),
                record_bullet_stop=(
                    lambda frame, offset=future_lead: bullet_stop_frames.add(
                        offset + frame
                    )
                ),
                record_bullet_release=(
                    lambda frame, offset=future_lead: bullet_release_frames.add(
                        offset + frame
                    )
                ),
                repeat_star_state=repeat_star_state,
            )
            births[future_lead].extend(future.births[0])
            if future.body_hazards:
                bodies[future_lead].extend(future.body_hazards[0])
            if future.covered_frames < 1:
                return result(
                    future_lead,
                    f"slot {emitter.slot} post-kill callback: "
                    f"{future.reason}",
                )
            if future.created_emitters:
                return result(
                    future_lead,
                    f"slot {emitter.slot} post-kill callback creates an "
                    "enemy",
                )
            if enemy_create_frames:
                return result(
                    future_lead,
                    f"slot {emitter.slot} post-kill callback inserts a "
                    "live slot",
                )
            if any(future.enemy_kill_all):
                return result(
                    future_lead,
                    f"slot {emitter.slot} post-kill callback mutates all "
                    "slots",
                )
            state = future.next_spawner
            repeat_star_state = future.repeat_star_state
            laser_world.advance_hazards()
            if _laser_world_changed(laser_world):
                return result(
                    future_lead,
                    f"slot {emitter.slot} post-kill callback changes the "
                    "shared laser world",
                )
            if state is None:
                if future.finished:
                    break
                return result(
                    future_lead + 1,
                    f"slot {emitter.slot} post-kill callback has no "
                    "continuation",
                )
    return result(horizon)


def _forecast_hard_timeline_births(
    snapshot: Snapshot,
    player_positions: tuple[tuple[float, float], ...],
) -> WorldBirthForecast:
    """Insert deterministic timeline children into the bounded Hard world."""
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    lasers: list[list[LaserHazard]] = [[] for _ in player_positions]
    horizon = len(player_positions)
    laser_births = 0
    mutated_initial_lasers: list[int] = []
    missing_laser_dereferences: list[int] = []
    retired_future_laser = False
    laser_effect_worlds = 0
    bullet_stop_frames: set[int] = set()
    bullet_release_frames: set[int] = set()
    replaces_live_from_lead: int | None = None
    known_boss_ids = {
        emitter.boss_id
        for emitter in snapshot.spawners
        if emitter.boss_id >= 0
    }
    # At lead zero RunEclTimeline runs before any enemy update, so a fresh
    # true byte proves every current-time spawn is suppressed. At later leads
    # the separately forecast live boss may despawn first; keep those births
    # instead of turning current state into unsafe future pruning. ``None`` is
    # an older artifact and likewise keeps the conservative insertion.
    boss_present = snapshot.boss_present
    earlier_timeline_spawn = False
    for lead, instruction in scheduled_timeline(
        snapshot.timeline_instructions,
        snapshot.timeline_time,
        stage=snapshot.stage,
        difficulty=snapshot.difficulty,
        character=snapshot.character,
        message_delays=snapshot.timeline_message_delays,
        current_message_waits=snapshot.timeline_current_message_waits,
    ):
        if lead >= horizon:
            break
        if instruction.opcode == 10:
            event = decode_boss_interrupt(instruction)
            if (
                event is None
                or not _timeline_interrupt_has_resolved_target(
                    snapshot, event, known_boss_ids
                )
            ):
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    lead,
                    "unresolved stage timeline boss interrupt opcode 10 "
                    f"at 0x{instruction.address:08x}",
                    tuple(map(tuple, bodies)),
                    laser_effect_worlds=laser_effect_worlds,
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            continue
        if lead == 0 and boss_present is True:
            continue
        spawn = decode_enemy_spawn(instruction)
        if spawn is None:
            if 0 <= instruction.opcode <= 7:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(births, True),
                    lead,
                    "invalid stage timeline enemy spawn record "
                    f"at 0x{instruction.address:08x}",
                    tuple(map(tuple, bodies)),
                    laser_effect_worlds=laser_effect_worlds,
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(bullet_release_frames)),
                )
            continue
        child_x = (
            SOURCE_RANDOM_TIMELINE_WIDTH / 2.0
            if spawn.random_x else spawn.x
        )
        child_y = (
            SOURCE_RANDOM_TIMELINE_HEIGHT / 2.0
            if spawn.random_y else spawn.y
        )
        child = source_enemy_template(
            snapshot.timeline_ecl_program,
            snapshot.ecl_subroutines,
            spawn.sub_id,
            child_x,
            child_y,
            spawn.life if spawn.life is not None else -1,
            spawn.item_drop,
        )
        if child is None:
            return WorldBirthForecast(
                tuple(map(tuple, births)),
                _project_hazards(births, True),
                lead,
                "timeline enemy ECL graph is unavailable "
                f"for sub {spawn.sub_id}",
                tuple(map(tuple, bodies)),
                laser_effect_worlds=laser_effect_worlds,
                bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                bullet_release_frames=tuple(sorted(bullet_release_frames)),
            )
        child = replace(
            child,
            forecast_position_uncertainty_x=(
                SOURCE_RANDOM_TIMELINE_WIDTH / 2.0
                if spawn.random_x else 0.0
            ),
            forecast_position_uncertainty_y=(
                SOURCE_RANDOM_TIMELINE_HEIGHT / 2.0
                if spawn.random_y else 0.0
            ),
        )

        # SpawnEnemy executes time-zero ECL inline without the manager's
        # movement, bounds, callback, body/damage, or boss-timer work. The
        # manager then starts its slot loop at zero, so every timeline child
        # receives one ordinary update in the same source frame.
        laser_world = HardLaserWorld(snapshot.lasers)
        inline = forecast_ecl_births(
            child,
            (player_positions[lead],),
            snapshot.difficulty,
            snapshot.rank,
            snapshot.bullet_sizes,
            snapshot.frame_multiplier,
            allow_player_variables=False,
            radial_births=True,
            abstract_rng=True,
            # A true flag delegates pre-existing slot effects to the exact
            # forced-life replay below.  The interpreter still audits the
            # newborn's own same-call order and records whether kill-all was
            # reached.  A prior compact timeline child makes that external
            # replay unavailable because its slot state is not shared here.
            enemy_kill_all_is_noop=not earlier_timeline_spawn,
            model_player_damage=False,
            laser_world=laser_world,
            spawn_inline=True,
            record_bullet_stop=(
                lambda frame, offset=lead: bullet_stop_frames.add(offset + frame)
            ),
            record_bullet_release=(
                lambda frame, offset=lead: bullet_release_frames.add(
                    offset + frame
                )
            ),
            # Timeline children are forecast in a compact world separate from
            # the live slot loop.  Letting either world mutate the same retail
            # globals would invent an ordering, so Hard stops if the child
            # activates EXINSREPEAT(2).  Nominal replay owns the exact shared
            # state and models the real frame-first/slot-second order.
            repeat_star_state=None,
        )
        if inline.covered_frames < 1:
            return WorldBirthForecast(
                tuple(map(tuple, births)),
                _project_hazards(births, True),
                lead,
                f"timeline emitter {spawn.sub_id}: {inline.reason}",
                tuple(map(tuple, bodies)),
                laser_effect_worlds=(
                    laser_effect_worlds
                    + int(_laser_world_changed(laser_world))
                ),
                bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                bullet_release_frames=tuple(sorted(bullet_release_frames)),
            )
        if any(inline.enemy_kill_all):
            external = _forecast_live_slots_after_kill_all(
                snapshot,
                player_positions,
                lead,
            )
            for frame_index, frame_births in enumerate(external.births):
                births[frame_index].extend(frame_births)
            for frame_index, frame_bodies in enumerate(
                external.body_hazards
            ):
                bodies[frame_index].extend(frame_bodies)
            bullet_stop_frames.update(external.bullet_stop_frames)
            bullet_release_frames.update(external.bullet_release_frames)
            if external.covered_frames < horizon:
                return WorldBirthForecast(
                    tuple(map(tuple, births)),
                    _project_hazards(
                        births,
                        True,
                        tuple(sorted(bullet_release_frames)),
                    ),
                    external.covered_frames,
                    f"timeline emitter {spawn.sub_id} external kill-all: "
                    f"{external.reason}",
                    tuple(map(tuple, bodies)),
                    laser_effect_worlds=(
                        laser_effect_worlds
                        + int(_laser_world_changed(laser_world))
                    ),
                    bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                    bullet_release_frames=tuple(sorted(
                        bullet_release_frames
                    )),
                )
            replaces_live_from_lead = (
                lead
                if replaces_live_from_lead is None
                else min(replaces_live_from_lead, lead)
            )
        earlier_timeline_spawn = True
        births[lead].extend(inline.births[0])
        if inline.next_spawner is None:
            if inline.finished:
                for frame_index in range(lead, horizon):
                    laser_boxes, oriented_lasers = (
                        laser_world.advance_hazards()
                    )
                    bodies[frame_index].extend(laser_boxes)
                    lasers[frame_index].extend(oriented_lasers)
                laser_births += laser_world.created_count
                mutated_initial_lasers.extend(
                    laser_world.mutated_initial_slots
                )
                missing_laser_dereferences.extend(
                    laser_world.missing_dereferences
                )
                retired_future_laser |= laser_world.retired_created
                laser_effect_worlds += int(_laser_world_changed(laser_world))
                continue
            return WorldBirthForecast(
                tuple(map(tuple, births)),
                _project_hazards(births, True),
                lead + 1,
                f"timeline emitter {spawn.sub_id}: {inline.reason}",
                tuple(map(tuple, bodies)),
                laser_effect_worlds=(
                    laser_effect_worlds
                    + int(_laser_world_changed(laser_world))
                ),
                bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                bullet_release_frames=tuple(sorted(bullet_release_frames)),
            )

        child = replace(
            inline.next_spawner,
            invert_x=spawn.invert_x,
                life=(
                    spawn.life
                    if spawn.life is not None
                    else inline.next_spawner.life
                ),
                item_drop=spawn.item_drop,
        )
        if (
            any(inline.enemy_kill_all)
            and not child.is_boss
            and child.interactable
            and child.life <= 0
        ):
            return WorldBirthForecast(
                tuple(map(tuple, births)),
                _project_hazards(
                    births,
                    True,
                    tuple(sorted(bullet_release_frames)),
                ),
                lead,
                f"timeline emitter {spawn.sub_id}: interactive newborn "
                "kill-all death needs a shared ordinary slot replay",
                tuple(map(tuple, bodies)),
                laser_effect_worlds=(
                    laser_effect_worlds
                    + int(_laser_world_changed(laser_world))
                ),
                bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                bullet_release_frames=tuple(sorted(bullet_release_frames)),
            )
        if lead == 0 and boss_present is False and child.is_boss:
            # SpawnEnemy executes time-zero BOSSSET inline. A following
            # timeline record at the same timer observes the written byte.
            boss_present = True
        if child.boss_id >= 0:
            known_boss_ids.add(child.boss_id)
        ordinary = _forecast_hard_emitter(
            snapshot,
            child,
            player_positions,
            start_lead=lead,
            enemy_kill_all_is_noop=False,
            laser_world=laser_world,
            allow_repeat_star=False,
        )
        for offset, frame_births in enumerate(ordinary.births):
            births[offset].extend(frame_births)
        for offset, frame_bodies in enumerate(ordinary.body_hazards):
            bodies[offset].extend(frame_bodies)
        for offset, frame_lasers in enumerate(ordinary.laser_hazards):
            lasers[offset].extend(frame_lasers)
        laser_births += ordinary.laser_births
        mutated_initial_lasers.extend(ordinary.mutated_initial_lasers)
        missing_laser_dereferences.extend(
            ordinary.missing_laser_dereferences
        )
        retired_future_laser |= ordinary.retired_future_laser
        laser_effect_worlds += ordinary.laser_effect_worlds
        bullet_stop_frames.update(ordinary.bullet_stop_frames)
        bullet_release_frames.update(ordinary.bullet_release_frames)
        if ordinary.covered_frames < horizon:
            return WorldBirthForecast(
                tuple(map(tuple, births)),
                _project_hazards(births, True),
                ordinary.covered_frames,
                f"timeline emitter {spawn.sub_id}: {ordinary.reason}",
                tuple(map(tuple, bodies)),
                laser_effect_worlds=laser_effect_worlds,
                bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
                bullet_release_frames=tuple(sorted(bullet_release_frames)),
            )
    return WorldBirthForecast(
        tuple(map(tuple, births)),
        _project_hazards(births, True),
        horizon,
        body_hazards=tuple(map(tuple, bodies)),
        laser_births=laser_births,
        mutated_initial_lasers=tuple(sorted(mutated_initial_lasers)),
        missing_laser_dereferences=tuple(sorted(
            missing_laser_dereferences
        )),
        retired_future_laser=retired_future_laser,
        laser_effect_worlds=laser_effect_worlds,
        laser_hazards=tuple(map(tuple, lasers)),
        bullet_stop_frames=tuple(sorted(bullet_stop_frames)),
        bullet_release_frames=tuple(sorted(bullet_release_frames)),
        replaces_live_from_lead=replaces_live_from_lead,
    )


def _forecast_nominal_without_shared_rng(
    snapshot: Snapshot,
    player_positions: tuple[tuple[float, float], ...],
    emitters: tuple[EnemySpawner, ...],
    rng: RngState,
    boss_present: bool | None,
    repeat_star_state: RepeatStarState | None,
) -> WorldBirthForecast | None:
    """Batch independent emitters only after proving that none reads RNG.

    EnemyManager normally advances ECL frame-first and slot-second because all
    emitters share one RNG.  A no-RNG interval is commutative: each captured
    emitter can advance across the whole interval once.  Unsupported global
    behavior, incomplete coverage, or the first RNG read discards this fast
    path and leaves the ordinary framewise model authoritative.
    """
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    if any(
        _program_can_create_enemy(emitter)
        or _program_can_repeat_star(emitter)
        for emitter in emitters
    ):
        # A child must join the manager's slot-ordered loop immediately; a
        # whole-emitter batch cannot preserve that interleaving.
        return None
    next_emitters = []
    for emitter in emitters:
        try:
            forecast = forecast_ecl_births(
                emitter,
                player_positions,
                snapshot.difficulty,
                snapshot.rank,
                snapshot.bullet_sizes,
                snapshot.frame_multiplier,
                _NoRngState(rng.seed, rng.generation_count),
                allow_player_variables=True,
                radial_births=False,
                repeat_star_state=repeat_star_state,
            )
        except (_NominalRngConsumed, UnsupportedBirthModel):
            return None
        if (
            forecast.covered_frames != len(player_positions)
            or (
                forecast.next_spawner is None
                and not forecast.finished
            )
        ):
            return None
        for frame_index, frame_births in enumerate(forecast.births):
            births[frame_index].extend(frame_births)
        for frame_index, frame_bodies in enumerate(forecast.body_hazards):
            bodies[frame_index].extend(frame_bodies)
        if forecast.next_spawner is not None:
            next_emitters.append(forecast.next_spawner)
        boss_present = _boss_present_after_emitter(
            boss_present, emitter, forecast.next_spawner
        )
    return WorldBirthForecast(
        tuple(tuple(frame) for frame in births),
        _project_hazards(births, False),
        len(player_positions),
        body_hazards=tuple(tuple(frame) for frame in bodies),
        continuation=WorldForecastContinuation(
            tuple(next_emitters),
            rng.seed,
            rng.generation_count,
            True,
            len(player_positions),
            boss_present,
            repeat_star_state,
        ),
    )


def _nominal_timeline_transitions(
    snapshot: Snapshot,
    start_lead: int,
    horizon: int,
) -> tuple[tuple[int, StageTimelineInstruction], ...]:
    """Return source world writes inside one nominal continuation slice."""
    end_lead = start_lead + horizon
    return tuple(
        (lead, instruction)
        for lead, instruction in scheduled_timeline(
            snapshot.timeline_instructions,
            snapshot.timeline_time,
            stage=snapshot.stage,
            difficulty=snapshot.difficulty,
            character=snapshot.character,
            message_delays=snapshot.timeline_message_delays,
            current_message_waits=snapshot.timeline_current_message_waits,
        )
        if start_lead <= lead < end_lead
        and instruction.opcode in (*range(8), 10)
    )


def _timeline_random_position(
    spawn: TimelineEnemySpawn,
    rng: RngState,
) -> tuple[float, float]:
    """Consume RunEclTimeline's x/y/z RNG in exact source order."""
    x = (
        rng.f32_in_range(SOURCE_RANDOM_TIMELINE_WIDTH)
        if spawn.random_x else spawn.x
    )
    y = (
        rng.f32_in_range(SOURCE_RANDOM_TIMELINE_HEIGHT)
        if spawn.random_y else spawn.y
    )
    if spawn.random_z:
        rng.f32_in_range(800.0)
    return x, y


def _forecast_nominal_from_state(
    snapshot: Snapshot,
    player_positions: tuple[tuple[float, float], ...],
    emitters: tuple[EnemySpawner, ...],
    rng: RngState,
    *,
    framewise: bool,
    start_lead: int = 0,
    combat=None,
    boss_present: bool | None = None,
    repeat_star_state: RepeatStarState | None = None,
) -> WorldBirthForecast:
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    timeline_transitions = _nominal_timeline_transitions(
        snapshot, start_lead, len(player_positions)
    )
    if boss_present is None:
        boss_present = snapshot.boss_present
    if repeat_star_state is None:
        repeat_star_state = snapshot.repeat_star_state
    if timeline_transitions:
        framewise = True
    if combat is not None:
        framewise = True
    if combat is None and timeline_transitions:
        # Preserve the frame/slot ordering through the last timeline write,
        # and let the existing no-shared-RNG guard batch untouched prefixes
        # and tails. Previously one spawn anywhere in the window forced every
        # frame through the 255-slot loop even when the exact ECL slices on
        # either side were commutative. If a slice reads RNG or creates a
        # child, that guard returns ``None`` and the ordinary framewise path
        # remains authoritative.
        transition_prefix_start = (
            min(lead for lead, _instruction in timeline_transitions)
            - start_lead
        )
        if transition_prefix_start > 0:
            prefix = _forecast_nominal_from_state(
                snapshot,
                player_positions[:transition_prefix_start],
                emitters,
                rng,
                framewise=framewise,
                start_lead=start_lead,
                boss_present=boss_present,
                repeat_star_state=repeat_star_state,
            )
            if (
                prefix.covered_frames == transition_prefix_start
                and prefix.continuation is not None
            ):
                return extend_nominal_world_births(
                    snapshot,
                    prefix,
                    player_positions[transition_prefix_start:],
                )
        transition_prefix_frames = (
            max(lead for lead, _instruction in timeline_transitions)
            - start_lead
            + 1
        )
        if transition_prefix_frames < len(player_positions):
            prefix = _forecast_nominal_from_state(
                snapshot,
                player_positions[:transition_prefix_frames],
                emitters,
                rng,
                framewise=True,
                start_lead=start_lead,
                boss_present=boss_present,
                repeat_star_state=repeat_star_state,
            )
            if (
                prefix.covered_frames == transition_prefix_frames
                and prefix.continuation is not None
            ):
                return extend_nominal_world_births(
                    snapshot,
                    prefix,
                    player_positions[transition_prefix_frames:],
                )
    if not framewise:
        if not emitters:
            return WorldBirthForecast(
                tuple(tuple(frame) for frame in births),
                _project_hazards(births, False),
                len(player_positions),
                body_hazards=tuple(tuple(frame) for frame in bodies),
                continuation=WorldForecastContinuation(
                    (), rng.seed, rng.generation_count, False,
                    start_lead + len(player_positions),
                    boss_present,
                    repeat_star_state,
                ),
            )
        if len(emitters) != 1:
            raise ValueError("batched nominal continuation needs one emitter")
        emitter = emitters[0]
        try:
            forecast = forecast_ecl_births(
                emitter,
                player_positions,
                snapshot.difficulty,
                snapshot.rank,
                snapshot.bullet_sizes,
                snapshot.frame_multiplier,
                rng,
                allow_player_variables=True,
                radial_births=False,
                # The original single-emitter path intentionally assumes no
                # unknown future player damage. Preserve that exact contract
                # when an already-started forecast is extended.
                model_player_damage=False,
                repeat_star_state=repeat_star_state,
            )
        except UnsupportedBirthModel as error:
            return WorldBirthForecast(
                tuple(tuple(frame) for frame in births),
                _project_hazards(births, False),
                0,
                f"emitter {emitter.slot}: {error}",
            )
        for frame_index, frame_births in enumerate(forecast.births):
            births[frame_index].extend(frame_births)
        for frame_index, frame_bodies in enumerate(forecast.body_hazards):
            bodies[frame_index].extend(frame_bodies)
        next_emitters = (
            (forecast.next_spawner,)
            if forecast.next_spawner is not None
            else ()
        )
        continuation = (
            WorldForecastContinuation(
                next_emitters,
                rng.seed,
                rng.generation_count,
                False,
                start_lead + len(player_positions),
                _boss_present_after_emitter(
                    boss_present, emitter, forecast.next_spawner
                ),
                forecast.repeat_star_state,
            )
            if (
                forecast.covered_frames == len(player_positions)
                and (
                    forecast.next_spawner is not None
                    or forecast.finished
                )
            )
            else None
        )
        return WorldBirthForecast(
            tuple(tuple(frame) for frame in births),
            _project_hazards(births, False),
            forecast.covered_frames,
            forecast.reason,
            tuple(tuple(frame) for frame in bodies),
            continuation,
        )

    if not timeline_transitions and combat is None:
        batched = _forecast_nominal_without_shared_rng(
            snapshot,
            player_positions,
            emitters,
            rng,
            boss_present,
            repeat_star_state,
        )
        if batched is not None:
            return replace(
                batched,
                continuation=replace(
                    batched.continuation,
                    elapsed_frames=start_lead + len(player_positions),
                ),
            )

    if (
        any(
            not 0 <= emitter.slot < SOURCE_ENEMY_SLOT_COUNT
            for emitter in emitters
        )
        or len({emitter.slot for emitter in emitters}) != len(emitters)
    ):
        return WorldBirthForecast(
            tuple(tuple(frame) for frame in births),
            _project_hazards(births, False),
            0,
            "nominal enemy slot occupancy is incomplete",
        )
    slots = {emitter.slot: emitter for emitter in emitters}
    timeline_by_lead: dict[int, list] = {}
    for lead, instruction in timeline_transitions:
        timeline_by_lead.setdefault(lead, []).append(instruction)
    for frame_index, player in enumerate(player_positions):
        source_lead = start_lead + frame_index
        for instruction in timeline_by_lead.get(source_lead, ()):
            if instruction.opcode == 10:
                event = decode_boss_interrupt(instruction)
                matching = [
                    slot for slot, emitter in slots.items()
                    if event is not None and emitter.boss_id == event.boss_id
                ]
                if event is None or len(matching) != 1:
                    return WorldBirthForecast(
                        tuple(tuple(frame) for frame in births),
                        _project_hazards(births, False),
                        frame_index,
                        "nominal timeline boss interrupt target is unresolved "
                        f"at 0x{instruction.address:08x}",
                        tuple(tuple(frame) for frame in bodies),
                    )
                slot = matching[0]
                slots[slot] = replace(
                    slots[slot], run_interrupt=event.interrupt_id
                )
                continue

            # The source gate is Gui::bossPresent. Older artifacts did not
            # capture it, so retain the previous slot-based fallback only for
            # those records rather than inventing a byte value.
            timeline_boss_present = (
                boss_present
                if boss_present is not None
                else any(emitter.is_boss for emitter in slots.values())
            )
            if timeline_boss_present:
                continue
            spawn = decode_enemy_spawn(instruction)
            if spawn is None:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    "invalid nominal stage timeline enemy spawn record "
                    f"at 0x{instruction.address:08x}",
                    tuple(tuple(frame) for frame in bodies),
                )
            free_slot = next(
                (
                    slot for slot in range(SOURCE_ENEMY_SLOT_COUNT)
                    if slot not in slots
                ),
                None,
            )
            if free_slot is None:
                # SpawnEnemy returns without initializing another occupied
                # slot when the 255-entry source pool is full.
                continue
            x, y = _timeline_random_position(spawn, rng)
            child = source_enemy_template(
                snapshot.timeline_ecl_program,
                snapshot.ecl_subroutines,
                spawn.sub_id,
                x,
                y,
                spawn.life if spawn.life is not None else -1,
                spawn.item_drop,
            )
            if child is None:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    "nominal timeline enemy ECL graph is unavailable "
                    f"for sub {spawn.sub_id}",
                    tuple(tuple(frame) for frame in bodies),
                )
            child = replace(child, slot=free_slot)
            inline = forecast_ecl_births(
                child,
                (player,),
                snapshot.difficulty,
                snapshot.rank,
                snapshot.bullet_sizes,
                snapshot.frame_multiplier,
                rng,
                allow_player_variables=True,
                radial_births=False,
                model_player_damage=False,
                record_enemy_kill_all=combat is not None,
                laser_world=combat,
                spawn_inline=True,
                repeat_star_state=repeat_star_state,
            )
            if inline.covered_frames < 1:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    f"nominal timeline emitter {spawn.sub_id}: "
                    f"{inline.reason}",
                    tuple(tuple(frame) for frame in bodies),
                )
            births[frame_index].extend(inline.births[0])
            repeat_star_state = inline.repeat_star_state
            if combat is not None and inline.effect_spawns:
                combat.observe_effect_spawns(inline.effect_spawns[0])
            if combat is not None and inline.item_spawns:
                combat.observe_item_spawns(
                    inline.item_births[0] if inline.item_births else (),
                    inline.item_spawns[0],
                    rng,
                )
            if combat is not None and (
                inline.enemy_kill_all and inline.enemy_kill_all[0]
            ):
                combat.enemy_kill_all(slots)
            if inline.body_hazards:
                bodies[frame_index].extend(inline.body_hazards[0])
            if inline.created_emitters:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    "timeline newborn creates a nested enemy inline before "
                    "nominal slot insertion is resolved",
                    tuple(tuple(frame) for frame in bodies),
                )
            if inline.next_spawner is not None:
                inserted = replace(
                    inline.next_spawner,
                    slot=free_slot,
                    invert_x=spawn.invert_x,
                    life=(
                        spawn.life
                        if spawn.life is not None
                        else inline.next_spawner.life
                    ),
                    item_drop=spawn.item_drop,
                )
                slots[free_slot] = inserted
                boss_present = _boss_present_after_emitter(
                    boss_present, child, inserted
                )
            elif not inline.finished:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index + 1,
                    f"nominal timeline emitter {spawn.sub_id}: "
                    f"{inline.reason}",
                    tuple(tuple(frame) for frame in bodies),
                )
            else:
                boss_present = _boss_present_after_emitter(
                    boss_present, child, None
                )

        for slot in range(SOURCE_ENEMY_SLOT_COUNT):
            emitter = slots.get(slot)
            if emitter is None:
                continue
            if combat is not None:
                emitter = combat.pre_emitter(emitter, slots)
                slots[slot] = emitter
            try:
                forecast = forecast_ecl_births(
                    emitter,
                    (player,),
                    snapshot.difficulty,
                    snapshot.rank,
                    snapshot.bullet_sizes,
                    snapshot.frame_multiplier,
                    rng,
                    allow_player_variables=True,
                    radial_births=False,
                    model_player_damage=combat is None,
                    record_enemy_kill_all=combat is not None,
                    laser_world=combat,
                    repeat_star_state=repeat_star_state,
                )
            except UnsupportedBirthModel as error:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    f"emitter {emitter.slot}: {error}",
                )
            if forecast.covered_frames < 1:
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    f"emitter {emitter.slot}: {forecast.reason}",
                )
            births[frame_index].extend(forecast.births[0])
            repeat_star_state = forecast.repeat_star_state
            if combat is not None and forecast.effect_spawns:
                combat.observe_effect_spawns(forecast.effect_spawns[0])
            if combat is not None and forecast.item_spawns:
                combat.observe_item_spawns(
                    forecast.item_births[0] if forecast.item_births else (),
                    forecast.item_spawns[0],
                    rng,
                )
            if combat is not None and (
                forecast.enemy_kill_all
                and forecast.enemy_kill_all[0]
            ):
                combat.enemy_kill_all(slots)
            if forecast.body_hazards:
                bodies[frame_index].extend(forecast.body_hazards[0])
            free_slots = [
                index for index in range(SOURCE_ENEMY_SLOT_COUNT)
                if index not in slots
            ]
            if len(forecast.created_emitters) > len(free_slots):
                births[frame_index].clear()
                bodies[frame_index].clear()
                return WorldBirthForecast(
                    tuple(tuple(frame) for frame in births),
                    _project_hazards(births, False),
                    frame_index,
                    "future ECL enemy creation exceeds the free slot pool",
                )
            # SpawnEnemy allocates and runs each child inline while the parent
            # remains occupied. Assign in creation order before retiring the
            # parent. A lower slot has already missed this manager pass; a
            # higher slot is reached later by this same loop.
            for child, child_slot in zip(
                forecast.created_emitters, free_slots
            ):
                slots[child_slot] = replace(child, slot=child_slot)
                if child.is_boss:
                    boss_present = True
            if forecast.next_spawner is None:
                if not forecast.finished:
                    return WorldBirthForecast(
                        tuple(tuple(frame) for frame in births),
                        _project_hazards(births, False),
                        frame_index + 1,
                        f"emitter {emitter.slot}: {forecast.reason}",
                    )
                slots.pop(slot, None)
                boss_present = _boss_present_after_emitter(
                    boss_present, emitter, None
                )
            else:
                next_emitter = replace(forecast.next_spawner, slot=slot)
                boss_present = _boss_present_after_emitter(
                    boss_present, emitter, next_emitter
                )
                if combat is not None:
                    next_emitter = combat.post_emitter(next_emitter, rng)
                    combat_write = combat.consume_boss_present_write()
                    if combat_write is not None:
                        boss_present = combat_write
                if next_emitter is None:
                    slots.pop(slot, None)
                else:
                    slots[slot] = next_emitter
        if combat is not None:
            combat.finish_frame(rng)
        emitters = tuple(slots[index] for index in sorted(slots))

    return WorldBirthForecast(
        tuple(tuple(frame) for frame in births),
        _project_hazards(births, False),
        len(player_positions),
        body_hazards=tuple(tuple(frame) for frame in bodies),
        continuation=WorldForecastContinuation(
            emitters,
            rng.seed,
            rng.generation_count,
            True,
            start_lead + len(player_positions),
            boss_present,
            repeat_star_state,
        ),
    )


def extend_nominal_world_births(
    snapshot: Snapshot,
    prefix: WorldBirthForecast,
    player_positions: tuple[tuple[float, float], ...],
) -> WorldBirthForecast:
    """Extend one complete nominal prefix from its exact ECL/RNG state."""
    if prefix.continuation is None:
        raise ValueError("nominal world prefix has no exact continuation")
    if prefix.covered_frames != len(prefix.births):
        raise ValueError("cannot extend a partially covered nominal prefix")
    continuation = prefix.continuation
    tail = _forecast_nominal_from_state(
        snapshot,
        player_positions,
        continuation.emitters,
        RngState(
            continuation.rng_seed,
            continuation.rng_generation,
        ),
        framewise=continuation.framewise,
        start_lead=continuation.elapsed_frames,
        boss_present=continuation.boss_present,
        repeat_star_state=continuation.repeat_star_state,
    )
    births = prefix.births + tail.births
    # An empty body_hazards tuple is the compact representation for "no body
    # hazards in this slice". A partial tail can legitimately use that form
    # after stopping on an unsupported future instruction; materialize both
    # slices before concatenation so frame indices remain aligned.
    prefix_bodies = (
        prefix.body_hazards
        if prefix.body_hazards
        else ((),) * len(prefix.births)
    )
    tail_bodies = (
        tail.body_hazards
        if tail.body_hazards
        else ((),) * len(tail.births)
    )
    if (
        len(prefix_bodies) != len(prefix.births)
        or len(tail_bodies) != len(tail.births)
    ):
        raise ValueError("nominal body forecast is not frame-aligned")
    bodies = prefix_bodies + tail_bodies
    prefix_horizon = len(prefix.births)
    total_horizon = len(births)
    hazards = [list(frame) for frame in prefix.hazards]
    hazards.extend([] for _ in tail.births)
    # Preserve every already-published prefix box. Only project older births
    # into the appended frames, then append the tail's newly born bullets in
    # the same source birth-frame order as one full forecast.
    for birth_frame, frame_births in enumerate(prefix.births):
        remaining = total_horizon - birth_frame
        for bullet in frame_births:
            projected = hazard_boxes(bullet, remaining)
            for frame_index in range(prefix_horizon, total_horizon):
                hazards[frame_index].append(
                    projected[frame_index - birth_frame]
                )
    for frame_index, frame_hazards in enumerate(
        tail.hazards,
        prefix_horizon,
    ):
        hazards[frame_index].extend(frame_hazards)
    return WorldBirthForecast(
        births,
        tuple(tuple(frame) for frame in hazards),
        prefix.covered_frames + tail.covered_frames,
        tail.reason,
        bodies,
        tail.continuation,
    )


def forecast_world_births(
    snapshot: Snapshot,
    player_positions: tuple[tuple[float, float], ...],
    rng_mode: Literal["fail-closed", "nominal"] = "fail-closed",
    nominal_combat=None,
) -> WorldBirthForecast:
    """Advance emitters frame-first and slot-second, matching EnemyManager.

    ``nominal`` reproduces the observed RNG stream for validation and proposal
    ranking. Hard authority uses ``fail-closed``: reaching a future random
    consumer ends coverage instead of assuming that no other subsystem has
    advanced the global RNG.
    """
    if rng_mode not in ("fail-closed", "nominal"):
        raise ValueError(f"unknown RNG mode {rng_mode}")
    if nominal_combat is not None and (
        rng_mode != "nominal" or len(player_positions) != 1
    ):
        raise ValueError("nominal combat advances exactly one nominal frame")
    births: list[list[Bullet]] = [[] for _ in player_positions]
    bodies: list[list[tuple[float, float, float, float]]] = [
        [] for _ in player_positions
    ]
    lasers: list[list[LaserHazard]] = [[] for _ in player_positions]
    emitters = tuple(sorted(snapshot.spawners, key=lambda item: item.slot))
    if rng_mode == "fail-closed":
        star_emitters = tuple(
            emitter for emitter in emitters
            if _program_can_repeat_star(emitter)
        )
        if len(star_emitters) > 1:
            return WorldBirthForecast(
                tuple(tuple(frame) for frame in births),
                _project_hazards(births, True),
                0,
                "multiple emitters can overwrite shared repeating-star globals",
                tuple(tuple(frame) for frame in bodies),
            )
        # Independent emitter envelopes may only record a live boss's
        # ENEMYKILLALL when the capture and stage timeline prove there is no
        # external non-boss target. ECL-created children revoke that proof
        # inside the emitter forecaster. Every other world keeps the ordinary
        # fail-closed opcode behavior.
        enemy_kill_all_is_noop = _live_kill_all_has_no_external_target(
            snapshot,
            len(player_positions),
        )
        # A boss in the first occupied slot executes before every possible
        # ENEMYKILLALL target.  In that one source-proven ordering, the
        # existing pre-slot forced-life replay also models the live ECL event.
        # Any earlier occupied slot keeps the old fail-closed behavior.
        live_kill_all_issuer = (
            emitters[0]
            if (
                len(emitters) > 1
                and emitters[0].is_boss
                and snapshot.timeline_complete
            )
            else None
        )
        live_kill_all_frames: list[int] = []
        covered_frames = len(player_positions)
        reason = ""
        laser_births = 0
        laser_creating_worlds = 0
        mutated_initial_lasers: list[int] = []
        missing_laser_dereferences: list[int] = []
        retired_future_laser = False
        laser_effect_worlds = 0
        bullet_stop_frames: list[int] = []
        bullet_release_frames: list[int] = []
        emitter_failures: list[tuple[int, str, bool]] = []
        timeline = _forecast_hard_timeline_births(
            snapshot,
            player_positions,
        )
        emitter_forecasts: list[
            tuple[EnemySpawner, WorldBirthForecast, bool]
        ] = []
        for emitter in emitters:
            reachable_laser_creates: set[int] = set()
            forecast = _forecast_hard_emitter(
                snapshot,
                emitter,
                player_positions,
                enemy_kill_all_is_noop=(
                    enemy_kill_all_is_noop
                    or emitter is live_kill_all_issuer
                ),
                record_laser_create=reachable_laser_creates.add,
            )
            emitter_forecasts.append((
                emitter,
                forecast,
                bool(reachable_laser_creates),
            ))
        # A lone create-only world is already conservatively enclosed by the
        # branch-unioned static beam projection; its pool identity cannot be
        # observed elsewhere.  With two independent owners, allocation order
        # becomes source-visible, so every statically projected creator must
        # be replayed through its mutable pool before cross-world validation.
        independent_laser_owners = (
            timeline.laser_effect_worlds
            + sum(
                int(has_create or forecast.laser_effect_worlds > 0)
                for _emitter, forecast, has_create in emitter_forecasts
            )
        )
        if independent_laser_owners > 1:
            emitter_forecasts = [
                (
                    emitter,
                    _forecast_hard_emitter(
                        snapshot,
                        emitter,
                        player_positions,
                        enemy_kill_all_is_noop=(
                            enemy_kill_all_is_noop
                            or emitter is live_kill_all_issuer
                        ),
                        force_mutable_laser_pool=True,
                    ),
                    has_create,
                )
                if has_create and forecast.laser_effect_worlds == 0
                else (emitter, forecast, has_create)
                for emitter, forecast, has_create in emitter_forecasts
            ]
        for emitter, forecast, _has_create in emitter_forecasts:
            emitter_coverage = forecast.covered_frames
            emitter_reason = forecast.reason
            for frame_index, frame_births in enumerate(forecast.births):
                births[frame_index].extend(frame_births)
            for frame_index, frame_bodies in enumerate(forecast.body_hazards):
                bodies[frame_index].extend(frame_bodies)
            for frame_index, frame_lasers in enumerate(
                forecast.laser_hazards
            ):
                lasers[frame_index].extend(frame_lasers)
            laser_births += forecast.laser_births
            laser_creating_worlds += int(forecast.laser_births > 0)
            mutated_initial_lasers.extend(forecast.mutated_initial_lasers)
            missing_laser_dereferences.extend(
                forecast.missing_laser_dereferences
            )
            retired_future_laser |= forecast.retired_future_laser
            laser_effect_worlds += forecast.laser_effect_worlds
            bullet_stop_frames.extend(forecast.bullet_stop_frames)
            bullet_release_frames.extend(forecast.bullet_release_frames)
            live_kill_all_frames.extend(forecast.enemy_kill_all_frames)
            if emitter_coverage < len(player_positions):
                emitter_failures.append((
                    emitter_coverage,
                    f"emitter {emitter.slot}: {emitter_reason}",
                    emitter.is_boss,
                ))
        for frame_index, frame_births in enumerate(timeline.births):
            births[frame_index].extend(frame_births)
        for frame_index, frame_bodies in enumerate(timeline.body_hazards):
            bodies[frame_index].extend(frame_bodies)
        for frame_index, frame_lasers in enumerate(timeline.laser_hazards):
            lasers[frame_index].extend(frame_lasers)
        laser_births += timeline.laser_births
        laser_creating_worlds += int(timeline.laser_births > 0)
        mutated_initial_lasers.extend(timeline.mutated_initial_lasers)
        missing_laser_dereferences.extend(
            timeline.missing_laser_dereferences
        )
        retired_future_laser |= timeline.retired_future_laser
        laser_effect_worlds += timeline.laser_effect_worlds
        bullet_stop_frames.extend(timeline.bullet_stop_frames)
        bullet_release_frames.extend(timeline.bullet_release_frames)
        live_replaces_from_lead: int | None = None
        if live_kill_all_frames:
            kill_lead = min(live_kill_all_frames)
            prior_timeline_spawn = next((
                lead
                for lead, instruction in scheduled_timeline(
                    snapshot.timeline_instructions,
                    snapshot.timeline_time,
                    stage=snapshot.stage,
                    difficulty=snapshot.difficulty,
                    character=snapshot.character,
                    message_delays=snapshot.timeline_message_delays,
                    current_message_waits=snapshot.timeline_current_message_waits,
                )
                if lead <= kill_lead and 0 <= instruction.opcode <= 7
            ), None)
            if prior_timeline_spawn is not None:
                covered_frames = min(covered_frames, kill_lead)
                reason = (
                    "live ENEMYKILLALL follows a timeline-created target "
                    "inside the source window"
                )
            elif timeline.replaces_live_from_lead is not None:
                covered_frames = min(covered_frames, kill_lead)
                reason = "multiple kill-all world transitions share one source window"
            else:
                external = _forecast_live_slots_after_kill_all(
                    snapshot,
                    player_positions,
                    kill_lead,
                )
                for frame_index, frame_births in enumerate(external.births):
                    births[frame_index].extend(frame_births)
                for frame_index, frame_bodies in enumerate(
                    external.body_hazards
                ):
                    bodies[frame_index].extend(frame_bodies)
                bullet_stop_frames.extend(external.bullet_stop_frames)
                bullet_release_frames.extend(external.bullet_release_frames)
                if external.covered_frames < len(player_positions):
                    covered_frames = min(
                        covered_frames,
                        external.covered_frames,
                    )
                    reason = (
                        "live emitter external kill-all: "
                        f"{external.reason}"
                    )
                else:
                    live_replaces_from_lead = kill_lead
        for emitter_coverage, emitter_reason, emitter_is_boss in emitter_failures:
            replacement_leads = tuple(
                lead
                for lead in (
                    timeline.replaces_live_from_lead,
                    live_replaces_from_lead,
                )
                if lead is not None
            )
            if (
                not emitter_is_boss
                and replacement_leads
                and emitter_coverage >= min(replacement_leads)
            ):
                continue
            if emitter_coverage < covered_frames:
                covered_frames = emitter_coverage
                reason = emitter_reason
        if timeline.covered_frames < covered_frames:
            covered_frames = timeline.covered_frames
            reason = timeline.reason
        active_laser_slots = {laser.slot for laser in snapshot.lasers}
        allocated_future_slots = set(
            slot
            for slot in range(64)
            if slot not in active_laser_slots
        )
        allocated_future_slots = set(sorted(allocated_future_slots)[:laser_births])
        stale_alias = (
            laser_effect_worlds > 1
            and bool(allocated_future_slots.intersection(
                missing_laser_dereferences
            ))
        )
        aliased_mutation = (
            len(mutated_initial_lasers)
            != len(set(mutated_initial_lasers))
        )
        if stale_alias:
            covered_frames = 0
            reason = "future laser allocation may alias a stale ECL pointer"
        elif snapshot.laser_count + laser_births > 64:
            covered_frames = 0
            reason = "future laser allocation exceeds the source pool"
        elif aliased_mutation:
            covered_frames = 0
            reason = "multiple emitters mutate one aliased source laser"
        elif retired_future_laser and laser_creating_worlds > 1:
            covered_frames = 0
            reason = "future laser retirement may change cross-emitter allocation"
        return WorldBirthForecast(
            tuple(tuple(frame) for frame in births),
            _project_hazards(
                births,
                True,
                tuple(sorted(set(bullet_release_frames))),
            ),
            covered_frames,
            reason,
            tuple(tuple(frame) for frame in bodies),
            laser_births=laser_births,
            mutated_initial_lasers=tuple(sorted(mutated_initial_lasers)),
            missing_laser_dereferences=tuple(sorted(
                missing_laser_dereferences
            )),
            retired_future_laser=retired_future_laser,
            laser_effect_worlds=laser_effect_worlds,
            laser_hazards=tuple(map(tuple, lasers)),
            bullet_stop_frames=tuple(sorted(set(bullet_stop_frames))),
            bullet_release_frames=tuple(sorted(set(bullet_release_frames))),
        )
    return _forecast_nominal_from_state(
        snapshot,
        player_positions,
        emitters,
        RngState(snapshot.rng_seed, snapshot.rng_generation),
        framewise=(
            len(emitters) != 1
            or any(
                _program_can_create_enemy(emitter)
                for emitter in emitters
            )
        ),
        combat=nominal_combat,
    )
