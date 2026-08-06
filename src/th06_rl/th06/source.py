"""Lower one coherent TH06 snapshot into compact native hazard frames."""

from __future__ import annotations

from dataclasses import dataclass

from ..core.model import Action, Kinematics, movement_actions
from ..native import Aabb, LaserRect, PackedHazards
from .donor import enable_donor_imports
from .observed_lasers import laser_rects_by_frame


HARD_HORIZON = 4
COLLISION_MARGIN = 0.35
ALL_ACTIONS = movement_actions()
ACTION_BY_STATE = {
    (action.dx, action.dy, action.focused): action for action in ALL_ACTIONS
}


class AuthorityUnavailable(RuntimeError):
    pass


@dataclass(frozen=True)
class SourceForecast:
    hazards: PackedHazards
    hard_horizon: int
    requested_horizon: int
    source_coverage: int
    coverage_reason: str

    @property
    def full_horizon(self) -> bool:
        return self.source_coverage == self.requested_horizon


def core_action_from_input(input_mask: int) -> Action:
    enable_donor_imports()
    from th06.model import action_from_input

    source = action_from_input(input_mask)
    return ACTION_BY_STATE[(source.dx, source.dy, source.focused)]


def donor_action(action: Action):
    enable_donor_imports()
    from th06.model import CONTROL_ACTIONS

    return next(
        item for item in CONTROL_ACTIONS
        if (
            item.dx,
            item.dy,
            item.focused,
        ) == (action.dx, action.dy, action.focused)
    )


def kinematics_from_snapshot(snapshot) -> Kinematics:
    return Kinematics(
        normal_speed=snapshot.normal_speed,
        focus_speed=snapshot.focus_speed,
        normal_diagonal_speed=snapshot.normal_diagonal_speed,
        focus_diagonal_speed=snapshot.focus_diagonal_speed,
    )


def automatic_source_context(snapshot) -> str:
    """Stable source identity for data partitioning, never movement control."""
    direct = getattr(snapshot, "source_context", None)
    if direct:
        return str(direct)
    bosses = tuple(
        sorted(
            (spawner for spawner in snapshot.spawners if spawner.is_boss),
            key=lambda item: (item.boss_id, item.slot),
        )
    )
    if bosses:
        boss = bosses[0]
        instruction = boss.next_instruction
        containing = tuple(
            (address, index)
            for index, address in enumerate(boss.ecl_subroutines)
            if instruction is not None and address <= instruction.address
        )
        subroutine = str(max(containing)[1]) if containing else "unknown"
        spell = bool(
            snapshot.player_attack is not None
            and snapshot.player_attack.spell_active
        )
        return ":".join((
            "boss",
            str(boss.boss_id),
            f"sub{subroutine}",
            f"life_cb{boss.life_callback_sub}",
            f"timer_cb{boss.timer_callback_sub}",
            "spell" if spell else "nonspell",
        ))
    if snapshot.timeline_instructions:
        instruction = snapshot.timeline_instructions[0]
        return (
            f"timeline:before-t{instruction.time}:"
            f"op{instruction.opcode}:arg{instruction.arg0}"
        )
    return (
        "timeline-complete"
        if snapshot.timeline_complete
        else "timeline-unknown"
    )


def _reachable_aabbs(snapshot, frames, margin: float):
    speed = max(snapshot.normal_speed, snapshot.focus_speed)
    result = []
    for index, frame in enumerate(frames):
        steps = index + 1
        minimum_x = max(8.0, snapshot.x - speed * steps) \
            - snapshot.half_width - margin
        maximum_x = min(376.0, snapshot.x + speed * steps) \
            + snapshot.half_width + margin
        minimum_y = max(16.0, snapshot.y - speed * steps) \
            - snapshot.half_height - margin
        maximum_y = min(432.0, snapshot.y + speed * steps) \
            + snapshot.half_height + margin
        result.append(tuple(
            hazard for hazard in frame
            if not (
                hazard[2] < minimum_x
                or hazard[0] > maximum_x
                or hazard[3] < minimum_y
                or hazard[1] > maximum_y
            )
        ))
    return tuple(result)


def lower_source_forecast(
    snapshot,
    requested_horizon: int = 12,
    *,
    collision_margin: float = COLLISION_MARGIN,
) -> SourceForecast:
    """Project live hazards and source ECL births without phase control flow."""
    if requested_horizon < HARD_HORIZON:
        raise ValueError("source forecast must cover Hard-4")
    enable_donor_imports()
    from th06.hazards.bullets import reachable_hazards_by_frame
    from th06.hazards.enemies import hazards_by_frame as enemy_hazards_by_frame
    from th06.hazards.lasers import hazards_by_frame as laser_hazards_by_frame
    from th06.hazards.world import forecast_world_births

    player_positions = ((snapshot.x, snapshot.y),) * requested_horizon
    hard_births = forecast_world_births(
        snapshot,
        player_positions[:HARD_HORIZON],
    )
    if hard_births.covered_frames < HARD_HORIZON:
        raise AuthorityUnavailable(
            "Hard source birth coverage ended at "
            f"h{hard_births.covered_frames}: {hard_births.reason}"
        )
    nominal_births = (
        forecast_world_births(
            snapshot,
            player_positions,
            rng_mode="nominal",
        )
        if requested_horizon > HARD_HORIZON
        else hard_births
    )
    source_coverage = min(requested_horizon, nominal_births.covered_frames)
    if source_coverage < HARD_HORIZON:
        source_coverage = HARD_HORIZON
    bullet_frames = reachable_hazards_by_frame(
        snapshot,
        source_coverage,
        collision_margin,
    )[:source_coverage]
    enemy_frames = enemy_hazards_by_frame(
        snapshot.enemies,
        source_coverage,
    )
    live_laser_frames = laser_hazards_by_frame(
        snapshot.lasers,
        source_coverage,
    )

    aabb_frames = []
    laser_frames = []
    for index in range(source_coverage):
        births = hard_births if index < HARD_HORIZON else nominal_births
        birth_aabbs = (
            births.hazards[index]
            if index < births.covered_frames
            else ()
        )
        birth_bodies = (
            births.body_hazards[index]
            if births.body_hazards and index < births.covered_frames
            else ()
        )
        birth_lasers = (
            births.laser_hazards[index]
            if births.laser_hazards and index < births.covered_frames
            else ()
        )
        aabb_frames.append(
            bullet_frames[index]
            + enemy_frames[index]
            + birth_aabbs
            + birth_bodies
        )
        laser_frames.append(live_laser_frames[index] + birth_lasers)

    reachable_frames = _reachable_aabbs(
        snapshot,
        tuple(aabb_frames),
        collision_margin,
    )
    packed = PackedHazards(
        aabb_frames=tuple(
            tuple(Aabb(*hazard) for hazard in frame)
            for frame in reachable_frames
        ),
        laser_frames=tuple(
            tuple(LaserRect(
                hazard.origin_x,
                hazard.origin_y,
                hazard.angle,
                hazard.center_offset,
                hazard.size_x,
                hazard.size_y,
            ) for hazard in frame)
            for frame in laser_frames
        ),
    )
    return SourceForecast(
        hazards=packed,
        hard_horizon=HARD_HORIZON,
        requested_horizon=requested_horizon,
        source_coverage=source_coverage,
        coverage_reason=(
            ""
            if source_coverage == requested_horizon
            else nominal_births.reason
        ),
    )


def lower_observed_hazards(
    snapshot,
    requested_horizon: int = 12,
    *,
    collision_margin: float = COLLISION_MARGIN,
) -> SourceForecast:
    """Project only already-observed physical hazards for the online gate.

    This deliberately performs no timeline/ECL interpretation and predicts no
    future births. Long-horizon birth/route reasoning belongs to learned policy
    data and offline training; the resident loop only needs a small, bounded
    native legality frontier over live bullets, enemy bodies, and lasers.
    """
    if requested_horizon < HARD_HORIZON:
        raise ValueError("observed hazard gate must cover Hard-4")
    enable_donor_imports()
    from th06.hazards.bullets import reachable_hazards_by_frame
    from th06.hazards.enemies import hazards_by_frame as enemy_hazards_by_frame

    bullet_frames = reachable_hazards_by_frame(
        snapshot,
        requested_horizon,
        collision_margin,
    )
    enemy_frames = enemy_hazards_by_frame(
        snapshot.enemies,
        requested_horizon,
    )
    laser_frames = laser_rects_by_frame(
        snapshot.lasers,
        requested_horizon,
    )
    aabb_frames = tuple(
        bullet_frames[index] + enemy_frames[index]
        for index in range(requested_horizon)
    )
    reachable_frames = _reachable_aabbs(
        snapshot,
        aabb_frames,
        collision_margin,
    )
    return SourceForecast(
        hazards=PackedHazards(
            aabb_frames=tuple(
                tuple(Aabb(*hazard) for hazard in frame)
                for frame in reachable_frames
            ),
            laser_frames=laser_frames,
        ),
        hard_horizon=HARD_HORIZON,
        requested_horizon=requested_horizon,
        source_coverage=requested_horizon,
        coverage_reason="observed-physical-hazards-only",
    )
