"""Source-defined enemy births in the loaded ECL stage timeline."""

from __future__ import annotations

from dataclasses import dataclass
import math
import struct

from ..model import StageTimelineInstruction


WORLD_TRANSITION_OPCODES = frozenset((*range(8), 10))


def timeline_message_index(
    instruction: StageTimelineInstruction,
    stage: int,
    difficulty: int,
    character: int,
) -> int | None:
    """Apply EnemyManager::RunEclTimeline's source-defined MSGREAD index."""
    if instruction.opcode != 8:
        return None
    if difficulty == 0 and stage == 5 and instruction.arg0 == 1:
        return character * 10 + 3
    return instruction.arg0 + character * 10


def scheduled_timeline(
    instructions: tuple[StageTimelineInstruction, ...],
    current_time: int,
    *,
    stage: int = 0,
    difficulty: int = 0,
    character: int = 0,
    message_delays: tuple[tuple[int, int], ...] = (),
    current_message_waits: int = 0,
) -> tuple[tuple[int, StageTimelineInstruction], ...]:
    """Return earliest source-frame leads, including proved MSGWAIT stalls.

    Each ``message_delays`` entry is the minimum number of priority-9 waits
    proved from immutable bytecode even under maximally fast dialogue input.
    ``current_message_waits`` applies when the captured pointer is already on
    MSGWAIT and the initiating MSGREAD is no longer in the remaining stream.
    """
    result: list[tuple[int, StageTimelineInstruction]] = []
    delay = 0
    pending_message_waits = 0
    proved_delays = dict(message_delays)
    used_current_wait = False
    for instruction in instructions:
        if instruction.time < 0:
            break
        lead = max(0, instruction.time - current_time) + delay
        result.append((lead, instruction))
        message_index = timeline_message_index(
            instruction,
            stage,
            difficulty,
            character,
        )
        if message_index is not None:
            pending_message_waits = proved_delays.get(message_index, 0)
        elif instruction.opcode == 9:
            if pending_message_waits:
                delay += pending_message_waits
            elif not used_current_wait:
                delay += max(0, current_message_waits)
                used_current_wait = True
            pending_message_waits = 0
    return tuple(result)


@dataclass(frozen=True)
class TimelineEnemySpawn:
    """The route-neutral source semantics of one timeline spawn opcode."""

    instruction_address: int
    time: int
    sub_id: int
    x: float
    y: float
    life: int | None
    item_drop: int
    invert_x: bool
    random_x: bool
    random_y: bool
    random_z: bool


@dataclass(frozen=True)
class TimelineBossInterrupt:
    """One source timeline write to ``bosses[id]->runInterrupt``."""

    instruction_address: int
    boss_id: int
    interrupt_id: int


def decode_boss_interrupt(
    instruction: StageTimelineInstruction,
) -> TimelineBossInterrupt | None:
    if instruction.opcode != 10:
        return None
    raw = bytes.fromhex(instruction.raw_hex)
    if len(raw) < 16:
        return None
    boss_id, interrupt_id = struct.unpack_from("<II", raw, 8)
    if boss_id >= 8 or interrupt_id >= 8:
        return None
    return TimelineBossInterrupt(
        instruction.address,
        boss_id,
        interrupt_id,
    )


def first_world_transition(
    instructions: tuple[StageTimelineInstruction, ...],
    current_time: int,
    horizon: int,
    *,
    stage: int = 0,
    difficulty: int = 0,
    character: int = 0,
    message_delays: tuple[tuple[int, int], ...] = (),
    current_message_waits: int = 0,
) -> tuple[int, StageTimelineInstruction] | None:
    """Return the first uninserted hazard-world transition in the window.

    EnemyManager runs the timeline before its enemy slots.  A record whose
    source time equals the captured post-update timeline timer therefore runs
    on forecast frame zero.  Spawn opcodes 0..7 can add a body and execute
    newborn time-zero ECL; opcode 10 changes a boss ECL interrupt before that
    boss is updated.  Dialogue/power/wait records do not directly add or
    redirect a hazard in the same update.
    """
    if horizon <= 0:
        return None
    for lead, instruction in scheduled_timeline(
        instructions,
        current_time,
        stage=stage,
        difficulty=difficulty,
        character=character,
        message_delays=message_delays,
        current_message_waits=current_message_waits,
    ):
        if lead >= horizon:
            return None
        if instruction.opcode in WORLD_TRANSITION_OPCODES:
            return lead, instruction
    return None


def decode_enemy_spawn(
    instruction: StageTimelineInstruction,
) -> TimelineEnemySpawn | None:
    """Decode EnemyManager::RunEclTimeline opcodes 0..7.

    Opcodes 0/2/4/6 carry an explicit life override.  The odd opcodes use
    the ECL-initialized life, which is not present in the timeline record and
    therefore remains unknown here.  Random-coordinate sentinels are retained
    instead of being replaced with a nominal position.
    """
    if not 0 <= instruction.opcode <= 7:
        return None
    raw = bytes.fromhex(instruction.raw_hex)
    if len(raw) < 20:
        return None
    x, y, z = struct.unpack_from("<fff", raw, 8)
    if not math.isfinite(x) or not math.isfinite(y) or not math.isfinite(z):
        return None
    explicit = instruction.opcode % 2 == 0
    life = struct.unpack_from("<h", raw, 20)[0] if explicit and len(raw) >= 22 else None
    item_drop = (
        struct.unpack_from("<h", raw, 22)[0]
        if explicit and len(raw) >= 24
        else -1
    )
    random_position = instruction.opcode >= 4
    return TimelineEnemySpawn(
        instruction.address,
        instruction.time,
        instruction.arg0,
        x,
        y,
        life,
        item_drop,
        bool(instruction.opcode & 0x02),
        random_position and x <= -990.0,
        random_position and y <= -990.0,
        random_position and z <= -990.0,
    )
