"""Decode runtime snapshots for offline barrage-lab replay."""

from __future__ import annotations

import json
from pathlib import Path

from ..model import (
    Bullet,
    BulletPattern,
    EnemyBody,
    EnemyEclContext,
    EnemySpawner,
    EclInstruction,
    ItemState,
    Laser,
    PlayerAttackState,
    PlayerShot,
    Snapshot,
    StageTimelineInstruction,
)


def decode_snapshot(raw: dict) -> Snapshot:
    """Restore the immutable runtime model written by ``agent.py``."""
    values = dict(raw)
    values["bullets"] = tuple(Bullet(**item) for item in values["bullets"])
    values["despawning_bullets"] = tuple(
        Bullet(**item) for item in values["despawning_bullets"]
    )
    values["lasers"] = tuple(Laser(**item) for item in values["lasers"])
    values["enemies"] = tuple(
        EnemyBody(**item) for item in values["enemies"]
    )
    values["item_states"] = tuple(
        ItemState(**item) for item in values.get("item_states", ())
    )
    values["spawners"] = tuple(
        EnemySpawner(
            **{
                **item,
                "ecl_compare": item.get("ecl_compare", 0),
                "ecl_ints": tuple(item.get("ecl_ints", ())),
                "ecl_floats": tuple(item.get("ecl_floats", ())),
                "ecl_stack": tuple(
                    EnemyEclContext(**{
                        **context,
                        "ints": tuple(context["ints"]),
                        "floats": tuple(context["floats"]),
                    })
                    for context in item.get("ecl_stack", ())
                ),
                "pattern": (
                    BulletPattern(**{
                        **item["pattern"],
                        "ex_floats": tuple(item["pattern"]["ex_floats"]),
                        "ex_ints": tuple(item["pattern"]["ex_ints"]),
                    })
                    if item.get("pattern") is not None
                    else None
                ),
                "next_instruction": (
                    EclInstruction(**item["next_instruction"])
                    if item.get("next_instruction") is not None
                    else None
                ),
                "ecl_program": tuple(
                    EclInstruction(**instruction)
                    for instruction in item.get("ecl_program", ())
                ),
                "ecl_subroutines": tuple(item.get("ecl_subroutines", ())),
                "bullet_effect_floats": tuple(
                    item.get("bullet_effect_floats", (0.0,) * 4)
                ),
                "bullet_effect_ints": tuple(
                    item.get("bullet_effect_ints", (0,) * 4)
                ),
                "interrupts": tuple(item.get("interrupts", (-1,) * 8)),
                "laser_slots": tuple(item.get("laser_slots", (-1,) * 32)),
                "laser_store": item.get("laser_store", 0),
            }
        )
        for item in values.get("spawners", ())
    )
    values["timeline_instructions"] = tuple(
        StageTimelineInstruction(**item)
        for item in values.get("timeline_instructions", ())
    )
    values["timeline_ecl_program"] = tuple(
        EclInstruction(**item)
        for item in values.get("timeline_ecl_program", ())
    )
    for field in (
        "bullet_sizes",
        "timeline_message_delays",
    ):
        values[field] = tuple(
            tuple(item) for item in values.get(field, ())
        )
    for field in (
        "ecl_subroutines",
        "timeline_emitter_subs",
        "timeline_boss_subs",
        "timeline_boss_slots",
        "pending_effect_rng_ids",
        "simulated_effect_expiry_updates",
    ):
        values[field] = tuple(values.get(field, ()))
    attack = values.get("player_attack")
    if attack is not None:
        values["player_attack"] = PlayerAttackState(
            **{
                **attack,
                "shots": tuple(
                    PlayerShot(**shot) for shot in attack.get("shots", ())
                ),
                "orb_positions": tuple(
                    tuple(position)
                    for position in attack.get("orb_positions", ())
                ),
            }
        )
    return Snapshot(**values)


def load_failure_history(path: Path) -> tuple[Snapshot, ...]:
    """Load ordered physical snapshots retained by a runtime diagnostic."""
    artifact = json.loads(path.read_text(encoding="utf-8"))
    raw_history = (
        artifact.get("snapshot_history")
        or artifact.get("mismatch_snapshots")
        or (artifact["snapshot"],)
    )
    history = tuple(decode_snapshot(raw) for raw in raw_history)
    if any(right.frame <= left.frame for left, right in zip(history, history[1:])):
        raise ValueError("runtime snapshot history must have increasing frames")
    return history
