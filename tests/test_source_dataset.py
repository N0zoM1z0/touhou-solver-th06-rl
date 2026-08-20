from __future__ import annotations

from dataclasses import replace
import struct

import pytest

from th06_rl.retail import native
from th06_rl.retail.model import PlayerAttackState, Snapshot
from th06_rl.th06.source_dataset import SourceDatasetError, validate_frame_authority
from th06_rl.th06.control_capture import (
    CONTROL_CAPTURE_TIER,
    OFFLINE_FACT_SCHEMA,
    SOURCE_RECORD_SCHEMA,
    ControlSnapshot,
    decode_control_snapshot,
)


def _pair() -> tuple[ControlSnapshot, Snapshot]:
    ex_addresses = tuple(
        0x401000 + index * 4 for index in range(native.ECL_EX_COUNT)
    )
    attack = PlayerAttackState(
        (), 0.0, 0.0, 0, False,
        0, 0, 0.0, 0, 0, 0.0,
        ((0.0, 0.0), (0.0, 0.0)), 0, False, False,
    )
    tail_size = (
        native.ENEMY_MANAGER_SIZE
        - native.ENEMY_ARRAY_OFFSET
        - native.ENEMY_COUNT * native.ENEMY_STRIDE
    )
    control = ControlSnapshot(
        capture_tier=CONTROL_CAPTURE_TIER,
        frame=10,
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
        bullets=(),
        live_bullet_count=0,
        raw_bullet_tails=b"",
        bullet_sprite_dimensions=(),
        bullets_are_reachable_subset=False,
        laser_count=0,
        in_menu=False,
        time_stopped=False,
        replay_or_demo=False,
        lasers=(),
        enemies=(),
        difficulty=3,
        character=0,
        shot_type=0,
        bomb_active=False,
        spell_active=False,
        rank=0,
        subrank=0,
        max_rank=32,
        min_rank=0,
        rng_seed=1,
        rng_generation=2,
        current_power=0,
        lives_remaining=2,
        source_context="timeline-complete",
        boss_life=None,
        timeline_time=10,
        timeline_time_float=10.0,
        capture_attempts=1,
        bullet_read_retries=0,
        raw_enemy_manager_tail=bytes(tail_size),
        source_record_schema=SOURCE_RECORD_SCHEMA,
        factual_state_schema=OFFLINE_FACT_SCHEMA,
        player_attack=attack,
        effect_active_upper_bound=0,
        item_active_upper_bound=0,
        ecl_ex_function_addresses=ex_addresses,
        timeline_boss_slots=(-1,) * 8,
        timeline_time_previous=9,
        boss_present=False,
    )
    control = decode_control_snapshot({
        field.name: getattr(control, field.name)
        for field in __import__("dataclasses").fields(control)
    })
    anchor = Snapshot(
        frame=10,
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
        bullets=(),
        laser_count=0,
        in_menu=False,
        time_stopped=False,
        replay_or_demo=False,
        difficulty=3,
        character=0,
        player_attack=attack,
        timeline_complete=True,
        ecl_ex_function_addresses=ex_addresses,
    )
    return control, anchor


def test_control_v4_frame_is_self_contained_after_wine_exit() -> None:
    control, anchor = _pair()

    validate_frame_authority(control, anchor)


def test_source_dataset_rejects_uncovered_timeline_pointer() -> None:
    control, anchor = _pair()
    tail = bytearray(control.raw_enemy_manager_tail)
    pointer_offset = (
        native.ENEMY_TIMELINE_INSTRUCTION_OFFSET
        - native.ENEMY_ARRAY_OFFSET
        - native.ENEMY_COUNT * native.ENEMY_STRIDE
    )
    struct.pack_into("<I", tail, pointer_offset, 0x700000)

    with pytest.raises(SourceDatasetError, match="timeline pointer"):
        validate_frame_authority(
            replace(control, raw_enemy_manager_tail=bytes(tail)),
            anchor,
        )


def test_source_dataset_rejects_external_ex_dispatch_assumption() -> None:
    control, anchor = _pair()

    with pytest.raises(SourceDatasetError, match="EX callback"):
        validate_frame_authority(
            replace(control, ecl_ex_function_addresses=(0x500000,) * 17),
            anchor,
        )
