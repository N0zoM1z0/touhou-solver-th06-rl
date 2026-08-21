from __future__ import annotations

from dataclasses import replace
import struct

import pytest

from th06_rl.corpus import CorpusRecorder, FrameEvidence, RunMetadata
from th06_rl.retail import native
from th06_rl.retail.model import PlayerAttackState, RepeatStarState, Snapshot
from th06_rl.th06.source_dataset import (
    SourceDatasetError,
    iter_source_diagnostic_frames,
    iter_source_frames,
    validate_frame_authority,
)
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
    repeat_star_state = RepeatStarState(
        (0.0,) * 6, 192.0, 128.0, 192.0, 400.0
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
        repeat_star_state=repeat_star_state,
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
        repeat_star_state=repeat_star_state,
    )
    return control, anchor


def _source_run(
    tmp_path,
    *,
    episode_complete: bool,
    route: bool = False,
    authority_valid: bool = True,
):
    control, anchor = _pair()
    if not authority_valid:
        assert anchor.repeat_star_state is not None
        anchor = replace(
            anchor,
            repeat_star_state=replace(anchor.repeat_star_state, enemy_x=193.0),
        )
    planner = {
        "algorithm": "source-hard4-paused-publication-v2",
        "source_commitment": "source-complete-hard-v1",
        "publication_epoch": "source-root-process-suspended-v1",
        "factual_state_schema": OFFLINE_FACT_SCHEMA,
        "hard_horizon": 4,
        "learner_feature_horizon": 4,
        "minimum_collision_margin": 0.35,
        "zero_margin_fallback": False,
    }
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test",
            native.TARGET_SHA256,
            "native",
            "test",
            3,
            0,
            0,
            1,
            planner,
            episode_unit="route" if route else "practice-stage",
            expected_stages=tuple(range(1, 7)) if route else (1,),
        ),
    )
    evidence = FrameEvidence(
        phase_id=control.source_context,
        current_action="stay",
        hard_actions=(("stay", None, control.x, control.y),),
        baseline_action="stay",
        locally_admissible_actions=("stay",),
        proposed_action="stay",
        published_action=None,
        behavior_probability=1.0,
        policy_id="fixture",
        policy_generation=1,
        policy_sha256="abc",
        effort_horizon=4,
        plan_min_clearance=None,
        cumulative_risk=None,
        terminal_x=control.x,
        terminal_y=control.y,
        endpoint_count=1,
        continuation_action_count=1,
        capture_ms=0.0,
        solve_ms=0.0,
        reason="ok",
        snapshot_tier=CONTROL_CAPTURE_TIER,
        hard_collision_margin=0.35,
    )
    snapshot_ref = recorder.record(control, evidence)
    recorder.record_anchor(
        anchor,
        phase_id=control.source_context,
        reason="stage-root",
        control_snapshot_ref=snapshot_ref,
    )
    return recorder.close({
        "stage_completed": episode_complete,
        "termination_reason": (
            "route-complete"
            if route and episode_complete
            else "practice-stage-complete"
            if episode_complete
            else "authority-stop"
        ),
    })


def test_control_v5_frame_is_self_contained_after_wine_exit() -> None:
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


def test_source_dataset_rejects_mismatched_repeat_star_globals() -> None:
    control, anchor = _pair()
    assert anchor.repeat_star_state is not None
    anchor = replace(
        anchor,
        repeat_star_state=replace(anchor.repeat_star_state, enemy_x=193.0),
    )

    with pytest.raises(SourceDatasetError, match="repeating-star globals"):
        validate_frame_authority(control, anchor)


def test_source_dataset_accepts_dense_repeat_globals_after_anchor_pause() -> None:
    control, anchor = _pair()
    assert control.repeat_star_state is not None
    control = replace(
        control,
        repeat_star_state=replace(control.repeat_star_state, enemy_x=193.0),
    )

    validate_frame_authority(control, anchor, same_pause=False)


def test_training_iterator_rejects_retained_incomplete_episode(tmp_path) -> None:
    run_dir = _source_run(tmp_path, episode_complete=False)

    with pytest.raises(SourceDatasetError, match="completely observed"):
        tuple(iter_source_frames(run_dir))


def test_diagnostic_iterator_retains_valid_nontraining_prefix(tmp_path) -> None:
    run_dir = _source_run(tmp_path, episode_complete=False)

    rows = tuple(iter_source_diagnostic_frames(run_dir))

    assert len(rows) == 1
    assert rows[0].frame.control.frame == 10
    assert rows[0].storage_complete is True
    assert rows[0].trajectory_complete is False
    assert rows[0].episode_complete is False
    assert rows[0].training_eligible is False


def test_diagnostic_iterator_still_rejects_invalid_authority(tmp_path) -> None:
    run_dir = _source_run(
        tmp_path,
        episode_complete=False,
        authority_valid=False,
    )

    with pytest.raises(SourceDatasetError, match="repeating-star globals"):
        tuple(iter_source_diagnostic_frames(run_dir))


def test_training_iterator_accepts_complete_practice_episode(tmp_path) -> None:
    run_dir = _source_run(tmp_path, episode_complete=True)

    rows = tuple(iter_source_frames(run_dir))

    assert len(rows) == 1
    assert rows[0].control.frame == 10


def test_training_iterator_requires_all_declared_route_stages(tmp_path) -> None:
    run_dir = _source_run(tmp_path, episode_complete=True, route=True)

    with pytest.raises(SourceDatasetError, match="declared physical episode stages"):
        tuple(iter_source_frames(run_dir))
