from __future__ import annotations

import gzip
import hashlib
import json
import struct
from dataclasses import replace

import pytest

from th06_rl.corpus import (
    CorpusRecorder,
    DialogueDeliverySample,
    FrameEvidence,
    RunMetadata,
    expand_compact,
)
from th06_rl.policy_api import PolicyOptionTrace
from th06_rl.retail import native
from th06_rl.retail.model import (
    Bullet,
    ItemState,
    PlayerAttackState,
    Snapshot,
)
from th06_rl.th06.control_capture import (
    CONTROL_CAPTURE_TIER,
    OFFLINE_FACT_SCHEMA,
    SOURCE_RECORD_SCHEMA,
    ControlSnapshot,
    decode_control_snapshot,
)
from th06_rl.th06.source import automatic_source_context


def _packed_control_bullet(
    *,
    slot: int = 7,
    sprite_pointer: int = 0x123456,
    ex_flags: int = 0x800,
) -> bytes:
    tail = bytearray(native.BULLET_STRIDE - native.BULLET_SIZE_OFFSET)
    relative = lambda absolute: absolute - native.BULLET_SIZE_OFFSET
    struct.pack_into("<ff", tail, 0, 2.0, 2.0)
    struct.pack_into("<ff", tail, relative(native.BULLET_POSITION_OFFSET), 100.0, 120.0)
    struct.pack_into("<ff", tail, relative(native.BULLET_VELOCITY_OFFSET), 1.0, 0.0)
    struct.pack_into("<f", tail, relative(native.BULLET_SPEED_OFFSET), 1.0)
    struct.pack_into("<f", tail, relative(native.BULLET_TURN_SPEED_OFFSET), 1.0)
    struct.pack_into("<H", tail, relative(native.BULLET_EX_FLAGS_OFFSET), ex_flags)
    struct.pack_into("<H", tail, relative(native.BULLET_STATE_OFFSET), 1)
    return struct.pack("<HI", slot, sprite_pointer) + tail


def test_manifest_distinguishes_storage_from_complete_stage(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            code_commit="test",
            executable_sha256="exe",
            native_kernel_sha256="native",
            input_backend="test",
            difficulty=2,
            character=0,
            shot_type=0,
            stage=4,
            planner={},
        ),
    )
    outcome = {
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "controller_exit_code": 0,
        "physical_hits": 3,
        "control_dead_ends": 1,
    }

    run_dir = recorder.close(outcome)
    manifest = json.loads((run_dir / "manifest.json").read_text())

    assert manifest["complete"] is True
    assert manifest["stage_trajectory_complete"] is True
    assert manifest["episode"] == {
        "id": recorder.run_id,
        "unit": "practice-stage",
        "complete": True,
        "termination_reason": "practice-stage-complete",
    }
    assert manifest["run_outcome"] == outcome
    assert manifest["summary"]["frames"] == 0
    assert manifest["summary"]["learning_eligible_transitions"] == 0


def test_route_corpus_retains_one_physical_episode_across_six_stages(
    tmp_path,
) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test",
            "exe",
            "native",
            "test",
            3,
            0,
            0,
            1,
            {},
            episode_unit="route",
            expected_stages=(1, 2, 3, 4, 5, 6),
        ),
    )
    run_dir = recorder.close({
        "termination_reason": "route-complete",
        "stage_completed": True,
        "physical_hits": 12,
    })

    run = json.loads((run_dir / "run.json").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert run["metadata"]["episode_unit"] == "route"
    assert run["metadata"]["expected_stages"] == [1, 2, 3, 4, 5, 6]
    assert manifest["episode"] == {
        "id": recorder.run_id,
        "unit": "route",
        "complete": True,
        "termination_reason": "route-complete",
    }


def test_practice_metadata_cannot_claim_multiple_stages(tmp_path) -> None:
    with pytest.raises(ValueError, match="episode metadata"):
        CorpusRecorder(
            tmp_path,
            RunMetadata(
                "test", "exe", "native", "test", 3, 0, 0, 1, {},
                expected_stages=(1, 2),
            ),
        )


def test_compact_frame_round_trips_repeated_dataclasses(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata("test", "exe", "native", "test", 2, 0, 0, 4, {}),
    )
    snapshot = Snapshot(
        frame=10,
        stage=4,
        player_state=0,
        x=192.0,
        y=400.0,
        half_width=1.5,
        half_height=1.5,
        normal_speed=4.0,
        focus_speed=2.0,
        normal_diagonal_speed=2.8,
        focus_diagonal_speed=1.4,
        frame_multiplier=1.0,
        input_mask=1,
        bullets=(Bullet(10.0, 20.0, 1.0, 2.0, 3.0, 4.0, 1, slot=7),),
        laser_count=0,
        in_menu=False,
        time_stopped=False,
        replay_or_demo=False,
    )
    recorder.record(snapshot, FrameEvidence(
        phase_id="timeline:test",
        current_action="stay",
        hard_actions=(("stay", 10.0, 192.0, 400.0),),
        baseline_action="stay",
        locally_admissible_actions=("stay",),
        proposed_action="stay",
        published_action="stay",
        behavior_probability=1.0,
        policy_id="test",
        policy_generation=1,
        policy_sha256="abc",
        effort_horizon=4,
        plan_min_clearance=10.0,
        cumulative_risk=None,
        terminal_x=192.0,
        terminal_y=400.0,
        endpoint_count=1,
        continuation_action_count=1,
        capture_ms=1.0,
        solve_ms=0.1,
        reason="ok",
        dialogue_delivery=(
            DialogueDeliverySample(
                stage=1,
                game_frame=7,
                current_input_mask=0x101,
                previous_input_mask=0x001,
                published_input_mask=0x101,
                held_repeat=0,
                held_frames=1,
                active=True,
                skippable=True,
                pulsed_shoot=False,
            ),
            DialogueDeliverySample(
                stage=1,
                game_frame=8,
                current_input_mask=0x001,
                previous_input_mask=0x101,
                published_input_mask=0x001,
                held_repeat=0,
                held_frames=0,
                active=False,
                skippable=False,
                pulsed_shoot=False,
            ),
        ),
    ))
    run_dir = recorder.close({"stage_completed": True})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["summary"]["frames"] == 1
    assert manifest["summary"]["reason_counts"] == {"ok": 1}
    objects = {}
    for path in run_dir.glob("objects-*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                objects[row["object_id"]] = row["payload"]
    frame_path = next(run_dir.glob("frames-*.jsonl.gz"))
    with gzip.open(frame_path, "rt", encoding="utf-8") as source:
        frame = json.loads(next(source))
    assert frame["schema_version"] == "th06-rl-authoritative-frame-v8"
    assert frame["decision"]["dialogue_delivery"] == [
        {
            "stage": 1,
            "game_frame": 7,
            "current_input_mask": 0x101,
            "previous_input_mask": 0x001,
            "published_input_mask": 0x101,
            "held_repeat": 0,
            "held_frames": 1,
            "active": True,
            "skippable": True,
            "pulsed_shoot": False,
        },
        {
            "stage": 1,
            "game_frame": 8,
            "current_input_mask": 0x001,
            "previous_input_mask": 0x101,
            "published_input_mask": 0x001,
            "held_repeat": 0,
            "held_frames": 0,
            "active": False,
            "skippable": False,
            "pulsed_shoot": False,
        },
    ]
    encoded_bullets = frame["snapshot"]["bullets"]
    assert encoded_bullets["codec"] == "dataclass-rows-v1"
    hydrated = expand_compact(frame["snapshot"], objects)
    assert len(hydrated["bullets"]) == 1
    assert hydrated["bullets"][0]["x"] == 10.0
    assert hydrated["bullets"][0]["slot"] == 7


def test_dialogue_delivery_rejects_bomb_and_out_of_order_samples() -> None:
    with pytest.raises(ValueError, match="Bomb-bearing"):
        DialogueDeliverySample(
            stage=1,
            game_frame=1,
            current_input_mask=0x02,
            previous_input_mask=0,
            published_input_mask=0,
            held_repeat=0,
            held_frames=0,
            active=True,
            skippable=False,
            pulsed_shoot=False,
        )

    later = DialogueDeliverySample(
        stage=1,
        game_frame=2,
        current_input_mask=1,
        previous_input_mask=0,
        published_input_mask=1,
        held_repeat=0,
        held_frames=1,
        active=True,
        skippable=False,
        pulsed_shoot=True,
    )
    earlier = replace(later, game_frame=1)
    with pytest.raises(ValueError, match="frame ordered"):
        FrameEvidence(
            phase_id="timeline:test",
            current_action="stay",
            hard_actions=(("stay", 10.0, 192.0, 400.0),),
            baseline_action="stay",
            locally_admissible_actions=("stay",),
            proposed_action="stay",
            published_action="stay",
            behavior_probability=1.0,
            policy_id="test",
            policy_generation=1,
            policy_sha256="abc",
            effort_horizon=4,
            plan_min_clearance=10.0,
            cumulative_risk=None,
            terminal_x=192.0,
            terminal_y=400.0,
            endpoint_count=1,
            continuation_action_count=1,
            capture_ms=1.0,
            solve_ms=0.1,
            reason="ok",
            dialogue_delivery=(later, earlier),
        )

    stage_reset = replace(later, stage=2, game_frame=0)
    FrameEvidence(
        phase_id="timeline:test",
        current_action="stay",
        hard_actions=(("stay", 10.0, 192.0, 400.0),),
        baseline_action="stay",
        locally_admissible_actions=("stay",),
        proposed_action="stay",
        published_action="stay",
        behavior_probability=1.0,
        policy_id="test",
        policy_generation=1,
        policy_sha256="abc",
        effort_horizon=4,
        plan_min_clearance=10.0,
        cumulative_risk=None,
        terminal_x=192.0,
        terminal_y=400.0,
        endpoint_count=1,
        continuation_action_count=1,
        capture_ms=1.0,
        solve_ms=0.1,
        reason="ok",
        dialogue_delivery=(later, stage_reset),
    )


def test_control_frames_exclude_latency_gaps_and_retain_full_anchor(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    control = ControlSnapshot(
        capture_tier="control-v2",
        frame=20,
        stage=4,
        player_state=0,
        x=192.0,
        y=400.0,
        half_width=1.5,
        half_height=1.5,
        normal_speed=4.0,
        focus_speed=2.0,
        normal_diagonal_speed=2.8,
        focus_diagonal_speed=1.4,
        frame_multiplier=1.0,
        input_mask=1,
        bullets=(),
        live_bullet_count=1,
        raw_bullet_tails=_packed_control_bullet(),
        bullet_sprite_dimensions=((0x123456, 8.0, 16.0),),
        bullets_are_reachable_subset=True,
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
        rank=10,
        subrank=0,
        max_rank=32,
        min_rank=0,
        rng_seed=123,
        rng_generation=456,
        current_power=64,
        lives_remaining=2,
        source_context="timeline:before-t100:op0:arg7",
        boss_life=None,
        timeline_time=90,
        timeline_time_float=90.0,
        capture_attempts=1,
        bullet_read_retries=0,
    )
    evidence = FrameEvidence(
        phase_id=control.source_context,
        current_action="stay",
        hard_actions=(("stay", 10.0, 192.0, 400.0),),
        baseline_action="stay",
        locally_admissible_actions=("stay",),
        proposed_action="stay",
        published_action=None,
        behavior_probability=1.0,
        policy_id="safe-option-exploration-v1",
        policy_generation=1,
        policy_sha256="abc",
        effort_horizon=4,
        plan_min_clearance=10.0,
        cumulative_risk=None,
        terminal_x=192.0,
        terminal_y=400.0,
        endpoint_count=1,
        continuation_action_count=1,
        capture_ms=2.0,
        solve_ms=0.1,
        reason="stale-retry",
        snapshot_tier="control-v2",
        option=PolicyOptionTrace(
            option_id="option-1",
            intent="stay",
            boundary=True,
            boundary_probability=1.0,
            elapsed_frames=1,
        ),
        hard_collision_margin=0.0,
    )
    root_ref = recorder.record(control, evidence)
    recorder.record_anchor(
        Snapshot(
            frame=20,
            stage=4,
            player_state=0,
            x=192.0,
            y=400.0,
            half_width=1.5,
            half_height=1.5,
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
        ),
        phase_id=control.source_context,
        reason="stage-root",
        control_snapshot_ref=root_ref,
    )
    recorder.record(
        replace(control, frame=22),
        replace(
            evidence,
            observation_gap=2,
            option=PolicyOptionTrace(
                option_id="option-2",
                intent="stay",
                boundary=True,
                boundary_probability=1.0,
                elapsed_frames=1,
                preceding_termination_reason="observation-gap",
            ),
        ),
    )
    run_dir = recorder.close({"stage_completed": True})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    run = json.loads((run_dir / "run.json").read_text())
    assert run["storage"]["queue_records"] == 512
    assert manifest["queue_capacity"] == 512
    assert 0 <= manifest["queue_high_watermark"] <= 512
    for shard in manifest["shards"]:
        payload = (run_dir / shard["path"]).read_bytes()
        assert hashlib.sha256(payload).hexdigest() == shard["sha256"]
    assert manifest["records"]["anchors"] == 1
    assert manifest["summary"]["observation_gap_rate"] == 0.5
    assert manifest["summary"]["dense_frame_samples"][0] == {
        "bullets": 1,
        "sequence": 1,
        "frame": 22,
    }
    assert automatic_source_context(control) == control.source_context
    frame_path = next(run_dir.glob("frames-*.jsonl.gz"))
    with gzip.open(frame_path, "rt", encoding="utf-8") as source:
        frame = json.loads(next(source))
    objects = {}
    for path in run_dir.glob("objects-*.jsonl.gz"):
        with gzip.open(path, "rt", encoding="utf-8") as source:
            for line in source:
                row = json.loads(line)
                objects[row["object_id"]] = row["payload"]
    hydrated = expand_compact(frame["snapshot"], objects)
    assert frame["decision"]["hard_collision_margin"] == 0.0
    assert hydrated["bullets"] == []
    assert hydrated["raw_bullet_tails"] == _packed_control_bullet()
    decoded = decode_control_snapshot(hydrated)
    assert len(decoded.bullets) == 1
    assert decoded.bullets[0].slot == 7
    assert decoded.bullets[0].ex_flags == 0x800
    assert decoded.bullets[0].sprite_half_width == 4.0
    assert decoded.bullets[0].sprite_half_height == 8.0
    transition_path = next(run_dir.glob("transitions-*.jsonl.gz"))
    with gzip.open(transition_path, "rt", encoding="utf-8") as source:
        transition = json.loads(next(source))
    assert transition["learning_eligible"] is False
    assert transition["published_action"] is None
    assert "action-not-published" in transition["learning_exclusion_reasons"]
    assert "observation-gap" in transition["learning_exclusion_reasons"]
    assert transition["episode"] == {
        "id": recorder.run_id,
        "unit": "practice-stage",
        "step": 0,
        "done": False,
    }
    assert transition["boundary"] == {
        "source_context_changed": False,
        "source_context": "3/0/0/4/timeline:before-t100:op0:arg7",
        "next_source_context": "3/0/0/4/timeline:before-t100:op0:arg7",
        "failure": None,
    }
    assert transition["policy_context"] == {
        "action_features": [],
        "current_action": "stay",
        "effort_horizon": 4,
        "hard_admissible_actions": ["stay"],
        "phase_elapsed_frames": 0,
        "player_x": 192.0,
        "player_y": 400.0,
        "power": 64,
        "bullet_count": 1,
        "laser_count": 0,
        "observation_features": [],
        "hazard_primitives": [],
        "history_features": [],
        "hard_action_count": 1,
        "hard_collision_margin": 0.0,
    }
    assert transition["policy_id"] == "safe-option-exploration-v1"
    assert transition["executed_action"] == "stay"
    assert transition["option"] == {
        "option_id": "option-1",
        "boundary": True,
        "intent": "stay",
        "boundary_probability": 1.0,
        "conditional_probability": 1.0,
        "elapsed_frames_at_decision": 1,
        "physical_elapsed_frames": 2,
            "termination_reason": "observation-gap",
            "preceding_termination_reason": None,
            "behavior_probabilities": [],
            "information_weights": [],
            "propensity_ess": [],
        }


def test_control_v3_retains_hazard_source_records() -> None:
    tail_size = (
        native.ENEMY_MANAGER_SIZE
        - native.ENEMY_ARRAY_OFFSET
        - native.ENEMY_COUNT * native.ENEMY_STRIDE
    )
    spawn = struct.pack("<H", 7) + bytes(native.BULLET_STRIDE)
    enemy = struct.pack("<H", 3) + bytes(native.ENEMY_STRIDE)
    laser = struct.pack("<H", 5) + bytes(native.LASER_STRIDE)
    base = ControlSnapshot(
        capture_tier=CONTROL_CAPTURE_TIER,
        frame=1,
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
        bullets_are_reachable_subset=True,
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
        timeline_time=0,
        timeline_time_float=0.0,
        capture_attempts=1,
        bullet_read_retries=0,
        reachable_bullet_slots=(7,),
        raw_spawn_bullet_records=spawn,
        raw_enemy_records=enemy,
        raw_laser_records=laser,
        raw_enemy_manager_tail=bytes(tail_size),
        source_record_schema=SOURCE_RECORD_SCHEMA,
        factual_state_schema=OFFLINE_FACT_SCHEMA,
        player_attack=PlayerAttackState(
            (), 10.0, 20.0, 0, False,
            0, 1, 1.0, 2, 3, 3.0,
            ((190.0, 400.0), (194.0, 400.0)), 0, False, False,
        ),
        item_states=(ItemState(
            9, 100.0, 120.0, 90.0, 100.0, 192.0, 400.0,
            3, 4, 4.0, 1, 2,
        ),),
        item_next_index=10,
        effect_active_upper_bound=2,
        item_active_upper_bound=1,
        pending_effect_rng_ids=(4,),
        random_item_spawn_index=3,
        random_item_table_index=2,
        score=123456,
        graze_in_stage=7,
    )

    decoded = decode_control_snapshot({
        field.name: getattr(base, field.name)
        for field in __import__("dataclasses").fields(base)
    })

    assert decoded.reachable_bullet_slots == (7,)
    assert decoded.raw_spawn_bullet_records == spawn
    assert decoded.raw_enemy_records == enemy
    assert decoded.raw_laser_records == laser
    assert len(decoded.raw_enemy_manager_tail) == tail_size
    assert decoded.factual_state_schema == OFFLINE_FACT_SCHEMA
    assert decoded.player_attack == base.player_attack
    assert decoded.item_states == base.item_states
    assert decoded.score == 123456
    assert decoded.graze_in_stage == 7
