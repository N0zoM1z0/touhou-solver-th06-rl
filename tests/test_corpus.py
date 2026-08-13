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
from th06_rl.th06.donor import enable_donor_imports
from th06_rl.th06.control_capture import ControlSnapshot, decode_control_snapshot
from th06_rl.th06.source import automatic_source_context


enable_donor_imports()
from th06.model import Bullet, Snapshot  # noqa: E402
import th06.native as native  # noqa: E402


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
    assert frame["schema_version"] == "th06-rl-authoritative-frame-v7"
    assert frame["decision"]["dialogue_delivery"] == [
        {
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
