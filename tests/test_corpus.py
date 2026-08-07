from __future__ import annotations

import gzip
import hashlib
import json
from dataclasses import replace

from th06_rl.corpus import (
    CorpusRecorder,
    FrameEvidence,
    RunMetadata,
    expand_compact,
)
from th06_rl.th06.donor import enable_donor_imports
from th06_rl.th06.control_capture import ControlSnapshot
from th06_rl.th06.source import automatic_source_context


enable_donor_imports()
from th06.model import Bullet, Snapshot  # noqa: E402


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
    encoded_bullets = frame["snapshot"]["bullets"]
    assert encoded_bullets["codec"] == "dataclass-rows-v1"
    hydrated = expand_compact(frame["snapshot"], objects)
    assert len(hydrated["bullets"]) == 1
    assert hydrated["bullets"][0]["x"] == 10.0
    assert hydrated["bullets"][0]["slot"] == 7


def test_control_frames_exclude_latency_gaps_and_retain_full_anchor(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    control = ControlSnapshot(
        capture_tier="control-v1",
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
        live_bullet_count=0,
        raw_bullet_tails=b"\x00\x01\x02",
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
        capture_ms=2.0,
        solve_ms=0.1,
        reason="ok",
        snapshot_tier="control-v1",
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
        replace(evidence, observation_gap=2),
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
        "bullets": 0,
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
    assert hydrated["raw_bullet_tails"] == b"\x00\x01\x02"
    transition_path = next(run_dir.glob("transitions-*.jsonl.gz"))
    with gzip.open(transition_path, "rt", encoding="utf-8") as source:
        transition = json.loads(next(source))
    assert transition["learning_eligible"] is False
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
        "current_action": "stay",
        "hard_admissible_actions": ["stay"],
        "phase_elapsed_frames": 0,
        "player_x": 192.0,
        "player_y": 400.0,
        "power": 64,
        "bullet_count": 0,
        "laser_count": 0,
        "hard_action_count": 1,
    }
