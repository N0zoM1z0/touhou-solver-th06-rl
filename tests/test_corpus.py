from __future__ import annotations

import gzip
import json

from th06_rl.corpus import (
    CorpusRecorder,
    FrameEvidence,
    RunMetadata,
    expand_compact,
)
from th06_rl.th06.donor import enable_donor_imports


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
    assert manifest["run_outcome"] == outcome


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
