from __future__ import annotations

import json

from th06_rl.corpus import CorpusRecorder, RunMetadata


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
