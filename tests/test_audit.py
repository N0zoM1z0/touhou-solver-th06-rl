from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

from th06_rl.corpus import CorpusRecorder, RunMetadata


_SPEC = importlib.util.spec_from_file_location(
    "th06_rl_audit_script",
    Path(__file__).resolve().parents[1] / "scripts/audit_run.py",
)
assert _SPEC is not None and _SPEC.loader is not None
_AUDIT = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_AUDIT)
audit = _AUDIT.audit


def test_empty_complete_stage_audit_is_structurally_stable(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata("test", "exe", "native", "test", 3, 0, 0, 4, {}),
    )
    run_dir = recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })

    report = audit(run_dir)

    assert report["stage_completed"] is True
    assert report["physical_hits"] == 0
    assert report["integrity_errors"] == []
    assert report["infra_stable_for_learning"] is True


def test_collision_evidence_identifies_new_enemy_body() -> None:
    common = {
        "x": 193.0,
        "y": 121.0,
        "half_width": 1.25,
        "half_height": 1.25,
        "bullets": (),
        "lasers": (),
    }
    before = SimpleNamespace(**common, enemies=())
    after = SimpleNamespace(
        **common,
        enemies=(SimpleNamespace(
            x=192.0,
            y=128.0,
            half_width=16.0,
            half_height=18.0,
        ),),
    )

    evidence = _AUDIT._collision_evidence(before, after, 1)

    assert evidence["after_overlapping_enemy_indices"] == [0]
    assert evidence["new_after_overlapping_enemy_indices"] == [0]
