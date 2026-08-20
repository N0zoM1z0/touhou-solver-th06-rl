from __future__ import annotations

import importlib.util
import json
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
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 4,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v1",
            },
        ),
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


def test_route_audit_accepts_only_declared_complete_stage_coverage(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 1,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v1",
            },
            episode_unit="route",
            expected_stages=(1, 2, 3, 4, 5, 6),
        ),
    )
    run_dir = recorder.close({
        "termination_reason": "route-complete",
        "stage_completed": True,
        "physical_hits": 0,
    })
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["summary"]["phases"] = [
        {"scope": f"3/0/0/{stage}/source:test"}
        for stage in range(1, 7)
    ]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = audit(run_dir)

    assert report["integrity_errors"] == []
    assert report["scope"]["episode_unit"] == "route"
    assert report["scope"]["observed_stages"] == [1, 2, 3, 4, 5, 6]


def test_audit_rejects_physical_hit_count_disagreement(tmp_path) -> None:
    recorder = CorpusRecorder(
        tmp_path,
        RunMetadata(
            "test", "exe", "native", "test", 3, 0, 0, 4,
            {
                "source_commitment": "source-complete-hard-v1",
                "factual_state_schema": "th06-1.02h-offline-facts-v1",
            },
        ),
    )
    run_dir = recorder.close({
        "termination_reason": "practice-stage-complete",
        "stage_completed": True,
        "physical_hits": 1,
    })

    assert "physical-hit-count-mismatch" in audit(run_dir)["integrity_errors"]


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
