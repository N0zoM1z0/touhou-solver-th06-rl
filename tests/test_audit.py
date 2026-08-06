from __future__ import annotations

import importlib.util
from pathlib import Path

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
