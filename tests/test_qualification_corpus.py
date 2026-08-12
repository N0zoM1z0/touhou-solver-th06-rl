from __future__ import annotations

import json
from pathlib import Path

import pytest

from th06_rl.qualification_corpus import load_qualification_partition


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_qualification_split_is_chronological_and_manifest_bound(tmp_path: Path) -> None:
    import hashlib

    runs = []
    for index in range(3):
        run = tmp_path / "artifacts" / f"run-{index}"
        _write(run / "manifest.json", {"index": index})
        runs.append({
            "path": str(run),
            "manifest_sha256": hashlib.sha256(
                (run / "manifest.json").read_bytes()
            ).hexdigest(),
        })
    report = tmp_path / "artifacts" / "report.json"
    _write(report, {"schema": "report-v1", "runs": runs})
    contract = tmp_path / "config.json"
    _write(contract, {
        "schema": "autonomous-offline-learner-qualification-v1",
        "evidence_eligible": False,
        "sources": [{
            "name": "fixture",
            "report": "artifacts/report.json",
            "report_sha256": hashlib.sha256(report.read_bytes()).hexdigest(),
            "report_schema": "report-v1",
            "stage": 4,
            "development_prefix": 2,
            "qualification_suffix": 1,
        }],
        "expected": {
            "development_episode_groups": 2,
            "qualification_episode_groups": 1,
            "total_episode_groups": 3,
        },
    })

    _contract, partition = load_qualification_partition(
        contract, repository=tmp_path
    )

    assert [row.role for row in partition] == [
        "development", "development", "qualification"
    ]
    assert [row.chronological_index for row in partition] == [0, 1, 2]

    (tmp_path / "artifacts/run-2/manifest.json").write_text("{}")
    with pytest.raises(ValueError, match="manifest drifted"):
        load_qualification_partition(contract, repository=tmp_path)
