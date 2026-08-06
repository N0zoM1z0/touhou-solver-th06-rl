from __future__ import annotations

import json

from th06_rl.storage_budget import (
    RUN_METADATA_RESERVE_BYTES,
    accounted_artifact_bytes,
    can_reserve_run,
    tree_file_bytes,
)


def test_storage_budget_reserves_a_full_run(tmp_path) -> None:
    run = tmp_path / "corpus" / "run"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({"compressed_bytes": 80}))
    live = tmp_path / "live"
    live.mkdir()
    (live / "live.jsonl").write_bytes(b"y" * 20)

    used = 100 + RUN_METADATA_RESERVE_BYTES
    assert accounted_artifact_bytes(tmp_path) == used
    assert can_reserve_run(
        tmp_path, limit_bytes=used + 50, reserve_bytes=50
    ) == (True, used)
    assert can_reserve_run(
        tmp_path, limit_bytes=used + 49, reserve_bytes=50
    ) == (False, used)


def test_missing_storage_root_is_empty(tmp_path) -> None:
    assert tree_file_bytes(tmp_path / "missing") == 0
    assert accounted_artifact_bytes(tmp_path / "missing") == 0
