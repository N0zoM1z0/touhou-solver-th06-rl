from __future__ import annotations

from th06_rl.storage_budget import can_reserve_run, tree_file_bytes


def test_storage_budget_reserves_a_full_run(tmp_path) -> None:
    (tmp_path / "corpus").mkdir()
    (tmp_path / "corpus" / "shard.gz").write_bytes(b"x" * 80)
    (tmp_path / "live.jsonl").write_bytes(b"y" * 20)

    assert tree_file_bytes(tmp_path) == 100
    assert can_reserve_run(
        tmp_path, limit_bytes=150, reserve_bytes=50
    ) == (True, 100)
    assert can_reserve_run(
        tmp_path, limit_bytes=149, reserve_bytes=50
    ) == (False, 100)


def test_missing_storage_root_is_empty(tmp_path) -> None:
    assert tree_file_bytes(tmp_path / "missing") == 0
