from __future__ import annotations

from th06_rl.policy_transaction import StagePolicyTransaction


def test_complete_stage_commits_updated_checkpoint(tmp_path) -> None:
    state = tmp_path / "policy.json"
    state.write_bytes(b"before")
    transaction = StagePolicyTransaction(state)
    assert transaction.begin() is False
    state.write_bytes(b"after")
    transaction.commit()

    assert state.read_bytes() == b"after"
    assert not transaction.marker_path.exists()
    assert not transaction.backup_path.exists()


def test_incomplete_stage_restores_starting_checkpoint(tmp_path) -> None:
    state = tmp_path / "policy.json"
    state.write_bytes(b"before")
    transaction = StagePolicyTransaction(state)
    transaction.begin()
    state.write_bytes(b"partial-stage")
    transaction.rollback()

    assert state.read_bytes() == b"before"
    assert not transaction.marker_path.exists()
    assert not transaction.backup_path.exists()


def test_incomplete_first_stage_removes_new_checkpoint(tmp_path) -> None:
    state = tmp_path / "policy.json"
    transaction = StagePolicyTransaction(state)
    transaction.begin()
    state.write_bytes(b"partial-stage")
    transaction.rollback()

    assert not state.exists()


def test_next_controller_recovers_stale_transaction_before_loading(tmp_path) -> None:
    state = tmp_path / "policy.json"
    state.write_bytes(b"before")
    abandoned = StagePolicyTransaction(state)
    abandoned.begin()
    state.write_bytes(b"checkpoint-from-interrupted-stage")

    replacement = StagePolicyTransaction(state)
    assert replacement.begin() is True
    assert state.read_bytes() == b"before"
    replacement.rollback()
    assert state.read_bytes() == b"before"
