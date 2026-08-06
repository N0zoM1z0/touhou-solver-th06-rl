"""Crash-recoverable Stage boundary for an online policy checkpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile


TRANSACTION_SCHEMA = "th06-rl-stage-policy-transaction-v1"


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class StagePolicyTransaction:
    """Commit online updates only after one complete physical Stage."""

    def __init__(self, state_path: Path) -> None:
        self.state_path = state_path.resolve()
        self.marker_path = self.state_path.with_name(
            f".{self.state_path.name}.stage-transaction.json"
        )
        self.backup_path = self.state_path.with_name(
            f".{self.state_path.name}.stage-start"
        )
        self.active = False

    def recover_stale(self) -> bool:
        """Roll back a prior controller that did not close its Stage."""
        if not self.marker_path.is_file():
            # A crash between backup creation and marker publication never
            # exposed an updated policy, so this orphan is safe to discard.
            self.backup_path.unlink(missing_ok=True)
            return False
        marker = json.loads(self.marker_path.read_text(encoding="utf-8"))
        if marker.get("schema") != TRANSACTION_SCHEMA:
            raise ValueError("unknown Stage policy transaction marker")
        had_state = marker.get("had_state")
        if not isinstance(had_state, bool):
            raise ValueError("invalid Stage policy transaction marker")
        if had_state:
            if not self.backup_path.is_file():
                raise FileNotFoundError(
                    "Stage policy transaction lost its starting checkpoint"
                )
            _atomic_write(self.state_path, self.backup_path.read_bytes())
        else:
            self.state_path.unlink(missing_ok=True)
        # Delete the marker first. An orphan backup cannot trigger rollback.
        self.marker_path.unlink()
        self.backup_path.unlink(missing_ok=True)
        self.active = False
        return True

    def begin(self) -> bool:
        recovered = self.recover_stale()
        had_state = self.state_path.is_file()
        if had_state:
            _atomic_write(self.backup_path, self.state_path.read_bytes())
        else:
            self.backup_path.unlink(missing_ok=True)
        marker = json.dumps(
            {"schema": TRANSACTION_SCHEMA, "had_state": had_state},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8") + b"\n"
        _atomic_write(self.marker_path, marker)
        self.active = True
        return recovered

    def commit(self) -> None:
        if not self.active:
            return
        # Once the marker is gone, a reboot must retain the learned state.
        self.marker_path.unlink(missing_ok=True)
        self.backup_path.unlink(missing_ok=True)
        self.active = False

    def rollback(self) -> None:
        if not self.active and not self.marker_path.exists():
            return
        self.recover_stale()
