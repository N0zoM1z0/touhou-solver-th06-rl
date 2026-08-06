"""Small fail-closed storage guard for complete-Stage collection loops."""

from __future__ import annotations

import json
import os
from pathlib import Path


RUN_METADATA_RESERVE_BYTES = 2 * 1024 * 1024


def tree_file_bytes(root: Path) -> int:
    """Return logical bytes below *root* without following directory links."""
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    for directory, _subdirectories, files in os.walk(root, followlinks=False):
        base = Path(directory)
        for name in files:
            path = base / name
            if not path.is_symlink():
                total += path.stat().st_size
    return total


def accounted_artifact_bytes(root: Path) -> int:
    """Account artifacts with one UNC read per corpus run, not per shard."""
    root = Path(root)
    if not root.exists():
        return 0
    total = 0
    corpus = root / "corpus"
    if corpus.is_dir():
        for run_dir in corpus.iterdir():
            if not run_dir.is_dir():
                continue
            manifest_path = run_dir / "manifest.json"
            if not manifest_path.is_file():
                raise OSError(f"corpus run has no manifest: {run_dir}")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            compressed = int(manifest.get("compressed_bytes", -1))
            if compressed < 0:
                raise ValueError(f"invalid compressed byte count: {run_dir}")
            # Covers run/manifest/audit JSON and a currently open shard that
            # has not yet reached the atomic manifest callback.
            total += compressed + RUN_METADATA_RESERVE_BYTES
    # These directories contain only a few append-only traces/checkpoints;
    # scanning them is cheap even through the Windows WSL UNC provider.
    total += tree_file_bytes(root / "live")
    total += tree_file_bytes(root / "policy")
    for path in root.iterdir():
        if path.is_file() and not path.is_symlink():
            total += path.stat().st_size
    return total


def can_reserve_run(
    root: Path,
    *,
    limit_bytes: int,
    reserve_bytes: int,
) -> tuple[bool, int]:
    if limit_bytes <= 0 or reserve_bytes <= 0:
        raise ValueError("storage limit and run reserve must be positive")
    used = accounted_artifact_bytes(root)
    return used + reserve_bytes <= limit_bytes, used
