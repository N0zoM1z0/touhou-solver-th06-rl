"""Small fail-closed storage guard for complete-Stage collection loops."""

from __future__ import annotations

import os
from pathlib import Path


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


def can_reserve_run(
    root: Path,
    *,
    limit_bytes: int,
    reserve_bytes: int,
) -> tuple[bool, int]:
    if limit_bytes <= 0 or reserve_bytes <= 0:
        raise ValueError("storage limit and run reserve must be positive")
    used = tree_file_bytes(root)
    return used + reserve_bytes <= limit_bytes, used
