"""Explicit temporary import boundary to the verified old TH06 substrate.

Only coherent capture, source projection, lifecycle sensing, and reversible
input are imported. Routes, phase state machines, solver ranking, and learning
are never imported by the new runtime.
"""

from __future__ import annotations

import os
from pathlib import Path
import sys


def donor_scripts_path() -> Path:
    override = os.environ.get("TH06_RL_DONOR_SCRIPTS")
    if override:
        path = Path(override).expanduser().resolve()
    else:
        repository = Path(__file__).resolve().parents[3]
        path = repository.parent / "th06" / "scripts"
    if not (path / "th06" / "native.py").is_file():
        raise RuntimeError(
            "verified TH06 donor scripts are unavailable; set "
            f"TH06_RL_DONOR_SCRIPTS (tried {path})"
        )
    return path


def enable_donor_imports() -> Path:
    path = donor_scripts_path()
    value = str(path)
    if value not in sys.path:
        sys.path.insert(0, value)
    return path

