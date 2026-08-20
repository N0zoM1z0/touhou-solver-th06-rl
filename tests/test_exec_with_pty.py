from __future__ import annotations

from pathlib import Path
import subprocess
import sys


REPOSITORY = Path(__file__).resolve().parents[1]


def test_pty_bridge_gives_console_child_a_tty_and_propagates_status() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(REPOSITORY / "scripts/exec_with_pty.py"),
            "--",
            sys.executable,
            "-c",
            "import os, sys; print(os.isatty(0), os.isatty(1)); sys.exit(7)",
        ],
        check=False,
        capture_output=True,
        timeout=10,
    )

    assert completed.returncode == 7
    assert b"True True" in completed.stdout
