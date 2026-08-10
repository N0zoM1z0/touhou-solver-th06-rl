from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from th06_rl.headless import HeadlessClient, HeadlessProtocolError, HeadlessScope


def _fake_headless(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            import json
            import sys

            print(json.dumps({"tick": 1, "terminal_reason": None}), flush=True)
            for line in sys.stdin:
                print(json.dumps({
                    "tick": 2,
                    "action": line.strip(),
                    "terminal_reason": "physical-hit",
                }), flush=True)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_headless_client_steps_without_a_pty_and_rejects_bomb(tmp_path: Path) -> None:
    client = HeadlessClient(
        binary=_fake_headless(tmp_path / "fake-headless"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )
    try:
        assert client.start() == {"tick": 1, "terminal_reason": None}
        with pytest.raises(ValueError, match="forbidden"):
            client.step("bomb")
        observation = client.step("left")
        assert observation["action"] == "left"
        assert observation["terminal_reason"] == "physical-hit"
        with pytest.raises(HeadlessProtocolError, match="terminal"):
            client.step("stay")
    finally:
        client.close()


def test_headless_scope_does_not_silently_mix_learning_keys() -> None:
    with pytest.raises(ValueError, match="stage"):
        HeadlessScope(3, 0, 0, 7)


def test_continue_after_hit_is_an_explicit_headless_flag(tmp_path: Path) -> None:
    client = HeadlessClient(
        binary=_fake_headless(tmp_path / "fake-headless"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 1),
        seed=7,
        continue_after_hit=True,
    )

    assert "--continue-after-hit" in client._command()
