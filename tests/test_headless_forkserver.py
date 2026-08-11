from __future__ import annotations

from pathlib import Path
import textwrap

import pytest

from th06_rl.headless import HeadlessProtocolError, HeadlessScope
from th06_rl.headless_forkserver import HeadlessForkserver


def _fake_forkserver(path: Path) -> Path:
    path.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env python3
            from pathlib import Path
            import sys

            print("READY 1", flush=True)
            child = 100
            nested = False
            for line in sys.stdin:
                fields = line.split()
                if fields == ["QUIT"]:
                    if nested:
                        nested = False
                        child += 1
                        print(f"CHECKPOINT_DONE {child} 0", flush=True)
                        continue
                    raise SystemExit(0)
                if len(fields) == 3 and fields[0] == "CHECKPOINT":
                    nested = True
                    print(f"READY {fields[1]}", flush=True)
                    continue
                if fields == ["STEP", "10"]:
                    print('{"tick":1,"terminal_reason":null}', flush=True)
                    action = sys.stdin.readline().strip()
                    terminal = "input-error" if action == "__authority_abort__" else "tick-limit"
                    print(f'{{"tick":2,"terminal_reason":"{terminal}"}}', flush=True)
                    child += 1
                    status = 1 if terminal == "input-error" else 0
                    print(f"DONE {child} {status}", flush=True)
                    continue
                if len(fields) != 4 or fields[0] not in {"RUN", "RUN_FINAL"}:
                    print("ERROR bad-command", flush=True)
                    continue
                Path(fields[3]).write_text('{"tick":1}\\n')
                child += 1
                print(f"DONE {child} 0", flush=True)
            """
        ),
        encoding="utf-8",
    )
    path.chmod(0o755)
    return path


def test_forkserver_runs_children_without_a_pty(tmp_path: Path) -> None:
    actions = tmp_path / "actions.txt"
    actions.write_text("stay\n", encoding="utf-8")
    trace = tmp_path / "trace.jsonl"
    server = HeadlessForkserver(
        binary=_fake_forkserver(tmp_path / "fake-forkserver"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )
    try:
        assert server.start() == 1
        result = server.run(
            terminal_tick=10,
            actions_path=actions,
            trace_path=trace,
        )
        assert result.child_pid == 101
        assert result.status == 0
        assert result.summary_only is False
        assert trace.read_text(encoding="utf-8") == '{"tick":1}\n'
    finally:
        server.close()


def test_forkserver_passes_retail_delivery_options(tmp_path: Path) -> None:
    dialogue_inputs = tmp_path / "dialogue-inputs.txt"
    dialogue_inputs.write_text("4996 2 1\n", encoding="ascii")
    server = HeadlessForkserver(
        binary=_fake_forkserver(tmp_path / "fake-forkserver"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=0,
        stage_rng_seed=3193,
        auto_shoot=True,
        auto_shoot_after_tick=127,
        retail_dialogue_control=True,
        retail_dialogue_control_after_tick=4995,
        retail_dialogue_inputs_path=dialogue_inputs,
    )

    command = server._command()

    assert command[command.index("--stage-rng-seed") + 1] == "3193"
    assert command[command.index("--auto-shoot-after-tick") + 1] == "127"
    assert command[command.index("--retail-dialogue-control-after-tick") + 1] == "4995"
    assert command[command.index("--retail-dialogue-inputs") + 1] == str(
        dialogue_inputs.resolve()
    )

    with pytest.raises(ValueError, match="requires auto_shoot"):
        HeadlessForkserver(
            binary=server.binary,
            game_directory=tmp_path,
            scope=server.scope,
            seed=0,
            auto_shoot=False,
            auto_shoot_after_tick=127,
        )
    with pytest.raises(ValueError, match="requires auto_shoot"):
        HeadlessForkserver(
            binary=server.binary,
            game_directory=tmp_path,
            scope=server.scope,
            seed=0,
            auto_shoot=False,
            retail_dialogue_control=True,
        )
    with pytest.raises(ValueError, match="requires retail dialogue control"):
        HeadlessForkserver(
            binary=server.binary,
            game_directory=tmp_path,
            scope=server.scope,
            seed=0,
            auto_shoot=True,
            retail_dialogue_control_after_tick=4995,
        )
    with pytest.raises(ValueError, match="requires retail dialogue control"):
        HeadlessForkserver(
            binary=server.binary,
            game_directory=tmp_path,
            scope=server.scope,
            seed=0,
            auto_shoot=True,
            retail_dialogue_inputs_path=dialogue_inputs,
        )


def test_forkserver_rejects_bad_ticks_and_protocol_paths(tmp_path: Path) -> None:
    actions = tmp_path / "actions.txt"
    actions.write_text("stay\n", encoding="utf-8")
    server = HeadlessForkserver(
        binary=_fake_forkserver(tmp_path / "fake-forkserver"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )
    try:
        server.start()
        with pytest.raises(ValueError, match="follow"):
            server.run(
                terminal_tick=1,
                actions_path=actions,
                trace_path=tmp_path / "trace.jsonl",
            )
        with pytest.raises(ValueError, match="whitespace"):
            server.run(
                terminal_tick=2,
                actions_path=actions,
                trace_path=tmp_path / "bad trace.jsonl",
            )
    finally:
        server.close()


def test_forkserver_reports_a_bad_greeting(tmp_path: Path) -> None:
    binary = tmp_path / "bad-forkserver"
    binary.write_text("#!/bin/sh\necho WRONG\n", encoding="utf-8")
    binary.chmod(0o755)
    server = HeadlessForkserver(
        binary=binary,
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )

    with pytest.raises(HeadlessProtocolError, match="greeting"):
        server.start()


def test_forkserver_replays_one_prefix_for_multiple_summary_branches(
    tmp_path: Path,
) -> None:
    actions = tmp_path / "actions.txt"
    actions.write_text("stay\n", encoding="utf-8")
    server = HeadlessForkserver(
        binary=_fake_forkserver(tmp_path / "fake-forkserver"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )
    try:
        server.start()
        assert server.enter_checkpoint(terminal_tick=600, actions_path=actions) == 600
        first = server.run(
            terminal_tick=660,
            actions_path=actions,
            trace_path=tmp_path / "first.jsonl",
            summary_only=True,
        )
        second = server.run(
            terminal_tick=660,
            actions_path=actions,
            trace_path=tmp_path / "second.jsonl",
            summary_only=True,
        )
        assert first.summary_only and second.summary_only
        restored = server.leave_checkpoint()
        assert restored.restored_tick == 1
        assert server.checkpoint_tick == 1
    finally:
        server.close()


def test_forkserver_interactive_child_reuses_control_pipes(tmp_path: Path) -> None:
    server = HeadlessForkserver(
        binary=_fake_forkserver(tmp_path / "fake-forkserver"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )
    try:
        server.start()
        assert server.begin_step_session(terminal_tick=10)["tick"] == 1
        terminal = server.step_session("stay")
        assert terminal["terminal_reason"] == "tick-limit"
        result = server.finish_step_session()
        assert result.status == 0
        assert result.aborted is False
        assert result.terminal_observation == terminal
    finally:
        server.close()


def test_forkserver_interactive_child_can_fail_close_authority(tmp_path: Path) -> None:
    server = HeadlessForkserver(
        binary=_fake_forkserver(tmp_path / "fake-forkserver"),
        game_directory=tmp_path,
        scope=HeadlessScope(3, 0, 0, 6),
        seed=7,
    )
    try:
        server.start()
        server.begin_step_session(terminal_tick=10)
        result = server.abort_step_session()
        assert result.status == 1
        assert result.aborted is True
        assert result.terminal_observation["terminal_reason"] == "input-error"
    finally:
        server.close()
