from __future__ import annotations

import hashlib
import json

import pytest

from scripts.collect_route_parallel import (
    SCHEDULE_SCHEMA,
    _schedule,
    build_route_runner_command,
    derive_episode_policy_state,
)


def _state(seed: int = 7) -> dict[str, object]:
    return {
        "schema": "th06-rl-safe-option-exploration-v2",
        "policy_seed": seed,
        "exploration_probability": 0.2,
        "option_horizon_frames": 8,
    }


def test_episode_policy_streams_are_distinct_replayable_and_canonical() -> None:
    first = derive_episode_policy_state(_state(), episode=0)
    second = derive_episode_policy_state(_state(), episode=1)

    assert first == derive_episode_policy_state(_state(), episode=0)
    assert first["policy_seed"] != second["policy_seed"]
    assert first["exploration_probability"] == 0.2
    with pytest.raises(ValueError, match="safe-option"):
        derive_episode_policy_state({"schema": "wrong"}, episode=0)


def test_route_schedule_binds_each_distinct_policy_state() -> None:
    workers = [{"worker": index} for index in range(8)]
    schedule = _schedule(
        episodes=10,
        workers=workers,
        base_state=_state(),
        gate_sha256="a" * 64,
        commit="b" * 40,
    )

    assert schedule["schema"] == SCHEDULE_SCHEMA
    rows = schedule["episodes"]
    assert [row["worker"] for row in rows] == [0, 1, 2, 3, 4, 5, 6, 7, 0, 1]
    assert len({row["policy_seed"] for row in rows}) == 10
    for row in rows:
        state = derive_episode_policy_state(_state(), episode=row["episode"])
        payload = (json.dumps(
            state, indent=2, sort_keys=True, allow_nan=False
        ) + "\n").encode()
        assert row["policy_state_sha256"] == hashlib.sha256(payload).hexdigest()


def test_route_runner_never_changes_clock_or_enables_bomb(tmp_path) -> None:
    command = build_route_runner_command(
        worker={
            "game_dir": tmp_path / "game",
            "wine_prefix": tmp_path / "prefix",
            "display": ":91",
            "game_cpu_list": "0-7",
            "controller_cpu_list": "8-15",
        },
        score_template=tmp_path / "score.dat",
        policy_plugin=tmp_path / "policy.py",
        policy_state=tmp_path / "state.json",
        artifact_dir=tmp_path / "artifacts",
        corpus_root=tmp_path / "corpus",
    )

    assert "--start-route" in command
    assert command[command.index("--difficulty") + 1] == "lunatic"
    assert command[command.index("--seconds") + 1] == "0"
    assert "--diagnostic-frame-multiplier" not in command
    assert "--diagnostic-rng-seed" not in command
    assert "--stop-on-hit" not in command
    assert all("bomb" not in argument.lower() for argument in command)
