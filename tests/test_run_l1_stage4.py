from __future__ import annotations

import hashlib
import json
from pathlib import Path

import scripts.run_wine_retail as wine_runner
from scripts.run_l1_stage4 import (
    PREREG_SCHEMA,
    _fit_command,
    build_schedule,
    derive_episode_policy_state,
    load_prereg,
)
from scripts.gate_parallel_wine import build_runner_command


REPOSITORY = Path(__file__).resolve().parents[1]
PREREG = REPOSITORY / "experiments/l1-stage4-bc-v1.json"


def test_preregistration_freezes_serial_inventory_and_seeds() -> None:
    prereg = load_prereg(PREREG)
    rows = prereg["collection"]["episodes"]

    assert prereg["schema"] == PREREG_SCHEMA
    assert len(rows) == 12
    assert [row["index"] for row in rows if row["split"] == "validation"] == [
        2, 5, 8, 11,
    ]
    assert len({row["policy_seed"] for row in rows}) == 12
    assert prereg["collection"]["serial_wine_workers"] == 1
    assert prereg["collection"]["natural_retail_rng"] is True
    assert prereg["auxiliary_targets"] == []


def test_schedule_and_episode_states_are_replayable(tmp_path) -> None:
    prereg = load_prereg(PREREG)
    pool_path = tmp_path / "pool.json"
    pool_path.write_text("{}\n", encoding="utf-8")
    schedule = build_schedule(
        prereg,
        prereg_path=PREREG,
        pool_path=pool_path,
        commit="a" * 40,
        work_log_path=REPOSITORY / "work_log/fixture",
    )

    assert schedule == build_schedule(
        prereg,
        prereg_path=PREREG,
        pool_path=pool_path,
        commit="a" * 40,
        work_log_path=REPOSITORY / "work_log/fixture",
    )
    assert schedule["serial_wine_workers"] == 1
    for row in schedule["episodes"]:
        state = derive_episode_policy_state(prereg, episode=row["index"])
        assert state["policy_seed"] == row["policy_seed"]
        payload = (
            json.dumps(
                state, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode()
        assert hashlib.sha256(payload).hexdigest() == row["policy_state_sha256"]


def test_stage4_command_is_serial_normal_speed_complete_and_bomb_free(
    tmp_path, monkeypatch,
) -> None:
    command = build_runner_command(
        worker={
            "game_dir": tmp_path / "game",
            "wine_prefix": tmp_path / "prefix",
            "display": ":91",
            "game_cpu_list": "0-3",
            "controller_cpu_list": "4-7",
        },
        score_template=tmp_path / "score.dat",
        policy_plugin=tmp_path / "policy.py",
        policy_state=tmp_path / "state.json",
        stage=4,
        rng_seed=None,
        artifact_dir=tmp_path / "artifacts",
        corpus_root=tmp_path / "corpus",
    )

    assert command[command.index("--practice-stage") + 1] == "4"
    assert command[command.index("--seconds") + 1] == "0"
    assert "--complete-stage-training-corpus-root" in command
    assert "--diagnostic-rng-seed" not in command
    assert "--stop-on-hit" not in command
    assert all("bomb" not in argument.lower() for argument in command)
    monkeypatch.setattr(wine_runner.os, "sched_getaffinity", lambda _pid: set(range(8)))
    parsed = wine_runner.parse_args(command[2:])
    assert parsed.practice_stage == 4
    assert parsed.complete_stage_training_corpus_root == tmp_path / "corpus"


def test_fit_command_uses_only_frozen_whole_episode_split(tmp_path) -> None:
    prereg = load_prereg(PREREG)
    evidence = {
        index: {"run_dir": f"corpora/fixture/run-{index}"}
        for index in range(12)
    }
    command = _fit_command(prereg, evidence, tmp_path / "model.json")

    train = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--train-run"
    ]
    validation = [
        command[index + 1]
        for index, value in enumerate(command)
        if value == "--validation-run"
    ]
    assert len(train) == 8
    assert len(validation) == 4
    assert set(train).isdisjoint(validation)
    assert command[command.index("--epochs") + 1] == "100"
    assert command[command.index("--max-rows") + 1] == "400000"
