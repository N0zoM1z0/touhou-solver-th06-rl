from __future__ import annotations

import hashlib
import json

import pytest

from scripts.authorize_supported_implicit_q_canary import authorize
from scripts.run_autonomous_learning_v5 import _behavior_state, _collect_wave
from scripts.shadow_supported_implicit_q import SCHEMA as SHADOW_SCHEMA
from tests.test_autonomous_supported_implicit_q_policy import _implicit_state
from th06_rl.policies.propensity_aware_option_exploration import STATE_SCHEMA


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def test_generation5_authorization_hash_binds_native_shadow(tmp_path) -> None:
    state = _implicit_state("shadow")
    state["fit_report"] = {"episode_groups": ["episode-a"]}
    state_path = tmp_path / "state.json"
    _write(state_path, state)
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    shadow_path = tmp_path / "shadow.json"
    _write(shadow_path, {
        "schema": SHADOW_SCHEMA,
        "shadow_eligible": True,
        "policy_state_sha256": state_sha,
        "audit_episode_groups": ["episode-a"],
        "decisions": 10,
        "policy_metrics": {"shadow_proposals": 2},
        "native_scorer_sha256": "b" * 64,
        "latency": {"p95_ms": 1.5},
    })

    active = authorize(state_path, shadow_path)

    assert active["mode"] == "active"
    assert active["authorization"]["active_canary"]["shadow_decisions"] == 10


def test_behavior_state_accepts_only_fit_eligible_generation5_information(
    tmp_path,
) -> None:
    information = _implicit_state("shadow")
    information_path = tmp_path / "information.json"
    _write(information_path, information)
    behavior_path = tmp_path / "behavior.json"

    _behavior_state(
        behavior_path,
        policy_seed=123,
        information_policy=information_path,
    )

    behavior = json.loads(behavior_path.read_text(encoding="utf-8"))
    assert behavior["schema"] == STATE_SCHEMA
    assert behavior["information_policy"]["schema"] == information["schema"]


def test_collection_wave_refuses_two_runs_on_one_worker(tmp_path) -> None:
    with pytest.raises(ValueError, match="worker twice"):
        _collect_wave(
            root=tmp_path,
            stage=4,
            rows=[
                {"episode": 0, "worker": 0},
                {"episode": 1, "worker": 0},
            ],
            workers=[],
            information_policy=None,
            scorer=tmp_path / "unused.dll",
            parallel=True,
        )


def test_failed_parallel_gate_runs_frozen_wave_serially(monkeypatch, tmp_path) -> None:
    observed = []

    def complete_run(**arguments):
        observed.append(int(arguments["worker"]["worker"]))
        run_dir = tmp_path / f"run-{len(observed)}"
        return {"controller_completion": {"physical_hits": 0}}, run_dir

    monkeypatch.setattr(
        "scripts.run_autonomous_learning_v5.complete_run", complete_run
    )
    monkeypatch.setattr(
        "scripts.run_autonomous_learning_v5._behavior_state",
        lambda *args, **kwargs: None,
    )
    rows = [
        {"episode": 0, "worker": 0, "game_rng_seed": 1, "policy_seed": 3},
        {"episode": 1, "worker": 1, "game_rng_seed": 2, "policy_seed": 4},
    ]

    completed = _collect_wave(
        root=tmp_path,
        stage=4,
        rows=rows,
        workers=[
            {"worker": 0, "display": ":97"},
            {"worker": 1, "display": ":98"},
        ],
        information_policy=None,
        scorer=tmp_path / "unused.dll",
        parallel=False,
    )

    assert observed == [0, 1]
    assert [row["episode"] for row in completed] == [0, 1]
