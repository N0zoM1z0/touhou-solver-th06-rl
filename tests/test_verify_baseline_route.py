import pytest

from scripts.verify_baseline_route import verify
from th06_rl.th06.control_capture import OFFLINE_FACT_SCHEMA


def _documents(hits: int = 7):
    commit = "a" * 40
    report = {
        "error": None,
        "controller_returncode": 0,
        "gdb_normalized": True,
        "repository_worktree_clean": True,
        "repository_commit": commit,
        "diagnostic_rng_seed": None,
        "immutable_policy_state_equal": True,
        "leftover_prefix_processes": [],
        "controller_completion": {
            "route_completed": True,
            "physical_hits": hits,
        },
        "trace": {
            "physical_hits_in_run": hits,
            "zero_margin_frames": 0,
            "invalid_shield_collision_margin_frames": 0,
        },
    }
    run = {
        "metadata": {
            "code_commit": commit,
            "episode_unit": "route",
            "expected_stages": [1, 2, 3, 4, 5, 6],
            "online_contract": {
                "algorithm": "observed-shield4-paused-publication-v1",
                "shield_contract": "observed-hazard-kinematics-v1",
                "publication_epoch": "coherent-root-process-suspended-v1",
                "shield_horizon": 4,
                "predicts_future_births": False,
                "minimum_collision_margin": 0.35,
                "factual_state_schema": OFFLINE_FACT_SCHEMA,
            },
        },
    }
    outcome = {
        "stage_completed": True,
        "termination_reason": "route-complete",
        "physical_hits": hits,
        "corpus_failure": None,
        "background_reactivations": 0,
        "capture_failures": 0,
        "corpus_failures": 0,
        "infrastructure_failures": 0,
        "policy_failures": 0,
        "trace_failures": 0,
    }
    manifest = {
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "episode": {"unit": "route", "complete": True},
        "run_outcome": outcome,
        "records": {"frames": 101, "transitions": 100},
        "summary": {},
    }
    player = {
        "method": "contiguous-player-center-successor-v1",
        "arithmetic_comparison": "float32-bit-exact",
        "input_semantics": "next-completed-root-sampled-input",
        "checked_links": 90,
        "mismatches": 0,
    }
    shield = {
        "method": "stored-observed-primitives-native-replay-v1",
        "checked": 64,
        "unsafe_divergences": [],
        "conservative_divergences": [],
    }
    audit = {
        "scope": {"observed_stages": [1, 2, 3, 4, 5, 6]},
        "episode_dataset_admission": {
            "checked_frames": 101,
            "checked_transitions": 100,
            "passes": True,
            "error": None,
        },
        "player_successor_parity": player,
        "dense_shield_parity": shield,
        "bomb_events": 0,
        "physical_hits": hits,
        "integrity_errors": [],
        "hit_classifications": {"policy-outcome": hits},
        "latency": {},
    }
    return report, run, manifest, audit


def test_baseline_route_accepts_hits_when_infra_is_complete() -> None:
    result = verify(*_documents(11))

    assert result["passed"] is True
    assert result["physical_hits"] == 11
    assert result["schema"] == "th06-rl-baseline-route-verification-v2"


def test_baseline_route_rejects_incomplete_episode() -> None:
    documents = _documents()
    documents[2]["complete"] = False

    with pytest.raises(ValueError, match="durable_complete"):
        verify(*documents)


def test_baseline_route_rejects_shield_replay_divergence() -> None:
    documents = _documents()
    documents[3]["dense_shield_parity"]["unsafe_divergences"] = [
        {"sequence": 9, "extra": ["left"]}
    ]

    with pytest.raises(ValueError, match="observed_shield_replay"):
        verify(*documents)


def test_baseline_route_rejects_unloadable_generic_episode() -> None:
    documents = _documents()
    documents[3]["episode_dataset_admission"]["passes"] = False
    documents[3]["episode_dataset_admission"]["error"] = "orphan transition"

    with pytest.raises(ValueError, match="algorithm_independent_episode"):
        verify(*documents)
