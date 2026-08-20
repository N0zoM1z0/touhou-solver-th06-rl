from __future__ import annotations

import pytest

from scripts.verify_baseline_route import verify


def _documents(hits: int = 12):
    report = {
        "error": None,
        "controller_returncode": 0,
        "gdb_normalized": True,
        "repository_commit": "abc123",
        "repository_worktree_clean": True,
        "diagnostic_rng_seed": None,
        "immutable_policy_state_equal": True,
        "leftover_prefix_processes": [],
        "controller_completion": {"route_completed": True, "physical_hits": hits},
        "trace": {
            "physical_hits_in_run": hits,
            "zero_margin_frames": 0,
            "invalid_hard_collision_margin_frames": 0,
        },
    }
    run = {"metadata": {
        "episode_unit": "route",
        "code_commit": "abc123",
        "expected_stages": [1, 2, 3, 4, 5, 6],
        "planner": {
            "algorithm": "source-hard4-paused-publication-v2",
            "source_commitment": "source-complete-hard-v1",
            "factual_state_schema": "th06-1.02h-offline-facts-v2",
            "hard_horizon": 4,
            "learner_feature_horizon": 4,
            "minimum_collision_margin": 0.35,
            "zero_margin_fallback": False,
        },
    }}
    manifest = {
        "complete": True,
        "stage_trajectory_complete": True,
        "dropped_records": 0,
        "episode": {"unit": "route", "complete": True},
        "run_outcome": {
            "stage_completed": True,
            "termination_reason": "route-complete",
            "physical_hits": hits,
            "corpus_failure": None,
            "background_reactivations": 0,
            "capture_failures": 0,
            "corpus_failures": 0,
            "infrastructure_failures": 0,
            "trace_failures": 0,
        },
        "records": {"frames": 100, "transitions": 99, "anchors": 6},
        "summary": {"learning_eligible_transitions": 90},
    }
    audit = {
        "scope": {"observed_stages": [1, 2, 3, 4, 5, 6]},
        "physical_hits": hits,
        "bomb_events": 0,
        "integrity_errors": [],
        "hit_classifications": {},
        "latency": {},
        "source_successor_coverage": {
            "method": "retained-next-root-one-sided-coverage-v1",
            "checked_links": 99,
            "actual_lasers_checked": 25,
            "uncovered_aabbs": 0,
            "uncovered_lasers": 0,
            "retained_laser_geometry_unavailable": {},
        },
        "source_numeric_successor_parity": {
            "method": "stable-retained-bullet-center-successor-v2",
            "arithmetic_comparison": "float32-bit-exact",
            "required_collision_margin": 0.35,
            "transcendental_axis_error_budget": 0.01,
            "global_release_acceleration_axis_bound": 0.010000001,
            "global_mutation_semantics": "source branch union",
            "linear_exact_checked": 50,
            "acceleration_exact_checked": 25,
            "transcendental_checked": 25,
            "global_stop_union_checked": 10,
            "exact_mismatches": 0,
            "transcendental_budget_violations": 0,
            "nonfinite_successors": 0,
            "global_mutation_union_violations": 0,
        },
        "player_successor_parity": {
            "method": "contiguous-active-player-center-successor-v1",
            "arithmetic_comparison": "float32-bit-exact",
            "input_semantics": "next-completed-root-sampled-input",
            "movement_order": "Player-before-Enemy-before-Bullet",
            "checked_links": 90,
            "mismatches": 0,
        },
        "source_anchor_coverage": {
            "anchored_stages": [1, 2, 3, 4, 5, 6],
            "missing_observed_stages": [],
        },
        "source_dataset_admission": {
            "checked_frames": 100,
            "passes": True,
            "error": None,
        },
    }
    return report, run, manifest, audit


def test_baseline_route_verifier_conserves_all_factual_contracts() -> None:
    result = verify(*_documents())

    assert result["passed"] is True
    assert result["physical_hits"] == 12
    assert all(result["checks"].values())


def test_baseline_route_verifier_rejects_hit_disagreement() -> None:
    documents = list(_documents())
    documents[2]["run_outcome"]["physical_hits"] = 11

    with pytest.raises(ValueError, match="hit_conservation"):
        verify(*documents)


def test_baseline_route_verifier_rejects_dirty_or_misbound_code() -> None:
    documents = list(_documents())
    documents[0]["repository_worktree_clean"] = False

    with pytest.raises(ValueError, match="runner_clean"):
        verify(*documents)

    documents = list(_documents())
    documents[0]["repository_commit"] = "different"
    with pytest.raises(ValueError, match="runner_clean"):
        verify(*documents)


def test_baseline_route_verifier_rejects_missing_stage() -> None:
    documents = list(_documents())
    documents[3]["scope"]["observed_stages"] = [1, 2, 3, 4, 5]

    with pytest.raises(ValueError, match="route_scope"):
        verify(*documents)


def test_baseline_route_verifier_rejects_observed_only_authority() -> None:
    documents = list(_documents())
    documents[1]["metadata"]["planner"]["source_commitment"] = (
        "observed-only-unqualified"
    )

    with pytest.raises(ValueError, match="source_complete_online_authority"):
        verify(*documents)


def test_baseline_route_verifier_rejects_incomplete_offline_facts() -> None:
    documents = list(_documents())
    del documents[1]["metadata"]["planner"]["factual_state_schema"]

    with pytest.raises(ValueError, match="comprehensive_offline_facts"):
        verify(*documents)


def test_baseline_route_verifier_rejects_legacy_audit_without_causal_check() -> None:
    documents = list(_documents())
    del documents[3]["source_successor_coverage"]

    with pytest.raises(ValueError, match="source successor coverage"):
        verify(*documents)


def test_baseline_route_verifier_rejects_uncovered_successor_hazard() -> None:
    documents = list(_documents())
    documents[3]["source_successor_coverage"]["uncovered_aabbs"] = 1

    with pytest.raises(ValueError, match="causal_source_successors"):
        verify(*documents)


def test_baseline_route_verifier_rejects_numeric_successor_error() -> None:
    documents = list(_documents())
    documents[3]["source_numeric_successor_parity"]["exact_mismatches"] = 1

    with pytest.raises(ValueError, match="numeric_source_successors"):
        verify(*documents)


def test_baseline_route_verifier_rejects_player_successor_error() -> None:
    documents = list(_documents())
    documents[3]["player_successor_parity"]["mismatches"] = 1

    with pytest.raises(ValueError, match="player_input_successors"):
        verify(*documents)


def test_baseline_route_verifier_rejects_legacy_single_stage_anchor_check() -> None:
    documents = list(_documents())
    del documents[3]["source_anchor_coverage"]

    with pytest.raises(ValueError, match="source anchor coverage"):
        verify(*documents)


def test_baseline_route_verifier_rejects_unloadable_source_dataset() -> None:
    documents = list(_documents())
    documents[3]["source_dataset_admission"]["passes"] = False
    documents[3]["source_dataset_admission"]["error"] = "missing geometry"

    with pytest.raises(ValueError, match="self_contained_source_dataset"):
        verify(*documents)
