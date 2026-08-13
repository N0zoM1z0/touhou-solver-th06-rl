from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.materialize_generation6_decision_candidate import (
    materialize_candidate,
)
from scripts.smoke_generation6_online_policy import (
    DEFAULT_CONTEXT_LOAD_MAXIMUM_SECONDS,
    DEFAULT_PREFLIGHT_WORKERS,
)
from th06_rl.policies.autonomous_iql_actor import (
    ALLOWED_NATIVE_SCORER_SHA256,
    ALLOWED_PREFLIGHT_NATIVE_SCORER_SHA256,
)


def _inputs():
    registry = "1" * 64
    fit_sha = "2" * 64
    library = "3" * 64
    contract_sha = "4" * 64
    contract = {
        "schema": "autonomous-generation-6-decision-successor-contract-v2",
        "authorization_eligible": False,
        "bomb": "forbidden",
        "reused_training": {
            "registry_sha256": registry,
            "fit_checkpoint_sha256": fit_sha,
            "episodes": 56,
            "factual_options": 167250,
            "new_wine_collection": False,
            "refit": False,
            "manual_stage_phase_frame_rng_hit_or_failure_targeting": False,
        },
        "numeric_conformance": {
            "reference": "native-order-centered-float64-v1",
            "serialized_parameter_precision": "float32",
            "serving_intermediate_precision": "float64",
        },
        "audit_performance": {
            "worker_processes": 16,
            "math_library_threads_per_worker": 1,
            "full_corpus_and_panel_maximum_seconds": 180.0,
        },
    }
    fit = {
        "schema": "autonomous-generation-6-fit-checkpoint-v1",
        "training_identity": {"sha256": registry},
        "training_identity_sha256": "5" * 64,
        "representation": {},
        "support": {},
        "support_report": {},
        "actors": [{} for _ in range(7)],
        "actor_diagnostics": [{} for _ in range(7)],
        "actor_bootstrap": [{} for _ in range(7)],
        "advantage_crossfit": {},
    }
    audit = {
        "schema": "autonomous-generation-6-native-decision-conformance-v1",
        "contract_sha256": contract_sha,
        "fit_checkpoint_sha256": fit_sha,
        "training_registry_sha256": registry,
        "linux_library_sha256": library,
        "passed": True,
        "reference": {"kind": "native-order-centered-float64-v1"},
        "worker_processes": 16,
        "math_library_threads_per_worker": 1,
        "timing": {"total_seconds": 110.0},
        "full_linux": {
            "options": 167250,
            "exact_choices": 167250,
            "exact_support_masks": 167250,
            "finite_fixed_width_rows": 167250,
        },
        "wide_panel": {
            "cases": 320,
            "exact_choices": 320,
            "covered_target_errors": 320,
            "certified_decisions": 320,
            "minimum_margin_ratio": 500000.0,
        },
    }
    return contract, fit, audit, contract_sha, fit_sha, registry, library


def test_materializer_reuses_fit_only_after_all_decision_gates() -> None:
    contract, fit, audit, contract_sha, fit_sha, registry, library = _inputs()
    candidate = materialize_candidate(
        contract=contract, contract_sha256=contract_sha,
        fit=fit, fit_sha256=fit_sha, audit=audit, audit_sha256="6" * 64,
        registry_sha256=registry, linux_library_sha256=library,
    )
    assert candidate["passed"] is True
    assert candidate["actors"] is fit["actors"]
    assert candidate["selection"]["physical_safety"] == "native-safe-set-only"
    assert candidate["numeric_serving"]["intermediate_precision"] == "float64"

    slow = deepcopy(audit)
    slow["timing"]["total_seconds"] = 181.0
    with pytest.raises(ValueError, match="bounded_audit_performance_exact"):
        materialize_candidate(
            contract=contract, contract_sha256=contract_sha,
            fit=fit, fit_sha256=fit_sha, audit=slow, audit_sha256="6" * 64,
            registry_sha256=registry, linux_library_sha256=library,
        )


def test_new_float64_binaries_are_preflight_only_until_wine_gate() -> None:
    for digest in (
        "b0d8a2ea8efeb2e4d3b0798f109a9d2c5da992e8d78a3ab8434776590c88a283",
        "5fad7ae536f3933fbe467f6b485455937c200be9b960958207ba90a8e501bb27",
    ):
        assert digest in ALLOWED_PREFLIGHT_NATIVE_SCORER_SHA256
        assert digest not in ALLOWED_NATIVE_SCORER_SHA256


def test_frozen_panel_context_loading_keeps_bounded_parallel_contract() -> None:
    assert DEFAULT_PREFLIGHT_WORKERS == 16
    assert DEFAULT_PREFLIGHT_WORKERS <= 32
    assert DEFAULT_CONTEXT_LOAD_MAXIMUM_SECONDS == 120.0
