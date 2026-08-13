from scripts.audit_generation6_round_offline import (
    offline_gates,
    successor_identity_gates,
)


def _cohort():
    return {
        "policy_dr_hit_effect_bootstrap_upper_95": -0.1,
        "policy_loo_worst_bootstrap_upper_95": -0.05,
        "policy_model_hit_effect_mean": -0.2,
        "policy_dr_beneficial_episode_rate": 0.75,
        "mean_population_proposal_rate": 0.05,
        "policy_intervention_exposure_rate": 0.01,
        "policy_max_abs_correction": 1.5,
    }


def test_round_smoke_is_conjunctive_across_generic_stage_cohorts() -> None:
    cohorts = {"overall": _cohort(), "stage-4": _cohort(), "stage-5": _cohort()}
    gates = offline_gates(
        synthetic={"passed": True},
        crossfit={
            "input_identity": {"sha256": "r" * 64},
            "report": {"cohorts": cohorts},
        },
        candidate={
            "passed": True,
            "training_identity": {"sha256": "r" * 64},
        },
        online={"passed": True, "candidate_sha256": "c" * 64},
        candidate_sha256="c" * 64,
        registry_sha256="r" * 64,
        required_cohorts=("stage-4", "stage-5"),
    )
    assert all(gates.values())
    cohorts["stage-5"]["policy_model_hit_effect_mean"] = 0.1
    failed = offline_gates(
        synthetic={"passed": True},
        crossfit={
            "input_identity": {"sha256": "r" * 64},
            "report": {"cohorts": cohorts},
        },
        candidate={
            "passed": True,
            "training_identity": {"sha256": "r" * 64},
        },
        online={"passed": True, "candidate_sha256": "c" * 64},
        candidate_sha256="c" * 64,
        registry_sha256="r" * 64,
        required_cohorts=("stage-4", "stage-5"),
    )
    assert failed["stage_5_model_effect_below_zero"] is False


def test_successor_smoke_binds_decision_panel_and_native_pair() -> None:
    contract = {
        "reused_training": {
            "synthetic_sha256": "s" * 64,
            "crossfit_sha256": "x" * 64,
            "registry_sha256": "r" * 64,
        },
        "frozen_inputs": {
            "build/native/libth06_rl_ranker.so": "l" * 64,
            "build/native-win32-fully-static/libth06_rl_ranker.dll": "w" * 64,
        },
    }
    gates = successor_identity_gates(
        contract=contract, contract_sha256="c" * 64,
        candidate={
            "autonomous_round_contract_sha256": "c" * 64,
            "native_decision_conformance": {"sha256": "d" * 64},
        },
        decision={"passed": True, "contract_sha256": "c" * 64},
        decision_sha256="d" * 64,
        online={
            "schema": "autonomous-generation-6-decision-panel-preflight-v2",
            "decision_audit_sha256": "d" * 64,
            "linux_library_sha256": "l" * 64,
            "windows_library_sha256": "w" * 64,
            "factual_contexts": 320,
            "mismatches": [],
        },
        synthetic_sha256="s" * 64, crossfit_sha256="x" * 64,
        registry_sha256="r" * 64,
    )
    assert all(gates.values())
    drifted = dict(contract)
    drifted["frozen_inputs"] = dict(contract["frozen_inputs"])
    drifted["frozen_inputs"]["build/native/libth06_rl_ranker.so"] = "z" * 64
    assert successor_identity_gates(
        contract=drifted, contract_sha256="c" * 64,
        candidate={
            "autonomous_round_contract_sha256": "c" * 64,
            "native_decision_conformance": {"sha256": "d" * 64},
        },
        decision={"passed": True, "contract_sha256": "c" * 64},
        decision_sha256="d" * 64,
        online={
            "schema": "autonomous-generation-6-decision-panel-preflight-v2",
            "decision_audit_sha256": "d" * 64,
            "linux_library_sha256": "l" * 64,
            "windows_library_sha256": "w" * 64,
            "factual_contexts": 320, "mismatches": [],
        },
        synthetic_sha256="s" * 64, crossfit_sha256="x" * 64,
        registry_sha256="r" * 64,
    )["frozen_native_binary_hashes_exact"] is False
