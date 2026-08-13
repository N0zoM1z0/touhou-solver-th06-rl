from scripts.audit_generation6_round_offline import offline_gates


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
