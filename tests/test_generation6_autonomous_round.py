from copy import deepcopy
import os

import pytest

from scripts.run_generation6_autonomous_round import (
    _paired_verdict,
    _validate_contract_shape,
)


def _row(block: int, role: str, hits: int, interventions: int = 0):
    return {
        "block": block,
        "role": role,
        "physical_hits": hits,
        "interventions": interventions,
        "passed": True,
    }


def test_paired_round_requires_aggregate_block_and_exposure_gates() -> None:
    rows = []
    for block, (incumbent, candidate) in enumerate(
        ((10, 8), (9, 8), (11, 7), (7, 8), (10, 9), (8, 8))
    ):
        rows.extend((
            _row(block, "incumbent", incumbent),
            _row(block, "candidate", candidate, 1),
        ))
    result = _paired_verdict(rows)
    assert result["verdict"] == "effective-learning-signal"
    assert result["effect_hits"] == 7
    assert result["candidate_no_worse_blocks"] == 5
    assert result["promotion_eligible"] is False


def test_paired_round_rejects_positive_total_without_block_consistency() -> None:
    rows = []
    for block, (incumbent, candidate) in enumerate(
        ((20, 1), (1, 2), (1, 2), (1, 2), (1, 2), (1, 2))
    ):
        rows.extend((
            _row(block, "incumbent", incumbent),
            _row(block, "candidate", candidate, 1),
        ))
    result = _paired_verdict(rows)
    assert result["effect_hits"] > 0
    assert result["verdict"] == "no-effective-learning-signal"


def _contract():
    cpus = sorted(os.sched_getaffinity(0))[:4]
    return {
        "maximum_interventions": 64,
        "retail_executable_sha256": "e" * 64,
        "frozen_inputs": {"input": "f" * 64},
        "collection": {
            "episodes": 12,
            "stages": [4, 5, 6],
            "schedule": [
                {"episode": index + 1, "stage": (4, 5, 6)[index % 3],
                 "policy_seed": 1_000 + index}
                for index in range(12)
            ],
        },
        "canary": {"schedule": [
            {"trial": index + 1, "stage": 4, "policy_seed": 2_000 + index}
            for index in range(2)
        ]},
        "evaluation": {"schedule": [
            {"trial": block * 2 + role_index + 1, "block": block,
             "role": role, "policy_seed": 3_000 + block * 2 + role_index}
            for block in range(6)
            for role_index, role in enumerate(
                ("incumbent", "candidate")
                if block % 2 == 0 else ("candidate", "incumbent")
            )
        ]},
        "environment": {
            "game_cpu_list": ",".join(map(str, cpus[:2])),
            "controller_cpu_list": ",".join(map(str, cpus[2:])),
            "source_game_inventory_sha256": "g" * 64,
        },
        "offline": {
            "crossfit_seed": 4_000,
            "seed_offset": 5_000,
            "preflight_policy_seed": 6_000,
        },
        "registry_append": {
            "source_id": "round",
            "base_source_count": 5,
            "capabilities": [
                "complete_stage_observation", "physical_hit_outcome",
                "representation_pretraining", "behavior_state_value",
                "factual_semi_markov_options",
                "recorded_complete_behavior_propensity",
                "native_safe_candidates", "sequential_offline_rl",
                "natural_rng", "generation6_actor_ess_behavior",
            ],
        },
    }


def test_round_contract_requires_balanced_outcome_blind_collection() -> None:
    contract = _contract()
    _validate_contract_shape(contract)
    drifted = deepcopy(contract)
    drifted["collection"]["schedule"][0]["rng_seed"] = 99
    with pytest.raises(ValueError, match="balanced/frozen"):
        _validate_contract_shape(drifted)


def test_paired_round_reports_invalid_shape_without_partial_verdict() -> None:
    result = _paired_verdict([_row(0, "candidate", 1, 1)])
    assert result["verdict"] == "invalid"
    assert result["incumbent_hits"] == []
    assert result["candidate_hits"] == []
