#!/usr/bin/env python3
"""Fit and cross-evaluate Generation-7 orthogonal one-step improvement."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.generation7.offline_dataset import prepare_episode_arrays  # noqa: E402
from th06_rl.generation7.orthogonal_learning import (  # noqa: E402
    OrthogonalConfig,
    crossfit_orthogonal_policy,
    load_compact_episodes,
    orthogonal_randomization_nulls,
)
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _calibrated(
    left: dict[str, object],
    right: dict[str, object],
    *,
    standard_errors: float,
) -> bool:
    combined_standard_error = math.sqrt(
        float(left["episode_cluster_standard_error"]) ** 2
        + float(right["episode_cluster_standard_error"]) ** 2
    )
    return abs(
        float(left["episode_equal_mean"]) - float(right["episode_equal_mean"])
    ) <= standard_errors * combined_standard_error


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY / "config/wine_corpus_registry.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "config/generation7_offline_contract.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY / "artifacts/cache/generation7-factual-arrays",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "artifacts/generation7-offline/orthogonal.json",
    )
    parser.add_argument("--null-replicates", type=int)
    parser.add_argument(
        "--effect-representation",
        choices=("action_only", "compact_bilinear", "richer_bilinear"),
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
    os.environ.setdefault("MKL_NUM_THREADS", "16")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    null_replicates = (
        int(args.null_replicates)
        if args.null_replicates is not None
        else int(contract["orthogonal_null_replicates"])
    )
    if (
        contract.get("schema") != "generation7-offline-contract-v1"
        or contract.get("wine_outcome_facing_authorized") is not False
        or contract.get("new_collection_authorized") is not False
    ):
        raise ValueError("Generation-7 offline boundary drifted")
    _registry, all_entries = load_wine_corpus_registry(
        args.registry, repository=REPOSITORY
    )
    entries = select_wine_corpora(
        all_entries,
        required_capabilities=frozenset(contract["required_corpus_capabilities"]),
    )
    paths = tuple(
        prepare_episode_arrays(
            entry,
            repository=REPOSITORY,
            cache_root=args.cache_root,
        )[0]
        for entry in entries
    )
    values = contract["orthogonal_direct"]
    reference = contract["reference_policy"]
    support = contract["statistical_support"]
    fqe = contract["fqe"]
    config = OrthogonalConfig(
        folds=int(contract["crossfit_folds"]),
        fold_seed=int(contract["seeds"]["fold"]),
        horizon=int(values["proximal_horizon_options"]),
        nuisance_ridge_alpha=float(values["nuisance_ridge_alpha"]),
        effect_ridge_alpha=float(values["effect_ridge_alpha"]),
        reference_epsilon=float(reference["epsilon"]),
        policy_temperature=float(reference["temperature"]),
        maximum_log_tilt=float(reference["maximum_log_tilt"]),
        minimum_action_assignments=int(
            support["minimum_factual_assignments_per_action"]
        ),
        minimum_action_episodes=int(
            support["minimum_episode_groups_per_action"]
        ),
        fqe_horizon=int(fqe["horizon_options"]),
        fqe_ridge_alpha=float(fqe["ridge_alpha"]),
        effect_representation=(
            args.effect_representation or str(values["effect_representation"])
        ),
    )
    episodes = load_compact_episodes(paths, horizon=config.horizon)
    crossfit, diagnostics = crossfit_orthogonal_policy(episodes, config=config)
    nulls = orthogonal_randomization_nulls(
        episodes,
        diagnostics,
        replicates=null_replicates,
        seed=int(contract["seeds"]["null"]),
    )
    estimates = crossfit["estimates"]
    ope_gates = contract["ope_gates"]
    calibration_standard_errors = float(
        ope_gates["calibration_standard_errors"]
    )
    sequential_weights = tuple(
        values
        for fold in crossfit["fold_reports"]
        for values in fold["sequential_dr_cumulative_weights"].values()
    )
    aggregate_direction = math.copysign(
        1.0, float(estimates["one_step_direct"]["episode_equal_mean"])
    )
    stratum_directions = {
        name: math.copysign(1.0, float(summary["episode_equal_mean"]))
        for name, summary in crossfit["strata"]["one_step_direct"].items()
    }
    action_direction_report = {}
    for action, values_ in crossfit["action_specific_one_step_direct"].items():
        aggregate_action_direction = math.copysign(
            1.0, float(values_["aggregate"]["episode_equal_mean"])
        )
        eligible_strata = {
            name: summary
            for name, summary in values_["strata"].items()
            if int(summary["episodes"])
            >= int(support["minimum_episode_groups_per_action"])
        }
        action_direction_report[action] = {
            "aggregate_direction": aggregate_action_direction,
            "eligible_strata": len(eligible_strata),
            "consistent_eligible_strata": sum(
                math.copysign(1.0, float(summary["episode_equal_mean"]))
                == aggregate_action_direction
                for summary in eligible_strata.values()
            ),
            "all_eligible_strata_consistent": all(
                math.copysign(1.0, float(summary["episode_equal_mean"]))
                == aggregate_action_direction
                for summary in eligible_strata.values()
            ),
        }
    direction_values = tuple(
        float(estimates[name]["episode_equal_mean"])
        for name in (
            "one_step_direct",
            "one_step_ips",
            "one_step_dr",
            "one_step_fqe",
        )
    )
    direction_agreement = all(value < 0.0 for value in direction_values) or all(
        value > 0.0 for value in direction_values
    )
    gates = {
        "predeclared_null_replicate_count": (
            int(nulls["replicates"])
            >= int(contract["orthogonal_null_replicates"])
        ),
        "factual_action_randomization_null": (
            float(nulls["action_randomization"]["null_p_value"]) <= 0.05
        ),
        "reward_suffix_permutation_null": (
            float(nulls["reward_suffix"]["null_p_value"]) <= 0.05
        ),
        "cross_source_stage_policy_direction": all(
            direction == aggregate_direction
            for direction in stratum_directions.values()
        ),
        "action_specific_cross_source_stage_report_complete": (
            len(action_direction_report) == 18
            and all(row["eligible_strata"] > 0 for row in action_direction_report.values())
        ),
        "bounded_heldout_effect_predictions": max(
            float(fold["maximum_absolute_heldout_effect_prediction"])
            for fold in crossfit["fold_reports"]
        ) <= float(ope_gates["maximum_absolute_effect_prediction"]),
        "one_step_direction_agreement": direction_agreement,
        "one_step_direct_dr_calibration": _calibrated(
            estimates["one_step_direct"],
            estimates["one_step_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "one_step_ips_dr_calibration": _calibrated(
            estimates["one_step_ips"],
            estimates["one_step_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "one_step_fqe_dr_calibration": _calibrated(
            estimates["one_step_fqe"],
            estimates["one_step_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "one_step_dr_episode_sign_stability": (
            abs(float(estimates["one_step_dr"]["episode_equal_mean"]))
            >= 2.0
            * float(estimates["one_step_dr"]["episode_cluster_standard_error"])
        ),
        "exact_sequential_fqe_dr_direction_agreement": (
            math.copysign(
                1.0, float(estimates["sequential_fqe"]["episode_equal_mean"])
            )
            == math.copysign(
                1.0, float(estimates["sequential_dr"]["episode_equal_mean"])
            )
        ),
        "exact_sequential_fqe_dr_calibration": _calibrated(
            estimates["sequential_fqe"],
            estimates["sequential_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "exact_sequential_dr_support": (
            min(float(row["effective_sample_size"]) for row in sequential_weights)
            >= float(ope_gates["minimum_sequential_dr_effective_sample_size"])
            and max(float(row["maximum"]) for row in sequential_weights)
            <= float(ope_gates["maximum_sequential_dr_cumulative_weight"])
        ),
    }
    report = {
        "schema": "generation7-orthogonal-fit-report-v1",
        "evidence_eligible": False,
        "wine_outcome_facing_authorized": False,
        "registry_sha256": _sha256(args.registry),
        "contract_sha256": _sha256(args.contract),
        "configuration": config.__dict__,
        "crossfit": crossfit,
        "orthogonal_nulls": nulls,
        "direction_diagnostics": {
            "one_step_policy_stratum_directions": stratum_directions,
            "action_specific": action_direction_report,
        },
        "gates": gates,
        "passes_current_layer": all(gates.values()),
        "next_required_layer": (
            "proper-awr-and-iql-challengers"
            if all(gates.values())
            else "resolve-failed-identifiability-or-ope-calibration-gates"
        ),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if all(gates.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
