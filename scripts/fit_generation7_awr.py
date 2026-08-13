#!/usr/bin/env python3
"""Cross-fit the bounded proper AWR challenger on Generation-7 arrays."""

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

from th06_rl.generation7.awr_actor import AwrConfig, crossfit_proper_awr  # noqa: E402
from th06_rl.generation7.offline_dataset import (  # noqa: E402
    load_episode_arrays,
    prepare_episode_arrays,
)
from th06_rl.generation7.orthogonal_learning import (  # noqa: E402
    OrthogonalConfig,
    load_compact_episodes,
)
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paired_difference_calibrated(
    difference: dict[str, object],
    *,
    standard_errors: float,
) -> bool:
    return abs(float(difference["episode_equal_mean"])) <= (
        standard_errors
        * float(difference["episode_cluster_standard_error"])
    )


def _unit_mean_calibrated(
    summary: dict[str, object],
    *,
    expected: float,
    standard_errors: float,
) -> bool:
    return abs(float(summary["episode_equal_mean"]) - expected) <= (
        standard_errors * float(summary["episode_cluster_standard_error"])
    )


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
        default=REPOSITORY / "artifacts/generation7-offline/proper-awr.json",
    )
    parser.add_argument(
        "--effect-representation",
        choices=("action_only", "compact_bilinear", "richer_bilinear"),
    )
    args = parser.parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "16")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "16")
    os.environ.setdefault("MKL_NUM_THREADS", "16")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        contract.get("wine_outcome_facing_authorized") is not False
        or contract.get("treatment_unit")
        != "randomized-proposal-assignment-intention-to-treat"
        or contract.get("post_assignment_native_revalidation")
        != "factual-deployment-kernel-not-a-filter"
    ):
        raise ValueError("proper AWR fit cannot authorize Wine")
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
    orthogonal = OrthogonalConfig(
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
    actor_values = contract["actor"]
    awr = AwrConfig(
        temperature=float(actor_values["awr_temperature"]),
        maximum_weight=float(actor_values["maximum_weight"]),
        kl_coefficient=float(actor_values["kl_coefficient"]),
        epochs=int(values["actor_epochs"]),
        learning_rate=float(values["actor_learning_rate"]),
        l2=float(values["actor_l2"]),
    )
    episodes = load_compact_episodes(paths, horizon=orthogonal.horizon)
    feature_names = tuple(map(str, load_episode_arrays(paths[0])["feature_names"]))
    crossfit = crossfit_proper_awr(
        episodes,
        feature_names=feature_names,
        orthogonal_config=orthogonal,
        awr_config=awr,
    )
    ope_gates = contract["ope_gates"]
    calibration_standard_errors = float(
        ope_gates["calibration_standard_errors"]
    )
    sequential_weights = tuple(
        values
        for fold in crossfit["fold_reports"]
        for values in fold["sequential_dr_cumulative_weights"].values()
    )
    calibration = crossfit["paired_calibration_differences"]
    propensity_calibration = crossfit["proposal_propensity_calibration"]
    signs = tuple(
        float(crossfit["estimates"][name]["episode_equal_mean"])
        for name in (
            "one_step_direct",
            "one_step_ips",
            "one_step_dr",
            "one_step_fqe",
        )
    )
    gates = {
        "proposal_propensity_calibration": (
            _unit_mean_calibrated(
                propensity_calibration["aggregate"],
                expected=1.0,
                standard_errors=calibration_standard_errors,
            )
            and all(
                _unit_mean_calibrated(
                    summary,
                    expected=1.0,
                    standard_errors=calibration_standard_errors,
                )
                for summary in propensity_calibration["strata"].values()
            )
        ),
        "fixed_physical_time_or_semi_markov_value": False,
        "objective_finite_nonnegative": all(
            all(
                row["objective"] >= 0.0
                for row in fold["actor_fit"]["epochs"]
            )
            for fold in crossfit["fold_reports"]
        ),
        "objective_decreased_in_every_fold": all(
            fold["actor_fit"]["epochs"][-1]["objective"]
            < fold["actor_fit"]["epochs"][0]["objective"]
            for fold in crossfit["fold_reports"]
        ),
        "probability_conservation": all(
            max(
                row["maximum_probability_sum_error"]
                for row in fold["actor_fit"]["epochs"]
            ) < 1e-12
            for fold in crossfit["fold_reports"]
        ),
        "one_step_direction_agreement": (
            all(value < 0.0 for value in signs)
            or all(value > 0.0 for value in signs)
        ),
        "one_step_direct_dr_calibration": _paired_difference_calibrated(
            calibration["one_step_direct_minus_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "one_step_ips_dr_calibration": _paired_difference_calibrated(
            calibration["one_step_ips_minus_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "one_step_fqe_dr_calibration": _paired_difference_calibrated(
            calibration["one_step_fqe_minus_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "exact_sequential_fqe_dr_direction_agreement": (
            math.copysign(
                1.0,
                float(crossfit["estimates"]["sequential_fqe"]["episode_equal_mean"]),
            )
            == math.copysign(
                1.0,
                float(crossfit["estimates"]["sequential_dr"]["episode_equal_mean"]),
            )
        ),
        "exact_sequential_fqe_dr_calibration": _paired_difference_calibrated(
            calibration["sequential_fqe_minus_dr"],
            standard_errors=calibration_standard_errors,
        ),
        "exact_sequential_dr_support": (
            min(float(row["effective_sample_size"]) for row in sequential_weights)
            >= float(ope_gates["minimum_sequential_dr_effective_sample_size"])
            and max(float(row["maximum"]) for row in sequential_weights)
            <= float(ope_gates["maximum_sequential_dr_cumulative_weight"])
        ),
        "heldout_one_step_dr_sign_stability": (
            abs(float(crossfit["estimates"]["one_step_dr"]["episode_equal_mean"]))
            >= 2.0
            * float(
                crossfit["estimates"]["one_step_dr"][
                    "episode_cluster_standard_error"
                ]
            )
        ),
    }
    report = {
        "schema": "generation7-proper-awr-fit-report-v2",
        "evidence_eligible": False,
        "wine_outcome_facing_authorized": False,
        "registry_sha256": _sha256(args.registry),
        "contract_sha256": _sha256(args.contract),
        "orthogonal_configuration": orthogonal.__dict__,
        "awr_configuration": awr.__dict__,
        "crossfit": crossfit,
        "gates": gates,
        "passes": all(gates.values()),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if report["passes"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
