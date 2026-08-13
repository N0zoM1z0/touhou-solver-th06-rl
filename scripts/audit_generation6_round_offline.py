#!/usr/bin/env python3
"""Conjunctive synthetic, cross-fit, fit, and native G6 round smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def offline_gates(
    *, synthetic: dict[str, object], crossfit: dict[str, object],
    candidate: dict[str, object], online: dict[str, object],
    candidate_sha256: str, registry_sha256: str,
    required_cohorts: tuple[str, ...],
) -> dict[str, bool]:
    report = crossfit.get("report")
    cohorts = report.get("cohorts") if isinstance(report, dict) else None
    overall = cohorts.get("overall") if isinstance(cohorts, dict) else None
    identity = crossfit.get("input_identity")
    training = candidate.get("training_identity")
    gates = {
        "synthetic_causal_and_null_passed": synthetic.get("passed") is True,
        "crossfit_uses_exact_registry": (
            isinstance(identity, dict)
            and identity.get("sha256") == registry_sha256
        ),
        "candidate_uses_exact_registry": (
            isinstance(training, dict)
            and training.get("sha256") == registry_sha256
        ),
        "all_required_cohorts_present": (
            isinstance(cohorts, dict)
            and set(required_cohorts) <= set(cohorts)
        ),
        "overall_policy_bootstrap_upper_below_zero": (
            isinstance(overall, dict)
            and float(overall.get(
                "policy_dr_hit_effect_bootstrap_upper_95", float("inf")
            )) < 0.0
        ),
        "overall_worst_loo_upper_below_zero": (
            isinstance(overall, dict)
            and float(overall.get(
                "policy_loo_worst_bootstrap_upper_95", float("inf")
            )) < 0.0
        ),
        "candidate_fit_and_native_smoke_passed": candidate.get("passed") is True,
        "full_linux_and_wine_policy_preflight_passed": (
            online.get("passed") is True
            and online.get("candidate_sha256") == candidate_sha256
        ),
    }
    if isinstance(cohorts, dict):
        for name in required_cohorts:
            row = cohorts.get(name)
            prefix = name.replace("-", "_")
            gates[f"{prefix}_model_effect_below_zero"] = (
                isinstance(row, dict)
                and float(row.get("policy_model_hit_effect_mean", 0.0)) < 0.0
            )
            gates[f"{prefix}_majority_beneficial_episodes"] = (
                isinstance(row, dict)
                and float(row.get("policy_dr_beneficial_episode_rate", 0.0))
                >= 0.5
            )
            gates[f"{prefix}_proposal_exercised_and_not_constant"] = (
                isinstance(row, dict)
                and 0.0 < float(row.get("mean_population_proposal_rate", 0.0))
                < 1.0
            )
            gates[f"{prefix}_intervention_exercised_below_cap"] = (
                isinstance(row, dict)
                and 0.0 < float(row.get(
                    "policy_intervention_exposure_rate", 0.0
                )) <= 0.10
            )
            gates[f"{prefix}_bounded_density_correction"] = (
                isinstance(row, dict)
                and float(row.get("policy_max_abs_correction", float("inf")))
                <= 2.0 + 1e-9
            )
    return gates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic", type=Path, required=True)
    parser.add_argument("--crossfit", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = _object(args.contract)
    if contract.get("schema") != "autonomous-generation-6-round-contract-v1":
        raise ValueError("Generation-6 autonomous round contract is invalid")
    required = tuple(
        f"stage-{int(stage)}" for stage in contract["collection"]["stages"]
    )
    synthetic = _object(args.synthetic)
    crossfit = _object(args.crossfit)
    candidate = _object(args.candidate)
    online = _object(args.online)
    registry_sha = _sha256(args.registry)
    gates = offline_gates(
        synthetic=synthetic, crossfit=crossfit, candidate=candidate,
        online=online,
        candidate_sha256=_sha256(args.candidate),
        registry_sha256=registry_sha, required_cohorts=required,
    )
    result = {
        "schema": "autonomous-generation-6-round-offline-smoke-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "contract_sha256": _sha256(args.contract),
        "training_registry_sha256": registry_sha,
        "synthetic_sha256": _sha256(args.synthetic),
        "crossfit_sha256": _sha256(args.crossfit),
        "candidate_sha256": _sha256(args.candidate),
        "online_preflight_sha256": _sha256(args.online),
        "required_cohorts": list(required),
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    print(json.dumps({
        "output": str(args.output), "passed": result["passed"],
        "failed_gates": [name for name, value in gates.items() if not value],
    }, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
