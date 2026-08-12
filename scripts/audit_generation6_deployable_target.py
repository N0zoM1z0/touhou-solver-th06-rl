#!/usr/bin/env python3
"""Post-disclosure audit of the implementable Generation-6 intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.advantage_learning import _augment_steps  # noqa: E402
from th06_rl.audited_option_loader import (  # noqa: E402
    AUDITED_OPTION_LOADER_CONTRACT,
    load_audited_option_episode,
)
from th06_rl.iql_actor_learning import (  # noqa: E402
    IqlActorMember,
    evaluate_iql_actor_fold,
    iql_actor_model_from_artifact,
    summarize_iql_actor_episodes,
)
from th06_rl.low_rank_learning import named_feature_roles  # noqa: E402
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.qualification_corpus import load_qualification_partition  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402


CANDIDATE_SHA256 = "aea789ed9fe63aa4a2c0799092675fd287c9b66787ed968d82e82098fbb4ea64"
QUALIFICATION_SHA256 = "1da0212281902daf18c124d3e246a244ae19d4a92fa3177efd34711c460b3e34"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", default=32, type=int)
    parser.add_argument(
        "--candidate", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-candidate/candidate-v1.json",
    )
    parser.add_argument(
        "--qualification-result", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-qualification/qualification-v1.json",
    )
    parser.add_argument(
        "--partition", type=Path,
        default=REPOSITORY / "config/autonomous_generation6_qualification.json",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace audit: {args.output}")
    if _sha256(args.candidate) != CANDIDATE_SHA256:
        raise ValueError("deployable-target candidate drifted")
    if _sha256(args.qualification_result) != QUALIFICATION_SHA256:
        raise ValueError("consumed qualification result drifted")
    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    _contract, partition = load_qualification_partition(
        args.partition, repository=REPOSITORY
    )
    development_specs = tuple(row for row in partition if row.role == "development")
    disclosed_specs = tuple(row for row in partition if row.role == "qualification")
    development_loaded = [
        load_cached_option_episode(
            row.path, loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        ) for row in development_specs
    ]
    disclosed_loaded = [
        load_cached_option_episode(
            row.path, loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        ) for row in disclosed_specs
    ]
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    actors = [
        IqlActorMember(
            model=iql_actor_model_from_artifact(model),
            bootstrap=bootstrap,
            advantage_scale=float(diagnostic["advantage_rms"]),
            diagnostics=diagnostic,
        )
        for model, bootstrap, diagnostic in zip(
            candidate["actors"], candidate["actor_bootstrap"],
            candidate["actor_diagnostics"], strict=True,
        )
    ]
    development = _augment_steps([
        sample for rows, _report, _hit in development_loaded for sample in rows
    ], candidate["representation"])
    disclosed = _augment_steps([
        sample for rows, _report, _hit in disclosed_loaded for sample in rows
    ], candidate["representation"])
    cohorts = {
        rows[0].episode_id: f"stage-{spec.stage}"
        for spec, (rows, _report, _hit) in zip(
            disclosed_specs, disclosed_loaded, strict=True
        )
    }
    loaded_at = time.perf_counter()
    fold = evaluate_iql_actor_fold(
        development, disclosed,
        layout=named_feature_roles(tuple(candidate["feature_names"])),
        episode_cohorts=cohorts,
        fold=0,
        critic_iterations=2,
        n_step_options=8,
        q_trees=8,
        value_trees=8,
        seed=460_813,
        threads=args.threads,
        actor_hidden=64,
        actor_rank=24,
        actor_epochs=8,
        actor_batch_size=1024,
        actor_learning_rate=1e-3,
        actor_log_weight_clip=4.0,
        intervention_probability_cap=0.10,
        intervention_density_ratio_cap=2.0,
        intervention_minimum_uniform_mass=0.10,
        frozen_actors=actors,
        frozen_support=candidate["support"],
    )
    cohorts_report = summarize_iql_actor_episodes(
        fold["episodes"], cohort_names=("stage-4", "stage-6")
    )
    completed = time.perf_counter()
    gates = {}
    for name, row in cohorts_report.items():
        gates[f"{name}-full-upper-negative"] = (
            row["policy_dr_hit_effect_bootstrap_upper_95"] < 0.0
        )
        gates[f"{name}-worst-loo-upper-negative"] = (
            row["policy_loo_worst_bootstrap_upper_95"] < 0.0
        )
        gates[f"{name}-model-negative"] = (
            row["policy_model_hit_effect_mean"] < 0.0
        )
        gates[f"{name}-majority-beneficial"] = (
            row["policy_dr_beneficial_episode_rate"] >= 0.5
        )
        gates[f"{name}-intervention-exercised"] = (
            row["policy_intervention_exposure_rate"] > 0.0
        )
        gates[f"{name}-correction-bounded"] = (
            row["policy_max_abs_correction"] <= 2.0 + 1e-9
        )
    result = {
        "schema": "autonomous-generation-6-deployable-target-audit-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "qualification_status": "permanently-disclosed-development-only",
        "candidate_sha256": CANDIDATE_SHA256,
        "qualification_result_sha256": QUALIFICATION_SHA256,
        "target": {
            "intervention_probability": "min(0.10, 2 * 0.10 / safe_set_size)",
            "minimum_uniform_mass": 0.10,
            "maximum_density_ratio": 2.0,
            "online_reconstructible": True,
        },
        "cohorts": cohorts_report,
        "fold": fold,
        "resource_contract": affinity.as_dict(),
        "timing_seconds": {
            "load_and_augment": loaded_at - started,
            "evaluation": completed - loaded_at,
            "total": completed - started,
        },
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "passed": result["passed"],
        "cohorts": cohorts_report,
        "failed_gates": sorted(key for key, value in gates.items() if not value),
    }, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
