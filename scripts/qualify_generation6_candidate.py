#!/usr/bin/env python3
"""Single-disclosure offline qualification for the frozen Generation-6 actor."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
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


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace qualification: {args.output}")
    freeze = _object(args.freeze)
    if (
        freeze.get("schema") != "autonomous-generation-6-qualification-freeze-v1"
        or freeze.get("qualification_disclosure_count") != 1
        or freeze.get("authorization_eligible") is not False
    ):
        raise ValueError("Generation-6 qualification freeze is invalid")
    for raw_path, expected in freeze["source_sha256"].items():
        path = (REPOSITORY / raw_path).resolve()
        if _sha256(path) != expected:
            raise ValueError(f"qualification source drifted: {raw_path}")
    commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY, text=True
    ).strip()
    if commit != freeze["execution_commit"]:
        raise ValueError("qualification execution commit drifted")
    candidate_path = (REPOSITORY / freeze["candidate"]["path"]).resolve()
    partition_path = (REPOSITORY / freeze["partition"]["path"]).resolve()
    if _sha256(candidate_path) != freeze["candidate"]["sha256"]:
        raise ValueError("frozen candidate artifact drifted")
    if _sha256(partition_path) != freeze["partition"]["sha256"]:
        raise ValueError("frozen qualification partition drifted")

    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    _partition_contract, partition = load_qualification_partition(
        partition_path, repository=REPOSITORY
    )
    development_specs = tuple(
        row for row in partition if row.role == "development"
    )
    qualification_specs = tuple(
        row for row in partition if row.role == "qualification"
    )
    # This is the single deliberate sample disclosure. No qualification path
    # is loaded before every source/artifact/gate hash above has passed.
    development_loaded = [
        load_cached_option_episode(
            row.path,
            loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        )
        for row in development_specs
    ]
    qualification_loaded = [
        load_cached_option_episode(
            row.path,
            loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        )
        for row in qualification_specs
    ]
    candidate = _object(candidate_path)
    if candidate.get("qualification_samples_loaded") is not False:
        raise ValueError("candidate provenance already used qualification")
    actors = [
        IqlActorMember(
            model=iql_actor_model_from_artifact(model),
            bootstrap=bootstrap,
            advantage_scale=float(diagnostic["advantage_rms"]),
            diagnostics=diagnostic,
        )
        for model, bootstrap, diagnostic in zip(
            candidate["actors"],
            candidate["actor_bootstrap"],
            candidate["actor_diagnostics"],
            strict=True,
        )
    ]
    development_raw = [
        sample for rows, _report, _hit in development_loaded for sample in rows
    ]
    qualification_raw = [
        sample for rows, _report, _hit in qualification_loaded for sample in rows
    ]
    development = _augment_steps(development_raw, candidate["representation"])
    qualification = _augment_steps(
        qualification_raw, candidate["representation"]
    )
    cohorts = {
        rows[0].episode_id: f"stage-{spec.stage}"
        for spec, (rows, _report, _hit) in zip(
            qualification_specs, qualification_loaded, strict=True
        )
    }
    loaded_at = time.perf_counter()
    fold = evaluate_iql_actor_fold(
        development,
        qualification,
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
        frozen_actors=actors,
        frozen_support=candidate["support"],
    )
    cohorts_report = summarize_iql_actor_episodes(
        fold["episodes"], cohort_names=("stage-4", "stage-6")
    )
    completed = time.perf_counter()
    numerical = freeze["numerical_gates"]
    gates: dict[str, bool] = {
        "exact_qualification_episode_groups": (
            len(qualification_specs) == numerical["episode_groups"]
        ),
        "exact_stage4_episode_groups": (
            cohorts_report["stage-4"]["episode_groups"]
            == numerical["stage-4_episode_groups"]
        ),
        "exact_stage6_episode_groups": (
            cohorts_report["stage-6"]["episode_groups"]
            == numerical["stage-6_episode_groups"]
        ),
    }
    for name, row in cohorts_report.items():
        prefix = name.replace("-", "_")
        gates[f"{prefix}_policy_bootstrap_upper_below_zero"] = (
            row["policy_dr_hit_effect_bootstrap_upper_95"] < 0.0
        )
        gates[f"{prefix}_worst_loo_bootstrap_upper_below_zero"] = (
            row["policy_loo_worst_bootstrap_upper_95"] < 0.0
        )
        gates[f"{prefix}_model_effect_below_zero"] = (
            row["policy_model_hit_effect_mean"] < 0.0
        )
        gates[f"{prefix}_majority_beneficial_episodes"] = (
            row["policy_dr_beneficial_episode_rate"]
            >= numerical["minimum_beneficial_episode_rate"]
        )
        gates[f"{prefix}_proposal_exercised"] = (
            row["mean_population_proposal_rate"] > 0.0
        )
        gates[f"{prefix}_proposal_below_cap"] = (
            row["mean_population_proposal_rate"]
            <= numerical["maximum_proposal_rate"]
        )
        gates[f"{prefix}_intervention_exercised"] = (
            row["policy_intervention_exposure_rate"] > 0.0
        )
        gates[f"{prefix}_intervention_below_cap"] = (
            row["policy_intervention_exposure_rate"]
            <= numerical["maximum_intervention_exposure_rate"]
        )
        gates[f"{prefix}_bounded_correction"] = (
            row["policy_max_abs_correction"]
            <= numerical["maximum_density_ratio"] + 1e-9
        )
    result = {
        "schema": "autonomous-generation-6-qualification-result-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "qualification_disclosed": True,
        "freeze_sha256": _sha256(args.freeze),
        "candidate_sha256": _sha256(candidate_path),
        "partition_sha256": _sha256(partition_path),
        "execution_commit": commit,
        "development_episode_groups": len(development_specs),
        "qualification_episode_groups": len(qualification_specs),
        "development_options": len(development),
        "qualification_options": len(qualification),
        "resource_contract": affinity.as_dict(),
        "cache_hits": {
            "development": sum(hit for _rows, _report, hit in development_loaded),
            "qualification": sum(hit for _rows, _report, hit in qualification_loaded),
        },
        "timing_seconds": {
            "load_and_augment": loaded_at - started,
            "evaluation": completed - loaded_at,
            "total": completed - started,
        },
        "cohorts": cohorts_report,
        "fold": fold,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": result["passed"],
        "cohorts": cohorts_report,
        "failed_gates": sorted(name for name, passed in gates.items() if not passed),
    }, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
