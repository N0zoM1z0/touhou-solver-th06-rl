#!/usr/bin/env python3
"""Fit and native-smoke the frozen Generation-6 actor on development Wine."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.advantage_learning import (  # noqa: E402
    _augment_steps,
    fit_hazard_codebook,
    rich_candidate_vector_from_encoding,
    rich_feature_names,
)
from th06_rl.audited_option_loader import (  # noqa: E402
    AUDITED_OPTION_LOADER_CONTRACT,
    load_audited_option_episode,
)
from th06_rl.hazard_representation import NativeHazardCodebookEncoder  # noqa: E402
from th06_rl.iql_actor_learning import (  # noqa: E402
    IqlActorMember,
    NativeIqlActorPopulation,
    actor_population_choice,
    fit_cross_fitted_iql_actor_population,
    iql_actor_model_artifact,
    iql_actor_model_from_artifact,
    native_actor_prediction_tolerance_ratio,
)
from th06_rl.low_rank_learning import named_feature_roles  # noqa: E402
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.policies.offline_ranker import (  # noqa: E402
    NativePrototypeSupport,
    PortablePrototypeSupport,
)
from th06_rl.qualification_corpus import load_qualification_partition  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402
from th06_rl.sequential_learning import _support  # noqa: E402
from th06_rl.sequential_learning import OrthogonalOption  # noqa: E402
from th06_rl.offline import ACTION_NAMES  # noqa: E402
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--resume-fit", type=Path)
    parser.add_argument(
        "--partition", type=Path,
        default=REPOSITORY / "config/autonomous_generation6_qualification.json",
    )
    parser.add_argument(
        "--registry", type=Path,
        help=(
            "fit every registered training episode with sequential offline-RL "
            "capability instead of the historical development partition"
        ),
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--round-contract-sha256")
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    parser.add_argument(
        "--native-scorer", type=Path,
        default=REPOSITORY / "build/native/libth06_rl_ranker.so",
    )
    args = parser.parse_args(argv)
    if args.round_contract_sha256 is not None and (
        args.registry is None or len(args.round_contract_sha256) != 64
    ):
        parser.error("--round-contract-sha256 requires a registry and one SHA-256")
    if args.output.exists():
        raise FileExistsError(f"refusing to replace candidate: {args.output}")
    if not args.native_scorer.is_file():
        raise FileNotFoundError(f"native actor scorer is absent: {args.native_scorer}")
    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    if args.registry is None:
        _contract, partition = load_qualification_partition(
            args.partition, repository=REPOSITORY
        )
        development = tuple(row for row in partition if row.role == "development")
        training_identity = {
            "kind": "historical-development-partition",
            "path": str(args.partition.resolve()),
            "sha256": _sha256(args.partition),
            "historical_qualification_reused_as_training": False,
        }
    else:
        _registry, entries = load_wine_corpus_registry(
            args.registry, repository=REPOSITORY
        )
        development = select_wine_corpora(
            entries,
            required_capabilities=frozenset({"sequential_offline_rl"}),
        )
        if not development:
            raise ValueError("Generation-6 registry selected no sequential corpus")
        training_identity = {
            "kind": "immutable-capability-registry",
            "path": str(args.registry.resolve()),
            "sha256": _sha256(args.registry),
            "sources": sorted({row.source for row in development}),
            "manifest_sha256": [row.manifest_sha256 for row in development],
            "historical_qualification_reused_as_training": True,
        }
    training_identity_sha256 = hashlib.sha256(json.dumps(
        training_identity, sort_keys=True, separators=(",", ":")
    ).encode()).hexdigest()
    loaded = [
        load_cached_option_episode(
            row.path,
            loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        )
        for row in development
    ]
    raw = [sample for rows, _report, _hit in loaded for sample in rows]
    loaded_at = time.perf_counter()
    layout = named_feature_roles(rich_feature_names())
    if args.resume_fit is not None:
        frozen_fit = json.loads(args.resume_fit.read_text(encoding="utf-8"))
        checkpoint_identity_matches = (
            frozen_fit.get("training_identity_sha256")
            == training_identity_sha256
            or (
                args.registry is None
                and frozen_fit.get("partition_sha256") == _sha256(args.partition)
            )
        )
        if (
            frozen_fit.get("schema") != "autonomous-generation-6-fit-checkpoint-v1"
            or not checkpoint_identity_matches
        ):
            raise ValueError("Generation-6 fit checkpoint is incompatible")
        representation = frozen_fit["representation"]
        support = frozen_fit["support"]
        support_report = frozen_fit["support_report"]
        advantage_crossfit = frozen_fit["advantage_crossfit"]
        actors = [
            IqlActorMember(
                model=iql_actor_model_from_artifact(model),
                bootstrap=bootstrap,
                advantage_scale=float(diagnostic["advantage_rms"]),
                diagnostics=diagnostic,
            )
            for model, bootstrap, diagnostic in zip(
                frozen_fit["actors"],
                frozen_fit["actor_bootstrap"],
                frozen_fit["actor_diagnostics"],
                strict=True,
            )
        ]
        samples = _augment_steps(raw, representation)
    else:
        representation = fit_hazard_codebook(
            raw, seed=500_813 + args.seed_offset
        )
        samples = _augment_steps(raw, representation)
        actors, advantage_crossfit = fit_cross_fitted_iql_actor_population(
            samples,
            layout=layout,
            advantage_folds=3,
            critic_iterations=2,
            n_step_options=8,
            q_trees=8,
            value_trees=8,
            seed=510_813 + args.seed_offset,
            threads=args.threads,
            hidden=64,
            rank=24,
            epochs=8,
            batch_size=1024,
            learning_rate=1e-3,
            log_weight_clip=4.0,
        )
        support, support_report = _support([
            OrthogonalOption(
                step=sample, n_step_target=0.0, outcome_residual=0.0, fold=0
            )
            for sample in samples
        ], seed=540_813 + args.seed_offset)
        checkpoint = {
            "schema": "autonomous-generation-6-fit-checkpoint-v1",
            "evidence_eligible": False,
            "qualification_samples_loaded": False,
            "training_identity": training_identity,
            "training_identity_sha256": training_identity_sha256,
            "representation": representation,
            "support": support,
            "support_report": support_report,
            "actors": [
                iql_actor_model_artifact(actor.model) for actor in actors
            ],
            "actor_diagnostics": [actor.diagnostics for actor in actors],
            "actor_bootstrap": [actor.bootstrap for actor in actors],
            "advantage_crossfit": advantage_crossfit,
        }
        checkpoint_path = args.output.with_name(
            args.output.stem + ".fit.json"
        )
        if checkpoint_path.exists():
            raise FileExistsError(
                f"refusing to replace fit checkpoint: {checkpoint_path}"
            )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, sort_keys=True, allow_nan=False)
            + "\n",
            encoding="utf-8",
        )
    fitted_at = time.perf_counter()

    native_sha = _sha256(args.native_scorer)
    native_actor = NativeIqlActorPopulation(
        args.native_scorer,
        expected_sha256=native_sha,
        models=[actor.model for actor in actors],
    )
    portable_support = PortablePrototypeSupport(
        support, feature_count=len(layout.names)
    )
    native_support = NativePrototypeSupport(
        args.native_scorer,
        expected_sha256=native_sha,
        portable=portable_support,
    )
    native_encoder = NativeHazardCodebookEncoder(
        args.native_scorer,
        expected_sha256=native_sha,
        artifact=representation,
        output_count=len(rich_feature_names()) - len(raw[0].vector)
        - len(raw[0].history_features),
    )
    action_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    conformance = []
    maximum_prediction_error = 0.0
    maximum_prediction_tolerance_ratio = 0.0
    maximum_support_error = 0.0
    exact_choices = 0
    indices = [round(index * (len(raw) - 1) / 63) for index in range(64)]
    for index in indices:
        source = raw[index]
        expected_hazard = tuple(samples[index].candidate_vectors[0][
            len(source.vector):len(source.vector) + native_encoder.output_count
        ])
        actual_hazard = native_encoder.encode(source.hazard_primitives)
        if not all(math.isclose(left, right, rel_tol=2e-5, abs_tol=2e-5)
                   for left, right in zip(expected_hazard, actual_hazard, strict=True)):
            raise RuntimeError("native hazard encoding differs from portable")
        rows = [list(rich_candidate_vector_from_encoding(
            vector, actual_hazard, source.history_features
        )) for vector in source.candidate_vectors]
        expected = [actor.model.predict(rows) for actor in actors]
        actual = native_actor.predict(rows)
        maximum_prediction_error = max(
            maximum_prediction_error,
            max(abs(float(left) - float(right))
                for member_left, member_right in zip(expected, actual, strict=True)
                for left, right in zip(member_left, member_right, strict=True)),
        )
        maximum_prediction_tolerance_ratio = max(
            maximum_prediction_tolerance_ratio,
            native_actor_prediction_tolerance_ratio(expected, actual),
        )
        actions = [action_index[action] for action in source.legal_actions]
        expected_distances = portable_support.distances(rows, actions)
        actual_distances = native_support.distances(rows, actions)
        maximum_support_error = max(
            maximum_support_error,
            max(abs(left - right) for left, right in zip(
                expected_distances, actual_distances, strict=True
            )),
        )
        mask = [distance <= float(support["threshold"])
                for distance in actual_distances]
        expected_choice = actor_population_choice(
            np.asarray(expected).mean(axis=0, keepdims=True),
            source,
            supported=mask,
        )
        actual_choice = actor_population_choice(
            np.asarray(actual).mean(axis=0, keepdims=True),
            source,
            supported=mask,
        )
        exact_choices += expected_choice == actual_choice
        conformance.append({
            "episode_id": source.episode_id,
            "option_id": source.option_id,
            "portable_choice": expected_choice,
            "native_choice": actual_choice,
        })
    if maximum_prediction_tolerance_ratio > 1.0 or maximum_support_error > 2e-5:
        raise RuntimeError(
            "native candidate exceeds equivalence tolerance: "
            f"actor={maximum_prediction_error:.9g}, "
            f"actor_ratio={maximum_prediction_tolerance_ratio:.9g}, "
            f"support={maximum_support_error:.9g}"
        )
    if exact_choices != len(indices):
        raise RuntimeError("native candidate changed a conformance action")

    latencies = []
    for repeat in range(1200):
        source = raw[indices[repeat % len(indices)]]
        before = time.perf_counter_ns()
        hazard = native_encoder.encode(source.hazard_primitives)
        rows = [list(rich_candidate_vector_from_encoding(
            vector, hazard, source.history_features
        )) for vector in source.candidate_vectors]
        actions = [action_index[action] for action in source.legal_actions]
        distances = native_support.distances(rows, actions)
        mask = [distance <= float(support["threshold"]) for distance in distances]
        scores = native_actor.predict(rows)
        actor_population_choice(
            np.asarray(scores).mean(axis=0, keepdims=True),
            source,
            supported=mask,
        )
        latencies.append((time.perf_counter_ns() - before) / 1_000_000.0)
    completed = time.perf_counter()
    p95 = _percentile(latencies, 0.95)
    misses = sum(value > 1000.0 / 60.0 for value in latencies)
    result = {
        "schema": "autonomous-generation-6-candidate-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "autonomous_round_contract_sha256": args.round_contract_sha256,
        "qualification_samples_loaded": bool(
            training_identity["historical_qualification_reused_as_training"]
        ),
        "training_identity": training_identity,
        "training_identity_sha256": training_identity_sha256,
        "development_episode_groups": len(development),
        "development_options": len(samples),
        "parameters": {
            "representation_seed": 500813 + args.seed_offset,
            "actor_seed": 510813 + args.seed_offset,
            "support_seed": 540813 + args.seed_offset,
            "advantage_folds": 3,
            "critic_iterations": 2,
            "n_step_options": 8,
            "q_trees": 8,
            "value_trees": 8,
            "actor_population": 7,
            "actor_hidden": 64,
            "actor_rank": 24,
            "actor_epochs": 8,
            "actor_batch_size": 1024,
            "actor_learning_rate": 1e-3,
            "actor_log_weight_clip": 4.0,
            "intervention_probability_cap": 0.10,
            "intervention_density_ratio_cap": 2.0,
        },
        "selection": {
            "kind": "complete-population-mean-supported-intervention",
            "physical_safety": "native-safe-set-only",
            "bomb": "forbidden",
        },
        "feature_names": list(layout.names),
        "representation": representation,
        "support": support,
        "support_report": support_report,
        "actors": [iql_actor_model_artifact(actor.model) for actor in actors],
        "actor_diagnostics": [actor.diagnostics for actor in actors],
        "actor_bootstrap": [actor.bootstrap for actor in actors],
        "advantage_crossfit": advantage_crossfit,
        "native": {
            "schema": "th06-rl-native-iql-actor-population-v1",
            "sha256": native_sha,
            "conformance_cases": len(conformance),
            "exact_choices": exact_choices,
            "maximum_prediction_error": maximum_prediction_error,
            "maximum_prediction_tolerance_ratio": (
                maximum_prediction_tolerance_ratio
            ),
            "maximum_support_distance_error": maximum_support_error,
            "latency_samples": len(latencies),
            "latency_p50_ms": statistics.median(latencies),
            "latency_p95_ms": p95,
            "latency_max_ms": max(latencies),
            "deadline_ms": 1000.0 / 60.0,
            "deadline_misses": misses,
        },
        "conformance": conformance,
        "resource_contract": affinity.as_dict(),
        "cache_hits": sum(hit for _rows, _report, hit in loaded),
        "timing_seconds": {
            "load": loaded_at - started,
            "fit": fitted_at - loaded_at,
            "native_smoke": completed - fitted_at,
            "total": completed - started,
        },
        "gates": {
            "complete_seven_actor_population": len(actors) == 7,
            "all_advantage_labels_out_of_episode": advantage_crossfit[
                "all_labels_out_of_episode"
            ],
            "support_factual_coverage_at_least_99_percent": (
                support_report["coverage"] >= 0.99
            ),
            "native_prediction_within_float32_scale_bound": (
                maximum_prediction_tolerance_ratio <= 1.0
            ),
            "native_support_error_at_most_2e_5": maximum_support_error <= 2e-5,
            "native_exact_choices": exact_choices == len(conformance),
            "native_latency_p95_below_4_ms": p95 < 4.0,
            "native_zero_60hz_deadline_misses": misses == 0,
        },
    }
    result["passed"] = all(result["gates"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": result["passed"],
        "native": result["native"],
        "timing_seconds": result["timing_seconds"],
    }, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
