#!/usr/bin/env python3
"""Train CPU fitted-Q iteration on gap-safe physical TH06 trajectories."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import time

from th06_rl.offline import iter_run_transitions, load_dataset_index
from th06_rl.offline_learning import LabeledTransition, features_for_candidate, regression_metrics
from th06_rl.offline_rl import SequentialTransition, sequential_transitions

# Reuse the exact encoder/model serialization contract of the factual policy
# zoo.  Executing this file from scripts/ puts that directory on sys.path.
from train_offline_cpu import Encoder, _fit, _model, _predict  # noqa: E402


@dataclass(frozen=True)
class NStepSample:
    state: LabeledTransition
    observed_return: float
    next_state: LabeledTransition | None
    bootstrap_discount: float
    contains_hit: bool


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _n_step(
    transitions: list[SequentialTransition],
    *,
    horizon: int,
    gamma: float,
) -> list[NStepSample]:
    by_sequence = {row.state.sequence: index for index, row in enumerate(transitions)}
    samples = []
    for start in range(len(transitions)):
        total = 0.0
        discount = 1.0
        cursor = start
        next_state = None
        contains_hit = False
        for _ in range(horizon):
            row = transitions[cursor]
            total += discount * row.reward
            contains_hit = contains_hit or row.terminal_reason == "physical-hit"
            discount *= gamma ** row.elapsed_frames
            if row.next_state is None:
                discount = 0.0
                next_state = None
                break
            next_state = row.next_state
            next_index = by_sequence.get(next_state.sequence)
            if next_index is None:
                discount = 0.0
                next_state = None
                break
            cursor = next_index
        samples.append(NStepSample(
            state=transitions[start].state,
            observed_return=total,
            next_state=next_state,
            bootstrap_discount=discount,
            contains_hit=contains_hit,
        ))
    return samples


def _future_hit_labels(transitions: list[SequentialTransition], horizon: int = 120) -> dict[int, bool]:
    labels: dict[int, bool] = {}
    for start, row in enumerate(transitions):
        hit = False
        for candidate in transitions[start:]:
            if candidate.state.frame - row.state.frame > horizon:
                break
            if candidate.terminal_reason == "physical-hit":
                hit = True
                break
            if candidate.terminal:
                break
        labels[row.state.sequence] = hit
    return labels


def _sample_training(
    samples: list[NStepSample],
    *,
    limit: int,
    seed: int,
) -> tuple[list[NStepSample], list[float]]:
    if len(samples) <= limit:
        return samples, [1.0] * len(samples)
    rng = random.Random(seed)
    important = [row for row in samples if row.contains_hit or row.observed_return < 0.0]
    ordinary = [row for row in samples if not (row.contains_hit or row.observed_return < 0.0)]
    if len(important) >= limit:
        selected = rng.sample(important, limit)
        return selected, [len(important) / limit] * limit
    remaining = limit - len(important)
    selected_ordinary = rng.sample(ordinary, remaining)
    return (
        important + selected_ordinary,
        [1.0] * len(important) + [len(ordinary) / remaining] * remaining,
    )


def _next_values(
    model,
    encoder: Encoder,
    samples: list[NStepSample],
    support: Counter[tuple[str, str]],
    *,
    support_threshold: int,
) -> list[float]:
    output = [0.0] * len(samples)
    for batch_start in range(0, len(samples), 2048):
        batch = samples[batch_start:batch_start + 2048]
        features = []
        slices = []
        for sample in batch:
            start = len(features)
            if sample.next_state is not None:
                actions = [
                    action for action in sample.next_state.legal_actions
                    if support_threshold <= 0
                    or action == sample.next_state.baseline_action
                    or support[(sample.next_state.source_context, action)] >= support_threshold
                ]
                if not actions:
                    actions = [sample.next_state.baseline_action]
                features.extend(features_for_candidate(sample.next_state, action) for action in actions)
            slices.append((start, len(features)))
        predicted = _predict(model, encoder, features)
        for offset, (start, stop) in enumerate(slices):
            if stop > start:
                output[batch_start + offset] = max(float(value) for value in predicted[start:stop])
    return output


def _policy_summary(
    model,
    encoder: Encoder,
    samples: list[NStepSample],
    support: Counter[tuple[str, str]],
    hit_labels: dict[tuple[str, int], bool],
    *,
    support_threshold: int,
    margin: float,
) -> dict[str, object]:
    logged_q = _predict(model, encoder, [row.state.features for row in samples])
    chosen = []
    baseline = []
    chosen_q = []
    baseline_q = []
    outside = 0
    for batch_start in range(0, len(samples), 2048):
        batch = samples[batch_start:batch_start + 2048]
        features = []
        slices = []
        action_lists = []
        for sample in batch:
            actions = list(sample.state.legal_actions)
            start = len(features)
            features.extend(features_for_candidate(sample.state, action) for action in actions)
            slices.append((start, len(features)))
            action_lists.append(actions)
        q_values = _predict(model, encoder, features)
        for sample, actions, (start, stop) in zip(batch, action_lists, slices, strict=True):
            values = q_values[start:stop]
            baseline_index = actions.index(sample.state.baseline_action)
            supported = [
                index for index, action in enumerate(actions)
                if support_threshold <= 0
                or action == sample.state.baseline_action
                or support[(sample.state.source_context, action)] >= support_threshold
            ]
            best = max(supported, key=lambda index: (float(values[index]), actions[index]))
            if float(values[best]) < float(values[baseline_index]) + margin:
                best = baseline_index
            action = actions[best]
            chosen.append(action)
            baseline.append(sample.state.baseline_action)
            chosen_q.append(float(values[best]))
            baseline_q.append(float(values[baseline_index]))
            outside += action not in sample.state.legal_actions
    actual_hit = [hit_labels[(row.state.run_id, row.state.sequence)] for row in samples]
    hit_ranking = {"average_precision_from_negative_q": None, "roc_auc_from_negative_q": None}
    if len(set(actual_hit)) == 2:
        from sklearn.metrics import average_precision_score, roc_auc_score

        hit_ranking = {
            "average_precision_from_negative_q": float(average_precision_score(actual_hit, [-float(value) for value in logged_q])),
            "roc_auc_from_negative_q": float(roc_auc_score(actual_hit, [-float(value) for value in logged_q])),
        }
    return {
        "rows": len(samples),
        "support_threshold": support_threshold,
        "baseline_margin": margin,
        "action_counts": dict(Counter(chosen).most_common()),
        "logged_action_match_rate": sum(
            action == row.state.action for action, row in zip(chosen, samples, strict=True)
        ) / len(samples),
        "baseline_disagreements": sum(
            action != base for action, base in zip(chosen, baseline, strict=True)
        ),
        "mean_predicted_q": sum(chosen_q) / len(chosen_q),
        "mean_predicted_advantage_over_baseline": sum(
            left - right for left, right in zip(chosen_q, baseline_q, strict=True)
        ) / len(chosen_q),
        "selected_outside_native_safe_set": outside,
        "bomb_selections": sum("bomb" in action.casefold() for action in chosen),
        "hit_120_ranking_of_logged_q": hit_ranking,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scope", required=True)
    parser.add_argument("--view", choices=("exact-v5", "common"), default="exact-v5")
    parser.add_argument("--regressors", default="lightgbm,extra-trees,xgboost")
    parser.add_argument("--backup", choices=("max", "support"), default="support")
    parser.add_argument("--support-threshold", type=int, default=32)
    parser.add_argument("--bellman-iterations", type=int, default=8)
    parser.add_argument("--n-step", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--trees-per-iteration", type=int, default=160)
    parser.add_argument("--threads", type=int, default=min(12, os.cpu_count() or 1))
    parser.add_argument("--validation-runs", type=int, default=1)
    parser.add_argument("--max-train-rows", type=int, default=250_000)
    parser.add_argument("--seed", type=int, default=6006)
    args = parser.parse_args()
    if not 0.0 < args.gamma < 1.0:
        parser.error("gamma must be between zero and one")
    if min(
        args.support_threshold,
        args.bellman_iterations,
        args.n_step,
        args.trees_per_iteration,
        args.threads,
        args.validation_runs,
        args.max_train_rows,
    ) <= 0:
        parser.error("integer training parameters must be positive")
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain exactly four integers")
    _, indexed = load_dataset_index(args.dataset)
    runs = sorted([
        run for run in indexed
        if run.scope == scope
        and run.training_eligible
        and (args.view == "common" or run.transition_schema == "th06-rl-transition-v5")
    ], key=lambda run: run.run_id)
    if len(runs) <= args.validation_runs:
        raise SystemExit("scope/view needs at least one train and one validation Stage")
    train_runs = runs[:-args.validation_runs]
    validation_runs = runs[-args.validation_runs:]
    exact = args.view == "exact-v5"

    train_sequences = [
        sequential_transitions(
            iter_run_transitions(args.dataset, run, verify_sha256=False),
            run,
            exact_context_only=exact,
        )
        for run in train_runs
    ]
    validation_sequences = [
        sequential_transitions(
            iter_run_transitions(args.dataset, run, verify_sha256=False),
            run,
            exact_context_only=exact,
        )
        for run in validation_runs
    ]
    all_train_samples = [
        sample for sequence in train_sequences
        for sample in _n_step(sequence, horizon=args.n_step, gamma=args.gamma)
    ]
    validation_samples = [
        sample for sequence in validation_sequences
        for sample in _n_step(sequence, horizon=args.n_step, gamma=args.gamma)
    ]
    sampled_train, inclusion_weights = _sample_training(
        all_train_samples,
        limit=args.max_train_rows,
        seed=args.seed,
    )
    support = Counter(
        (sample.state.source_context, sample.state.action)
        for sample in all_train_samples
    )
    hit_labels = {
        (run.run_id, sequence): value
        for run, transitions in zip(validation_runs, validation_sequences, strict=True)
        for sequence, value in _future_hit_labels(transitions).items()
    }
    encoder = Encoder([sample.state for sample in sampled_train])
    x_train = encoder.encode([sample.state.features for sample in sampled_train])
    weights = [
        inclusion * math.sqrt(min(20.0, 1.0 / sample.state.behavior_probability))
        for sample, inclusion in zip(sampled_train, inclusion_weights, strict=True)
    ]
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "encoder.json").write_text(
        json.dumps(encoder.manifest(), sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    backup_support = args.support_threshold if args.backup == "support" else 0
    import joblib

    results = {}
    for regressor_index, regressor in enumerate(
        value.strip() for value in args.regressors.split(",") if value.strip()
    ):
        started = time.monotonic()
        targets = [sample.observed_return for sample in sampled_train]
        iterations = []
        model = None
        for bellman_iteration in range(args.bellman_iterations):
            model, fit_options = _model(
                regressor,
                threads=args.threads,
                iterations=args.trees_per_iteration,
                seed=args.seed + regressor_index * 100 + bellman_iteration,
            )
            _fit(model, regressor, x_train, targets, weights, fit_options)
            next_values = _next_values(
                model,
                encoder,
                sampled_train,
                support,
                support_threshold=backup_support,
            )
            updated = [
                sample.observed_return + sample.bootstrap_discount * continuation
                for sample, continuation in zip(sampled_train, next_values, strict=True)
            ]
            delta = sum(abs(left - right) for left, right in zip(updated, targets, strict=True)) / len(targets)
            iterations.append({
                "iteration": bellman_iteration + 1,
                "mean_absolute_target_delta": delta,
                "target_min": min(updated),
                "target_mean": sum(updated) / len(updated),
                "target_max": max(updated),
            })
            targets = updated
        assert model is not None
        validation_next = _next_values(
            model,
            encoder,
            validation_samples,
            support,
            support_threshold=backup_support,
        )
        validation_targets = [
            sample.observed_return + sample.bootstrap_discount * continuation
            for sample, continuation in zip(validation_samples, validation_next, strict=True)
        ]
        validation_q = [
            float(value) for value in _predict(
                model,
                encoder,
                [sample.state.features for sample in validation_samples],
            )
        ]
        model_path = args.output / f"fqi-{args.backup}-{regressor}.joblib"
        joblib.dump(model, model_path, compress=3)
        results[regressor] = {
            "model_file": model_path.name,
            "model_sha256": _sha256(model_path),
            "total_seconds": time.monotonic() - started,
            "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "bellman_iterations": iterations,
            "heldout_bellman_residual": regression_metrics(validation_targets, validation_q),
            "policy": _policy_summary(
                model,
                encoder,
                validation_samples,
                support,
                hit_labels,
                support_threshold=args.support_threshold,
                margin=0.5,
            ),
        }
    manifest = {
        "schema": "th06-rl-offline-fqi-v1",
        "dataset": {"path": str(args.dataset), "revision": args.revision},
        "scope": list(scope),
        "view": args.view,
        "algorithm": "n-step-fitted-q-iteration",
        "backup": args.backup,
        "support_threshold": args.support_threshold,
        "gamma_per_physical_frame": args.gamma,
        "n_step": args.n_step,
        "split": {
            "kind": "chronological-complete-practice-stage",
            "train_runs": [run.run_id for run in train_runs],
            "validation_runs": [run.run_id for run in validation_runs],
            "train_rows_before_cap": len(all_train_samples),
            "train_rows": len(sampled_train),
            "validation_rows": len(validation_samples),
            "maximum_inclusion_weight": max(inclusion_weights),
        },
        "resource_limits": {
            "threads": args.threads,
            "sequential_regressors": True,
            "max_train_rows": args.max_train_rows,
            "trees_per_bellman_iteration": args.trees_per_iteration,
        },
        "results": results,
        "evaluation_boundary": {
            "heldout_bellman_residual_is_not_counterfactual_policy_value": True,
            "no_transition_level_ips_claim": True,
            "physical_stage_evaluation_required": True,
            "native_gate_remains_authoritative": True,
            "fresh_issue_revalidation_required": True,
            "bomb_representable": False,
        },
    }
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "results": results}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
