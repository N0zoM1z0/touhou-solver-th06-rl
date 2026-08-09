#!/usr/bin/env python3
"""Distill the native-gated offline teacher into a small CPU action ranker."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import gzip
import hashlib
import json
import math
from pathlib import Path
import resource
import subprocess
import time
from typing import Any, Iterable, Mapping

from th06_rl.native import ACTIONS
from th06_rl.headless_corpus import (
    HAZARD_FEATURE_DEFAULTS,
    HAZARD_FEATURE_NAMES,
    PROFILE_FEATURE_NAMES,
)


ACTION_BY_NAME = {action.name: action for action in ACTIONS}
CATEGORICAL_FEATURES = ("action", "previous_action", "source_context")
NUMERIC_FEATURES = (
    "player_x",
    "player_y",
    "boundary_reserve",
    "game_frame",
    "lives",
    "power",
    "rank",
    "graze",
    "bullet_count",
    "laser_count",
    "enemy_count",
    "boss_count",
    "legal_count",
    "min_clearance",
    "clearance_infinite",
    "final_x",
    "final_y",
    "final_boundary_reserve",
    "action_dx",
    "action_dy",
    "action_focused",
    "changed_direction",
    "reversed_direction",
    "changed_focus",
    *HAZARD_FEATURE_NAMES,
    *PROFILE_FEATURE_NAMES,
)
FEATURE_NAMES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
CORRECTIVE_TERMINATIONS = frozenset({"authority-failure", "physical-hit"})
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def repository_commit() -> str:
    """Resolve training provenance independently of the caller's cwd."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


@dataclass(frozen=True)
class Decision:
    run: str
    seed: int
    sequence: int
    source_context: str
    state: Mapping[str, Any]
    legal_actions: tuple[str, ...]
    candidates: tuple[Mapping[str, Any], ...]
    teacher_action: str
    selected_action: str
    observation_sha256: str = ""
    terminal_failure_distance: int | None = None
    counterfactual_original_action: str | None = None
    counterfactual_acceptable_actions: tuple[str, ...] | None = None


def candidate_features(decision: Decision, candidate: Mapping[str, Any]) -> dict[str, str | float]:
    action_name = str(candidate["action"])
    action = ACTION_BY_NAME[action_name]
    previous_name = str(decision.state["previous_action"])
    previous = ACTION_BY_NAME[previous_name]
    raw_clearance = candidate.get("min_clearance")
    infinite = raw_clearance is None
    clearance = 1024.0 if infinite else float(raw_clearance)
    return {
        "action": action_name,
        "previous_action": previous_name,
        "source_context": decision.source_context,
        "player_x": float(decision.state["player_x"]),
        "player_y": float(decision.state["player_y"]),
        "boundary_reserve": float(decision.state["boundary_reserve"]),
        "game_frame": float(decision.state["game_frame"]),
        "lives": float(decision.state["lives"]),
        "power": float(decision.state["power"]),
        "rank": float(decision.state["rank"]),
        "graze": float(decision.state["graze"]),
        "bullet_count": float(decision.state["bullet_count"]),
        "laser_count": float(decision.state["laser_count"]),
        "enemy_count": float(decision.state["enemy_count"]),
        "boss_count": float(decision.state["boss_count"]),
        "legal_count": float(len(decision.legal_actions)),
        "min_clearance": clearance,
        "clearance_infinite": float(infinite),
        "final_x": float(candidate["final_x"]),
        "final_y": float(candidate["final_y"]),
        "final_boundary_reserve": float(candidate["final_boundary_reserve"]),
        "action_dx": float(action.dx),
        "action_dy": float(action.dy),
        "action_focused": float(action.focused),
        "changed_direction": float((action.dx, action.dy) != (previous.dx, previous.dy)),
        "reversed_direction": float(
            (previous.dx != 0 or previous.dy != 0)
            and (action.dx, action.dy) == (-previous.dx, -previous.dy)
        ),
        "changed_focus": float(action.focused != previous.focused),
        **{
            name: float(decision.state.get(name, HAZARD_FEATURE_DEFAULTS[name]))
            for name in HAZARD_FEATURE_NAMES
        },
        **{
            name: 1024.0 if candidate.get(name) is None else float(candidate[name])
            for name in PROFILE_FEATURE_NAMES
        },
    }


def generic_choice(decision: Decision) -> str:
    def key(candidate: Mapping[str, Any]) -> tuple[float | int | str, ...]:
        action = ACTION_BY_NAME[str(candidate["action"])]
        clearance = candidate.get("min_clearance")
        return (
            math.inf if clearance is None else float(clearance),
            float(candidate["final_boundary_reserve"]),
            int(action.name == decision.state["previous_action"]),
            int(action.dx == 0 and action.dy == 0),
            int(action.focused),
            action.name,
        )

    return str(max(decision.candidates, key=key)["action"])


def candidate_sample_weight(
    decision: Decision,
    candidate: Mapping[str, Any],
    *,
    failure_horizon: int,
    failure_weight: float,
    counterfactual_weight: float = 0.0,
) -> float:
    """Emphasize the corrective pair before a demonstrated HIT or dead-end.

    The terminal signal says that the behavior trajectory became uncertifiable;
    it does not say that every locally legal action near the end was bad.  Only
    weight the teacher/behavior pair when they disagree, leaving all other
    candidates and teacher-agreeing failures at their ordinary imitation weight.
    """
    weight = 1.0
    counterfactual_actions = decision.counterfactual_acceptable_actions
    if (
        counterfactual_weight > 0.0
        and counterfactual_actions is not None
        and str(candidate["action"])
        in set(counterfactual_actions).union(
            {decision.counterfactual_original_action}
            if decision.counterfactual_original_action is not None else set()
        )
    ):
        weight = max(weight, 1.0 + counterfactual_weight)
    distance = decision.terminal_failure_distance
    if not (
        failure_horizon <= 0
        or failure_weight <= 0.0
        or distance is None
        or distance <= 0
        or distance > failure_horizon
        or decision.teacher_action == decision.selected_action
        or str(candidate["action"]) not in {decision.teacher_action, decision.selected_action}
    ):
        proximity = (failure_horizon - distance + 1) / failure_horizon
        weight = max(weight, 1.0 + failure_weight * proximity)
    return weight


class Encoder:
    def __init__(
        self,
        decisions: Iterable[Decision],
        *,
        feature_names: Iterable[str] = FEATURE_NAMES,
    ) -> None:
        self.feature_names = tuple(feature_names)
        if not self.feature_names or not set(CATEGORICAL_FEATURES).issubset(self.feature_names):
            raise ValueError("feature schema must retain all categorical features")
        unknown = set(self.feature_names) - set(FEATURE_NAMES)
        if unknown:
            raise ValueError(f"feature schema contains unsupported fields: {sorted(unknown)}")
        values = {name: set() for name in CATEGORICAL_FEATURES}
        for decision in decisions:
            values["previous_action"].add(str(decision.state["previous_action"]))
            values["source_context"].add(decision.source_context)
            values["action"].update(decision.legal_actions)
        values["action"].update(ACTION_BY_NAME)
        values["previous_action"].update(ACTION_BY_NAME)
        self.categories = {
            name: {value: index for index, value in enumerate(sorted(items))}
            for name, items in values.items()
        }

    def encode(self, features: list[dict[str, str | float]]):
        import numpy as np

        output = np.empty((len(features), len(self.feature_names)), dtype=np.float32)
        for column, name in enumerate(self.feature_names):
            if name in self.categories:
                mapping = self.categories[name]
                output[:, column] = [mapping.get(str(row[name]), -1) for row in features]
            else:
                output[:, column] = [float(row[name]) for row in features]
        return output

    def manifest(self) -> dict[str, list[str]]:
        return {
            name: [value for value, _ in sorted(mapping.items(), key=lambda item: item[1])]
            for name, mapping in self.categories.items()
        }


def _run_directories(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if (path / "manifest.json").is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(item.parent for item in path.rglob("manifest.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def load_decisions(paths: Iterable[Path]) -> tuple[list[Decision], dict[str, Any]]:
    decisions = []
    index_by_observation: dict[str, int] = {}
    duplicate_decisions = 0
    scope: Mapping[str, Any] | None = None
    source: Mapping[str, Any] | None = None
    for run in _run_directories(paths):
        manifest = json.loads((run / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("transaction_complete") is not True or manifest.get("training_eligible") is not True:
            continue
        if scope is None:
            scope = manifest["scope"]
            source = manifest["source"]
        if manifest.get("scope") != scope:
            raise ValueError("headless teacher data silently mixes scopes")
        if manifest.get("source", {}).get("commit") != source.get("commit"):  # type: ignore[union-attr]
            raise ValueError("headless teacher data silently mixes source revisions")
        seed = int(manifest["initial_seed"])
        terminal_failure = manifest.get("termination_reason") in CORRECTIVE_TERMINATIONS
        transition_count = int(manifest.get("transition_count", 0))
        with gzip.open(run / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                behavior = row["behavior"]
                digest = str(row["observation_sha256"])
                teacher_action = str(behavior["teacher_action"])
                previous_index = index_by_observation.get(digest)
                failure_distance = (
                    transition_count - int(row["sequence"])
                    if terminal_failure
                    else None
                )
                if previous_index is not None:
                    previous = decisions[previous_index]
                    if previous.teacher_action != teacher_action:
                        raise ValueError("identical observation has conflicting teacher labels")
                    if (
                        failure_distance is not None
                        and (
                            previous.terminal_failure_distance is None
                            or failure_distance < previous.terminal_failure_distance
                        )
                    ):
                        decisions[previous_index] = replace(
                            previous,
                            run=run.name,
                            selected_action=str(behavior["selected_action"]),
                            terminal_failure_distance=failure_distance,
                        )
                    duplicate_decisions += 1
                    continue
                candidates = tuple(row["action_candidates"])
                legal = tuple(row["legal_actions"])
                if set(legal) != {str(candidate["action"]) for candidate in candidates}:
                    raise ValueError("candidate table does not equal the native legal set")
                index_by_observation[digest] = len(decisions)
                decisions.append(Decision(
                    run=run.name,
                    seed=seed,
                    sequence=int(row["sequence"]),
                    source_context=str(row["source_context"]),
                    state=row["state"],
                    legal_actions=legal,
                    candidates=candidates,
                    teacher_action=teacher_action,
                    selected_action=str(behavior["selected_action"]),
                    observation_sha256=digest,
                    terminal_failure_distance=failure_distance,
                ))
    if not decisions or scope is None or source is None:
        raise ValueError("no eligible compact headless decisions found")
    return decisions, {
        "scope": scope,
        "source": source,
        "duplicate_decisions_skipped": duplicate_decisions,
    }


def _counterfactual_files(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.rglob("*.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def _counterfactual_file_records(files: Iterable[Path]) -> tuple[list[dict[str, str]], str]:
    records = []
    for path in files:
        records.append({
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        })
    payload = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    return records, hashlib.sha256(payload).hexdigest()


def apply_counterfactual_labels(
    decisions: list[Decision],
    provenance: Mapping[str, Any],
    paths: Iterable[Path],
    *,
    target: str = "unique-best",
) -> tuple[list[Decision], dict[str, Any]]:
    if target not in {"unique-best", "survivable"}:
        raise ValueError("counterfactual target must be unique-best or survivable")
    files = _counterfactual_files(paths)
    if not files:
        raise ValueError("no counterfactual label files found")
    file_records, file_set_sha256 = _counterfactual_file_records(files)
    labels: dict[str, tuple[str, ...]] = {}
    ambiguous = 0
    checkpoints = 0
    runtime_sources = set()
    for path in files:
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("schema") != "th06-rl-headless-cow-counterfactual-v1":
            raise ValueError(f"unsupported counterfactual schema: {path}")
        if document.get("scope") != provenance["scope"]:
            raise ValueError("counterfactual labels silently mix scopes")
        if document.get("input_source", {}).get("commit") != provenance["source"].get("commit"):
            raise ValueError("counterfactual labels use a different factual source revision")
        runtime_sources.add(json.dumps(document.get("runtime_source"), sort_keys=True))
        for checkpoint in document.get("checkpoints", []):
            checkpoints += 1
            best = tuple(str(action) for action in checkpoint.get("best_actions", []))
            if target == "survivable":
                branch_frames = int(checkpoint["branch_frames"])
                selected = tuple(
                    str(outcome["first_action"])
                    for outcome in checkpoint["outcomes"]
                    if int(outcome["survival_ticks"]) == branch_frames
                    and int(outcome["physical_deaths_delta"]) == 0
                    and outcome["termination_reason"]
                    in {"tick-limit", "chain-exit-success", "stage-clear-success"}
                )
                selected = selected or best
            else:
                selected = best
            if not selected or (target == "unique-best" and len(selected) != 1):
                ambiguous += 1
                continue
            digest = str(checkpoint["observation_sha256"])
            previous = labels.get(digest)
            if previous is not None and previous != selected:
                raise ValueError("counterfactual checkpoints disagree on target actions")
            labels[digest] = selected

    matched = 0
    changed = 0
    output = []
    for decision in decisions:
        acceptable = labels.get(decision.observation_sha256)
        if acceptable is None:
            output.append(decision)
            continue
        if not set(acceptable).issubset(decision.legal_actions):
            raise ValueError("counterfactual target action escaped the factual native legal set")
        action = decision.teacher_action if decision.teacher_action in acceptable else acceptable[0]
        matched += 1
        changed += decision.teacher_action not in acceptable
        output.append(replace(
            decision,
            teacher_action=action,
            counterfactual_original_action=(
                decision.teacher_action if action != decision.teacher_action else None
            ),
            counterfactual_acceptable_actions=acceptable,
        ))
    return output, {
        "files": len(files),
        "files_used": file_records,
        "file_set_sha256": file_set_sha256,
        "checkpoints": checkpoints,
        "unique_unambiguous_labels": len(labels),
        "ambiguous_checkpoints_skipped": ambiguous,
        "matched_decisions": matched,
        "changed_local_teacher_labels": changed,
        "unmatched_labels": len(labels) - matched,
        "runtime_sources": [json.loads(item) for item in sorted(runtime_sources)],
        "target": target,
        "mean_acceptable_actions": (
            sum(len(actions) for actions in labels.values()) / len(labels) if labels else 0.0
        ),
    }


def _candidate_matrix(
    decisions: Iterable[Decision],
    *,
    failure_horizon: int,
    failure_weight: float,
    counterfactual_weight: float,
) -> tuple[list[dict[str, str | float]], list[int], list[float], dict[str, Any]]:
    features = []
    labels = []
    weights = []
    corrective_decisions = 0
    counterfactual_decisions = 0
    for decision in decisions:
        corrective = False
        for candidate in decision.candidates:
            features.append(candidate_features(decision, candidate))
            acceptable = decision.counterfactual_acceptable_actions
            labels.append(int(
                candidate["action"] in acceptable
                if acceptable is not None
                else candidate["action"] == decision.teacher_action
            ))
            weight = candidate_sample_weight(
                decision,
                candidate,
                failure_horizon=failure_horizon,
                failure_weight=failure_weight,
                counterfactual_weight=counterfactual_weight,
            )
            weights.append(weight)
            corrective = corrective or weight > 1.0
        corrective_decisions += int(corrective)
        counterfactual_decisions += int(decision.counterfactual_acceptable_actions is not None)
    return features, labels, weights, {
        "terminal_failure_horizon": failure_horizon,
        "terminal_failure_reasons": sorted(CORRECTIVE_TERMINATIONS),
        "corrective_pair_weight": failure_weight,
        "corrective_decisions": corrective_decisions,
        "counterfactual_pair_weight": counterfactual_weight,
        "counterfactual_decisions": counterfactual_decisions,
        "weighted_candidate_rows": sum(weight > 1.0 for weight in weights),
        "maximum_sample_weight": max(weights, default=1.0),
    }


def evaluate(
    model,
    encoder: Encoder,
    decisions: list[Decision],
    *,
    threads: int,
) -> dict[str, Any]:
    teacher_matches = 0
    behavior_matches = 0
    generic_matches = 0
    reciprocal_ranks = []
    acceptable_matches = 0
    acceptable_reciprocal_ranks = []
    action_counts: Counter[str] = Counter()
    features = [
        candidate_features(decision, candidate)
        for decision in decisions
        for candidate in decision.candidates
    ]
    probabilities = model.booster_.predict(encoder.encode(features), num_threads=threads)
    offset = 0
    for decision in decisions:
        count = len(decision.candidates)
        decision_probabilities = probabilities[offset:offset + count]
        offset += count
        ranked = sorted(
            zip(decision.candidates, decision_probabilities, strict=True),
            key=lambda item: (float(item[1]), str(item[0]["action"])),
            reverse=True,
        )
        selected = str(ranked[0][0]["action"])
        action_counts[selected] += 1
        teacher_matches += selected == decision.teacher_action
        acceptable = decision.counterfactual_acceptable_actions or (decision.teacher_action,)
        acceptable_matches += selected in acceptable
        behavior_matches += selected == decision.selected_action
        generic_matches += generic_choice(decision) == decision.teacher_action
        rank = next(
            index for index, item in enumerate(ranked, 1)
            if item[0]["action"] == decision.teacher_action
        )
        reciprocal_ranks.append(1.0 / rank)
        acceptable_rank = next(
            index for index, item in enumerate(ranked, 1)
            if item[0]["action"] in acceptable
        )
        acceptable_reciprocal_ranks.append(1.0 / acceptable_rank)
    count = len(decisions)
    return {
        "decisions": count,
        "teacher_top1_accuracy": teacher_matches / count,
        "teacher_mean_reciprocal_rank": sum(reciprocal_ranks) / count,
        "acceptable_top1_accuracy": acceptable_matches / count,
        "acceptable_mean_reciprocal_rank": sum(acceptable_reciprocal_ranks) / count,
        "behavior_action_match": behavior_matches / count,
        "generic_teacher_top1_accuracy": generic_matches / count,
        "native_legal_action_ratio": 1.0,
        "bomb_actions": 0,
        "selected_action_counts": dict(sorted(action_counts.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--holdout-seed", type=int, action="append")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--iterations", type=int, default=240)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--min-child-samples", type=int, default=40)
    parser.add_argument("--failure-horizon", type=int, default=0)
    parser.add_argument("--failure-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-labels", type=Path, action="append")
    parser.add_argument(
        "--counterfactual-target",
        choices=("unique-best", "survivable"),
        default="unique-best",
    )
    parser.add_argument("--counterfactual-weight", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12 on the shared VPS")
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    if args.learning_rate <= 0.0:
        parser.error("learning-rate must be positive")
    if args.num_leaves < 2 or args.max_depth < 1 or args.min_child_samples < 1:
        parser.error("tree capacity bounds must be positive")
    if min(args.failure_horizon, args.failure_weight, args.counterfactual_weight) < 0:
        parser.error("failure weighting bounds must be nonnegative")
    # Counterfactual generation may continue in parallel. Freeze its input
    # file set before corpus decoding so one model never consumes an
    # accidental mid-training mixture that cannot be reconstructed later.
    counterfactual_label_files = _counterfactual_files(args.counterfactual_labels or ())
    decisions, provenance = load_decisions(args.paths)
    counterfactuals = None
    if args.counterfactual_labels:
        decisions, counterfactuals = apply_counterfactual_labels(
            decisions,
            provenance,
            counterfactual_label_files,
            target=args.counterfactual_target,
        )
    seeds = sorted({decision.seed for decision in decisions})
    holdout = set(args.holdout_seed or seeds[-1:])
    train = [decision for decision in decisions if decision.seed not in holdout]
    test = [decision for decision in decisions if decision.seed in holdout]
    if not train or not test:
        parser.error("seed-grouped train and holdout sets must both be nonempty")
    encoder = Encoder(train)
    train_features, labels, sample_weights, weighting = _candidate_matrix(
        train,
        failure_horizon=args.failure_horizon,
        failure_weight=args.failure_weight,
        counterfactual_weight=args.counterfactual_weight,
    )
    x_train = encoder.encode(train_features)

    from lightgbm import LGBMClassifier
    import joblib
    import numpy as np

    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.iterations,
        learning_rate=args.learning_rate,
        num_leaves=args.num_leaves,
        max_depth=args.max_depth,
        min_child_samples=args.min_child_samples,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=args.threads,
        verbosity=-1,
    )
    started = time.perf_counter()
    model.fit(
        x_train,
        np.asarray(labels, dtype=np.int8),
        sample_weight=np.asarray(sample_weights, dtype=np.float32),
        categorical_feature=list(range(len(CATEGORICAL_FEATURES))),
    )
    elapsed = time.perf_counter() - started
    compatible_sources = [provenance["source"]]
    if counterfactuals is not None:
        for compatible in counterfactuals["runtime_sources"]:
            if compatible not in compatible_sources:
                compatible_sources.append(compatible)
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "categories": encoder.manifest(),
            "scope": provenance["scope"],
            "headless_source": provenance["source"],
            "compatible_headless_sources": compatible_sources,
        },
        args.output / "teacher-ranker.joblib",
        compress=3,
    )
    code_commit = repository_commit()
    report = {
        "schema": "th06-rl-headless-teacher-distillation-v1",
        "algorithm": "lightgbm-binary-candidate-ranker",
        "authority": "rank-native-legal-set-only",
        "scope": provenance["scope"],
        "headless_source": provenance["source"],
        "compatible_headless_sources": compatible_sources,
        "code_commit": code_commit,
        "train_seeds": sorted(set(seeds) - holdout),
        "holdout_seeds": sorted(holdout),
        "train_decisions": len(train),
        "holdout_decisions": len(test),
        "candidate_training_rows": len(labels),
        "duplicate_decisions_skipped": provenance["duplicate_decisions_skipped"],
        "iterations": args.iterations,
        "model_parameters": {
            "learning_rate": args.learning_rate,
            "num_leaves": args.num_leaves,
            "max_depth": args.max_depth,
            "min_child_samples": args.min_child_samples,
        },
        "failure_weighting": weighting,
        "counterfactual_labels": counterfactuals,
        "threads": args.threads,
        "fit_seconds": elapsed,
        "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
        "train": evaluate(model, encoder, train, threads=args.threads),
        "holdout": evaluate(model, encoder, test, threads=args.threads),
        "promotion_allowed": False,
        "promotion_blocker": (
            "teacher imitation is not an off-policy return estimate; require long multi-seed "
            "HIT/stage-clear trajectories and later Windows differential evidence"
        ),
    }
    rendered = json.dumps(report, indent=2, sort_keys=True) + "\n"
    (args.output / "report.json").write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
