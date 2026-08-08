#!/usr/bin/env python3
"""Distill the native-gated offline teacher into a small CPU action ranker."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import gzip
import json
import math
from pathlib import Path
import resource
import subprocess
import time
from typing import Any, Iterable, Mapping

from th06_rl.native import ACTIONS


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
)
FEATURE_NAMES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)


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


class Encoder:
    def __init__(self, decisions: Iterable[Decision]) -> None:
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

        output = np.empty((len(features), len(FEATURE_NAMES)), dtype=np.float32)
        for column, name in enumerate(FEATURE_NAMES):
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
    labels_by_observation: dict[str, str] = {}
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
        with gzip.open(run / "transitions.jsonl.gz", "rt", encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                behavior = row["behavior"]
                digest = str(row["observation_sha256"])
                teacher_action = str(behavior["teacher_action"])
                previous_label = labels_by_observation.get(digest)
                if previous_label is not None:
                    if previous_label != teacher_action:
                        raise ValueError("identical observation has conflicting teacher labels")
                    duplicate_decisions += 1
                    continue
                labels_by_observation[digest] = teacher_action
                candidates = tuple(row["action_candidates"])
                legal = tuple(row["legal_actions"])
                if set(legal) != {str(candidate["action"]) for candidate in candidates}:
                    raise ValueError("candidate table does not equal the native legal set")
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
                ))
    if not decisions or scope is None or source is None:
        raise ValueError("no eligible compact headless decisions found")
    return decisions, {
        "scope": scope,
        "source": source,
        "duplicate_decisions_skipped": duplicate_decisions,
    }


def _candidate_matrix(decisions: Iterable[Decision]) -> tuple[list[dict[str, str | float]], list[int]]:
    features = []
    labels = []
    for decision in decisions:
        for candidate in decision.candidates:
            features.append(candidate_features(decision, candidate))
            labels.append(int(candidate["action"] == decision.teacher_action))
    return features, labels


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
        behavior_matches += selected == decision.selected_action
        generic_matches += generic_choice(decision) == decision.teacher_action
        rank = next(
            index for index, item in enumerate(ranked, 1)
            if item[0]["action"] == decision.teacher_action
        )
        reciprocal_ranks.append(1.0 / rank)
    count = len(decisions)
    return {
        "decisions": count,
        "teacher_top1_accuracy": teacher_matches / count,
        "teacher_mean_reciprocal_rank": sum(reciprocal_ranks) / count,
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
    parser.add_argument("--seed", type=int, default=6006)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12 on the shared VPS")
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    decisions, provenance = load_decisions(args.paths)
    seeds = sorted({decision.seed for decision in decisions})
    holdout = set(args.holdout_seed or seeds[-1:])
    train = [decision for decision in decisions if decision.seed not in holdout]
    test = [decision for decision in decisions if decision.seed in holdout]
    if not train or not test:
        parser.error("seed-grouped train and holdout sets must both be nonempty")
    encoder = Encoder(train)
    train_features, labels = _candidate_matrix(train)
    x_train = encoder.encode(train_features)

    from lightgbm import LGBMClassifier
    import joblib
    import numpy as np

    model = LGBMClassifier(
        objective="binary",
        n_estimators=args.iterations,
        learning_rate=0.05,
        num_leaves=31,
        max_depth=10,
        min_child_samples=40,
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
        categorical_feature=list(range(len(CATEGORICAL_FEATURES))),
    )
    elapsed = time.perf_counter() - started
    args.output.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "categories": encoder.manifest(),
            "scope": provenance["scope"],
            "headless_source": provenance["source"],
        },
        args.output / "teacher-ranker.joblib",
        compress=3,
    )
    code_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    report = {
        "schema": "th06-rl-headless-teacher-distillation-v1",
        "algorithm": "lightgbm-binary-candidate-ranker",
        "authority": "rank-native-legal-set-only",
        "scope": provenance["scope"],
        "headless_source": provenance["source"],
        "code_commit": code_commit,
        "train_seeds": sorted(set(seeds) - holdout),
        "holdout_seeds": sorted(holdout),
        "train_decisions": len(train),
        "holdout_decisions": len(test),
        "candidate_training_rows": len(labels),
        "duplicate_decisions_skipped": provenance["duplicate_decisions_skipped"],
        "iterations": args.iterations,
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
