#!/usr/bin/env python3
"""Fit and evaluate Generation 7 from an immutable admitted route index."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.g7_dataset import load_admitted_episodes  # noqa: E402
from th06_rl.g7_ope import evaluate_candidate  # noqa: E402
from th06_rl.g7_training import fit_g7_candidate  # noqa: E402
from th06_rl.offline_options import OfflineOptionError, whole_episode_split  # noqa: E402


CONFIG_SCHEMA = "th06-rl-g7-training-config-v2"
TRAINING_RUN_SCHEMA = "th06-rl-g7-training-run-v1"


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPOSITORY))
    except ValueError as error:
        raise ValueError(f"training path is outside the repository: {path}") from error


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _config(path: Path) -> dict[str, object]:
    value = _object(path)
    required = {
        "schema", "seed", "validation_fraction", "reference_epsilon",
        "awr_temperature", "crossfit_folds", "critic_estimators", "n_jobs",
        "maximum_importance_ratio", "support_prototypes_per_action",
        "support_distance_quantile", "support_minimum_samples",
        "support_minimum_ess", "ensemble_members",
        "ensemble_episode_fraction", "required_vote_fraction", "target_max_kl",
        "maximum_step_ratio", "maximum_cumulative_ratio",
        "minimum_effective_sample_size", "minimum_validation_episodes",
        "ope_confidence", "bootstrap_resamples", "permutation_resamples",
        "maximum_null_p_value",
    }
    if value.get("schema") != CONFIG_SCHEMA or set(value) != required:
        raise ValueError("Generation-7 config schema/keys differ")
    return value


def train(
    dataset_path: Path,
    config_path: Path,
) -> dict[str, object]:
    dataset_path = dataset_path.resolve()
    config_path = config_path.resolve()
    _relative(dataset_path)
    _relative(config_path)
    config = _config(config_path)
    episodes = load_admitted_episodes(dataset_path, repository=REPOSITORY)
    by_id = {episode[0].episode_id: episode for episode in episodes}
    if len(by_id) != len(episodes):
        raise ValueError("admitted dataset repeats a physical episode")
    training_ids, validation_ids = whole_episode_split(
        tuple(by_id),
        validation_fraction=float(config["validation_fraction"]),
        seed=int(config["seed"]),
    )
    if len(validation_ids) < int(config["minimum_validation_episodes"]):
        raise ValueError("predeclared held-out episode minimum is not met")
    training = tuple(by_id[episode] for episode in sorted(training_ids))
    validation = tuple(by_id[episode] for episode in sorted(validation_ids))
    candidate = fit_g7_candidate(
        training,
        seed=int(config["seed"]),
        reference_epsilon=float(config["reference_epsilon"]),
        awr_temperature=float(config["awr_temperature"]),
        crossfit_folds=int(config["crossfit_folds"]),
        critic_estimators=int(config["critic_estimators"]),
        n_jobs=int(config["n_jobs"]),
        maximum_importance_ratio=float(config["maximum_importance_ratio"]),
        support_prototypes_per_action=int(config["support_prototypes_per_action"]),
        support_distance_quantile=float(config["support_distance_quantile"]),
        support_minimum_samples=int(config["support_minimum_samples"]),
        support_minimum_ess=float(config["support_minimum_ess"]),
        ensemble_members=int(config["ensemble_members"]),
        ensemble_episode_fraction=float(config["ensemble_episode_fraction"]),
        required_vote_fraction=float(config["required_vote_fraction"]),
    )
    evaluation = evaluate_candidate(
        candidate,
        validation,
        max_kl=float(config["target_max_kl"]),
        maximum_step_ratio=float(config["maximum_step_ratio"]),
        maximum_cumulative_ratio=float(config["maximum_cumulative_ratio"]),
        minimum_effective_sample_size=float(
            config["minimum_effective_sample_size"]
        ),
        minimum_episodes=int(config["minimum_validation_episodes"]),
        confidence=float(config["ope_confidence"]),
        bootstrap_resamples=int(config["bootstrap_resamples"]),
        permutation_resamples=int(config["permutation_resamples"]),
        maximum_null_p_value=float(config["maximum_null_p_value"]),
        seed=int(config["seed"]) + 20_000,
    )
    return {
        "schema": TRAINING_RUN_SCHEMA,
        "authorization": "wine-canary-forbidden",
        "authorization_reason": (
            "portable online integration and original-Wine shadow/latency gates "
            "have not run"
        ),
        "dataset": {
            "path": _relative(dataset_path),
            "sha256": _sha256(dataset_path),
        },
        "config": {
            "path": _relative(config_path),
            "sha256": _sha256(config_path),
            "values": config,
        },
        "split": {
            "unit": "complete-route",
            "training_episode_ids": sorted(training_ids),
            "validation_episode_ids": sorted(validation_ids),
            "disjoint": not training_ids & validation_ids,
        },
        "candidate": candidate,
        "heldout_evaluation": evaluation,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY / "config/g7_training_v2.json",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        output = args.output.resolve()
        _relative(output)
        result = train(args.dataset, args.config)
        _atomic_json(output, result)
    except (OSError, OfflineOptionError, RuntimeError, TypeError, ValueError) as error:
        parser.error(str(error))
    print(json.dumps({
        "authorization": result["authorization"],
        "heldout_evaluation_passed": result["heldout_evaluation"]["passed"],
        "output": _relative(output),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
