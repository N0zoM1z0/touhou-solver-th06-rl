"""Algorithm-independent admission index for complete causal Wine routes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from th06_rl.corpus import (
    FRAME_SCHEMA,
    MANIFEST_SCHEMA,
    RUN_SCHEMA,
    TRANSITION_SCHEMA,
)
from th06_rl.offline_options import (
    OPTION_DATASET_SCHEMA,
    OfflineOptionError,
    OfflineOptionTransition,
    iter_offline_options,
    validate_offline_episode,
)
from th06_rl.policies.safe_option_exploration import POLICY_NAME


DATASET_SCHEMA = "th06-rl-offline-route-dataset-v1"
COLLECTION_SCHEMA = "th06-rl-source-complete-parallel-route-collection-v1"


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


def _relative_path(repository: Path, value: object, *, label: str) -> Path:
    if not isinstance(value, str) or not value or Path(value).is_absolute():
        raise ValueError(f"{label} must be a repository-relative path")
    root = repository.resolve()
    path = (root / value).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes the repository") from error
    return path


def _relative(repository: Path, path: Path) -> str:
    try:
        return str(path.resolve().relative_to(repository.resolve()))
    except ValueError as error:
        raise ValueError(f"dataset input is outside the repository: {path}") from error


def _require_hash(path: Path, expected: object, *, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64 or _sha256(path) != expected:
        raise ValueError(f"{label} SHA-256 differs")


def _verify_shards(run_dir: Path, manifest: dict[str, object]) -> None:
    rows = manifest.get("shards")
    if not isinstance(rows, list) or not rows:
        raise ValueError("admitted route has no immutable corpus shards")
    seen = set()
    streams = set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("corpus shard declaration is not an object")
        relative = row.get("path")
        if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
            raise ValueError("corpus shard path is not relative")
        path = (run_dir / relative).resolve()
        try:
            path.relative_to(run_dir.resolve())
        except ValueError as error:
            raise ValueError("corpus shard path escapes its run") from error
        if relative in seen or not path.is_file():
            raise ValueError("corpus shard is duplicated or absent")
        seen.add(relative)
        streams.add(str(row.get("stream", "")))
        _require_hash(path, row.get("sha256"), label=f"corpus shard {relative}")
        if path.stat().st_size != int(row.get("compressed_bytes", -1)):
            raise ValueError(f"corpus shard byte count differs: {relative}")
    if not {"frames", "transitions", "objects"}.issubset(streams):
        raise ValueError("admitted route lacks required factual streams")


def _seed_from_options(options: tuple[OfflineOptionTransition, ...]) -> int:
    prefixes = set()
    for option in options:
        prefix, separator, _counter = option.option_id.partition(":")
        if not separator or len(prefix) != 16:
            raise OfflineOptionError("option identity does not bind its policy seed")
        try:
            prefixes.add(int(prefix, 16))
        except ValueError as error:
            raise OfflineOptionError("option policy seed is not hexadecimal") from error
    if len(prefixes) != 1:
        raise OfflineOptionError("one episode contains multiple policy seeds")
    return prefixes.pop()


def _exploration_probability(
    options: tuple[OfflineOptionTransition, ...],
) -> float:
    inferred = []
    for option in options:
        probabilities = dict(option.behavior_probabilities)
        legal = option.state.legal_actions
        if len(legal) <= 1:
            continue
        nonbaseline = [
            probabilities[action]
            for action in legal
            if action != option.state.baseline_action
        ]
        if not nonbaseline:
            raise OfflineOptionError("behavior baseline is outside its legal set")
        epsilon = len(legal) * nonbaseline[0]
        expected = {
            action: epsilon / len(legal)
            + (1.0 - epsilon if action == option.state.baseline_action else 0.0)
            for action in legal
        }
        if (
            not 0.0 <= epsilon <= 1.0
            or any(
                abs(probabilities[action] - expected[action]) > 1e-12
                for action in legal
            )
        ):
            raise OfflineOptionError("behavior distribution is not declared epsilon-safe")
        inferred.append(epsilon)
    if not inferred or any(abs(value - inferred[0]) > 1e-12 for value in inferred):
        raise OfflineOptionError("episode exploration probability is absent or changed")
    return inferred[0]


def _admit_episode(
    repository: Path,
    row: dict[str, object],
    *,
    collection_commit: str,
) -> tuple[dict[str, object], tuple[OfflineOptionTransition, ...]]:
    run_dir = _relative_path(repository, row.get("run_dir"), label="run_dir")
    report_path = _relative_path(
        repository, row.get("report_path"), label="report_path"
    )
    audit_path = _relative_path(
        repository, row.get("audit_path"), label="audit_path"
    )
    run_path = run_dir / "run.json"
    manifest_path = run_dir / "manifest.json"
    for path, expected, label in (
        (run_path, row.get("run_sha256"), "run"),
        (manifest_path, row.get("manifest_sha256"), "manifest"),
        (report_path, row.get("report_sha256"), "report"),
        (audit_path, row.get("audit_sha256"), "audit"),
    ):
        _require_hash(path, expected, label=label)
    run = _object(run_path)
    manifest = _object(manifest_path)
    report = _object(report_path)
    audit = _object(audit_path)
    metadata = run.get("metadata")
    schemas = run.get("schemas")
    planner = metadata.get("planner") if isinstance(metadata, dict) else None
    outcome = manifest.get("run_outcome")
    episode = manifest.get("episode")
    records = manifest.get("records")
    admission = audit.get("source_dataset_admission")
    verification = row.get("verification")
    checks = verification.get("checks") if isinstance(verification, dict) else None
    if not all(isinstance(value, dict) for value in (
        metadata, schemas, planner, outcome, episode, records, admission,
        verification, checks,
    )):
        raise ValueError("collection evidence lacks complete route contracts")
    assert isinstance(metadata, dict)
    assert isinstance(schemas, dict)
    assert isinstance(planner, dict)
    assert isinstance(outcome, dict)
    assert isinstance(episode, dict)
    assert isinstance(records, dict)
    assert isinstance(admission, dict)
    assert isinstance(verification, dict)
    assert isinstance(checks, dict)
    clean_fields = (
        "background_reactivations",
        "capture_failures",
        "corpus_failures",
        "infrastructure_failures",
        "policy_failures",
        "trace_failures",
    )
    if not (
        run.get("schema_version") == RUN_SCHEMA
        and manifest.get("schema_version") == MANIFEST_SCHEMA
        and run.get("run_id") == manifest.get("run_id")
        and schemas.get("frame") == FRAME_SCHEMA
        and schemas.get("transition") == TRANSITION_SCHEMA
        and metadata.get("code_commit") == collection_commit
        and metadata.get("difficulty") == 3
        and metadata.get("episode_unit") == "route"
        and metadata.get("expected_stages") == [1, 2, 3, 4, 5, 6]
        and planner.get("algorithm") == "source-hard4-paused-publication-v2"
        and planner.get("source_commitment") == "source-complete-hard-v1"
        and planner.get("factual_state_schema") == "th06-1.02h-offline-facts-v2"
        and planner.get("hard_horizon") == 4
        and planner.get("learner_feature_horizon") == 4
        and planner.get("minimum_collision_margin") == 0.35
        and planner.get("zero_margin_fallback") is False
        and report.get("repository_commit") == collection_commit
        and report.get("repository_worktree_clean") is True
        and report.get("diagnostic_rng_seed") is None
        and report.get("immutable_policy_state_equal") is True
        and manifest.get("complete") is True
        and manifest.get("stage_trajectory_complete") is True
        and manifest.get("dropped_records") == 0
        and episode.get("unit") == "route"
        and episode.get("complete") is True
        and outcome.get("stage_completed") is True
        and outcome.get("termination_reason") == "route-complete"
        and outcome.get("corpus_failure") is None
        and all(outcome.get(field) == 0 for field in clean_fields)
        and audit.get("bomb_events", 0) == 0
        and audit.get("integrity_errors") == []
        and admission.get("passes") is True
        and admission.get("checked_frames") == records.get("frames")
        and admission.get("error") is None
        and verification.get("passed") is True
        and checks
        and all(value is True for value in checks.values())
    ):
        raise ValueError("route does not pass strict data admission")
    _verify_shards(run_dir, manifest)
    options = tuple(iter_offline_options(run_dir))
    validate_offline_episode(options)
    behavior_policies = {option.behavior_policy_id for option in options}
    episode_ids = {option.episode_id for option in options}
    policy_seed = _seed_from_options(options)
    exploration_probability = _exploration_probability(options)
    physical_hits = sum(option.physical_hit_cost for option in options)
    factual_digest = row.get("digest")
    if not (
        behavior_policies == {POLICY_NAME}
        and len(episode_ids) == 1
        and policy_seed == row.get("policy_seed")
        and physical_hits == outcome.get("physical_hits")
        and physical_hits == audit.get("physical_hits")
        and isinstance(factual_digest, str)
        and len(factual_digest) == 64
    ):
        raise ValueError("route option identity/HIT conservation failed")
    return ({
        "episode_id": next(iter(episode_ids)),
        "run_dir": _relative(repository, run_dir),
        "report_path": _relative(repository, report_path),
        "audit_path": _relative(repository, audit_path),
        "run_sha256": row["run_sha256"],
        "manifest_sha256": row["manifest_sha256"],
        "report_sha256": row["report_sha256"],
        "audit_sha256": row["audit_sha256"],
        "factual_digest": factual_digest,
        "code_commit": collection_commit,
        "policy_seed": policy_seed,
        "exploration_probability": exploration_probability,
        "behavior_policy_id": POLICY_NAME,
        "options": len(options),
        "eligible_options": sum(option.eligible for option in options),
        "physical_hits": physical_hits,
        "controlled_hits": sum(option.controlled_hit_cost for option in options),
        "interstitial_hits": sum(
            option.interstitial_hit_cost for option in options
        ),
        "controlled_elapsed_frames": sum(option.elapsed_frames for option in options),
        "interstitial_elapsed_frames": sum(
            option.interstitial_elapsed_frames for option in options
        ),
    }, options)


def build_dataset_index(
    collection_paths: Iterable[Path],
    *,
    repository: Path,
) -> dict[str, object]:
    """Validate collection ledgers and create an algorithm-free data index."""
    collection_rows = []
    episode_rows = []
    for raw_path in collection_paths:
        collection_path = raw_path.resolve()
        relative = _relative(repository, collection_path)
        collection = _object(collection_path)
        episodes = collection.get("episodes")
        commit = collection.get("repository_commit")
        if not (
            collection.get("schema") == COLLECTION_SCHEMA
            and collection.get("complete") is True
            and collection.get("game_clock") == "original-retail-normal-speed"
            and collection.get("natural_rng") is True
            and collection.get("episode_unit") == "complete-route"
            and isinstance(commit, str)
            and len(commit) == 40
            and isinstance(episodes, list)
            and episodes
        ):
            raise ValueError(f"collection ledger is not admissible: {relative}")
        collection_rows.append({"path": relative, "sha256": _sha256(collection_path)})
        for row in episodes:
            if not isinstance(row, dict):
                raise ValueError("collection episode evidence is not an object")
            admitted, _options = _admit_episode(
                repository,
                row,
                collection_commit=commit,
            )
            episode_rows.append(admitted)
    episode_ids = [str(row["episode_id"]) for row in episode_rows]
    policy_seeds = [int(row["policy_seed"]) for row in episode_rows]
    digests = [row["factual_digest"] for row in episode_rows]
    if (
        len(episode_rows) < 2
        or len(set(episode_ids)) != len(episode_ids)
        or len(set(policy_seeds)) != len(policy_seeds)
        or len(set(digests)) != len(digests)
        or any(
            not isinstance(digest, str) or len(digest) != 64
            for digest in digests
        )
    ):
        raise ValueError("dataset repeats an episode, policy seed, or factual route")
    episode_rows.sort(key=lambda row: str(row["episode_id"]))
    return {
        "schema": DATASET_SCHEMA,
        "option_schema": OPTION_DATASET_SCHEMA,
        "frame_schema": FRAME_SCHEMA,
        "transition_schema": TRANSITION_SCHEMA,
        "episode_unit": "complete-route",
        "behavior_policy_id": POLICY_NAME,
        "collections": collection_rows,
        "episodes": episode_rows,
        "totals": {
            "episodes": len(episode_rows),
            "options": sum(int(row["options"]) for row in episode_rows),
            "eligible_options": sum(
                int(row["eligible_options"]) for row in episode_rows
            ),
            "physical_hits": sum(
                int(row["physical_hits"]) for row in episode_rows
            ),
            "controlled_hits": sum(
                int(row["controlled_hits"]) for row in episode_rows
            ),
            "interstitial_hits": sum(
                int(row["interstitial_hits"]) for row in episode_rows
            ),
        },
    }


def load_admitted_episodes(
    dataset_path: Path,
    *,
    repository: Path,
) -> tuple[tuple[OfflineOptionTransition, ...], ...]:
    """Revalidate immutable bytes before exposing episodes to any algorithm."""
    dataset = _object(dataset_path.resolve())
    rows = dataset.get("episodes")
    collections = dataset.get("collections")
    if not (
        dataset.get("schema") == DATASET_SCHEMA
        and dataset.get("option_schema") == OPTION_DATASET_SCHEMA
        and dataset.get("frame_schema") == FRAME_SCHEMA
        and dataset.get("transition_schema") == TRANSITION_SCHEMA
        and dataset.get("episode_unit") == "complete-route"
        and dataset.get("behavior_policy_id") == POLICY_NAME
        and isinstance(rows, list)
        and len(rows) >= 2
        and all(isinstance(row, dict) for row in rows)
        and isinstance(collections, list)
        and collections
    ):
        raise ValueError("offline dataset index contract mismatch")
    admitted_rows = []
    episodes = []
    for collection_binding in collections:
        if not isinstance(collection_binding, dict):
            raise ValueError("dataset collection binding is malformed")
        path = _relative_path(
            repository,
            collection_binding.get("path"),
            label="collection path",
        )
        _require_hash(
            path, collection_binding.get("sha256"), label="collection"
        )
        collection = _object(path)
        collection_episodes = collection.get("episodes")
        commit = collection.get("repository_commit")
        if not (
            collection.get("schema") == COLLECTION_SCHEMA
            and collection.get("complete") is True
            and collection.get("game_clock") == "original-retail-normal-speed"
            and collection.get("natural_rng") is True
            and collection.get("episode_unit") == "complete-route"
            and isinstance(commit, str)
            and len(commit) == 40
            and isinstance(collection_episodes, list)
        ):
            raise ValueError("bound collection ledger changed contract")
        for collection_row in collection_episodes:
            if not isinstance(collection_row, dict):
                raise ValueError("collection episode evidence is malformed")
            admitted, options = _admit_episode(
                repository,
                collection_row,
                collection_commit=commit,
            )
            admitted_rows.append(admitted)
            episodes.append(options)
    if sorted(
        admitted_rows, key=lambda row: str(row["episode_id"])
    ) != sorted(rows, key=lambda row: str(row.get("episode_id"))):
        raise ValueError("dataset index differs from re-admitted collection evidence")
    ids = set()
    seeds = set()
    digests = set()
    for row, options in zip(admitted_rows, episodes, strict=True):
        episode_id = options[0].episode_id
        seed = _seed_from_options(options)
        digest = row.get("factual_digest")
        if episode_id in ids or seed in seeds or digest in digests:
            raise ValueError("dataset repeats an episode, seed, or factual route")
        ids.add(episode_id)
        seeds.add(seed)
        digests.add(digest)
    return tuple(episodes)
