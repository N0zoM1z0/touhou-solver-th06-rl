"""Learner-independent compact arrays derived from factual Wine options."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile

from ..actions import ACTION_NAMES
from ..wine_corpus_registry import WineCorpusEntry
from .factual_options import load_factual_episode
from .feature_contract import (
    compact_actor_feature_names,
    richer_causal_context_feature_names,
)


DATASET_SCHEMA = "generation7-factual-proposal-itt-arrays-v3"


@dataclass(frozen=True)
class EpisodeArrayPaths:
    entry: WineCorpusEntry
    arrays: Path
    metadata: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_sha256(repository: Path) -> str:
    digest = hashlib.sha256()
    for relative in (
        "src/th06_rl/generation7/factual_options.py",
        "src/th06_rl/generation7/feature_contract.py",
        "src/th06_rl/generation7/offline_dataset.py",
        "src/th06_rl/hazard_representation.py",
        "src/th06_rl/learning_features.py",
        "src/th06_rl/th06/learning_adapter.py",
        "src/th06_rl/wine_transitions.py",
    ):
        digest.update((repository / relative).read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def episode_array_paths(
    entry: WineCorpusEntry,
    *,
    repository: Path,
    cache_root: Path,
) -> EpisodeArrayPaths:
    identity = hashlib.sha256(
        (
            f"{DATASET_SCHEMA}\0{entry.manifest_sha256}\0"
            f"{_contract_sha256(repository)}"
        ).encode()
    ).hexdigest()
    root = cache_root.resolve()
    return EpisodeArrayPaths(
        entry=entry,
        arrays=root / f"{identity}.npz",
        metadata=root / f"{identity}.json",
    )


def _valid(paths: EpisodeArrayPaths, *, contract_sha256: str) -> bool:
    if not paths.arrays.is_file() or not paths.metadata.is_file():
        if paths.arrays.exists() or paths.metadata.exists():
            raise ValueError(f"partial Generation-7 dataset cache: {paths.arrays}")
        return False
    metadata = json.loads(paths.metadata.read_text(encoding="utf-8"))
    if (
        metadata.get("schema") != DATASET_SCHEMA
        or metadata.get("manifest_sha256") != paths.entry.manifest_sha256
        or metadata.get("contract_sha256") != contract_sha256
        or metadata.get("arrays_sha256") != _sha256(paths.arrays)
    ):
        raise ValueError(f"invalid Generation-7 dataset cache: {paths.arrays}")
    return True


def prepare_episode_arrays(
    entry: WineCorpusEntry,
    *,
    repository: Path,
    cache_root: Path,
) -> tuple[EpisodeArrayPaths, bool]:
    """Materialize one hash-bound episode without changing corpus facts."""
    import numpy as np

    repository = repository.resolve()
    paths = episode_array_paths(
        entry,
        repository=repository,
        cache_root=cache_root,
    )
    contract_sha256 = _contract_sha256(repository)
    if _valid(paths, contract_sha256=contract_sha256):
        return paths, True
    episode = load_factual_episode(entry)
    offsets = [0]
    candidate_features = []
    candidate_action_indices = []
    probabilities = []
    factual_positions = []
    baseline_positions = []
    for option in episode.options:
        candidate_features.extend(option.candidate_features)
        candidate_action_indices.extend(
            ACTION_NAMES.index(action) for action in option.legal_actions
        )
        probabilities.extend(option.behavior_probabilities)
        factual_positions.append(option.factual_index)
        baseline_positions.append(option.baseline_index)
        offsets.append(len(candidate_features))
    arrays = {
        "schema": np.asarray(DATASET_SCHEMA),
        "episode_id": np.asarray(episode.episode_id),
        "source_id": np.asarray(episode.source_id),
        "cohort_id": np.asarray(episode.cohort_id),
        "transition_schema": np.asarray(episode.transition_schema),
        "stage": np.asarray(episode.stage, dtype=np.int16),
        "manifest_hits": np.asarray(episode.manifest_hits, dtype=np.int16),
        "pre_option_hits": np.asarray(episode.pre_option_hits, dtype=np.int16),
        "feature_names": np.asarray(compact_actor_feature_names()),
        "causal_context_feature_names": np.asarray(
            richer_causal_context_feature_names()
        ),
        "candidate_features": np.asarray(candidate_features, dtype=np.float32),
        "candidate_action_indices": np.asarray(
            candidate_action_indices, dtype=np.int16
        ),
        "boundary_executed_action_indices": np.asarray(
            [
                ACTION_NAMES.index(option.boundary_executed_action)
                for option in episode.options
            ],
            dtype=np.int16,
        ),
        "proposal_complied": np.asarray(
            [option.complied for option in episode.options], dtype=np.bool_
        ),
        "causal_context_features": np.asarray(
            [option.causal_context_features for option in episode.options],
            dtype=np.float32,
        ),
        "offsets": np.asarray(offsets, dtype=np.int64),
        "factual_positions": np.asarray(factual_positions, dtype=np.int16),
        "baseline_positions": np.asarray(baseline_positions, dtype=np.int16),
        "behavior_probabilities": np.asarray(probabilities, dtype=np.float64),
        "hit_costs": np.asarray(
            [option.hit_cost for option in episode.options], dtype=np.int16
        ),
        "durations": np.asarray(
            [option.duration_frames for option in episode.options], dtype=np.int32
        ),
    }
    paths.arrays.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{paths.arrays.name}.", suffix=".tmp", dir=paths.arrays.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as output:
            np.savez(output, **arrays)
            output.flush()
            os.fsync(output.fileno())
        arrays_sha256 = _sha256(temporary)
        os.replace(temporary, paths.arrays)
    finally:
        temporary.unlink(missing_ok=True)
    metadata = {
        "schema": DATASET_SCHEMA,
        "manifest_sha256": entry.manifest_sha256,
        "contract_sha256": contract_sha256,
        "arrays_sha256": arrays_sha256,
        "episode_id": episode.episode_id,
        "options": len(episode.options),
        "proposal_assignments": len(episode.options),
        "complied_assignments": sum(
            option.complied for option in episode.options
        ),
        "candidate_rows": len(candidate_features),
        "feature_count": len(compact_actor_feature_names()),
        "causal_context_feature_count": len(
            richer_causal_context_feature_names()
        ),
    }
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{paths.metadata.name}.", suffix=".tmp", dir=paths.metadata.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(metadata, output, indent=2, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, paths.metadata)
    finally:
        temporary.unlink(missing_ok=True)
    return paths, False


def load_episode_arrays(paths: EpisodeArrayPaths) -> dict[str, object]:
    import numpy as np

    with np.load(paths.arrays, allow_pickle=False) as raw:
        result = {name: raw[name] for name in raw.files}
    if (
        str(result.get("schema")) != DATASET_SCHEMA
        or tuple(map(str, result.get("feature_names", ())))
        != compact_actor_feature_names()
        or tuple(map(str, result.get("causal_context_feature_names", ())))
        != richer_causal_context_feature_names()
    ):
        raise ValueError("Generation-7 episode array schema drifted")
    offsets = result["offsets"]
    features = result["candidate_features"]
    probabilities = result["behavior_probabilities"]
    action_indices = result["candidate_action_indices"]
    factual = result["factual_positions"]
    baseline = result["baseline_positions"]
    executed = result["boundary_executed_action_indices"]
    complied = result["proposal_complied"]
    option_count = len(offsets) - 1
    if (
        option_count <= 0
        or len(factual) != option_count
        or len(baseline) != option_count
        or len(result["hit_costs"]) != option_count
        or len(result["boundary_executed_action_indices"]) != option_count
        or len(result["proposal_complied"]) != option_count
        or bool(((executed < 0) | (executed >= len(ACTION_NAMES))).any())
        or result["causal_context_features"].shape
        != (option_count, len(richer_causal_context_feature_names()))
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(features)
        or len(probabilities) != len(features)
        or len(action_indices) != len(features)
    ):
        raise ValueError("Generation-7 episode arrays are inconsistent")
    for index in range(option_count):
        start, stop = int(offsets[index]), int(offsets[index + 1])
        if (
            stop <= start
            or not 0 <= int(factual[index]) < stop - start
            or not 0 <= int(baseline[index]) < stop - start
            or not abs(float(probabilities[start:stop].sum()) - 1.0) <= 1e-9
        ):
            raise ValueError("Generation-7 candidate group is invalid")
    factual_rows = offsets[:-1] + factual
    if not (complied == (executed == action_indices[factual_rows])).all():
        raise ValueError("proposal compliance metadata is inconsistent")
    return result


def proximal_targets(hit_costs, horizon: int):
    import numpy as np

    if horizon <= 0:
        raise ValueError("proximal horizon must be positive")
    costs = np.asarray(hit_costs, dtype=np.float64)
    prefix = np.concatenate((np.zeros(1, dtype=np.float64), np.cumsum(costs)))
    indices = np.arange(len(costs))
    return prefix[np.minimum(len(costs), indices + horizon)] - prefix[indices]
