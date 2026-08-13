#!/usr/bin/env python3
"""Run Generation-7 factual action-effect gates on the immutable registry."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.actions import ACTION_NAMES  # noqa: E402
from th06_rl.generation7.effect_diagnostics import (  # noqa: E402
    action_specific_ipw_effect,
    binary_ipw_effect,
    bootstrap_sign_stability,
    effect_episode,
    permutation_null,
    synthetic_delayed_effect,
)
from th06_rl.generation7.factual_options import load_factual_episode  # noqa: E402
from th06_rl.generation7.feature_contract import (  # noqa: E402
    DEFAULT_FEATURE_CATALOG,
    FeatureUse,
    compact_actor_feature_names,
)
from th06_rl.generation7.objectives import extreme_logit_smoke  # noqa: E402
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


SCHEMA = "generation7-identifiability-report-v1"
CACHE_SCHEMA = "generation7-effect-episode-cache-v1"
MAX_WORKERS = 16


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_effect_episode(entry):
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    return effect_episode(load_factual_episode(entry))


def _cache_identity(registry: Path, contract: Path) -> str:
    digest = hashlib.sha256()
    for path in (
        registry,
        contract,
        REPOSITORY / "src/th06_rl/generation7/factual_options.py",
        REPOSITORY / "src/th06_rl/generation7/feature_contract.py",
        REPOSITORY / "src/th06_rl/hazard_representation.py",
        REPOSITORY / "src/th06_rl/wine_transitions.py",
        REPOSITORY / "src/th06_rl/learning_features.py",
        REPOSITORY / "src/th06_rl/th06/learning_adapter.py",
    ):
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(payload)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _episodes(entries, *, workers: int, cache_root: Path, identity: str):
    payload_path = cache_root / f"{identity}.pickle"
    metadata_path = cache_root / f"{identity}.json"
    if payload_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema") == CACHE_SCHEMA
            and metadata.get("identity") == identity
            and metadata.get("payload_sha256") == _sha256(payload_path)
        ):
            with payload_path.open("rb") as source:
                episodes = pickle.load(source)  # noqa: S301 - bound local cache
            if isinstance(episodes, tuple) and len(episodes) == len(entries):
                return episodes, True
        raise ValueError("Generation-7 effect cache is invalid")
    with ProcessPoolExecutor(max_workers=workers) as executor:
        episodes = tuple(executor.map(_load_effect_episode, entries))
    payload = pickle.dumps(episodes, protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "schema": CACHE_SCHEMA,
        "identity": identity,
        "episodes": len(episodes),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
    }
    _atomic_write(payload_path, payload)
    _atomic_write(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    return episodes, False


def _strata(episodes):
    result = defaultdict(list)
    for episode in episodes:
        result[f"source:{episode.source_id}"].append(episode)
        result[f"stage:{episode.stage}"].append(episode)
    return {
        name: {
            "episodes": len(rows),
            "horizon_8": binary_ipw_effect(tuple(rows), horizon=8),
            "horizon_32": binary_ipw_effect(tuple(rows), horizon=32),
        }
        for name, rows in sorted(result.items())
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--registry",
        type=Path,
        default=REPOSITORY / "config/wine_corpus_registry.json",
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=REPOSITORY / "config/generation7_offline_contract.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPOSITORY / "artifacts/generation7-offline/identifiability.json",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=REPOSITORY / "artifacts/cache/generation7-effect-episodes",
    )
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--permutations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.workers <= MAX_WORKERS or args.permutations < 20:
        parser.error("workers must be 1..16 and permutations at least 20")
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    if (
        contract.get("schema") != "generation7-offline-contract-v1"
        or contract.get("wine_outcome_facing_authorized") is not False
        or contract.get("new_collection_authorized") is not False
    ):
        raise ValueError("Generation-7 offline contract drifted")
    _registry, all_entries = load_wine_corpus_registry(
        args.registry, repository=REPOSITORY
    )
    required = frozenset(contract["required_corpus_capabilities"])
    entries = select_wine_corpora(
        all_entries,
        required_capabilities=required,
    )
    identity = _cache_identity(args.registry, args.contract)
    episodes, cache_hit = _episodes(
        entries,
        workers=args.workers,
        cache_root=args.cache_root,
        identity=identity,
    )
    horizons = tuple(map(int, contract["proximal_horizons_options"]))
    binary = {
        str(horizon): binary_ipw_effect(episodes, horizon=horizon)
        for horizon in horizons
    }
    seeds = contract["seeds"]
    bootstrap = bootstrap_sign_stability(
        episodes,
        horizon=32,
        replicates=int(contract["bootstrap_replicates"]),
        seed=int(seeds["bootstrap"]),
    )
    nulls = {
        kind: permutation_null(
            episodes,
            horizon=32,
            replicates=args.permutations,
            seed=int(seeds["null"]) + index,
            kind=kind,
        )
        for index, kind in enumerate(("action", "reward-suffix"))
    }
    synthetic = synthetic_delayed_effect(
        episodes=80,
        rows_per_episode=200,
        delay=4,
        seed=int(seeds["synthetic"]),
    )
    action_specific = {
        action: {
            str(horizon): action_specific_ipw_effect(
                episodes,
                action=action,
                horizon=horizon,
            )
            for horizon in (8, 32)
        }
        for action in ACTION_NAMES
    }
    DEFAULT_FEATURE_CATALOG.lint(
        compact_actor_feature_names(),
        use=FeatureUse.ACTOR,
        require_online=True,
    )
    proper = extreme_logit_smoke()
    counts = Counter()
    for episode in episodes:
        counts["episodes"] += 1
        counts["options"] += len(episode.rows)
        counts["hits"] += sum(row.hit_cost for row in episode.rows)
        counts["nonbaseline"] += sum(row.treated for row in episode.rows)
    gates = {
        "feature_availability_lint": True,
        "extreme_logit_proper_loss": proper["passes"],
        "factual_action_permutation_null": nulls["action"]["passes"],
        "reward_suffix_permutation_null": nulls["reward-suffix"]["passes"],
        "synthetic_delayed_effect": synthetic["passes"],
        "episode_bootstrap_sign_stability": (
            float(bootstrap["same_sign_fraction"]) >= 0.80
        ),
        "cross_source_stage_direction_report": True,
        "exact_policy_direct_dr_fqe_crosscheck": False,
    }
    report = {
        "schema": SCHEMA,
        "evidence_eligible": False,
        "wine_outcome_facing_authorized": False,
        "registry_sha256": _sha256(args.registry),
        "contract_sha256": _sha256(args.contract),
        "cache": {"identity": identity, "hit": cache_hit},
        "workers": args.workers,
        "counts": dict(counts),
        "binary_ipw": binary,
        "bootstrap": bootstrap,
        "permutation_nulls": nulls,
        "synthetic_delayed_effect": synthetic,
        "strata": _strata(episodes),
        "action_specific": action_specific,
        "proper_objective": proper,
        "gates": gates,
        "passes_current_layer": all(gates.values()),
        "next_required_layer": "exact-policy-direct-dr-fqe-crosscheck",
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if all(value for name, value in gates.items() if name != "exact_policy_direct_dr_fqe_crosscheck") else 1


if __name__ == "__main__":
    raise SystemExit(main())
