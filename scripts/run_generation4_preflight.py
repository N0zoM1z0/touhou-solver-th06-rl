#!/usr/bin/env python3
"""Hard-gate Generation 4 with causal, native-latency, and Wine-v10 smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.hazard_representation import (  # noqa: E402
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
)
from th06_rl.offline import ACTION_NAMES  # noqa: E402
from th06_rl.policies.autonomous_sequential_r_critic import (  # noqa: E402
    AutonomousSequentialRCriticPolicy,
)
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV  # noqa: E402
from th06_rl.policies.propensity_aware_option_exploration import (  # noqa: E402
    INCUMBENT_MASS,
    INFORMATION_MASS,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA as EXPLORATION_STATE_SCHEMA,
    UNIFORM_MASS,
)
from th06_rl.policy_api import PolicyContext  # noqa: E402
from th06_rl.sequential_learning import (  # noqa: E402
    audit_propensity_wine_smoke,
    audit_sequential_causal_fixture,
    fit_sequential_causal_fixture,
)
from th06_rl.th06.learning_adapter import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


SCHEMA = "autonomous-generation-4-preflight-v1"
SEEDS = REPOSITORY / "config/autonomous_generation4_seeds.json"
HISTORICAL = REPOSITORY / "config/autonomous_generation4_historical.json"
HISTORICAL_CORPUS = (
    REPOSITORY / "artifacts/autonomous-wine-generation-3/collection-corpus"
)
EXPLORATION_PLUGIN = (
    REPOSITORY / "src/th06_rl/policies/propensity_aware_option_exploration.py"
)
CONTRACT_FILES = (
    SEEDS,
    HISTORICAL,
    REPOSITORY / "src/th06_rl/advantage_learning.py",
    REPOSITORY / "src/th06_rl/sequential_learning.py",
    REPOSITORY / "src/th06_rl/corpus.py",
    REPOSITORY / "src/th06_rl/hazard_representation.py",
    EXPLORATION_PLUGIN,
    REPOSITORY / "src/th06_rl/policies/autonomous_sequential_r_critic.py",
    REPOSITORY / "src/th06_rl/th06/controller.py",
    REPOSITORY / "src/th06_rl/th06/learning_adapter.py",
    REPOSITORY / "scripts/fit_sequential_r_critic.py",
    REPOSITORY / "scripts/shadow_sequential_r_critic.py",
    REPOSITORY / "scripts/authorize_sequential_r_canary.py",
    REPOSITORY / "scripts/run_autonomous_learning_v4.py",
)
NATIVE_DECISIONS = 1_200
MAXIMUM_P95_MS = 4.0


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _contract_sha256() -> str:
    digest = hashlib.sha256()
    for path in CONTRACT_FILES:
        digest.update(str(path.relative_to(REPOSITORY)).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
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


def _archive_incomplete(path: Path) -> None:
    if not path.exists():
        return
    for index in range(1, 1_000):
        destination = path.with_name(f"{path.name}.incomplete-{index:03d}")
        if not destination.exists():
            path.rename(destination)
            return
    raise RuntimeError(f"too many incomplete artifacts beside {path}")


def _validate_seed_contract() -> dict[str, object]:
    seeds = _object(SEEDS)
    collection = seeds.get("collection")
    canary = seeds.get("canary")
    if (
        seeds.get("schema")
        != "autonomous-generation-4-precommitted-seeds-v1"
        or seeds.get("generation_seed") != 260812
        or not isinstance(collection, list)
        or len(collection) != 16
        or [row.get("episode") for row in collection if isinstance(row, dict)]
        != list(range(16))
        or not isinstance(canary, list)
        or len(canary) != 9
        or [(row.get("round"), row.get("pair")) for row in canary if isinstance(row, dict)]
        != [(round_index, pair) for round_index in range(1, 4) for pair in range(3)]
    ):
        raise ValueError("Generation-4 seed schedule is invalid")
    values = [int(row["game_rng_seed"]) for row in collection + canary]
    policy = [int(row["policy_seed"]) for row in collection]
    if (
        len(values) != len(set(values))
        or len(policy) != len(set(policy))
        or any(not 0 < value < 2**16 for value in values + policy)
    ):
        raise ValueError("Generation-4 seed schedule is not unique/bounded")
    return seeds


def _validate_historical_contract() -> list[Path]:
    contract = _object(HISTORICAL)
    rows = contract.get("episodes")
    if (
        contract.get("schema")
        != "autonomous-generation-4-frozen-historical-corpus-v1"
        or not isinstance(rows, list)
        or len(rows) != 13
        or [row.get("episode") for row in rows if isinstance(row, dict)]
        != list(range(13))
    ):
        raise ValueError("Generation-4 historical corpus contract is invalid")
    result = []
    for row in rows:
        path = HISTORICAL_CORPUS / str(row["run_id"])
        if (
            not (path / "manifest.json").is_file()
            or _sha256(path / "manifest.json") != row["manifest_sha256"]
        ):
            raise ValueError(f"frozen historical episode differs: {path}")
        result.append(path)
    return result


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)]


def _latency_context(frame: int) -> PolicyContext:
    observation = tuple((name, 0.0) for name in OBSERVATION_FEATURE_NAMES)
    action_features = tuple(
        (action, tuple((name, 0.0) for name in ACTION_FEATURE_NAMES))
        for action in ACTION_NAMES
    )
    hazards = []
    for index in range(256):
        primitive = [0.0] * len(HAZARD_PRIMITIVE_FEATURE_NAMES)
        primitive[index % len(primitive)] = (index % 17) / 16.0
        hazards.append(tuple(primitive))
    return PolicyContext(
        frame=frame,
        scope=(0, 0, 0, 6),
        source_context="preflight-hidden",
        baseline_action="stay",
        locally_admissible_actions=tuple(ACTION_NAMES),
        player_x=192.0,
        player_y=400.0,
        power=128,
        bullet_count=256,
        laser_count=0,
        hard_action_count=len(ACTION_NAMES),
        exploration_rate=0.0,
        current_action="stay",
        observation_features=observation,
        action_features=action_features,
        hazard_primitives=tuple(hazards),
        history_features=tuple((name, 0.0) for name in HISTORY_FEATURE_NAMES),
    )


def _native_latency_smoke(
    state: dict[str, object],
    scorer: Path,
) -> dict[str, object]:
    prior = os.environ.get(NATIVE_SCORER_ENV)
    try:
        os.environ[NATIVE_SCORER_ENV] = str(scorer.resolve())
        policy = AutonomousSequentialRCriticPolicy()
        policy.import_state(state)
    finally:
        if prior is None:
            os.environ.pop(NATIVE_SCORER_ENV, None)
        else:
            os.environ[NATIVE_SCORER_ENV] = prior
    timings = []
    context = _latency_context(0)
    for index in range(NATIVE_DECISIONS):
        started = time.perf_counter()
        policy.decide(context)
        timings.append((time.perf_counter() - started) * 1_000.0)
    metrics = policy.metrics()
    p95 = _p95(timings)
    gates = {
        "native_batch_backend": metrics["scorer_backend"] == "native-batch",
        "complete_seven_member_population": metrics["population_members"] == 7,
        "full_128_trees_per_member": metrics["trees_per_member"] == 128,
        "exact_decision_count": metrics["decisions"] == NATIVE_DECISIONS,
        "p95_at_most_four_ms": p95 <= MAXIMUM_P95_MS,
        "zero_controller_deadline_misses": metrics["controller_deadline_misses"] == 0,
    }
    return {
        "schema": "autonomous-generation-4-native-population-smoke-v1",
        "decisions": NATIVE_DECISIONS,
        "legal_actions_per_decision": len(ACTION_NAMES),
        "hazard_primitives_per_decision": 256,
        "mean_ms": statistics.fmean(timings),
        "p95_ms": p95,
        "max_ms": max(timings),
        "policy_metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
    }


def _validate_cached(
    root: Path,
    *,
    seconds: float,
    wine_scorer: Path,
    host_scorer: Path,
) -> dict[str, object]:
    state = _object(root / "preflight.json")
    expected = {
        "schema": SCHEMA,
        "passed": True,
        "evidence_eligible": False,
        "wine_seconds": seconds,
        "seed_schedule_sha256": _sha256(SEEDS),
        "historical_contract_sha256": _sha256(HISTORICAL),
        "preflight_contract_sha256": _contract_sha256(),
        "wine_native_scorer_sha256": _sha256(wine_scorer),
        "host_native_scorer_sha256": _sha256(host_scorer),
    }
    if any(state.get(key) != value for key, value in expected.items()):
        raise ValueError("cached Generation-4 preflight contract differs")
    for name in (
        "causal-smoke.json",
        "native-population-smoke.json",
        "wine-smoke-audit.json",
    ):
        if _object(root / name).get("passed") is not True:
            raise ValueError(f"cached Generation-4 gate failed: {name}")
    return state


def run(
    root: Path,
    *,
    threads: int,
    seconds: float,
    wine_scorer: Path,
    host_scorer: Path,
) -> dict[str, object]:
    root = root.resolve()
    state_path = root / "preflight.json"
    if state_path.is_file():
        try:
            return _validate_cached(
                root,
                seconds=seconds,
                wine_scorer=wine_scorer,
                host_scorer=host_scorer,
            )
        except (FileNotFoundError, TypeError, ValueError):
            _archive_incomplete(root)
    elif root.exists():
        _archive_incomplete(root)
    root.mkdir(parents=True)
    seeds = _validate_seed_contract()
    historical = _validate_historical_contract()

    samples, policy_state = fit_sequential_causal_fixture(
        threads=threads,
        native_scorer_sha256=_sha256(wine_scorer),
        compatible_native_scorer_sha256=(_sha256(host_scorer),),
    )
    causal = audit_sequential_causal_fixture(samples, policy_state)
    _atomic_json(root / "causal-smoke.json", causal)
    if causal.get("passed") is not True:
        raise RuntimeError("Generation-4 sequential causal smoke failed")
    native = _native_latency_smoke(policy_state, host_scorer)
    _atomic_json(root / "native-population-smoke.json", native)
    if native.get("passed") is not True:
        raise RuntimeError("Generation-4 full-population native smoke failed")

    exploration_state = root / "exploration-smoke-policy.json"
    _atomic_json(exploration_state, {
        "schema": EXPLORATION_STATE_SCHEMA,
        "policy_seed": int(seeds["generation_seed"]),
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
        "mixture": {
            "incumbent": INCUMBENT_MASS,
            "uniform": UNIFORM_MASS,
            "information": INFORMATION_MASS,
        },
    })
    artifact = root / "wine"
    corpus = root / "wine-corpus"
    command = [
        sys.executable,
        str(REPOSITORY / "scripts/run_wine_retail.py"),
        "--practice-stage", "6",
        "--difficulty", "lunatic",
        "--seconds", str(seconds),
        "--artifact-dir", str(artifact),
        "--policy-plugin", str(EXPLORATION_PLUGIN),
        "--policy-state", str(exploration_state),
        "--immutable-policy",
        "--exploration-rate", "0",
        "--option-smoke-corpus-root", str(corpus),
        "--diagnostic-rng-seed", hex(int(seeds["smoke_game_rng_seed"])),
    ]
    completed = subprocess.run(command, cwd=REPOSITORY, check=False)
    report = _object(artifact / "report.json")
    trace = report.get("trace")
    run_ids = trace.get("corpus_run_ids") if isinstance(trace, dict) else None
    if (
        completed.returncode != 0
        or report.get("error") is not None
        or report.get("evaluation_mode") != "fixed-rng-option-smoke-non-evidence"
        or report.get("diagnostic_rng_seed") != seeds["smoke_game_rng_seed"]
        or report.get("immutable_policy_state_equal") is not True
        or report.get("leftover_prefix_processes") != []
        or int(report.get("controller_returncode", -1)) != 0
        or not isinstance(run_ids, list)
        or len(run_ids) != 1
    ):
        raise RuntimeError("Generation-4 short Wine runner provenance failed")
    run_dir = corpus / str(run_ids[0])
    wine_audit = audit_propensity_wine_smoke(run_dir)
    _atomic_json(root / "wine-smoke-audit.json", wine_audit)
    if wine_audit.get("passed") is not True:
        raise RuntimeError("Generation-4 Wine v10 propensity audit failed")

    state = {
        "schema": SCHEMA,
        "passed": True,
        "evidence_eligible": False,
        "wine_seconds": seconds,
        "seed_schedule_sha256": _sha256(SEEDS),
        "historical_contract_sha256": _sha256(HISTORICAL),
        "frozen_historical_episodes": len(historical),
        "preflight_contract_sha256": _contract_sha256(),
        "wine_native_scorer_sha256": _sha256(wine_scorer),
        "host_native_scorer_sha256": _sha256(host_scorer),
        "causal_smoke": str(root / "causal-smoke.json"),
        "native_population_smoke": str(root / "native-population-smoke.json"),
        "wine_smoke_audit": str(root / "wine-smoke-audit.json"),
        "wine_corpus_run": str(run_dir),
    }
    _atomic_json(state_path, state)
    return state


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--wine-seconds", type=float, default=45.0)
    parser.add_argument(
        "--wine-native-scorer",
        type=Path,
        default=REPOSITORY / "build/native-win32-fully-static/libth06_rl_ranker.dll",
    )
    parser.add_argument(
        "--host-native-scorer",
        type=Path,
        default=REPOSITORY / "build/native/libth06_rl_ranker.so",
    )
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("thread count must be positive")
    if not 30.0 <= args.wine_seconds <= 120.0:
        parser.error("Wine smoke must be between 30 and 120 seconds")
    if not args.wine_native_scorer.is_file() or not args.host_native_scorer.is_file():
        parser.error("native scorer library is absent")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.output_root,
        threads=args.threads,
        seconds=args.wine_seconds,
        wine_scorer=args.wine_native_scorer.resolve(),
        host_scorer=args.host_native_scorer.resolve(),
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
