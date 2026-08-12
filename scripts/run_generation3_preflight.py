#!/usr/bin/env python3
"""Hard-gate long Generation-3 collection with causal and short Wine smokes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.advantage_learning import (  # noqa: E402
    audit_wine_option_smoke,
    run_causal_recovery_smoke,
)
from th06_rl.policies.safe_option_exploration import (  # noqa: E402
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA as OPTION_STATE_SCHEMA,
)


SCHEMA = "autonomous-generation-3-preflight-v1"
SEEDS = REPOSITORY / "config/autonomous_generation3_seeds.json"
OPTION_PLUGIN = REPOSITORY / "src/th06_rl/policies/safe_option_exploration.py"
CONTRACT_FILES = (
    REPOSITORY / "src/th06_rl/advantage_learning.py",
    REPOSITORY / "src/th06_rl/corpus.py",
    REPOSITORY / "src/th06_rl/hazard_representation.py",
    REPOSITORY / "src/th06_rl/policies/safe_option_exploration.py",
    REPOSITORY / "src/th06_rl/th06/controller.py",
)


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
        digest.update(str(path.relative_to(REPOSITORY)).encode("utf-8"))
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
    for index in range(1, 1000):
        archived = path.with_name(f"{path.name}.incomplete-{index:03d}")
        if not archived.exists():
            path.rename(archived)
            return
    raise RuntimeError(f"too many incomplete smoke artifacts beside {path}")


def _validate_seed_contract() -> dict[str, object]:
    seeds = _object(SEEDS)
    collection = seeds.get("collection")
    canary = seeds.get("canary")
    collection_rows = [row for row in collection or () if isinstance(row, dict)]
    if (
        seeds.get("schema") != "autonomous-generation-3-precommitted-seeds-v1"
        or seeds.get("generation_seed") != 260812
        or not isinstance(collection, list)
        or len(collection) != 24
        or [row.get("episode") for row in collection_rows] != list(range(24))
        or not isinstance(canary, list)
        or len(canary) != 12
        or len({row.get("game_rng_seed") for row in collection_rows}) != 24
    ):
        raise ValueError("Generation-3 precommitted seed schedule is invalid")
    return seeds


def _validate_cached(root: Path, *, seconds: float) -> dict[str, object]:
    state = _object(root / "preflight.json")
    if (
        state.get("schema") != SCHEMA
        or state.get("passed") is not True
        or state.get("evidence_eligible") is not False
        or state.get("wine_seconds") != seconds
        or state.get("seed_schedule_sha256") != _sha256(SEEDS)
        or state.get("preflight_contract_sha256") != _contract_sha256()
    ):
        raise ValueError("cached Generation-3 preflight does not match the contract")
    for name in ("causal-smoke.json", "wine-smoke-audit.json"):
        report = _object(root / name)
        if report.get("passed") is not True:
            raise ValueError(f"cached preflight gate is not passing: {name}")
    return state


def run(root: Path, *, threads: int, seconds: float) -> dict[str, object]:
    root = root.resolve()
    state_path = root / "preflight.json"
    if state_path.is_file():
        return _validate_cached(root, seconds=seconds)
    if root.exists():
        _archive_incomplete(root)
    root.mkdir(parents=True)
    seeds = _validate_seed_contract()

    causal = run_causal_recovery_smoke(threads=threads)
    _atomic_json(root / "causal-smoke.json", causal)
    if causal.get("passed") is not True:
        raise RuntimeError("Generation-3 causal learner smoke failed")

    option_state = root / "option-smoke-policy.json"
    _atomic_json(option_state, {
        "schema": OPTION_STATE_SCHEMA,
        "policy_seed": int(seeds["generation_seed"]),
        "exploration_probability": 0.10,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
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
        "--policy-plugin", str(OPTION_PLUGIN),
        "--policy-state", str(option_state),
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
        report.get("error") is not None
        or report.get("evaluation_mode") != "fixed-rng-option-smoke-non-evidence"
        or report.get("diagnostic_rng_seed") != seeds["smoke_game_rng_seed"]
        or report.get("immutable_policy_state_equal") is not True
        or report.get("leftover_prefix_processes") != []
        or int(report.get("controller_returncode", -1)) != 0
        or completed.returncode != 0
        or not isinstance(run_ids, list)
        or len(run_ids) != 1
    ):
        raise RuntimeError("short Wine option smoke runner failed its provenance gate")
    run_dir = corpus / str(run_ids[0])
    audit = audit_wine_option_smoke(run_dir)
    _atomic_json(root / "wine-smoke-audit.json", audit)
    if audit.get("passed") is not True:
        raise RuntimeError("short Wine option corpus failed its wiring audit")

    state = {
        "schema": SCHEMA,
        "passed": True,
        "evidence_eligible": False,
        "wine_seconds": seconds,
        "seed_schedule": str(SEEDS),
        "seed_schedule_sha256": _sha256(SEEDS),
        "preflight_contract_sha256": _contract_sha256(),
        "causal_smoke": str(root / "causal-smoke.json"),
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
    args = parser.parse_args(argv)
    if args.threads <= 0:
        parser.error("thread count must be positive")
    if args.wine_seconds < 30.0 or args.wine_seconds > 120.0:
        parser.error("Wine smoke must be between 30 and 120 seconds")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run(
        args.output_root,
        threads=args.threads,
        seconds=args.wine_seconds,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
