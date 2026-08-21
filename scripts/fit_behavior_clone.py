#!/usr/bin/env python3
"""Fit the first frozen behavior-cloning artifact from complete Wine episodes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from th06_rl.resource_control import enforce_training_cpu_affinity


REPOSITORY = Path(__file__).resolve().parents[1]
POLICY_PLUGIN = REPOSITORY / "src/th06_rl/policies/linear_behavior_clone.py"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _clean_commit() -> str:
    status = subprocess.run(
        ("git", "status", "--porcelain", "--untracked-files=all"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError("evidence fit requires a clean committed worktree")
    return subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=REPOSITORY,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _atomic_new_json(path: Path, value: dict[str, object]) -> None:
    path = path.resolve()
    if path.exists():
        raise FileExistsError(f"refusing to replace immutable fit artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, sort_keys=True, separators=(",", ":"), allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, action="append", required=True)
    parser.add_argument("--validation-run", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--calibration-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-updates", type=int, default=0)
    parser.add_argument("--relative-gradient-l2-tolerance", type=float)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    affinity = enforce_training_cpu_affinity()
    commit = _clean_commit()
    from th06_rl.bc_training import fit_behavior_clone

    state = fit_behavior_clone(
        tuple(args.train_run),
        tuple(args.validation_run),
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        calibration_tolerance=args.calibration_tolerance,
        minimum_updates=args.minimum_updates,
        relative_gradient_l2_tolerance=args.relative_gradient_l2_tolerance,
        max_rows=args.max_rows,
        code_commit=commit,
        policy_plugin_sha256=_sha256(POLICY_PLUGIN),
    )
    _atomic_new_json(args.output, state)
    print(json.dumps({
        "output": str(args.output.resolve()),
        "policy_plugin": str(POLICY_PLUGIN),
        "policy_id": state["policy_id"],
        "learnability_gate_passed": state["fit"]["learnability_gate_passed"],
        "train_rows": state["fit"]["train"]["rows"],
        "validation_rows": state["fit"]["validation"]["rows"],
        "optimization": state["fit"]["optimization"],
        "cpu_affinity": list(affinity.effective),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
