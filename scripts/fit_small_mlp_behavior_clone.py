#!/usr/bin/env python3
"""Fit the frozen small-MLP ablation from complete Wine episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.fit_behavior_clone import _atomic_new_json, _clean_commit, _sha256  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402


POLICY_PLUGIN = REPOSITORY / "src/th06_rl/policies/small_mlp_behavior_clone.py"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-run", type=Path, action="append", required=True)
    parser.add_argument("--validation-run", type=Path, action="append", required=True)
    parser.add_argument("--linear-comparator-state", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-samples", type=int, default=2_000)
    parser.add_argument("--calibration-tolerance", type=float, default=0.02)
    parser.add_argument("--minimum-updates", type=int, default=100)
    parser.add_argument("--relative-gradient-l2-tolerance", type=float, default=0.01)
    parser.add_argument("--hidden-width", type=int, default=32)
    parser.add_argument("--max-rows", type=int, default=2_000_000)
    args = parser.parse_args()

    affinity = enforce_training_cpu_affinity()
    commit = _clean_commit()
    from th06_rl.mlp_bc_training import fit_small_mlp_behavior_clone

    state = fit_small_mlp_behavior_clone(
        tuple(args.train_run),
        tuple(args.validation_run),
        linear_comparator_state=args.linear_comparator_state,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        l2=args.l2,
        seed=args.seed,
        bootstrap_samples=args.bootstrap_samples,
        calibration_tolerance=args.calibration_tolerance,
        minimum_updates=args.minimum_updates,
        relative_gradient_l2_tolerance=args.relative_gradient_l2_tolerance,
        hidden_width=args.hidden_width,
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
