#!/usr/bin/env python3
"""Run Generation-6 action-centered IQL actor causal/null contracts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.iql_actor_learning import run_iql_actor_causal_smoke  # noqa: E402
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace smoke report: {args.output}")
    affinity = enforce_training_cpu_affinity(args.threads)
    report = run_iql_actor_causal_smoke(threads=args.threads)
    report["resource_contract"] = affinity.as_dict()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": report["passed"],
        "beneficial_overrides": report["beneficial_overrides"],
        "null_overrides": report["null_overrides"],
        "actor_member_mean_logit_effects": (
            report["actor_member_mean_logit_effects"]
        ),
    }, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
