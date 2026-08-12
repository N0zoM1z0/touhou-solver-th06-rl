#!/usr/bin/env python3
"""Replay a Generation-5 native population on exact factual Wine options."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.shadow_sequential_r_critic import shadow as _shadow  # noqa: E402
from th06_rl.implicit_learning import (  # noqa: E402
    POPULATION_MEMBERS,
    Q_TREES,
    STATE_SCHEMA,
)
from th06_rl.policies.autonomous_supported_implicit_q import (  # noqa: E402
    AutonomousSupportedImplicitQPolicy,
)


SCHEMA = "autonomous-generation-5-native-shadow-audit-v1"


def shadow(
    state_path: Path,
    run_dirs: list[Path],
    *,
    native_scorer: Path,
    maximum_p95_ms: float = 4.0,
) -> dict[str, object]:
    return _shadow(
        state_path,
        run_dirs,
        native_scorer=native_scorer,
        maximum_p95_ms=maximum_p95_ms,
        policy_type=AutonomousSupportedImplicitQPolicy,
        state_schema=STATE_SCHEMA,
        population_members=POPULATION_MEMBERS,
        full_trees=Q_TREES,
        report_schema=SCHEMA,
        generation_label="Generation-5",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--native-scorer", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--maximum-p95-ms", type=float, default=4.0)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace shadow audit: {args.output}")
    report = shadow(
        args.state,
        args.runs,
        native_scorer=args.native_scorer,
        maximum_p95_ms=args.maximum_p95_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "shadow_eligible": report["shadow_eligible"],
        "decisions": report["decisions"],
        "proposals": report["policy_metrics"]["shadow_proposals"],
        "p95_ms": report["latency"]["p95_ms"],
    }, sort_keys=True))
    return 0 if report["shadow_eligible"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
