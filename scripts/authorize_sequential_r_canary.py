#!/usr/bin/env python3
"""Hash-bind a clean Generation-4 native shadow into active canary state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.shadow_sequential_r_critic import SCHEMA as SHADOW_SCHEMA  # noqa: E402
from th06_rl.sequential_learning import STATE_SCHEMA  # noqa: E402


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authorize(state_path: Path, shadow_path: Path) -> dict[str, object]:
    state_path = state_path.resolve()
    shadow_path = shadow_path.resolve()
    state = _object(state_path)
    shadow = _object(shadow_path)
    authorization = state.get("authorization")
    fit = state.get("fit_report")
    groups = shadow.get("audit_episode_groups")
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("mode") != "shadow"
        or not isinstance(authorization, dict)
        or authorization.get("fit_eligible") is not True
        or authorization.get("active_canary") is not None
        or not isinstance(fit, dict)
        or not isinstance(groups, list)
        or not groups
    ):
        raise ValueError("candidate did not pass Generation-4 fit authorization")
    fit_groups = set(map(str, fit.get("episode_groups", ())))
    if (
        shadow.get("schema") != SHADOW_SCHEMA
        or shadow.get("shadow_eligible") is not True
        or shadow.get("policy_state_sha256") != _sha256(state_path)
        or not set(map(str, groups)) <= fit_groups
        or int(shadow.get("decisions", 0)) <= 0
    ):
        raise ValueError("Generation-4 native shadow is absent, stale, or invalid")
    state["mode"] = "active"
    authorization["active_canary"] = {
        "schema": "autonomous-generation-4-active-canary-authorization-v1",
        "shadow_audit_sha256": _sha256(shadow_path),
        "shadow_policy_state_sha256": _sha256(state_path),
        "audit_episode_groups": list(map(str, groups)),
        "shadow_decisions": int(shadow["decisions"]),
        "shadow_proposals": int(shadow["policy_metrics"]["shadow_proposals"]),
        "native_scorer_sha256": str(shadow["native_scorer_sha256"]),
        "shadow_latency_p95_ms": float(shadow["latency"]["p95_ms"]),
    }
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("shadow", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace canary state: {args.output}")
    active = authorize(args.state, args.shadow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(active, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
