#!/usr/bin/env python3
"""Hash-bind a clean Generation-3 native shadow into active canary state."""

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

from scripts.shadow_option_advantage import SCHEMA as SHADOW_SCHEMA  # noqa: E402
from th06_rl.advantage_learning import STATE_SCHEMA  # noqa: E402


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
    if (
        state.get("schema") != STATE_SCHEMA
        or state.get("mode") != "shadow"
        or not isinstance(authorization, dict)
        or authorization.get("fit_eligible") is not True
        or authorization.get("active_canary") is not None
        or not isinstance(fit, dict)
    ):
        raise ValueError("candidate did not pass Generation-3 fit authorization")
    expected_groups = sorted(map(str, fit.get("validation_groups", ())))
    if (
        shadow.get("schema") != SHADOW_SCHEMA
        or shadow.get("shadow_eligible") is not True
        or shadow.get("policy_state_sha256") != _sha256(state_path)
        or shadow.get("heldout_episode_groups") != expected_groups
        or int(shadow.get("decisions", -1))
        != int(fit.get("validation_options", -2))
    ):
        raise ValueError("Generation-3 shadow is absent, stale, or ineligible")
    state["mode"] = "active"
    authorization["active_canary"] = {
        "schema": "autonomous-generation-3-active-canary-authorization-v1",
        "shadow_audit_sha256": _sha256(shadow_path),
        "shadow_policy_state_sha256": _sha256(state_path),
        "heldout_episode_groups": expected_groups,
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
