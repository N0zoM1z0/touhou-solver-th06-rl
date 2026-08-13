#!/usr/bin/env python3
"""Hash-bind a clean generation-2 shadow into a bounded active state."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
from scripts.shadow_conservative_q import SCHEMA as SHADOW_SCHEMA


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def authorize(state_path: Path, shadow_path: Path) -> dict[str, object]:
    state = _object(state_path)
    shadow = _object(shadow_path)
    authorization = state.get("authorization")
    selection = state.get("selection")
    if (
        state.get("mode") != "shadow"
        or not isinstance(authorization, dict)
        or authorization.get("fit_eligible") is not True
        or not isinstance(selection, dict)
    ):
        raise ValueError("candidate did not pass grouped fit authorization")
    if (
        shadow.get("schema") != SHADOW_SCHEMA
        or shadow.get("shadow_eligible") is not True
        or shadow.get("policy_state_sha256") != _sha256(state_path)
    ):
        raise ValueError("candidate shadow audit is absent, stale, or ineligible")
    if int(selection.get("active_override_budget", 0)) <= 0:
        raise ValueError("candidate has no bounded canary budget")
    state["mode"] = "active"
    authorization["active_canary"] = {
        "schema": "autonomous-conservative-active-authorization-v2",
        "shadow_audit_sha256": _sha256(shadow_path),
        "shadow_policy_state_sha256": _sha256(state_path),
        "heldout_episode_groups": shadow["heldout_episode_groups"],
        "shadow_decisions": shadow["decisions"],
        "shadow_proposals": shadow["policy_metrics"]["shadow_proposals"],
    }
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("state", type=Path)
    parser.add_argument("shadow", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace canary state: {args.output}")
    active = authorize(args.state, args.shadow)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(active, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
