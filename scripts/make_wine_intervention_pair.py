#!/usr/bin/env python3
"""Create two immutable, balanced state files for one Wine intervention pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th06_rl.policies.wine_intervention import ARMS, STATE_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--min-player-y", type=float, default=420.0)
    parser.add_argument("--min-bullets", type=int, default=256)
    parser.add_argument("--max-hard-actions", type=int, default=12)
    parser.add_argument("--min-reserve-gain", type=float, default=4.0)
    args = parser.parse_args()

    incumbent = json.loads(args.incumbent_state.read_text(encoding="utf-8"))
    if not isinstance(incumbent, dict):
        parser.error("incumbent state root must be an object")
    if args.output_dir.exists():
        parser.error("output directory must not already exist")
    args.output_dir.mkdir(parents=True)
    eligibility = {
        "min_player_y": args.min_player_y,
        "min_bullets": args.min_bullets,
        "max_hard_actions": args.max_hard_actions,
        "min_reserve_gain": args.min_reserve_gain,
    }
    for arm in ARMS:
        state = {
            "schema": STATE_SCHEMA,
            "pair_id": args.pair_id,
            "arm": arm,
            "alternative_probability": 0.5,
            "eligibility": eligibility,
            "incumbent_state": incumbent,
        }
        (args.output_dir / f"{arm}.json").write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
