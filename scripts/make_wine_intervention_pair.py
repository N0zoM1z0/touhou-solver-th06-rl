#!/usr/bin/env python3
"""Create two immutable, balanced state files for one Wine intervention pair."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th06_rl.policies.wine_intervention import ARMS, STATE_SCHEMA


def write_pair(
    *,
    incumbent: dict[str, object],
    output_dir: Path,
    pair_id: str,
    min_player_y: float = 420.0,
    min_bullets: int = 256,
    max_hard_actions: int = 18,
    max_local_actions: int = 6,
    max_reserve_deficit: float = 4.0,
) -> None:
    if output_dir.exists():
        raise FileExistsError("output directory must not already exist")
    output_dir.mkdir(parents=True)
    eligibility = {
        "min_player_y": min_player_y,
        "min_bullets": min_bullets,
        "max_hard_actions": max_hard_actions,
        "max_local_actions": max_local_actions,
        "max_reserve_deficit": max_reserve_deficit,
    }
    for arm in ARMS:
        state = {
            "schema": STATE_SCHEMA,
            "pair_id": pair_id,
            "arm": arm,
            "alternative_probability": 0.5,
            "eligibility": eligibility,
            "incumbent_state": incumbent,
        }
        (output_dir / f"{arm}.json").write_text(
            json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-state", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pair-id", required=True)
    parser.add_argument("--min-player-y", type=float, default=420.0)
    parser.add_argument("--min-bullets", type=int, default=256)
    parser.add_argument("--max-hard-actions", type=int, default=18)
    parser.add_argument("--max-local-actions", type=int, default=6)
    parser.add_argument("--max-reserve-deficit", type=float, default=4.0)
    args = parser.parse_args()

    incumbent = json.loads(args.incumbent_state.read_text(encoding="utf-8"))
    if not isinstance(incumbent, dict):
        parser.error("incumbent state root must be an object")
    try:
        write_pair(
            incumbent=incumbent,
            output_dir=args.output_dir,
            pair_id=args.pair_id,
            min_player_y=args.min_player_y,
            min_bullets=args.min_bullets,
            max_hard_actions=args.max_hard_actions,
            max_local_actions=args.max_local_actions,
            max_reserve_deficit=args.max_reserve_deficit,
        )
    except FileExistsError as error:
        parser.error(str(error))
    print(args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
