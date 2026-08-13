#!/usr/bin/env python3
"""Canonicalize Generation-6 episode metrics independent of mapping order."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.iql_actor_learning import summarize_iql_actor_episodes  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to replace summary: {args.output}")
    source = json.loads(args.input.read_text(encoding="utf-8"))
    episodes = source["report"]["episodes"]
    cohorts = tuple(sorted({row["cohort"] for row in episodes.values()}))
    result = {
        "schema": "autonomous-generation-6-canonical-policy-summary-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "source_report_sha256": hashlib.sha256(
            args.input.read_bytes()
        ).hexdigest(),
        "episode_order": "immutable-episode-id-ascending",
        "bootstrap_resamples": 4096,
        "cohorts": summarize_iql_actor_episodes(
            episodes, cohort_names=cohorts
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output), "cohorts": result["cohorts"]
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
