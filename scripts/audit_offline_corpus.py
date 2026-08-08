#!/usr/bin/env python3
"""Audit effective offline-training coverage in a fixed TH06-RL snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th06_rl.offline import audit_dataset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--revision", required=True, help="immutable Hugging Face commit")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--skip-sha256",
        action="store_true",
        help="size/schema audit only; never use before publishing or pruning",
    )
    args = parser.parse_args()
    result = audit_dataset(args.dataset, verify_sha256=not args.skip_sha256)
    result["dataset_revision"] = args.revision
    payload = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
