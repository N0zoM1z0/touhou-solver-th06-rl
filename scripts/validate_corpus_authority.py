#!/usr/bin/env python3
"""Validate that a corpus is self-contained after the Wine process exits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from th06_rl.th06.source_dataset import iter_source_frames


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    frames = 0
    stages: set[int] = set()
    anchors: set[tuple[int, int]] = set()
    for bundle in iter_source_frames(args.run_dir):
        frames += 1
        stages.add(bundle.control.stage)
        anchors.add((bundle.control.stage, bundle.anchor_sequence))
    report = {
        "schema_version": "th06-source-dataset-admission-v1",
        "passes": True,
        "frames": frames,
        "stages": sorted(stages),
        "active_anchor_roots": len(anchors),
    }
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
