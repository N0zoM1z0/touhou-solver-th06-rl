#!/usr/bin/env python3
"""Find the first physical-state divergence between two TH06 JSONL traces."""

from __future__ import annotations

import argparse
from itertools import zip_longest
import json
import math
from pathlib import Path
from typing import Any


_MISSING = object()


def first_difference(
    left: Any,
    right: Any,
    *,
    path: str = "$",
    absolute_tolerance: float = 0.0,
) -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right, "reason": "type"}
    if isinstance(left, dict):
        if left.keys() != right.keys():
            return {
                "path": path,
                "left": sorted(left),
                "right": sorted(right),
                "reason": "keys",
            }
        for key in left:
            difference = first_difference(
                left[key],
                right[key],
                path=f"{path}.{key}",
                absolute_tolerance=absolute_tolerance,
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {
                "path": path,
                "left": len(left),
                "right": len(right),
                "reason": "length",
            }
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            difference = first_difference(
                left_item,
                right_item,
                path=f"{path}[{index}]",
                absolute_tolerance=absolute_tolerance,
            )
            if difference is not None:
                return difference
        return None
    if isinstance(left, float):
        if math.isfinite(left) and math.isfinite(right):
            equal = abs(left - right) <= absolute_tolerance
        else:
            equal = left == right
    else:
        equal = left == right
    if not equal:
        return {"path": path, "left": left, "right": right, "reason": "value"}
    return None


def compare_traces(left_path: Path, right_path: Path, *, absolute_tolerance: float) -> dict[str, Any]:
    rows = 0
    with left_path.open(encoding="utf-8") as left_file, right_path.open(encoding="utf-8") as right_file:
        for line_number, (left_line, right_line) in enumerate(
            zip_longest(left_file, right_file, fillvalue=_MISSING),
            start=1,
        ):
            if left_line is _MISSING or right_line is _MISSING:
                return {
                    "equal": False,
                    "matched_rows": rows,
                    "line": line_number,
                    "difference": {
                        "path": "$",
                        "left": "missing" if left_line is _MISSING else "present",
                        "right": "missing" if right_line is _MISSING else "present",
                        "reason": "trace-length",
                    },
                }
            try:
                left = json.loads(left_line)
                right = json.loads(right_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSONL at line {line_number}: {error}") from error
            difference = first_difference(
                left,
                right,
                absolute_tolerance=absolute_tolerance,
            )
            if difference is not None:
                return {
                    "equal": False,
                    "matched_rows": rows,
                    "line": line_number,
                    "tick": left.get("tick") if isinstance(left, dict) else None,
                    "difference": difference,
                }
            rows += 1
    return {"equal": True, "matched_rows": rows, "line": None, "difference": None}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    parser.add_argument("--absolute-tolerance", type=float, default=0.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.absolute_tolerance < 0.0:
        parser.error("absolute tolerance must be nonnegative")
    result = compare_traces(args.left, args.right, absolute_tolerance=args.absolute_tolerance)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
