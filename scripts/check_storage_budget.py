#!/usr/bin/env python3
"""Refuse to start another Stage when its full corpus reserve will not fit."""

from __future__ import annotations

import argparse
from pathlib import Path

from th06_rl.storage_budget import can_reserve_run


GIB = 1024**3
MIB = 1024**2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--limit-gib", type=float, default=45.0)
    parser.add_argument("--reserve-mib", type=float, default=512.0)
    args = parser.parse_args()
    try:
        allowed, used = can_reserve_run(
            args.root,
            limit_bytes=int(args.limit_gib * GIB),
            reserve_bytes=int(args.reserve_mib * MIB),
        )
    except (OSError, ValueError) as error:
        print(f"storage budget check failed closed: {type(error).__name__}: {error}")
        return 2
    projected = used + int(args.reserve_mib * MIB)
    print(
        "artifact storage: "
        f"used={used / GIB:.3f} GiB, "
        f"next-run-reserve={args.reserve_mib / 1024.0:.3f} GiB, "
        f"projected={projected / GIB:.3f} GiB, "
        f"limit={args.limit_gib:.3f} GiB"
    )
    if not allowed:
        print("storage budget exhausted; refusing to start another Stage")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
