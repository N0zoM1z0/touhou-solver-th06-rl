#!/usr/bin/env python3
"""Pause a physical batch before launch when Windows commit is too low."""

from __future__ import annotations

import argparse

from th06_rl.th06.system_health import GIB, below_commit_reserve, read_system_memory


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve-gib", type=float, default=4.0)
    args = parser.parse_args()
    if args.reserve_gib < 0:
        parser.error("--reserve-gib cannot be negative")
    sample = read_system_memory()
    reserve = int(args.reserve_gib * GIB)
    print(
        "host commit: "
        f"{sample.commit_total_bytes / GIB:.2f}/"
        f"{sample.commit_limit_bytes / GIB:.2f} GiB; "
        f"headroom={sample.commit_headroom_bytes / GIB:.2f} GiB; "
        f"controller={sample.controller_private_bytes / GIB:.3f} GiB"
    )
    return 1 if below_commit_reserve(sample, reserve) else 0


if __name__ == "__main__":
    raise SystemExit(main())
