#!/usr/bin/env python3
"""Stdlib-only 32-bit Windows runner for one IQL actor kernel fixture."""

from __future__ import annotations

import argparse
import ctypes
import json


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", required=True)
    parser.add_argument("--library", required=True)
    args = parser.parse_args()
    fixture = json.load(open(args.fixture, encoding="utf-8"))
    pointer = ctypes.POINTER(ctypes.c_float)
    function = ctypes.CDLL(args.library).th06_rl_score_iql_actor_population_v1
    function.argtypes = [
        pointer, pointer,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
        *(pointer for _index in range(10)),
        pointer,
    ]
    function.restype = ctypes.c_int

    def floats(values):
        return (ctypes.c_float * len(values))(*values)

    state = floats(fixture["state"])
    actions = floats(fixture["actions"])
    arrays = [floats(fixture[name]) for name in fixture["array_names"]]
    output = (ctypes.c_float * (
        fixture["model_count"] * fixture["row_count"]
    ))()
    status = function(
        state,
        actions,
        fixture["row_count"],
        fixture["state_count"],
        fixture["action_count"],
        fixture["model_count"],
        fixture["hidden_count"],
        fixture["rank_count"],
        *arrays,
        output,
    )
    print(json.dumps({"status": status, "outputs": list(output)}))
    return 0 if status == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
