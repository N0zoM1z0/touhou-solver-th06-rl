#!/usr/bin/env python3
"""Fail closed unless a Wine runtime smoke reached real controlled gameplay."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def verify(report: dict[str, object]) -> dict[str, object]:
    trace = report.get("trace")
    if not isinstance(trace, dict):
        raise ValueError("run report has no trace summary")
    metrics = trace.get("last_policy_metrics")
    checks = {
        "runner_error": report.get("error") is None,
        "controller_exit": report.get("controller_returncode") == 0,
        "startup_normalized": report.get("gdb_normalized") is True,
        "retail_identity": (
            report.get("retail_sha256") == report.get("expected_retail_sha256")
        ),
        "policy_immutable": report.get("immutable_policy_state_equal") is True,
        "exact_cleanup": report.get("leftover_prefix_processes") == [],
        "coherent_gameplay_trace": (
            isinstance(trace.get("rows"), int) and trace["rows"] > 1
            and isinstance(trace.get("first_frame"), int)
            and isinstance(trace.get("last_frame"), int)
            and trace["last_frame"] >= trace["first_frame"]
        ),
        "agent_decisions": (
            isinstance(trace.get("decisions"), int) and trace["decisions"] > 0
        ),
        "smoke_policy_identity": (
            isinstance(metrics, dict)
            and metrics.get("purpose") == "infrastructure-smoke-only"
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError("Wine runtime smoke failed: " + ", ".join(failed))
    return {
        "schema": "th06-rl-wine-runtime-smoke-verification-v1",
        "passed": True,
        "artifact_dir": report.get("artifact_dir"),
        "wine_version": report.get("wine_version"),
        "first_frame": trace["first_frame"],
        "last_frame": trace["last_frame"],
        "trace_rows": trace["rows"],
        "decisions": trace["decisions"],
        "max_bullets": trace.get("max_bullets"),
        "checks": checks,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    args = parser.parse_args(argv)
    value = json.loads(args.report.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        parser.error("run report root must be an object")
    try:
        result = verify(value)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
