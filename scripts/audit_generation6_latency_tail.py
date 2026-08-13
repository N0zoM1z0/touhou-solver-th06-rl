#!/usr/bin/env python3
"""Reproduce and verify Generation-6 CFS tail-latency isolation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.process_priority import ProcessPriorityContract  # noqa: E402


SCHEMA = "generation6-scheduler-tail-latency-audit-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"latency audit input is not an object: {path}")
    return value


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _smoke_command(args: argparse.Namespace, output: Path) -> list[str]:
    return [
        sys.executable,
        str(REPOSITORY / "scripts/smoke_generation6_online_policy.py"),
        "--candidate", str(args.candidate),
        "--linux-state", str(args.linux_state),
        "--windows-state", str(args.windows_state),
        "--linux-library", str(args.linux_library),
        "--windows-library", str(args.windows_library),
        "--windows-python", str(args.windows_python),
        "--contexts", str(args.contexts),
        "--repetitions", str(args.repetitions),
        "--output", str(output),
    ]


def _under_contention(
    command: list[str], *, cpu_list: str, workers: int, seconds: int,
) -> int:
    load = subprocess.Popen(
        [
            "taskset", "-c", cpu_list,
            "stress-ng", "--cpu", str(workers), "--cpu-load", "100",
            "--timeout", f"{seconds}s", "--quiet",
        ],
        cwd=REPOSITORY,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        return subprocess.run(command, cwd=REPOSITORY, check=False).returncode
    finally:
        if load.poll() is None:
            load.terminate()
        try:
            load.wait(timeout=10)
        except subprocess.TimeoutExpired:
            load.kill()
            load.wait(timeout=10)


def run(args: argparse.Namespace) -> int:
    if args.output.exists():
        raise FileExistsError(f"refusing to replace latency audit: {args.output}")
    contract = ProcessPriorityContract.from_values(
        nice=args.nice, cpu_list=args.cpu_list
    )
    contract.verify_available(os.sched_getaffinity(0))
    root = args.output.parent / f".{args.output.stem}-work"
    root.mkdir(parents=True, exist_ok=False)
    baseline_path = root / "baseline.json"
    protected_path = root / "protected.json"
    attestation_path = root / "priority.json"

    baseline_rc = _under_contention(
        _smoke_command(args, baseline_path),
        cpu_list=args.cpu_list,
        workers=args.stress_workers,
        seconds=args.stress_seconds,
    )
    protected_command = [
        "sudo", "-n", "--preserve-env=PATH",
        sys.executable,
        str(REPOSITORY / "scripts/exec_bounded_priority.py"),
        "--nice", str(args.nice),
        "--cpu-list", args.cpu_list,
        "--attestation", str(attestation_path),
        "--",
        *_smoke_command(args, protected_path),
    ]
    protected_rc = _under_contention(
        protected_command,
        cpu_list=args.cpu_list,
        workers=args.stress_workers,
        seconds=args.stress_seconds,
    )
    baseline = _object(baseline_path)
    protected = _object(protected_path)
    attestation = _object(attestation_path)
    baseline_windows = baseline.get("windows")
    protected_windows = protected.get("windows")
    if not isinstance(baseline_windows, dict) or not isinstance(
        protected_windows, dict
    ):
        raise ValueError("latency smoke lacks Windows measurements")
    gates = {
        "baseline_reproduces_scheduler_deadline_tail": (
            baseline_rc == 1
            and int(baseline_windows.get("deadline_misses", 0)) > 0
        ),
        "protected_preflight_passes": protected_rc == 0
        and protected.get("passed") is True,
        "protected_zero_deadline_misses": int(
            protected_windows.get("deadline_misses", -1)
        ) == 0,
        "protected_p95_below_4_ms": float(
            protected_windows.get("latency_p95_ms", float("inf"))
        ) < 4.0,
        "bounded_non_realtime_priority_attested": (
            attestation.get("scheduler") == "SCHED_OTHER"
            and attestation.get("nice") == args.nice
            and attestation.get("effective_nice") == args.nice
            and attestation.get("cpus") == list(contract.cpus)
            and attestation.get("effective_cpus") == list(contract.cpus)
            and attestation.get("uid") == attestation.get("target_uid")
            and attestation.get("gid") == attestation.get("target_gid")
        ),
    }
    report = {
        "schema": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_eligible": False,
        "cpu_list": args.cpu_list,
        "stress_workers": args.stress_workers,
        "stress_seconds": args.stress_seconds,
        "repetitions": args.repetitions,
        "priority": contract.as_dict(),
        "inputs": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in (
                ("candidate", args.candidate),
                ("linux_state", args.linux_state),
                ("windows_state", args.windows_state),
                ("linux_library", args.linux_library),
                ("windows_library", args.windows_library),
            )
        },
        "baseline": baseline,
        "protected": protected,
        "attestation": attestation,
        "gates": gates,
        "passed": all(gates.values()),
    }
    _atomic_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    root = REPOSITORY / "artifacts/autonomous-generation-6-round-preflight"
    parser.add_argument(
        "--candidate", type=Path, default=root / "authorized-candidate.json"
    )
    parser.add_argument(
        "--linux-state", type=Path, default=root / "authorized-linux-state.json"
    )
    parser.add_argument(
        "--windows-state", type=Path,
        default=root / "authorized-windows-state.json",
    )
    parser.add_argument(
        "--linux-library", type=Path,
        default=REPOSITORY / "build/native/libth06_rl_ranker.so",
    )
    parser.add_argument(
        "--windows-library", type=Path,
        default=(
            REPOSITORY
            / "build/native-win32-fully-static/libth06_rl_ranker.dll"
        ),
    )
    parser.add_argument(
        "--windows-python", type=Path,
        default=(
            REPOSITORY
            / "reference/tools/windows-python-3.11.9-embed-win32/python.exe"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu-list", default="0-31")
    parser.add_argument("--nice", type=int, default=-10)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=10_000)
    parser.add_argument("--stress-workers", type=int, default=32)
    parser.add_argument("--stress-seconds", type=int, default=45)
    args = parser.parse_args(argv)
    if args.repetitions < 1 or args.stress_workers < 1 or args.stress_seconds < 1:
        parser.error("latency audit counts must be positive")
    return args


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
