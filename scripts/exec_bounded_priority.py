#!/usr/bin/env python3
"""Apply a bounded nice/CPU contract, drop sudo authority, and exec a child."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import pwd
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.process_priority import ProcessPriorityContract  # noqa: E402


SCHEMA = "bounded-wine-process-priority-v1"


def _sudo_identity() -> tuple[int, int]:
    if os.geteuid() != 0:
        raise PermissionError("bounded priority executor must run through sudo")
    try:
        uid = int(os.environ["SUDO_UID"])
        gid = int(os.environ["SUDO_GID"])
    except (KeyError, ValueError) as error:
        raise PermissionError("bounded priority executor lacks sudo identity") from error
    if uid <= 0 or gid <= 0:
        raise PermissionError("bounded priority executor refuses a root target")
    return uid, gid


def _write_attestation(
    path: Path, *, contract: ProcessPriorityContract, uid: int, gid: int,
) -> None:
    value = {
        "schema": SCHEMA,
        **contract.as_dict(),
        "pid": os.getpid(),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "effective_nice": os.getpriority(os.PRIO_PROCESS, 0),
        "effective_cpus": sorted(os.sched_getaffinity(0)),
        "target_uid": uid,
        "target_gid": gid,
    }
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(value, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
        output.flush()
        os.fsync(output.fileno())


def run(args: argparse.Namespace) -> None:
    uid, gid = _sudo_identity()
    contract = ProcessPriorityContract.from_values(
        nice=args.nice, cpu_list=args.cpu_list
    )
    contract.verify_available(os.sched_getaffinity(0))
    os.setpriority(os.PRIO_PROCESS, 0, contract.nice)
    os.sched_setaffinity(0, contract.cpus)
    if os.getpriority(os.PRIO_PROCESS, 0) != contract.nice:
        raise RuntimeError("bounded process priority was not applied")
    if tuple(sorted(os.sched_getaffinity(0))) != contract.cpus:
        raise RuntimeError("bounded process affinity was not applied")

    account = pwd.getpwuid(uid)
    os.initgroups(account.pw_name, gid)
    os.setgid(gid)
    os.setuid(uid)
    os.environ.update({
        "HOME": account.pw_dir,
        "LOGNAME": account.pw_name,
        "USER": account.pw_name,
    })
    _write_attestation(
        args.attestation.resolve(), contract=contract, uid=uid, gid=gid
    )
    os.execvpe(args.command[0], args.command, os.environ)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nice", type=int, required=True)
    parser.add_argument("--cpu-list", required=True)
    parser.add_argument("--attestation", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command[:1] == ["--"]:
        args.command = args.command[1:]
    if not args.command:
        parser.error("a child command is required after --")
    return args


if __name__ == "__main__":
    run(parse_args())
