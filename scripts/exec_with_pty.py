#!/usr/bin/env python3
"""Run a console-subsystem child on a PTY and forward its complete output."""

from __future__ import annotations

import argparse
import errno
import os
import signal
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("a command is required after --")

    pid, descriptor = os.forkpty()
    if pid == 0:
        os.execvpe(command[0], command, os.environ)

    def forward(signum: int, _frame) -> None:
        try:
            os.killpg(pid, signum)
        except ProcessLookupError:
            pass

    signal.signal(signal.SIGINT, forward)
    signal.signal(signal.SIGTERM, forward)
    try:
        while True:
            try:
                chunk = os.read(descriptor, 64 * 1024)
            except OSError as error:
                if error.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
    finally:
        os.close(descriptor)
    _pid, status = os.waitpid(pid, 0)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    raise SystemExit(main())
