#!/usr/bin/env python3
"""Prepare a hash-attested, resource-bounded normal-speed Wine worker pool."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from th06_rl.wine_workers import (  # noqa: E402
    allocate_worker_specifications,
    prepare_retail_template,
    prepare_wine_workers,
)


POOL_SCHEMA = "th06-rl-normal-speed-wine-pool-v1"
PREFIX_SCHEMA = "th06-rl-isolated-wine-prefix-v1"
SCORE_SHA256 = "54cd436d5d8a7a904190c792a977bf270ab1cb759fd72101e51e94d26b749c71"
GIB = 1024**3


def linux_mem_available_bytes(path: Path = Path("/proc/meminfo")) -> int:
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("MemAvailable:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    raise OSError("Linux MemAvailable telemetry is absent")


def require_host_capacity(
    *, workers: int, memory_available: int, disk_available: int,
    memory_per_worker: int, disk_per_worker: int,
) -> None:
    if min(memory_available, disk_available, memory_per_worker, disk_per_worker) < 0:
        raise ValueError("worker capacity values cannot be negative")
    if memory_available < workers * memory_per_worker:
        raise RuntimeError("insufficient host memory for isolated Wine workers")
    if disk_available < workers * disk_per_worker:
        raise RuntimeError("insufficient host storage for isolated Wine workers")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
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


def _prepare_score_template(template: Path, destination: Path) -> None:
    candidates = [
        path for path in template.rglob("score.dat")
        if path.is_file() and not path.is_symlink() and _sha256(path) == SCORE_SHA256
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"retail template contains {len(candidates)} full-unlock score candidates"
        )
    if destination.is_file():
        if destination.is_symlink() or _sha256(destination) != SCORE_SHA256:
            raise ValueError("worker-pool score template differs")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise FileExistsError(temporary)
    try:
        shutil.copy2(candidates[0], temporary)
        if _sha256(temporary) != SCORE_SHA256:
            raise ValueError("copied full-unlock score differs")
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()


def _initialize_wine_prefix(worker: dict[str, object]) -> None:
    """Initialize every pool prefix before the serial/concurrent differential."""
    prefix = Path(str(worker["wine_prefix"])).resolve()
    display = str(worker["display"])
    marker = prefix / ".th06-rl-retail-ready-v1"
    version = subprocess.run(
        ["wine", "--version"], check=True, capture_output=True, text=True,
    ).stdout.strip()
    expected = {
        "schema": PREFIX_SCHEMA,
        "wine_version": version,
        "winearch": "win32",
    }
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != expected:
            raise ValueError(f"Wine worker prefix contract differs: {prefix}")
        return
    socket = Path("/tmp/.X11-unix") / f"X{display[1:]}"
    if socket.exists():
        raise RuntimeError(f"worker X display socket already exists: {socket}")
    environment = os.environ.copy()
    environment.update({
        "WINEPREFIX": str(prefix),
        "DISPLAY": display,
        "WINEARCH": "win32",
        "WINEDEBUG": "-all",
        "WINEDLLOVERRIDES": "mscoree,mshtml=",
        "LANG": "ja_JP.UTF-8",
        "LC_ALL": "ja_JP.UTF-8",
    })
    xvfb_log = (prefix / "pool-xvfb.log").open("wb")
    wineboot_log = (prefix / "pool-wineboot.log").open("wb")
    xvfb = None
    try:
        xvfb = subprocess.Popen(
            ["Xvfb", display, "-screen", "0", "1024x768x24", "-nolisten", "tcp"],
            stdin=subprocess.DEVNULL,
            stdout=xvfb_log,
            stderr=subprocess.STDOUT,
        )
        time.sleep(0.5)
        if xvfb.poll() is not None:
            raise RuntimeError(f"worker Xvfb exited early: {xvfb.returncode}")
        completed = subprocess.run(
            ["wineboot", "-u"],
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=wineboot_log,
            stderr=subprocess.STDOUT,
            check=False,
            timeout=180,
        )
        if completed.returncode:
            raise RuntimeError(
                f"Wine worker prefix initialization failed: {completed.returncode}"
            )
        subprocess.run(
            ["wineserver", "-k"], env=environment, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        subprocess.run(
            ["wineserver", "-w"], env=environment, check=True, timeout=30,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        _atomic_json(marker, expected)
    finally:
        subprocess.run(
            ["wineserver", "-k"], env=environment, check=False,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
        if xvfb is not None:
            xvfb.terminate()
            try:
                xvfb.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb.kill()
                xvfb.wait(timeout=5)
        wineboot_log.close()
        xvfb_log.close()
    if socket.exists():
        raise RuntimeError(f"worker X display socket survived initialization: {socket}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--archive", type=Path, default=REPOSITORY / "../game-exe/th06.rar",
    )
    parser.add_argument(
        "--template-game-dir", type=Path,
        default=REPOSITORY / "reference/th06-game-original/template",
    )
    parser.add_argument(
        "--worker-root", type=Path,
        default=REPOSITORY / "reference/wine-workers-v2",
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--cpus-per-worker", type=int, default=8)
    parser.add_argument("--display-base", type=int, default=107)
    parser.add_argument("--min-memory-gib-per-worker", type=float, default=8.0)
    parser.add_argument("--min-disk-gib-per-worker", type=float, default=8.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.cpus_per_worker < 4:
        parser.error("--cpus-per-worker must be at least 4")
    if args.min_memory_gib_per_worker <= 0 or args.min_disk_gib_per_worker <= 0:
        parser.error("per-worker memory and disk reserves must be positive")
    if args.output is None:
        args.output = args.worker_root / "pool.json"
    return args


def run(args: argparse.Namespace) -> dict[str, object]:
    worker_root = args.worker_root.resolve()
    require_host_capacity(
        workers=args.workers,
        memory_available=linux_mem_available_bytes(),
        disk_available=shutil.disk_usage(worker_root.parent).free,
        memory_per_worker=int(args.min_memory_gib_per_worker * GIB),
        disk_per_worker=int(args.min_disk_gib_per_worker * GIB),
    )
    specifications = allocate_worker_specifications(
        available_cpus=tuple(sorted(os.sched_getaffinity(0))),
        workers=args.workers,
        cpus_per_worker=args.cpus_per_worker,
        display_base=args.display_base,
    )
    for row in specifications:
        socket = Path("/tmp/.X11-unix") / f"X{str(row['display'])[1:]}"
        if socket.exists():
            raise RuntimeError(f"worker X display socket already exists: {socket}")
    template = args.template_game_dir.resolve()
    template_contract = prepare_retail_template(
        archive=args.archive, template_game_dir=template,
    )
    workers = prepare_wine_workers(
        root=worker_root,
        source_game_dir=template,
        specifications=specifications,
    )
    for worker in workers:
        subprocess.run(
            [
                sys.executable,
                str(REPOSITORY / "scripts/configure_wine_retail.py"),
                str(worker["game_dir"]),
                "--initialize",
            ],
            check=True,
            stdout=subprocess.DEVNULL,
        )
        _initialize_wine_prefix(worker)
    score_template = worker_root / "full-unlock-score.dat"
    _prepare_score_template(template, score_template)
    rows = []
    for specification, worker in zip(specifications, workers, strict=True):
        rows.append({**worker, **{
            "game_cpu_list": specification["game_cpu_list"],
            "controller_cpu_list": specification["controller_cpu_list"],
        }})
    result = {
        "schema": POOL_SCHEMA,
        "repository": str(REPOSITORY),
        "template": template_contract,
        "score_template": str(score_template.resolve()),
        "score_template_sha256": _sha256(score_template),
        "workers": rows,
        "resource_contract": {
            "maximum_total_cpus": 32,
            "cpus_per_worker": args.cpus_per_worker,
            "minimum_memory_gib_per_worker": args.min_memory_gib_per_worker,
            "minimum_disk_gib_per_worker": args.min_disk_gib_per_worker,
            "game_clock": "original-retail-normal-speed",
        },
    }
    output = args.output.resolve()
    if output.is_file():
        actual = json.loads(output.read_text(encoding="utf-8"))
        if actual != result:
            raise ValueError("Wine worker pool contract differs")
    else:
        _atomic_json(output, result)
    return result


def main(argv: list[str] | None = None) -> int:
    result = run(parse_args(argv))
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
