from pathlib import Path

import pytest

from scripts.prepare_wine_workers import (
    linux_mem_available_bytes,
    parse_args,
    require_host_capacity,
)


def test_linux_memory_reader_requires_memavailable(tmp_path: Path) -> None:
    path = tmp_path / "meminfo"
    path.write_text("MemTotal: 99 kB\nMemAvailable: 42 kB\n")
    assert linux_mem_available_bytes(path) == 42 * 1024
    path.write_text("MemTotal: 99 kB\n")
    with pytest.raises(OSError, match="MemAvailable"):
        linux_mem_available_bytes(path)


def test_worker_capacity_is_fail_closed_per_complete_pool() -> None:
    require_host_capacity(
        workers=2,
        memory_available=16,
        disk_available=20,
        memory_per_worker=8,
        disk_per_worker=10,
    )
    with pytest.raises(RuntimeError, match="memory"):
        require_host_capacity(
            workers=2,
            memory_available=15,
            disk_available=20,
            memory_per_worker=8,
            disk_per_worker=10,
        )
    with pytest.raises(RuntimeError, match="storage"):
        require_host_capacity(
            workers=2,
            memory_available=16,
            disk_available=19,
            memory_per_worker=8,
            disk_per_worker=10,
        )


def test_worker_pool_defaults_are_repo_relative_and_bounded() -> None:
    args = parse_args([])
    assert args.workers == 2
    assert args.cpus_per_worker == 8
    assert args.archive.name == "th06.rar"
    assert args.worker_root.name == "wine-workers-v2"


def test_worker_pool_accepts_resource_bounded_parallel_width() -> None:
    args = parse_args(["--workers", "8"])
    assert args.workers == 8
