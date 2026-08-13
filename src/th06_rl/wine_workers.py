"""Isolated original-Wine worker workspaces for parallel factual collection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


WORKER_SCHEMA = "th06-rl-isolated-wine-worker-v1"
RETAIL_EXECUTABLE = "東方紅魔郷.exe"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(_sha256(path).encode())
        digest.update(b"\0")
    return digest.hexdigest()


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


def _archive_partial(path: Path) -> None:
    for index in range(1, 1_000):
        destination = path.with_name(f"{path.name}.incomplete-{index:03d}")
        if not destination.exists():
            path.rename(destination)
            return
    raise RuntimeError(f"too many partial Wine workers beside {path}")


def prepare_wine_worker(
    *,
    root: Path,
    source_game_dir: Path,
    worker: int,
    directory: str,
    display: str,
    source_inventory_sha256: str | None = None,
) -> dict[str, object]:
    """Create or validate one game/prefix/display isolation boundary."""
    root = root.resolve()
    source_game_dir = source_game_dir.resolve()
    destination = root / directory
    marker = destination / "worker.json"
    executable = source_game_dir / RETAIL_EXECUTABLE
    if not executable.is_file() or executable.is_symlink():
        raise FileNotFoundError(executable)
    actual_inventory = _inventory_sha256(source_game_dir)
    if (
        source_inventory_sha256 is not None
        and actual_inventory != source_inventory_sha256
    ):
        raise ValueError("Wine worker source-game inventory differs")
    inventory = source_inventory_sha256 or actual_inventory
    expected = {
        "schema": WORKER_SCHEMA,
        "worker": worker,
        "display": display,
        "source_game_dir": str(source_game_dir),
        "source_inventory_sha256": inventory,
        "retail_executable_sha256": _sha256(executable),
        "game_dir": str((destination / "game").resolve()),
        "wine_prefix": str((destination / "prefix").resolve()),
    }
    if marker.is_file():
        actual = json.loads(marker.read_text(encoding="utf-8"))
        if actual != expected:
            raise ValueError(f"Wine worker contract differs: {destination}")
        if _sha256(destination / "game" / RETAIL_EXECUTABLE) != expected[
            "retail_executable_sha256"
        ]:
            raise ValueError(f"Wine worker executable differs: {destination}")
        if not (destination / "prefix").is_dir():
            raise FileNotFoundError(destination / "prefix")
        return expected
    if destination.exists():
        _archive_partial(destination)
    destination.mkdir(parents=True)
    try:
        shutil.copytree(source_game_dir, destination / "game")
        (destination / "prefix").mkdir()
        if _sha256(destination / "game" / RETAIL_EXECUTABLE) != expected[
            "retail_executable_sha256"
        ]:
            raise ValueError("copied Wine worker executable differs")
        _atomic_json(marker, expected)
    except BaseException:
        _archive_partial(destination)
        raise
    return expected


def prepare_wine_workers(
    *, root: Path, source_game_dir: Path, specifications: list[dict[str, object]]
) -> list[dict[str, object]]:
    inventory = _inventory_sha256(source_game_dir.resolve())
    return [
        prepare_wine_worker(
            root=root,
            source_game_dir=source_game_dir,
            worker=int(row["worker"]),
            directory=str(row["directory"]),
            display=str(row["display"]),
            source_inventory_sha256=inventory,
        )
        for row in specifications
    ]
