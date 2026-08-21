"""Isolated original-Wine worker workspaces for parallel factual collection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


WORKER_SCHEMA = "th06-rl-isolated-wine-worker-v2"
TEMPLATE_SCHEMA = "th06-rl-immutable-retail-template-v1"
RETAIL_EXECUTABLE = "東方紅魔郷.exe"
RETAIL_EXECUTABLE_SHA256 = (
    "9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245"
)
RETAIL_ARCHIVE_SHA256 = (
    "6b013b24c101ae846b97a2778abf461d537640611a835824a42533c692be55d6"
)


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


def _template_marker(root: Path) -> Path:
    return root.with_name(f"{root.name}.template.json")


def attest_retail_template(
    template_game_dir: Path,
    *,
    archive_sha256: str,
    executable_sha256: str = RETAIL_EXECUTABLE_SHA256,
) -> dict[str, object]:
    """Freeze an already extracted retail directory as a copy-only template."""
    template_game_dir = template_game_dir.resolve()
    executable = template_game_dir / RETAIL_EXECUTABLE
    if (
        len(archive_sha256) != 64
        or len(executable_sha256) != 64
        or not executable.is_file()
        or executable.is_symlink()
    ):
        raise ValueError("retail template inputs are invalid")
    if any(path.is_symlink() for path in template_game_dir.rglob("*")):
        raise ValueError("retail template cannot contain symlinks")
    actual_executable = _sha256(executable)
    if actual_executable != executable_sha256:
        raise ValueError("retail template executable differs")
    contract = {
        "schema": TEMPLATE_SCHEMA,
        "template_game_dir": str(template_game_dir),
        "source_archive_sha256": archive_sha256,
        "source_inventory_sha256": _inventory_sha256(template_game_dir),
        "retail_executable_sha256": actual_executable,
    }
    marker = _template_marker(template_game_dir)
    if marker.exists():
        actual = json.loads(marker.read_text(encoding="utf-8"))
        if actual != contract:
            raise ValueError("retail template contract differs")
    else:
        _atomic_json(marker, contract)
    return contract


def validate_retail_template(template_game_dir: Path) -> dict[str, object]:
    template_game_dir = template_game_dir.resolve()
    marker = _template_marker(template_game_dir)
    if not marker.is_file() or marker.is_symlink():
        raise FileNotFoundError(f"retail template marker is absent: {marker}")
    contract = json.loads(marker.read_text(encoding="utf-8"))
    executable = template_game_dir / RETAIL_EXECUTABLE
    if (
        not isinstance(contract, dict)
        or contract.get("schema") != TEMPLATE_SCHEMA
        or contract.get("template_game_dir") != str(template_game_dir)
        or not executable.is_file()
        or executable.is_symlink()
        or contract.get("retail_executable_sha256") != _sha256(executable)
        or contract.get("source_inventory_sha256")
        != _inventory_sha256(template_game_dir)
        or any(path.is_symlink() for path in template_game_dir.rglob("*"))
    ):
        raise ValueError("retail template integrity differs")
    return contract


def prepare_retail_template(
    *,
    archive: Path,
    template_game_dir: Path,
    extractor: str = "unar",
) -> dict[str, object]:
    """Extract and attest one immutable copy source from the shipped archive."""
    archive = archive.resolve()
    template_game_dir = template_game_dir.resolve()
    if not archive.is_file() or archive.is_symlink():
        raise FileNotFoundError(archive)
    if _sha256(archive) != RETAIL_ARCHIVE_SHA256:
        raise ValueError("retail source archive differs")
    if _template_marker(template_game_dir).is_file():
        contract = validate_retail_template(template_game_dir)
        if contract.get("source_archive_sha256") != RETAIL_ARCHIVE_SHA256:
            raise ValueError("retail template archive provenance differs")
        return contract
    if template_game_dir.exists():
        _archive_partial(template_game_dir)
    template_game_dir.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".th06-template-extract.", dir=template_game_dir.parent
    ) as temporary:
        extraction = Path(temporary)
        subprocess.run(
            [
                extractor,
                "-quiet",
                "-force-overwrite",
                "-output-directory",
                str(extraction),
                str(archive),
            ],
            check=True,
            stdin=subprocess.DEVNULL,
        )
        candidates = sorted(extraction.rglob(RETAIL_EXECUTABLE))
        if len(candidates) != 1:
            raise ValueError(
                f"retail archive contains {len(candidates)} executable candidates"
            )
        if _sha256(candidates[0]) != RETAIL_EXECUTABLE_SHA256:
            raise ValueError("retail archive executable differs")
        shutil.copytree(candidates[0].parent, template_game_dir)
    try:
        return attest_retail_template(
            template_game_dir,
            archive_sha256=RETAIL_ARCHIVE_SHA256,
        )
    except BaseException:
        _archive_partial(template_game_dir)
        raise


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
    if (
        worker < 0
        or not directory
        or Path(directory).name != directory
        or directory in {".", ".."}
        or not display.startswith(":")
        or not display[1:].isdigit()
    ):
        raise ValueError("Wine worker identity is invalid")
    root = root.resolve()
    source_game_dir = source_game_dir.resolve()
    template = validate_retail_template(source_game_dir)
    destination = root / directory
    marker = destination / "worker.json"
    executable = source_game_dir / RETAIL_EXECUTABLE
    if not executable.is_file() or executable.is_symlink():
        raise FileNotFoundError(executable)
    actual_inventory = str(template["source_inventory_sha256"])
    if (
        source_inventory_sha256 is not None
        and actual_inventory != source_inventory_sha256
    ):
        raise ValueError("Wine worker source-game inventory differs")
    inventory = source_inventory_sha256 or actual_inventory
    expected = {
        "schema": WORKER_SCHEMA,
        "worker": worker,
        "directory": directory,
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


def validate_wine_worker(worker: dict[str, object]) -> dict[str, object]:
    """Validate one prepared mutable worker against its immutable marker."""
    keys = (
        "schema", "worker", "directory", "display", "source_game_dir",
        "source_inventory_sha256", "retail_executable_sha256", "game_dir",
        "wine_prefix",
    )
    expected = {key: worker.get(key) for key in keys}
    game_dir = Path(str(expected["game_dir"])).resolve()
    prefix = Path(str(expected["wine_prefix"])).resolve()
    destination = game_dir.parent
    marker = destination / "worker.json"
    source = Path(str(expected["source_game_dir"])).resolve()
    template = validate_retail_template(source)
    if (
        expected.get("schema") != WORKER_SCHEMA
        or expected.get("directory") != destination.name
        or game_dir != destination / "game"
        or prefix != destination / "prefix"
        or not marker.is_file()
        or marker.is_symlink()
        or json.loads(marker.read_text(encoding="utf-8")) != expected
        or template.get("source_inventory_sha256")
        != expected.get("source_inventory_sha256")
        or not (game_dir / RETAIL_EXECUTABLE).is_file()
        or (game_dir / RETAIL_EXECUTABLE).is_symlink()
        or _sha256(game_dir / RETAIL_EXECUTABLE)
        != expected.get("retail_executable_sha256")
        or not prefix.is_dir()
        or prefix.is_symlink()
    ):
        raise ValueError(f"Wine worker integrity differs: {destination}")
    return expected


def prepare_wine_workers(
    *, root: Path, source_game_dir: Path, specifications: list[dict[str, object]]
) -> list[dict[str, object]]:
    validate_worker_specifications(specifications)
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


def _cpu_set(value: str) -> set[int]:
    result: set[int] = set()
    for part in value.split(","):
        if not part:
            raise ValueError("empty CPU-list component")
        bounds = part.split("-", 1)
        start = int(bounds[0])
        stop = int(bounds[-1])
        if start < 0 or stop < start:
            raise ValueError("invalid CPU-list range")
        result.update(range(start, stop + 1))
    return result


def validate_worker_specifications(
    specifications: list[dict[str, object]],
) -> None:
    if not specifications:
        raise ValueError("normal-speed collection requires at least one worker")
    identities: set[int] = set()
    directories: set[str] = set()
    displays: set[str] = set()
    assigned_cpus: set[int] = set()
    for row in specifications:
        worker = int(row.get("worker", -1))
        directory = str(row.get("directory", ""))
        display = str(row.get("display", ""))
        game_cpus = _cpu_set(str(row.get("game_cpu_list", "")))
        controller_cpus = _cpu_set(str(row.get("controller_cpu_list", "")))
        if (
            worker < 0
            or worker in identities
            or not directory
            or Path(directory).name != directory
            or directory in directories
            or not display.startswith(":")
            or not display[1:].isdigit()
            or display in displays
            or not game_cpus
            or not controller_cpus
            or game_cpus & controller_cpus
            or assigned_cpus & (game_cpus | controller_cpus)
        ):
            raise ValueError("Wine worker specifications overlap or are invalid")
        identities.add(worker)
        directories.add(directory)
        displays.add(display)
        assigned_cpus.update(game_cpus | controller_cpus)
    if identities != set(range(len(specifications))):
        raise ValueError("Wine worker identities must be contiguous from zero")


def assigned_worker_cpus(
    specifications: list[dict[str, object]],
) -> frozenset[int]:
    """Return the exact disjoint CPU ownership after structural validation."""
    validate_worker_specifications(specifications)
    assigned: set[int] = set()
    for row in specifications:
        assigned.update(_cpu_set(str(row["game_cpu_list"])))
        assigned.update(_cpu_set(str(row["controller_cpu_list"])))
    return frozenset(assigned)


def allocate_worker_specifications(
    *,
    available_cpus: tuple[int, ...],
    workers: int = 2,
    cpus_per_worker: int = 8,
    display_base: int = 107,
) -> list[dict[str, object]]:
    """Allocate deterministic disjoint game/controller ownership."""
    if (
        workers < 1
        or cpus_per_worker < 4
        or display_base < 1
        or len(set(available_cpus)) != len(available_cpus)
        or tuple(sorted(available_cpus)) != available_cpus
        or len(available_cpus) < workers * cpus_per_worker
    ):
        raise ValueError("host cannot satisfy the bounded Wine worker allocation")
    selected = available_cpus[: workers * cpus_per_worker]
    game_count = cpus_per_worker // 2
    rows: list[dict[str, object]] = []
    for worker in range(workers):
        cpus = selected[
            worker * cpus_per_worker : (worker + 1) * cpus_per_worker
        ]
        game = cpus[:game_count]
        controller = cpus[game_count:]
        rows.append({
            "worker": worker,
            "directory": f"wine-{worker}",
            "display": f":{display_base + worker}",
            "game_cpu_list": ",".join(map(str, game)),
            "controller_cpu_list": ",".join(map(str, controller)),
        })
    validate_worker_specifications(rows)
    return rows
