"""Hash-bound local cache for fully audited factual option episodes."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
from typing import Callable

from .advantage_learning import OptionStep


CACHE_SCHEMA = "th06-rl-audited-option-episode-cache-v1"


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _contract_sha256(paths: tuple[Path, ...]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        digest.update(str(path).encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, raw_temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(raw_temporary)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _identity(
    run_dir: Path,
    cache_root: Path,
    contract_files: tuple[Path, ...],
) -> tuple[Path, str, str, Path, Path]:
    run_dir = run_dir.resolve()
    manifest = run_dir / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise FileNotFoundError(manifest)
    manifest_sha256 = _sha256(manifest)
    contract_sha256 = _contract_sha256(contract_files)
    key = _sha256_bytes(
        f"{CACHE_SCHEMA}\0{manifest_sha256}\0{contract_sha256}".encode()
    )
    root = cache_root.resolve()
    return (
        run_dir,
        manifest_sha256,
        contract_sha256,
        root / f"{key}.pickle",
        root / f"{key}.json",
    )


def _validated_metadata(
    *,
    run_dir: Path,
    manifest_sha256: str,
    contract_sha256: str,
    payload_path: Path,
    metadata_path: Path,
) -> dict[str, object] | None:
    if payload_path.is_file() and metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if (
            metadata.get("schema") == CACHE_SCHEMA
            and metadata.get("manifest_sha256") == manifest_sha256
            and metadata.get("loader_contract_sha256") == contract_sha256
            and metadata.get("run_dir") == str(run_dir)
            and metadata.get("payload_sha256") == _sha256(payload_path)
        ):
            return metadata
        raise ValueError(f"invalid option cache metadata: {metadata_path}")
    if payload_path.exists() or metadata_path.exists():
        raise ValueError(f"partial option cache entry beside: {payload_path}")
    return None


def prime_option_episode_cache(
    run_dir: Path,
    *,
    loader: Callable[[Path], tuple[list[OptionStep], dict[str, object]]],
    cache_root: Path,
    contract_files: tuple[Path, ...],
) -> tuple[bool, int]:
    """Create one audited cache entry without returning its large payload."""
    (
        run_dir,
        manifest_sha256,
        contract_sha256,
        payload_path,
        metadata_path,
    ) = _identity(run_dir, cache_root, contract_files)
    metadata = _validated_metadata(
        run_dir=run_dir,
        manifest_sha256=manifest_sha256,
        contract_sha256=contract_sha256,
        payload_path=payload_path,
        metadata_path=metadata_path,
    )
    if metadata is not None:
        return True, int(metadata["options"])
    rows, report = loader(run_dir)
    payload = pickle.dumps((rows, report), protocol=pickle.HIGHEST_PROTOCOL)
    metadata = {
        "schema": CACHE_SCHEMA,
        "run_dir": str(run_dir),
        "manifest_sha256": manifest_sha256,
        "loader_contract_sha256": contract_sha256,
        "payload_sha256": _sha256_bytes(payload),
        "options": len(rows),
    }
    _atomic_bytes(payload_path, payload)
    _atomic_bytes(
        metadata_path,
        (json.dumps(metadata, indent=2, sort_keys=True) + "\n").encode(),
    )
    return False, len(rows)


def load_cached_option_episode(
    run_dir: Path,
    *,
    loader: Callable[[Path], tuple[list[OptionStep], dict[str, object]]],
    cache_root: Path,
    contract_files: tuple[Path, ...],
) -> tuple[list[OptionStep], dict[str, object], bool]:
    """Load an exact audited result; source or loader changes force a miss."""
    (
        run_dir,
        manifest_sha256,
        contract_sha256,
        payload_path,
        metadata_path,
    ) = _identity(run_dir, cache_root, contract_files)
    cache_hit, _options = prime_option_episode_cache(
        run_dir,
        loader=loader,
        cache_root=cache_root,
        contract_files=contract_files,
    )
    metadata = _validated_metadata(
        run_dir=run_dir,
        manifest_sha256=manifest_sha256,
        contract_sha256=contract_sha256,
        payload_path=payload_path,
        metadata_path=metadata_path,
    )
    if metadata is not None:
        with payload_path.open("rb") as source:
            value = pickle.load(source)  # noqa: S301 - hash-bound local cache
        if (
            isinstance(value, tuple)
            and len(value) == 2
            and isinstance(value[0], list)
            and value[0]
            and all(isinstance(row, OptionStep) for row in value[0])
            and isinstance(value[1], dict)
        ):
            return value[0], value[1], cache_hit
        raise ValueError(f"invalid option cache payload: {payload_path}")
    raise RuntimeError("option cache disappeared after successful prime")
