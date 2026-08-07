#!/usr/bin/env python3
"""Recompress closed local TH06 corpus spools and archive them atomically."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile


CHUNK_BYTES = 1024 * 1024


class _HashingWriter:
    def __init__(self, raw) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()

    def write(self, payload: bytes) -> int:
        self.digest.update(payload)
        return self.raw.write(payload)

    def flush(self) -> None:
        self.raw.flush()

    def tell(self) -> int:
        return self.raw.tell()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _atomic_json(path: Path, value: object) -> None:
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(_canonical(value) + b"\n")
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _recompress_shard(source: Path, temporary: Path) -> tuple[str, int, int]:
    uncompressed = 0
    with gzip.open(source, "rb") as input_stream, temporary.open("wb") as raw:
        hashing = _HashingWriter(raw)
        with gzip.GzipFile(
            fileobj=hashing,
            mode="wb",
            compresslevel=3,
            mtime=0,
            filename="",
        ) as output:
            while True:
                payload = input_stream.read(CHUNK_BYTES)
                if not payload:
                    break
                uncompressed += len(payload)
                output.write(payload)
        raw.flush()
        os.fsync(raw.fileno())
        digest = hashing.digest.hexdigest()
    return digest, temporary.stat().st_size, uncompressed


def finalize_run(run_dir: Path) -> dict[str, object]:
    manifest_path = run_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    compression = manifest.get("storage_compression", "gzip-3")
    if compression == "gzip-3":
        return manifest
    if compression != "gzip-0":
        raise ValueError(f"unsupported spool compression {compression!r}: {run_dir}")

    prepared = []
    new_shards = []
    for index, shard in enumerate(manifest.get("shards", ())):
        source = run_dir / str(shard["path"])
        temporary = run_dir / f".recompress-{index:06d}-{os.getpid()}.partial"
        digest, compressed_bytes, uncompressed_bytes = _recompress_shard(
            source, temporary
        )
        if uncompressed_bytes != int(shard["uncompressed_bytes"]):
            raise ValueError(f"uncompressed size mismatch while finalizing {source}")
        final = run_dir / (
            f"{shard['stream']}-{index:06d}-{digest[:16]}.jsonl.gz"
        )
        updated = dict(shard)
        updated.update({
            "path": final.name,
            "sha256": digest,
            "compressed_bytes": compressed_bytes,
        })
        prepared.append((source, temporary, final))
        new_shards.append(updated)

    compressed_total = sum(int(item["compressed_bytes"]) for item in new_shards)
    archive_limit = int(
        manifest.get("archive_max_run_bytes", manifest.get("max_run_bytes", 0))
    )
    if archive_limit and compressed_total > archive_limit:
        raise ValueError(
            f"final corpus exceeds archive limit: {compressed_total} > {archive_limit}"
        )
    for _source, temporary, final in prepared:
        os.replace(temporary, final)

    manifest["shards"] = new_shards
    manifest["compressed_bytes"] = compressed_total
    manifest["storage_compression"] = "gzip-3"
    _atomic_json(manifest_path, manifest)
    run_path = run_dir / "run.json"
    run = json.loads(run_path.read_text(encoding="utf-8"))
    run.setdefault("storage", {})["compression"] = "gzip-3"
    run["storage"]["finalized_from"] = "gzip-0-local-spool"
    _atomic_json(run_path, run)
    retained = {str(item["path"]) for item in new_shards}
    for source, _temporary, _final in prepared:
        if source.name not in retained:
            source.unlink()
    return manifest


def _copy_verified_run(
    source: Path,
    destination_root: Path,
    manifest: dict[str, object],
) -> Path:
    destination = destination_root / source.name
    declared = {
        str(item["path"]): str(item["sha256"])
        for item in manifest.get("shards", ())
    }
    if destination.exists():
        existing = json.loads(
            (destination / "manifest.json").read_text(encoding="utf-8")
        )
        if existing.get("run_id") != manifest.get("run_id"):
            raise ValueError(f"archive collision at {destination}")
        return destination

    destination_root.mkdir(parents=True, exist_ok=True)
    staging = destination_root / f".{source.name}.import-{os.getpid()}"
    staging.mkdir(parents=False, exist_ok=False)
    try:
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            relative = path.relative_to(source)
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with path.open("rb") as input_stream, target.open("wb") as output:
                while True:
                    payload = input_stream.read(CHUNK_BYTES)
                    if not payload:
                        break
                    output.write(payload)
                    if relative.as_posix() in declared:
                        digest.update(payload)
                output.flush()
                os.fsync(output.fileno())
            expected = declared.get(relative.as_posix())
            if expected is not None and digest.hexdigest() != expected:
                raise ValueError(f"archive copy hash mismatch for {path}")
        os.replace(staging, destination)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    return destination


def finalize_spool(spool_root: Path, archive_root: Path) -> list[Path]:
    if not spool_root.exists():
        return []
    archived = []
    for run_dir in sorted(path for path in spool_root.iterdir() if path.is_dir()):
        if not (run_dir / "manifest.json").is_file():
            continue
        manifest = finalize_run(run_dir)
        destination = _copy_verified_run(run_dir, archive_root, manifest)
        shutil.rmtree(run_dir)
        archived.append(destination)
    return archived


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spool_root", type=Path)
    parser.add_argument("archive_root", type=Path)
    args = parser.parse_args()
    try:
        archived = finalize_spool(args.spool_root, args.archive_root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"corpus spool finalization failed closed: {type(error).__name__}: {error}")
        return 1
    for path in archived:
        print(f"archived finalized corpus: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
