#!/usr/bin/env python3
"""Rebuild the ignored TH06 ECL reference cache from the retail ST archive."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = (
    REPOSITORY / "reference/th06-game-original/th06/紅魔郷ST.DAT"
)
DEFAULT_OUTPUT = REPOSITORY / "reference/th06-ecl-original"
DEFAULT_MANIFEST = REPOSITORY / "config/th06_ecl_reference.json"


@dataclass(frozen=True)
class Pbg3Entry:
    filename: str
    checksum: int
    data_offset: int
    uncompressed_size: int


class BitReader:
    def __init__(self, data: bytes) -> None:
        self.data = data
        self.bit_offset = 0

    def read_bits(self, count: int) -> int:
        if count < 0 or self.bit_offset + count > len(self.data) * 8:
            raise ValueError("truncated PBG3 bitstream")
        value = 0
        for _ in range(count):
            byte = self.data[self.bit_offset // 8]
            shift = 7 - self.bit_offset % 8
            value = (value << 1) | ((byte >> shift) & 1)
            self.bit_offset += 1
        return value

    def read_varint(self) -> int:
        width = (8, 16, 24, 32)[self.read_bits(2)]
        return self.read_bits(width)

    def read_string(self) -> str:
        raw = bytearray()
        for _ in range(256):
            value = self.read_bits(8)
            if value == 0:
                return raw.decode("ascii")
            raw.append(value)
        raise ValueError("PBG3 filename exceeds the source's 256-byte field")

    def seek_byte(self, offset: int) -> None:
        if not 0 <= offset < len(self.data):
            raise ValueError("PBG3 byte offset is outside the archive")
        self.bit_offset = offset * 8


def parse_entries(archive: bytes) -> tuple[tuple[Pbg3Entry, ...], int]:
    reader = BitReader(archive)
    magic = bytes(reader.read_bits(8) for _ in range(4))
    if magic != b"PBG3":
        raise ValueError("archive does not have the TH06 PBG3 magic")
    count = reader.read_varint()
    table_offset = reader.read_varint()
    if not 0 < count <= 4096:
        raise ValueError(f"invalid PBG3 entry count: {count}")
    reader.seek_byte(table_offset)
    entries = []
    names = set()
    for _ in range(count):
        reader.read_varint()  # source field unk2
        reader.read_varint()  # source field unk1
        checksum = reader.read_varint()
        data_offset = reader.read_varint()
        uncompressed_size = reader.read_varint()
        filename = reader.read_string()
        if filename in names:
            raise ValueError(f"duplicate PBG3 entry: {filename}")
        if not 0 <= data_offset < table_offset:
            raise ValueError(f"invalid data offset for PBG3 entry: {filename}")
        names.add(filename)
        entries.append(Pbg3Entry(
            filename, checksum, data_offset, uncompressed_size
        ))
    if any(
        left.data_offset >= right.data_offset
        for left, right in zip(entries, entries[1:])
    ):
        raise ValueError("PBG3 entries are not in increasing data-offset order")
    return tuple(entries), table_offset


def decompress_entry(
    archive: bytes,
    entries: tuple[Pbg3Entry, ...],
    table_offset: int,
    index: int,
) -> bytes:
    entry = entries[index]
    end = entries[index + 1].data_offset if index + 1 < len(entries) else table_offset
    compressed = archive[entry.data_offset:end]
    reader = BitReader(compressed)
    dictionary = bytearray(0x2000)
    dictionary_head = 1
    output = bytearray()

    def write(value: int) -> None:
        nonlocal dictionary_head
        if len(output) >= entry.uncompressed_size:
            raise ValueError(f"PBG3 entry expands past its declared size: {entry.filename}")
        output.append(value)
        dictionary[dictionary_head] = value
        dictionary_head = (dictionary_head + 1) & 0x1FFF

    while True:
        if reader.read_bits(1):
            write(reader.read_bits(8))
            continue
        match_offset = reader.read_bits(13)
        if match_offset == 0:
            break
        match_length = reader.read_bits(4) + 3
        for offset in range(match_length):
            write(dictionary[(match_offset + offset) & 0x1FFF])

    consumed_bytes = (reader.bit_offset + 7) // 8
    checksum = sum(compressed[:consumed_bytes]) & 0xFFFFFFFF
    if checksum != entry.checksum:
        raise ValueError(
            f"PBG3 checksum mismatch for {entry.filename}: "
            f"{checksum:#x} != {entry.checksum:#x}"
        )
    if len(output) != entry.uncompressed_size:
        raise ValueError(
            f"PBG3 size mismatch for {entry.filename}: "
            f"{len(output)} != {entry.uncompressed_size}"
        )
    return bytes(output)


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "th06-ecl-reference-v1":
        raise ValueError("unsupported TH06 ECL reference manifest")
    entries = manifest.get("entries")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("TH06 ECL reference manifest has no entries")
    for filename, expected in entries.items():
        if Path(filename).name != filename or not filename.endswith(".ecl"):
            raise ValueError(f"unsafe ECL manifest path: {filename}")
        if set(expected) != {"sha256", "size"}:
            raise ValueError(f"invalid ECL manifest row: {filename}")
    return manifest


def rebuild(
    archive_path: Path,
    output_dir: Path,
    manifest_path: Path,
    *,
    verify_only: bool,
) -> None:
    manifest = load_manifest(manifest_path)
    archive = archive_path.read_bytes()
    archive_hash = sha256(archive)
    if archive_hash != manifest["archive_sha256"]:
        raise ValueError(
            "retail ST archive hash mismatch: "
            f"{archive_hash} != {manifest['archive_sha256']}"
        )
    entries, table_offset = parse_entries(archive)
    index_by_name = {entry.filename: index for index, entry in enumerate(entries)}
    decoded = {}
    for filename, expected in manifest["entries"].items():
        if filename not in index_by_name:
            raise ValueError(f"retail ST archive is missing {filename}")
        payload = decompress_entry(
            archive, entries, table_offset, index_by_name[filename]
        )
        if len(payload) != expected["size"] or sha256(payload) != expected["sha256"]:
            raise ValueError(f"decoded ECL does not match manifest: {filename}")
        decoded[filename] = payload

    if verify_only:
        for filename, payload in decoded.items():
            target = output_dir / filename
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"ECL reference cache mismatch: {target}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, payload in decoded.items():
        target = output_dir / filename
        if target.exists():
            if not target.is_file() or target.read_bytes() != payload:
                raise ValueError(f"refusing to overwrite mismatched reference: {target}")
            continue
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_bytes(payload)
        temporary.replace(target)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rebuild(
        args.archive.resolve(),
        args.output_dir.resolve(),
        args.manifest.resolve(),
        verify_only=args.verify_only,
    )
    print(f"TH06 ECL reference cache verified: {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
