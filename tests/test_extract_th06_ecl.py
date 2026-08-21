from __future__ import annotations

import importlib.util
from pathlib import Path
import sys


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/extract_th06_ecl.py"
SPEC = importlib.util.spec_from_file_location("extract_th06_ecl", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


class BitWriter:
    def __init__(self) -> None:
        self.bits: list[int] = []

    def write(self, value: int, width: int) -> None:
        self.bits.extend((value >> shift) & 1 for shift in range(width - 1, -1, -1))

    def varint(self, value: int) -> None:
        if value < 1 << 8:
            header, width = 0, 8
        elif value < 1 << 16:
            header, width = 1, 16
        elif value < 1 << 24:
            header, width = 2, 24
        else:
            header, width = 3, 32
        self.write(header, 2)
        self.write(value, width)

    def bytes(self) -> bytes:
        result = bytearray((len(self.bits) + 7) // 8)
        for index, bit in enumerate(self.bits):
            result[index // 8] |= bit << (7 - index % 8)
        return bytes(result)


def _compressed_abcabc() -> bytes:
    writer = BitWriter()
    for value in b"ABC":
        writer.write(1, 1)
        writer.write(value, 8)
    writer.write(0, 1)
    writer.write(1, 13)
    writer.write(0, 4)
    writer.write(0, 1)
    writer.write(0, 13)
    return writer.bytes()


def _archive() -> bytes:
    payload = _compressed_abcabc()
    data_offset = 16
    table_offset = 64
    header = BitWriter()
    for value in b"PBG3":
        header.write(value, 8)
    header.varint(1)
    header.varint(table_offset)
    table = BitWriter()
    table.varint(0)
    table.varint(0)
    table.varint(sum(payload))
    table.varint(data_offset)
    table.varint(6)
    for value in b"ecldata1.ecl\0":
        table.write(value, 8)
    archive = bytearray(table_offset + len(table.bytes()))
    archive[:len(header.bytes())] = header.bytes()
    archive[data_offset:data_offset + len(payload)] = payload
    archive[table_offset:] = table.bytes()
    return bytes(archive)


def test_pbg3_parser_replays_literal_and_dictionary_tokens() -> None:
    archive = _archive()
    entries, table_offset = MODULE.parse_entries(archive)

    assert [entry.filename for entry in entries] == ["ecldata1.ecl"]
    assert MODULE.decompress_entry(archive, entries, table_offset, 0) == b"ABCABC"
