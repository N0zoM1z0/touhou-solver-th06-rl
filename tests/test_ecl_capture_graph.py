from __future__ import annotations

import pytest

from th06_rl.retail import native
from th06_rl.retail.model import EclInstruction


class _LinearEclProcess:
    def __init__(self, instructions: int | None) -> None:
        self.instructions = instructions
        self.ecl_program_cache = {}
        self.ecl_subroutines = ()

    def read_ecl_instruction(self, address: int) -> EclInstruction:
        index = (address - 0x10000) // 12
        terminal = self.instructions is not None and index == self.instructions
        return EclInstruction(
            address,
            -1 if terminal else 0,
            -1 if terminal else 0,
            12,
            0xFF,
            "00" * 12,
        )


def test_ecl_graph_capture_exhausts_more_than_old_256_limit() -> None:
    process = _LinearEclProcess(300)

    program = native._read_ecl_program(process, 0x10000)

    assert len(program) == 301
    assert program[-1].time == -1
    assert program[-1].address == 0x10000 + 300 * 12


def test_ecl_graph_capture_never_returns_a_silent_prefix() -> None:
    process = _LinearEclProcess(None)

    with pytest.raises(RuntimeError, match="source capture capacity"):
        native._read_ecl_program(process, 0x10000)
