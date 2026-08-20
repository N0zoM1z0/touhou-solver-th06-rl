"""Source-exact TH06 RNG state for nominal ECL forecasting."""

from __future__ import annotations

from dataclasses import dataclass
import struct


def _f32(value: float) -> float:
    return struct.unpack("<f", struct.pack("<f", value))[0]


@dataclass
class RngState:
    seed: int
    generation_count: int

    def u16(self) -> int:
        value = ((self.seed ^ 0x9630) - 0x6553) & 0xFFFF
        self.seed = (((value & 0xC000) >> 14) + value * 4) & 0xFFFF
        self.generation_count += 1
        return self.seed

    def u32(self) -> int:
        return (self.u16() << 16) | self.u16()

    def f32_zero_to_one(self) -> float:
        return _f32(_f32(float(self.u32())) / _f32(float(0xFFFFFFFF)))

    def f32_in_range(self, value: float) -> float:
        return _f32(self.f32_zero_to_one() * value)

    def u32_in_range(self, value: int) -> int:
        return self.u32() % value if value else 0
