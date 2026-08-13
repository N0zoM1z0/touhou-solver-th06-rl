"""Bounded host scheduling contract for latency-sensitive Wine children."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable


MINIMUM_NICE = -15
MAXIMUM_NICE = 0


def parse_cpu_list(value: str) -> tuple[int, ...]:
    """Parse the same explicit CPU-list grammar accepted by ``taskset``."""
    result: set[int] = set()
    for component in value.split(","):
        bounds = component.split("-", 1)
        if not component or len(bounds) not in (1, 2):
            raise ValueError("invalid bounded-priority CPU list")
        try:
            first = int(bounds[0])
            last = int(bounds[-1])
        except ValueError as error:
            raise ValueError("invalid bounded-priority CPU list") from error
        if first < 0 or last < first:
            raise ValueError("invalid bounded-priority CPU range")
        result.update(range(first, last + 1))
    if not result:
        raise ValueError("bounded-priority CPU list is empty")
    return tuple(sorted(result))


def validate_nice(value: int) -> int:
    """Keep latency priority useful without admitting real-time scheduling."""
    value = int(value)
    if not MINIMUM_NICE <= value <= MAXIMUM_NICE:
        raise ValueError(
            f"process nice must be between {MINIMUM_NICE} and {MAXIMUM_NICE}"
        )
    return value


@dataclass(frozen=True)
class ProcessPriorityContract:
    nice: int
    cpus: tuple[int, ...]

    def __post_init__(self) -> None:
        validate_nice(self.nice)
        normalized = tuple(sorted(set(self.cpus)))
        if not normalized or normalized != self.cpus or normalized[0] < 0:
            raise ValueError("bounded-priority CPU set is invalid")

    @classmethod
    def from_values(cls, *, nice: int, cpu_list: str) -> "ProcessPriorityContract":
        return cls(nice=validate_nice(nice), cpus=parse_cpu_list(cpu_list))

    def verify_available(self, available: Iterable[int]) -> None:
        if not set(self.cpus) <= set(available):
            raise ValueError("bounded-priority CPU set escapes inherited affinity")

    def as_dict(self) -> dict[str, object]:
        return {
            "authority": "linux-setpriority-and-sched-setaffinity",
            "scheduler": "SCHED_OTHER",
            "nice": self.nice,
            "cpus": list(self.cpus),
        }
