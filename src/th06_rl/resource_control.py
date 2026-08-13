"""Host-sharing resource contracts for offline learner processes."""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Iterable


MAXIMUM_TRAINING_CPUS = 32


@dataclass(frozen=True)
class CpuAffinityContract:
    requested_maximum: int
    inherited: tuple[int, ...]
    effective: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "requested_maximum": self.requested_maximum,
            "inherited": list(self.inherited),
            "effective": list(self.effective),
            "effective_count": len(self.effective),
            "authority": "linux-sched-setaffinity",
        }


def bounded_cpu_set(
    inherited: Iterable[int], *, maximum: int
) -> tuple[int, ...]:
    """Select a deterministic subset without escaping an inherited cgroup."""
    available = tuple(sorted(set(inherited)))
    if not 1 <= maximum <= MAXIMUM_TRAINING_CPUS:
        raise ValueError(
            "training CPU maximum must be between 1 and "
            f"{MAXIMUM_TRAINING_CPUS}"
        )
    if not available:
        raise RuntimeError("process inherited an empty CPU affinity set")
    return available[:maximum]


def enforce_training_cpu_affinity(
    maximum: int = MAXIMUM_TRAINING_CPUS,
) -> CpuAffinityContract:
    """Hard-limit this process and every future child to at most 32 CPUs."""
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        raise RuntimeError("offline training requires Linux CPU-affinity support")
    inherited = tuple(sorted(os.sched_getaffinity(0)))
    selected = bounded_cpu_set(inherited, maximum=maximum)
    os.sched_setaffinity(0, selected)
    effective = tuple(sorted(os.sched_getaffinity(0)))
    if effective != selected:
        raise RuntimeError(
            f"CPU-affinity enforcement failed: selected={selected}, effective={effective}"
        )
    return CpuAffinityContract(
        requested_maximum=maximum,
        inherited=inherited,
        effective=effective,
    )
