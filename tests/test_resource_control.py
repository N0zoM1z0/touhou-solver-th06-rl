from __future__ import annotations

import pytest

from th06_rl.resource_control import bounded_cpu_set
from th06_rl.process_priority import (
    ProcessPriorityContract,
    parse_cpu_list,
    validate_nice,
)


def test_cpu_affinity_selects_first_inherited_cpus_without_escaping() -> None:
    assert bounded_cpu_set((9, 4, 7, 4), maximum=2) == (4, 7)
    assert bounded_cpu_set((12, 15), maximum=32) == (12, 15)


@pytest.mark.parametrize("maximum", (0, 33))
def test_cpu_affinity_rejects_outside_host_sharing_contract(maximum: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 32"):
        bounded_cpu_set((0, 1), maximum=maximum)


def test_bounded_priority_is_non_realtime_and_affinity_scoped() -> None:
    contract = ProcessPriorityContract.from_values(
        nice=-10, cpu_list="8-10,12"
    )
    contract.verify_available(range(32))

    assert contract.cpus == (8, 9, 10, 12)
    assert contract.as_dict() == {
        "authority": "linux-setpriority-and-sched-setaffinity",
        "scheduler": "SCHED_OTHER",
        "nice": -10,
        "cpus": [8, 9, 10, 12],
    }


@pytest.mark.parametrize("value", (-16, 1))
def test_bounded_priority_rejects_unbounded_nice(value: int) -> None:
    with pytest.raises(ValueError, match="between -15 and 0"):
        validate_nice(value)


def test_bounded_priority_rejects_invalid_or_escaping_cpu_sets() -> None:
    with pytest.raises(ValueError, match="range"):
        parse_cpu_list("4-2")
    contract = ProcessPriorityContract.from_values(nice=-10, cpu_list="8-9")
    with pytest.raises(ValueError, match="escapes inherited"):
        contract.verify_available((0, 1, 8))
