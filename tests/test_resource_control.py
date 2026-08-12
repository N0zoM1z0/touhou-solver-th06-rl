from __future__ import annotations

import pytest

from th06_rl.resource_control import bounded_cpu_set


def test_cpu_affinity_selects_first_inherited_cpus_without_escaping() -> None:
    assert bounded_cpu_set((9, 4, 7, 4), maximum=2) == (4, 7)
    assert bounded_cpu_set((12, 15), maximum=32) == (12, 15)


@pytest.mark.parametrize("maximum", (0, 33))
def test_cpu_affinity_rejects_outside_host_sharing_contract(maximum: int) -> None:
    with pytest.raises(ValueError, match="between 1 and 32"):
        bounded_cpu_set((0, 1), maximum=maximum)
