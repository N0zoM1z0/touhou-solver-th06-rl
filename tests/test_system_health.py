from __future__ import annotations

import pytest

from th06_rl.th06.system_health import SystemMemorySample, below_commit_reserve


def sample(*, total: int = 6, limit: int = 10) -> SystemMemorySample:
    return SystemMemorySample(
        commit_total_bytes=total,
        commit_limit_bytes=limit,
        physical_available_bytes=3,
        controller_private_bytes=2,
    )


def test_commit_reserve_uses_system_commit_headroom() -> None:
    assert below_commit_reserve(sample(), 5) is True
    assert below_commit_reserve(sample(), 4) is False
    assert sample().commit_headroom_bytes == 4


def test_commit_headroom_clamps_incoherent_counter() -> None:
    assert sample(total=11).commit_headroom_bytes == 0


def test_negative_commit_reserve_is_invalid() -> None:
    with pytest.raises(ValueError):
        below_commit_reserve(sample(), -1)
