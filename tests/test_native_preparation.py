from __future__ import annotations

import pytest

from th06_rl.native import Aabb, LaserRect, NativeKernel, PackedHazards


def test_prepared_prefix_reuses_the_exact_native_buffers() -> None:
    hazards = PackedHazards(
        aabb_frames=(
            (Aabb(1.0, 2.0, 3.0, 4.0),),
            (Aabb(5.0, 6.0, 7.0, 8.0),),
        ),
        laser_frames=(
            (LaserRect(9.0, 10.0, 0.5, 11.0, 12.0, 13.0),),
            (LaserRect(14.0, 15.0, 1.5, 16.0, 17.0, 18.0),),
        ),
    )

    prepared = NativeKernel.prepare_hazards(hazards)
    hard = prepared.prefix(1)

    assert prepared.horizon == 2
    assert hard.horizon == 1
    assert all(
        hard_buffer is full_buffer
        for hard_buffer, full_buffer in zip(
            hard.native_args,
            prepared.native_args,
            strict=True,
        )
    )
    assert list(prepared.aabb_offsets) == [0, 1, 2]
    assert list(prepared.laser_offsets) == [0, 1, 2]


def test_prepared_prefix_cannot_expand_or_be_empty() -> None:
    hazards = PackedHazards(
        aabb_frames=((), ()),
        laser_frames=((), ()),
    )
    prepared = NativeKernel.prepare_hazards(hazards)

    with pytest.raises(ValueError):
        prepared.prefix(0)
    with pytest.raises(ValueError):
        prepared.prefix(3)
