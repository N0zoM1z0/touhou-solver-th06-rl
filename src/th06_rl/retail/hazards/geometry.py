"""Shared exact signed-clearance primitive."""

from __future__ import annotations

import math


def signed_clearance(
    player_x: float,
    player_y: float,
    player_half_width: float,
    player_half_height: float,
    hazard: tuple[float, float, float, float],
) -> float:
    left, top, right, bottom = hazard
    gap_x = max(left - (player_x + player_half_width), (player_x - player_half_width) - right)
    gap_y = max(top - (player_y + player_half_height), (player_y - player_half_height) - bottom)
    if gap_x <= 0.0 and gap_y <= 0.0:
        return max(gap_x, gap_y)
    return math.hypot(max(0.0, gap_x), max(0.0, gap_y))
