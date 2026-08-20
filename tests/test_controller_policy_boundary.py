from __future__ import annotations

import pytest

from th06_rl.retail.model import PLAYER_ALIVE, PLAYER_INVULNERABLE
from th06_rl.th06.controller import _nmnb_policy_actions


def test_nmnb_policy_choice_is_forced_after_a_hit() -> None:
    hard = ("left", "stay", "right")

    assert _nmnb_policy_actions(PLAYER_ALIVE, hard, "stay") == hard
    assert _nmnb_policy_actions(
        PLAYER_INVULNERABLE, hard, "stay"
    ) == ("stay",)


def test_nmnb_policy_boundary_rejects_invalid_lifecycle_or_baseline() -> None:
    with pytest.raises(ValueError, match="source-safe set"):
        _nmnb_policy_actions(PLAYER_ALIVE, ("left",), "stay")
    with pytest.raises(ValueError, match="inactive"):
        _nmnb_policy_actions(2, ("stay",), "stay")
