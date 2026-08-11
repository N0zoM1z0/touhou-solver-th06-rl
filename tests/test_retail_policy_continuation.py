from __future__ import annotations

import pytest

from scripts.audit_retail_policy_continuation import _is_recorded_policy_call
from th06_rl.headless_corpus import retail_policy_source_context_id
from th06_rl.headless_geometry import HeadlessAuthorityUnavailable


def test_policy_restore_excludes_input_lease_with_carried_proposal() -> None:
    transition = {"proposed_action": "down_left"}

    assert _is_recorded_policy_call(transition, {"reason": "ok"}) is True
    assert _is_recorded_policy_call(transition, {"reason": "input-lease"}) is False
    assert (
        _is_recorded_policy_call({"proposed_action": None}, {"reason": "ok"}) is False
    )


def test_retail_policy_source_context_reconstructs_boss_partition() -> None:
    observation = {
        "spell_active": True,
        "enemies": [
            {
                "slot": 0,
                "boss": True,
                "boss_id": 0,
                "ecl_sub": 31,
                "life_callback_sub": 31,
                "timer_callback_sub": 19,
            }
        ],
    }

    assert retail_policy_source_context_id(observation) == (
        "boss:0:sub31:life_cb31:timer_cb19:spell"
    )


def test_retail_policy_source_context_reconstructs_timeline_partition() -> None:
    observation = {
        "enemies": [],
        "source_context": {
            "next": {"time": 440, "opcode": 0, "arg0": 1},
        },
    }

    assert retail_policy_source_context_id(observation) == (
        "timeline:before-t440:op0:arg1"
    )


def test_retail_policy_source_context_fails_closed_on_partial_boss() -> None:
    observation = {
        "spell_active": True,
        "enemies": [
            {
                "slot": 0,
                "boss": True,
                "boss_id": 0,
                "ecl_sub": 31,
                "life_callback_sub": 31,
            }
        ],
    }

    with pytest.raises(HeadlessAuthorityUnavailable, match="boss policy"):
        retail_policy_source_context_id(observation)
