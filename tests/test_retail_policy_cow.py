from __future__ import annotations

import pytest

from scripts.label_retail_policy_cow import (
    _action_sha256,
    _validate_recorded_local_actions,
)


def test_policy_cow_action_digest_is_ordered_and_delimited() -> None:
    assert _action_sha256(("down", "left")) != _action_sha256(("down_left",))
    assert _action_sha256(("down", "left")) != _action_sha256(("left", "down"))


def test_policy_cow_rejects_hard_only_action_before_source_branching() -> None:
    transition = {"legal_actions": ["down_left"]}

    assert _validate_recorded_local_actions(("down_left",), transition) == (
        "down_left",
    )
    with pytest.raises(ValueError, match="outside the recorded local set"):
        _validate_recorded_local_actions(("stay", "down_left"), transition)
