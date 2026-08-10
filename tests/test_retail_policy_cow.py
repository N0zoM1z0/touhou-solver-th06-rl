from __future__ import annotations

from scripts.label_retail_policy_cow import _action_sha256


def test_policy_cow_action_digest_is_ordered_and_delimited() -> None:
    assert _action_sha256(("down", "left")) != _action_sha256(("down_left",))
    assert _action_sha256(("down", "left")) != _action_sha256(("left", "down"))
