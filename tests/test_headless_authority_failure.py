from __future__ import annotations

import pytest

from scripts.audit_headless_authority_failure import (
    classify_differential,
    expand_action_trace,
)
from scripts.label_headless_feasibility_oracle import action_trace_sha256


def test_expand_action_trace_rechecks_count_and_digest() -> None:
    actions = ["stay", "left", "left"]
    branch = {
        "actions_issued": len(actions),
        "action_trace_rle": [
            {"action": "stay", "ticks": 1},
            {"action": "left", "ticks": 2},
        ],
        "action_trace_sha256": action_trace_sha256(actions),
    }

    assert expand_action_trace(branch) == actions

    branch["action_trace_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="SHA-256"):
        expand_action_trace(branch)


def test_authority_failure_classification_separates_geometry_and_margin() -> None:
    assert classify_differential(
        configured=set(),
        margin_zero={"left"},
        source_safe={"left"},
    ) == "conservative-margin-closure"
    assert classify_differential(
        configured=set(),
        margin_zero=set(),
        source_safe={"left"},
    ) == "geometry-model-mismatch"
    assert classify_differential(
        configured=set(),
        margin_zero=set(),
        source_safe=set(),
    ) == "source-immediate-dead-end-under-constant-actions"
    assert classify_differential(
        configured=set(),
        margin_zero=set(),
        source_safe={"left"},
        authority_error="newborn laser lacks history",
    ) == "source-safe-but-native-observation-incomplete"
