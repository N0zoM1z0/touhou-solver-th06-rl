from __future__ import annotations

import pytest

from th06_rl.feature_contract import (
    ACTOR_FEATURES,
    actor_feature_manifest,
    validate_actor_feature_rows,
)
from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.th06.learning_adapter import (
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)


def _rows(names):
    return tuple((name, float(index)) for index, name in enumerate(names))


def test_actor_contract_contains_only_predecision_online_features() -> None:
    manifest = actor_feature_manifest()

    assert manifest["schema"] == "th06-rl-feature-availability-v1"
    assert len(manifest["features"]) == len(ACTOR_FEATURES)
    assert all(specification.online_deployable for specification in ACTOR_FEATURES)
    assert not any(
        specification.privileged_diagnostic_only for specification in ACTOR_FEATURES
    )
    assert {specification.earliest_availability for specification in ACTOR_FEATURES} == {
        "before-policy-decision"
    }


def test_actor_rows_match_the_online_interface_exactly() -> None:
    validate_actor_feature_rows(
        _rows(OBSERVATION_FEATURE_NAMES),
        (("stay", _rows(ACTION_FEATURE_NAMES)),),
        _rows(HISTORY_FEATURE_NAMES),
    )


def test_future_length_or_privileged_feature_cannot_enter_actor() -> None:
    leaked = (*_rows(OBSERVATION_FEATURE_NAMES), ("final_episode_length", 10.0))

    with pytest.raises(ValueError, match="future, privileged"):
        validate_actor_feature_rows(
            leaked,
            (("stay", _rows(ACTION_FEATURE_NAMES)),),
            _rows(HISTORY_FEATURE_NAMES),
        )
