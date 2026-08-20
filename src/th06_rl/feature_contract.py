"""Deployment-time availability contract for offline actor features."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

from th06_rl.hazard_representation import HISTORY_FEATURE_NAMES
from th06_rl.th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES


FEATURE_AVAILABILITY_SCHEMA = "th06-rl-feature-availability-v1"


@dataclass(frozen=True)
class FeatureAvailability:
    name: str
    domain: str
    source: str
    earliest_availability: str
    online_deployable: bool
    privileged_diagnostic_only: bool


def _specs(names: tuple[str, ...], domain: str) -> tuple[FeatureAvailability, ...]:
    return tuple(
        FeatureAvailability(
            name,
            domain,
            "same-paused-epoch-online-adapter",
            "before-policy-decision",
            True,
            False,
        )
        for name in names
    )


ACTOR_FEATURES = (
    *_specs(OBSERVATION_FEATURE_NAMES, "observation"),
    *_specs(ACTION_FEATURE_NAMES, "action"),
    *_specs(HISTORY_FEATURE_NAMES, "causal-history"),
)


def actor_feature_manifest() -> dict[str, object]:
    return {
        "schema": FEATURE_AVAILABILITY_SCHEMA,
        "features": [asdict(specification) for specification in ACTOR_FEATURES],
    }


def _validate_named_values(
    rows,
    expected: tuple[str, ...],
    *,
    label: str,
) -> None:
    names = []
    for row in rows:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError(f"{label} row is not a name/value pair")
        name, value = str(row[0]), float(row[1])
        if not math.isfinite(value):
            raise ValueError(f"{label} feature {name!r} is non-finite")
        names.append(name)
    if tuple(names) != expected:
        raise ValueError(
            f"{label} feature availability/order mismatch; "
            "future, privileged, missing, and renamed fields are forbidden"
        )


def validate_actor_feature_rows(
    observation_features,
    action_features,
    history_features,
) -> None:
    """Reject any actor input absent from the immutable online interface."""
    _validate_named_values(
        observation_features,
        OBSERVATION_FEATURE_NAMES,
        label="observation",
    )
    action_names = []
    for row in action_features:
        if not isinstance(row, (tuple, list)) or len(row) != 2:
            raise ValueError("action feature row lacks action/value pairs")
        action = str(row[0])
        if not action or action in action_names:
            raise ValueError("action feature actions are empty or duplicated")
        action_names.append(action)
        _validate_named_values(
            row[1],
            ACTION_FEATURE_NAMES,
            label=f"action:{action}",
        )
    if not action_names:
        raise ValueError("actor requires at least one candidate action")
    _validate_named_values(
        history_features,
        HISTORY_FEATURE_NAMES,
        label="causal-history",
    )
