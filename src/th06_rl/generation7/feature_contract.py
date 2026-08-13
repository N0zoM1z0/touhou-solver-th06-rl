"""Availability metadata and fail-closed feature linting for Generation 7."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from ..hazard_representation import (
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    HAZARD_SUMMARY_FEATURE_NAMES,
    HISTORY_FEATURE_NAMES,
)
from ..th06.learning_adapter import ACTION_FEATURE_NAMES, OBSERVATION_FEATURE_NAMES


class Availability(str, Enum):
    DECISION = "decision"
    AFTER_ACTION = "after-action"
    EPISODE_END = "episode-end"
    PRIVILEGED_DIAGNOSTIC = "privileged-diagnostic"


class FeatureUse(str, Enum):
    ACTOR = "actor"
    NUISANCE = "nuisance"
    DIAGNOSTIC = "diagnostic"


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    family: str
    availability: Availability
    online_available: bool
    allowed_uses: frozenset[FeatureUse]

    def __post_init__(self) -> None:
        if not self.name or not self.family or not self.allowed_uses:
            raise ValueError("feature metadata is incomplete")
        if (
            self.online_available
            and self.availability is not Availability.DECISION
        ):
            raise ValueError("post-decision feature cannot be online-available")


class FeatureCatalog:
    def __init__(self, specs: tuple[FeatureSpec, ...]) -> None:
        by_name = {spec.name: spec for spec in specs}
        if len(by_name) != len(specs):
            raise ValueError("feature catalog contains duplicate names")
        self._specs = by_name

    def spec(self, name: str) -> FeatureSpec:
        try:
            return self._specs[name]
        except KeyError as error:
            raise ValueError(f"feature has no availability metadata: {name}") from error

    def lint(
        self,
        names: tuple[str, ...],
        *,
        use: FeatureUse,
        require_online: bool,
    ) -> tuple[FeatureSpec, ...]:
        if not names or len(set(names)) != len(names):
            raise ValueError("feature list must be nonempty and unique")
        result = tuple(self.spec(name) for name in names)
        forbidden = tuple(
            spec.name
            for spec in result
            if use not in spec.allowed_uses
            or (
                require_online
                and (
                    not spec.online_available
                    or spec.availability is not Availability.DECISION
                )
            )
        )
        if forbidden:
            raise ValueError(
                f"features are unavailable for {use.value}: {', '.join(forbidden)}"
            )
        return result

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._specs)


_ACTOR_AND_NUISANCE = frozenset({FeatureUse.ACTOR, FeatureUse.NUISANCE})
_NUISANCE_AND_DIAGNOSTIC = frozenset({
    FeatureUse.NUISANCE,
    FeatureUse.DIAGNOSTIC,
})
_DIAGNOSTIC_ONLY = frozenset({FeatureUse.DIAGNOSTIC})


def _decision_spec(name: str, family: str) -> FeatureSpec:
    return FeatureSpec(
        name=name,
        family=family,
        availability=Availability.DECISION,
        online_available=True,
        allowed_uses=_ACTOR_AND_NUISANCE,
    )


def _default_specs() -> tuple[FeatureSpec, ...]:
    specs = []
    specs.extend(
        _decision_spec(f"observation:{name}", "observation")
        for name in OBSERVATION_FEATURE_NAMES
    )
    specs.extend(
        _decision_spec(f"action:{name}", "action")
        for name in ACTION_FEATURE_NAMES
    )
    specs.extend(
        _decision_spec(f"delta_from_baseline:{name}", "action-delta")
        for name in ACTION_FEATURE_NAMES
    )
    specs.extend((
        _decision_spec("matches_baseline", "action-identity"),
        _decision_spec("matches_current", "action-identity"),
        _decision_spec("option_index_log", "causal-time"),
    ))
    specs.extend(
        _decision_spec(f"history:{name}", "causal-history")
        for name in HISTORY_FEATURE_NAMES
    )
    specs.extend(
        _decision_spec(f"hazard:{name}", "observed-hazard")
        for name in HAZARD_PRIMITIVE_FEATURE_NAMES
    )
    specs.extend(
        _decision_spec(f"hazard_summary:{name}", "observed-hazard-summary")
        for name in HAZARD_SUMMARY_FEATURE_NAMES
    )
    specs.extend((
        FeatureSpec(
            "cohort_id",
            "behavior-nuisance",
            Availability.DECISION,
            False,
            _NUISANCE_AND_DIAGNOSTIC,
        ),
        FeatureSpec(
            "stage_id",
            "scope-nuisance",
            Availability.DECISION,
            False,
            _NUISANCE_AND_DIAGNOSTIC,
        ),
        FeatureSpec(
            "source_context",
            "privileged-source",
            Availability.PRIVILEGED_DIAGNOSTIC,
            False,
            _DIAGNOSTIC_ONLY,
        ),
        FeatureSpec(
            "rng_seed",
            "privileged-rng",
            Availability.PRIVILEGED_DIAGNOSTIC,
            False,
            _DIAGNOSTIC_ONLY,
        ),
        FeatureSpec(
            "episode_option_count",
            "future-length",
            Availability.EPISODE_END,
            False,
            _DIAGNOSTIC_ONLY,
        ),
        FeatureSpec(
            "remaining_option_count",
            "future-length",
            Availability.EPISODE_END,
            False,
            _DIAGNOSTIC_ONLY,
        ),
        FeatureSpec(
            "future_hit_suffix",
            "outcome",
            Availability.AFTER_ACTION,
            False,
            _DIAGNOSTIC_ONLY,
        ),
    ))
    return tuple(specs)


DEFAULT_FEATURE_CATALOG = FeatureCatalog(_default_specs())


def compact_actor_feature_names() -> tuple[str, ...]:
    names = (
        *(f"observation:{name}" for name in OBSERVATION_FEATURE_NAMES),
        *(f"action:{name}" for name in ACTION_FEATURE_NAMES),
        *(f"delta_from_baseline:{name}" for name in ACTION_FEATURE_NAMES),
        "matches_baseline",
        "matches_current",
        "option_index_log",
    )
    DEFAULT_FEATURE_CATALOG.lint(
        names,
        use=FeatureUse.ACTOR,
        require_online=True,
    )
    return names


def richer_causal_context_feature_names() -> tuple[str, ...]:
    names = (
        *(f"history:{name}" for name in HISTORY_FEATURE_NAMES),
        *(f"hazard_summary:{name}" for name in HAZARD_SUMMARY_FEATURE_NAMES),
    )
    DEFAULT_FEATURE_CATALOG.lint(
        names,
        use=FeatureUse.ACTOR,
        require_online=True,
    )
    return names
