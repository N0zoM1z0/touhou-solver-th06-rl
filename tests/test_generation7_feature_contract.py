import pytest

from th06_rl.generation7.feature_contract import (
    DEFAULT_FEATURE_CATALOG,
    FeatureUse,
    compact_actor_feature_names,
    richer_causal_context_feature_names,
)


def test_compact_actor_features_are_causal_and_online_available() -> None:
    specs = DEFAULT_FEATURE_CATALOG.lint(
        compact_actor_feature_names(),
        use=FeatureUse.ACTOR,
        require_online=True,
    )
    assert all(spec.online_available for spec in specs)
    assert "option_index_log" in {spec.name for spec in specs}


def test_richer_history_and_hazard_features_are_decision_available() -> None:
    specs = DEFAULT_FEATURE_CATALOG.lint(
        richer_causal_context_feature_names(),
        use=FeatureUse.ACTOR,
        require_online=True,
    )
    assert specs
    assert {spec.family for spec in specs} == {
        "causal-history",
        "observed-hazard-summary",
    }


@pytest.mark.parametrize(
    "name",
    (
        "episode_option_count",
        "remaining_option_count",
        "future_hit_suffix",
        "rng_seed",
        "source_context",
        "stage_id",
        "cohort_id",
    ),
)
def test_actor_linter_rejects_future_or_privileged_features(name: str) -> None:
    with pytest.raises(ValueError, match="unavailable"):
        DEFAULT_FEATURE_CATALOG.lint(
            (name,),
            use=FeatureUse.ACTOR,
            require_online=True,
        )


def test_feature_without_metadata_fails_closed() -> None:
    with pytest.raises(ValueError, match="no availability metadata"):
        DEFAULT_FEATURE_CATALOG.lint(
            ("mystery_feature",),
            use=FeatureUse.ACTOR,
            require_online=True,
        )
