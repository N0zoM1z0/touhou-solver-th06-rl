from __future__ import annotations

import pytest

from scripts.extend_headless_ranker_compatibility import validate_upgrade_evidence
from th06_rl.native import ACTIONS


OLD = {"commit": "old", "binary_sha256": "old-bin", "clean": True}


def evidence() -> dict[str, object]:
    return {
        "schema": "th06-rl-headless-authority-failure-differential-v1",
        "classification": "source-safe-but-native-observation-incomplete",
        "native_comparison_available": False,
        "native_authority_error": "laser slot 0 lacks angular history",
        "runtime_source": OLD,
        "source_safe_constant_actions": [action.name for action in ACTIONS],
    }


def test_compatibility_extension_is_bounded_to_audited_additive_abi() -> None:
    validate_upgrade_evidence(
        evidence(),
        old_sources=[OLD],
        changed_paths=["src/HeadlessRuntime.cpp", "src/HeadlessRuntime.hpp", "HEADLESS.md"],
    )

    with pytest.raises(ValueError, match="outside"):
        validate_upgrade_evidence(
            evidence(),
            old_sources=[OLD],
            changed_paths=["src/BulletManager.cpp"],
        )


def test_compatibility_extension_requires_all_source_safe_actions() -> None:
    value = evidence()
    value["source_safe_constant_actions"] = ["stay"]

    with pytest.raises(ValueError, match="every ordinary action"):
        validate_upgrade_evidence(
            value,
            old_sources=[OLD],
            changed_paths=["src/HeadlessRuntime.cpp"],
        )


def event_evidence() -> dict[str, object]:
    actions = [action.name for action in ACTIONS]
    return {
        "schema": "th06-rl-headless-event-observation-differential-v1",
        "authority": "additive-diagnostics-only-no-native-set-revision",
        "removed_observation_members": ["events"],
        "physical_observations_equal": True,
        "actions": actions,
        "branches": [
            {
                "action": action,
                "physical_observations_equal": True,
                "mismatch_offsets": [],
                "old_observation_count": 2,
                "new_observation_count": 2,
                "old_physical_sha256": action,
                "new_physical_sha256": action,
            }
            for action in actions
        ],
        "eventful_observation_count": 1,
        "hit_kinds": ["enemy"],
        "old_runtime_source": OLD,
    }


def test_event_compatibility_requires_physical_equivalence_and_exact_patch() -> None:
    paths = [
        "HEADLESS.md",
        "src/BulletManager.cpp",
        "src/EnemyManager.cpp",
        "src/GameWindow.cpp",
        "src/HeadlessRuntime.cpp",
        "src/HeadlessRuntime.hpp",
        "src/Player.cpp",
    ]
    validate_upgrade_evidence(event_evidence(), old_sources=[OLD], changed_paths=paths)

    mismatched = event_evidence()
    mismatched["physical_observations_equal"] = False
    with pytest.raises(ValueError, match="physical observation"):
        validate_upgrade_evidence(mismatched, old_sources=[OLD], changed_paths=paths)

    with pytest.raises(ValueError, match="event-only"):
        validate_upgrade_evidence(
            event_evidence(), old_sources=[OLD], changed_paths=paths[:-1]
        )
