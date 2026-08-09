from __future__ import annotations

import json

from scripts.train_headless_cow_value import (
    behavior_value_groups,
    delivery_contract,
    load_value_groups,
    ordinal_outcome_labels,
)
from scripts.train_headless_teacher import Decision


def outcome(terminal: str, survival: int, legal: int, reserve: float):
    return {
        "termination_reason": terminal,
        "survival_ticks": survival,
        "minimum_native_legal_actions": legal,
        "terminal_boundary_reserve": reserve,
    }


def test_ordinal_value_labels_preserve_physical_priority() -> None:
    labels = ordinal_outcome_labels([
        outcome("authority-failure", 179, 18, 100.0),
        outcome("tick-limit", 180, 2, 5.0),
        outcome("tick-limit", 180, 12, 40.0),
    ])

    assert labels[0] < labels[1] < labels[2]


def test_ordinal_value_labels_retain_exact_ties() -> None:
    value = outcome("tick-limit", 180, 12, 40.0)

    assert ordinal_outcome_labels([value, value]) == (0, 0)


def test_ordinal_value_labels_do_not_invent_value_inside_equal_dead_end() -> None:
    labels = ordinal_outcome_labels([
        outcome("authority-failure", 1, 18, 4.0),
        outcome("authority-failure", 1, 2, 100.0),
    ])

    assert labels == (0, 0)


def test_ordinal_value_labels_retain_failed_survival_signal() -> None:
    labels = ordinal_outcome_labels([
        outcome("authority-failure", 61, 18, 100.0),
        outcome("authority-failure", 240, 1, 0.0),
    ])

    assert labels[0] < labels[1]


def test_ordinal_value_labels_keep_equivalent_completed_routes_tied() -> None:
    labels = ordinal_outcome_labels([
        outcome("tick-limit", 240, 16, 25.0),
        outcome("tick-limit", 240, 18, 31.0),
    ])

    assert labels == (0, 0)


def test_cow_value_delivery_contract_is_explicit_and_backward_auditable() -> None:
    assert delivery_contract({
        "runtime_delivery_contract": "synchronous-step-v1",
        "runtime_delivery_delays": [0],
    }) == ("synchronous-step-v1", (0,))
    assert delivery_contract({}) == ("legacy-unspecified-v0", ())


def test_behavior_regularization_yields_to_cow_observation() -> None:
    decision = Decision(
        run="run",
        seed=7,
        sequence=4,
        source_context="boss:0/1",
        state={},
        legal_actions=("left", "right"),
        candidates=(),
        teacher_action="left",
        selected_action="right",
        observation_sha256="digest",
    )

    assert behavior_value_groups([decision], stride=2)[0].labels == (0, 1)
    assert behavior_value_groups(
        [decision],
        excluded_observations=frozenset({"digest"}),
    ) == []


def test_longer_duplicate_cow_horizon_supersedes_shorter(tmp_path) -> None:
    decision = Decision(
        run="run",
        seed=7,
        sequence=4,
        source_context="boss:0/1",
        state={},
        legal_actions=("left", "right"),
        candidates=(),
        teacher_action="left",
        selected_action="left",
        observation_sha256="digest",
    )
    provenance = {
        "scope": {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 2},
        "source": {"commit": "source", "clean": True, "binary_sha256": "binary"},
        "native_delivery_contract": "synchronous-step-v1",
        "native_delivery_delays": [0],
        "observation_digest_contract": "physical-v1",
    }

    def document(branch_frames: int, completed_action: str):
        outcomes = []
        for action in decision.legal_actions:
            completed = action == completed_action
            outcomes.append({
                "first_action": action,
                "termination_reason": "tick-limit" if completed else "authority-failure",
                "survival_ticks": branch_frames if completed else 5,
                "minimum_native_legal_actions": 8 if completed else 1,
                "terminal_boundary_reserve": 32.0 if completed else 0.0,
            })
        return {
            "schema": "th06-rl-headless-cow-counterfactual-v1",
            "scope": provenance["scope"],
            "input_source": provenance["source"],
            "runtime_source": provenance["source"],
            "runtime_delivery_contract": "synchronous-step-v1",
            "runtime_delivery_delays": [0],
            "observation_digest_contract": "physical-v1",
            "initial_seed": 7,
            "checkpoints": [{
                "observation_sha256": "digest",
                "branch_frames": branch_frames,
                "outcomes": outcomes,
            }],
        }

    (tmp_path / "a-short.json").write_text(
        json.dumps(document(240, "left")), encoding="utf-8"
    )
    (tmp_path / "b-long.json").write_text(
        json.dumps(document(1200, "right")), encoding="utf-8"
    )

    groups, report = load_value_groups([decision], provenance, [tmp_path])

    assert groups[0].best_actions == ("right",)
    assert report["groups"] == 1
    assert report["duplicate_checkpoints"] == 1
    assert report["longer_horizon_replacements"] == 1
