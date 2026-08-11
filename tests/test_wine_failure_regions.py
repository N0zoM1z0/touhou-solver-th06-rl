from __future__ import annotations

from scripts.audit_wine_failure_regions import (
    aggregate_memberships,
    opportunity_region,
    terminal_family,
    window_atom,
)


def _features(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "edge_reserve": 8.0,
        "bullet_count": 420.0,
        "laser_count": 0.0,
        "hard_action_count": 6.0,
        "incumbent_hard_clearance": 1.5,
        "action": "right_fast",
        "baseline_action": "left_fast",
    }
    result.update(overrides)
    return result


def test_fixed_region_bins_are_generic_and_action_opportunity_is_explicit() -> None:
    features = _features()

    assert terminal_family(features) == "boundary/dense-bullets"
    assert window_atom(features) == "boundary/dense-bullets/broad-5-plus"
    assert opportunity_region(features).endswith(
        "incumbent=right_fast/baseline=left_fast"
    )
    assert terminal_family(_features(edge_reserve=40.0, laser_count=48.0)) == (
        "interior/lasers-present"
    )


def test_region_support_counts_each_episode_once() -> None:
    episodes = [
        {
            "run_id": "a",
            "failure_context": "ctx-a",
            "failure_frame": 100,
            "fallback_opportunity_rows": 20,
            "window_atoms": ["region", "region"],
        },
        {
            "run_id": "b",
            "failure_context": "ctx-b",
            "failure_frame": 110,
            "fallback_opportunity_rows": 0,
            "window_atoms": ["region"],
        },
    ]

    result = aggregate_memberships(
        episodes, membership_key="window_atoms", plural=True
    )

    assert result[0]["episode_support"] == 2
    assert result[0]["classification"] == "repeated"
    assert result[0]["episodes_with_any_fallback_opportunity"] == 1
    assert result[0]["contexts"] == {"ctx-a": 1, "ctx-b": 1}
