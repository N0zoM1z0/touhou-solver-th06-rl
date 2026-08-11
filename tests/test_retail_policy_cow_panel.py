from __future__ import annotations

from scripts.audit_retail_policy_cow_panel import panel_gate, parse_episode


def test_panel_gate_counts_episode_level_nonincumbent_winners() -> None:
    rows = (
        {"robust_winners": ["up"], "incumbent_action": "down"},
        {"robust_winners": ["left"], "incumbent_action": "right"},
        {"robust_winners": ["stay"], "incumbent_action": "up"},
        {"robust_winners": ["down", "left"], "incumbent_action": "down"},
    )

    gate = panel_gate(rows, minimum_unique_nonincumbent=3)

    assert gate["unique_winner_episodes"] == 3
    assert gate["unique_nonincumbent_winner_episodes"] == 3
    assert gate["targeted_action_relative_fit_allowed"] is True
    assert gate["candidate_population_cap"] == 3


def test_panel_gate_rejects_unique_incumbent_wins() -> None:
    rows = (
        {"robust_winners": ["down"], "incumbent_action": "down"},
        {"robust_winners": ["left"], "incumbent_action": "right"},
        {"robust_winners": ["stay", "up"], "incumbent_action": "stay"},
    )

    gate = panel_gate(rows, minimum_unique_nonincumbent=2)

    assert gate["unique_nonincumbent_winner_episodes"] == 1
    assert gate["targeted_action_relative_fit_allowed"] is False
    assert gate["candidate_population_cap"] == 0


def test_episode_spec_preserves_action_order() -> None:
    row = parse_episode("result.json::run-1::42::down::down,stay,up")

    assert row["sequence"] == 42
    assert row["actions"] == ["down", "stay", "up"]
