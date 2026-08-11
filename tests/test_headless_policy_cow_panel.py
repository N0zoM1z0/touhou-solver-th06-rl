from __future__ import annotations

from scripts.run_headless_policy_cow_panel import (
    _action_sha256,
    is_target_checkpoint,
    summarize_gate,
    unique_robust_winner,
)
from th06_rl.policy_api import PolicyContext


def _context(**overrides: object) -> PolicyContext:
    values: dict[str, object] = {
        "frame": 3200,
        "scope": (3, 0, 0, 6),
        "source_context": "boss:0:sub10:life_cb14:timer_cb13:nonspell",
        "baseline_action": "down",
        "locally_admissible_actions": ("stay", "up", "down", "left", "right"),
        "player_x": 8.0,
        "player_y": 410.0,
        "power": 128,
        "bullet_count": 384,
        "laser_count": 0,
        "hard_action_count": 5,
        "exploration_rate": 0.0,
    }
    values.update(overrides)
    return PolicyContext(**values)  # type: ignore[arg-type]


def _outcome(action: str, survival: int) -> dict[str, object]:
    return {
        "first_action": action,
        "termination_reason": "tick-limit" if survival == 600 else "authority-failure",
        "survival_ticks": survival,
        "minimum_native_legal_actions": 5,
        "terminal_boundary_reserve": 16.0,
        "physical_deaths_delta": 0,
    }


def _seed(seed: int, *, selected: bool, winner: bool) -> dict[str, object]:
    checkpoint = None
    if selected:
        checkpoint = {"unique_non_incumbent_winner": winner}
    return {
        "root": {
            "split": "development-odd-seeds" if seed % 2 else "confirmation-even-seeds"
        },
        "checkpoint": checkpoint,
    }


def test_target_selector_uses_generic_geometry_and_automatic_context() -> None:
    assert is_target_checkpoint(_context())
    assert not is_target_checkpoint(_context(bullet_count=383))
    assert not is_target_checkpoint(_context(laser_count=1))
    assert not is_target_checkpoint(_context(player_x=25.0))
    assert not is_target_checkpoint(_context(player_x=7.0))
    assert not is_target_checkpoint(
        _context(locally_admissible_actions=("stay", "up", "down", "left"))
    )
    assert not is_target_checkpoint(_context(source_context="boss:0:sub11"))


def test_robust_winner_requires_a_unique_top_tier() -> None:
    assert unique_robust_winner((_outcome("up", 600), _outcome("down", 7))) == "up"
    assert unique_robust_winner((_outcome("up", 600), _outcome("down", 600))) is None
    assert unique_robust_winner(()) is None


def test_source_gate_is_episode_and_split_grouped() -> None:
    rows = [
        _seed(201, selected=True, winner=True),
        _seed(202, selected=True, winner=True),
        _seed(203, selected=True, winner=True),
        _seed(204, selected=True, winner=True),
        _seed(205, selected=True, winner=False),
        _seed(206, selected=True, winner=False),
    ]
    assert summarize_gate(rows)["source_support_gate_passed"] is True
    rows[-1] = _seed(206, selected=False, winner=False)
    assert summarize_gate(rows)["source_support_gate_passed"] is False


def test_action_digest_is_ordered_and_delimited() -> None:
    assert _action_sha256(("down", "left")) != _action_sha256(("down_left",))
    assert _action_sha256(("down", "left")) != _action_sha256(("left", "down"))
