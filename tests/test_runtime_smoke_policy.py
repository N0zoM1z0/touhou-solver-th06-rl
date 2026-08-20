from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from th06_rl.policy_loader import ImmutablePolicy


REPOSITORY = Path(__file__).resolve().parents[1]


def test_runtime_smoke_policy_delegates_only_to_the_native_baseline() -> None:
    policy = ImmutablePolicy(
        REPOSITORY / "scripts/policies/runtime_smoke_policy.py",
        state_path=REPOSITORY / "config/runtime_smoke_policy.json",
    )
    context = SimpleNamespace(
        baseline_action="focus_left",
        locally_admissible_actions=("stay", "focus_left"),
    )

    decision = policy.decide(context)

    assert decision.action == "focus_left"
    assert decision.behavior_probability == 1.0
    assert policy.status()["metrics"] == {
        "decisions": 1,
        "purpose": "infrastructure-smoke-only",
    }
