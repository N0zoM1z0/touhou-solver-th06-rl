from __future__ import annotations

import json
from types import SimpleNamespace

from th06_rl.policy_api import PolicyDecision
from th06_rl.policy_loader import ImmutablePolicy


POLICY = b"""
from th06_rl.policy_api import POLICY_API_VERSION, PolicyDecision

class Policy:
    api_version = POLICY_API_VERSION
    name = 'test-immutable'
    def __init__(self):
        self.loaded = None
    def import_state(self, state):
        self.loaded = state['token']
    def decide(self, context):
        return PolicyDecision(
            context.baseline_action,
            self.name,
            1.0,
            tuple((action, float(action == context.baseline_action))
                  for action in context.locally_admissible_actions),
        )
    def metrics(self):
        return {'loaded': self.loaded}

def create_policy():
    return Policy()
"""


def _loader(tmp_path) -> ImmutablePolicy:
    plugin = tmp_path / "policy.py"
    state = tmp_path / "state.json"
    plugin.write_bytes(POLICY)
    state.write_text(json.dumps({"token": "frozen"}), encoding="utf-8")
    return ImmutablePolicy(plugin, state_path=state)


def _context():
    return SimpleNamespace(
        baseline_action="stay",
        locally_admissible_actions=("stay",),
    )


def test_immutable_policy_loads_explicit_state_once(tmp_path) -> None:
    loader = _loader(tmp_path)

    assert loader.decide(_context()).action == "stay"
    assert loader.status()["metrics"]["loaded"] == "frozen"
    assert loader.status()["immutable"] is True


def test_policy_source_and_state_changes_cannot_affect_live_run(tmp_path) -> None:
    loader = _loader(tmp_path)
    loader.path.write_text("raise RuntimeError('mutated')", encoding="utf-8")
    loader.state_path.write_text("{}", encoding="utf-8")

    assert loader.decide(_context()).policy_id == "test-immutable"


def test_identity_status_does_not_materialize_metrics(tmp_path) -> None:
    loader = _loader(tmp_path)

    identity = loader.status(include_metrics=False)

    assert "metrics" not in identity


def test_failed_decision_is_explicit_and_counted(tmp_path) -> None:
    loader = _loader(tmp_path)

    def fail(_context):
        raise RuntimeError("broken decision")

    loader.policy.decide = fail

    decision = loader.decide(_context())

    assert decision.action == "stay"
    assert decision.policy_id == "reactive-baseline-policy-error"
    assert loader.status(include_metrics=False)["policy_failures"] == 1
    assert loader.status(include_metrics=False)["last_error"] == (
        "RuntimeError: broken decision"
    )


def test_incomplete_behavior_distribution_falls_back_and_is_counted(tmp_path) -> None:
    loader = _loader(tmp_path)
    loader.policy.decide = lambda context: PolicyDecision(
        context.baseline_action, "bad"
    )

    decision = loader.decide(_context())

    assert decision.policy_id == "reactive-baseline-policy-error"
    assert decision.behavior_probabilities == (("stay", 1.0),)
    assert loader.status(include_metrics=False)["policy_failures"] == 1
