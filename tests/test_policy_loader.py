from __future__ import annotations

import json
from types import SimpleNamespace

from th06_rl.policy_loader import ImmutablePolicy


POLICY = b"""
from th06_rl.policy_api import POLICY_API_VERSION, PolicyDecision

class Policy:
    api_version = POLICY_API_VERSION
    name = 'test-immutable'
    def __init__(self):
        self.loaded = None
        self.rejected = 0
    def import_state(self, state):
        self.loaded = state['token']
    def decide(self, context):
        return PolicyDecision(context.baseline_action, self.name)
    def reject_publication(self, decision):
        self.rejected += 1
    def metrics(self):
        return {'loaded': self.loaded, 'rejected': self.rejected}

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


def test_immutable_policy_receives_operational_publication_rejection(
    tmp_path,
) -> None:
    loader = _loader(tmp_path)
    decision = loader.decide(_context())

    loader.reject_publication(decision)

    assert loader.status()["metrics"]["rejected"] == 1


def test_identity_status_does_not_materialize_metrics(tmp_path) -> None:
    loader = _loader(tmp_path)

    identity = loader.status(include_metrics=False)

    assert "metrics" not in identity
