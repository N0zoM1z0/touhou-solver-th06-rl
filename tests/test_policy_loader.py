from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from th06_rl.policy_loader import HotReloadPolicy
from th06_rl.policy_api import PolicyFailureEvent


POLICY = b"""
from th06_rl.policy_api import POLICY_API_VERSION, PolicyDecision

class Policy:
    api_version = POLICY_API_VERSION
    name = 'test'
    def decide(self, context):
        return PolicyDecision(context.baseline_action, self.name)

def create_policy():
    return Policy()
"""

OPTION_POLICY = b"""
from th06_rl.policy_api import POLICY_API_VERSION, PolicyDecision

class Policy:
    api_version = POLICY_API_VERSION
    name = 'test-option'
    def __init__(self):
        self.rejected = 0
    def decide(self, context):
        return PolicyDecision(context.baseline_action, self.name)
    def reject_publication(self, decision):
        self.rejected += 1

def create_policy():
    return Policy()
"""

METRICS_POLICY = b"""
from th06_rl.policy_api import POLICY_API_VERSION, PolicyDecision

class Policy:
    api_version = POLICY_API_VERSION
    name = 'test-metrics'
    def __init__(self):
        self.calls = 0
    def decide(self, context):
        return PolicyDecision(context.baseline_action, self.name)
    def metrics(self):
        self.calls += 1
        return {'calls': self.calls}

def create_policy():
    return Policy()
"""


def test_unchanged_policy_does_not_cross_unc_read_boundary(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "policy.py"
    path.write_bytes(POLICY)
    loader = HotReloadPolicy(path, check_interval_frames=30)
    original = Path.read_bytes
    reads = 0

    def counted(value):
        nonlocal reads
        reads += 1
        return original(value)

    monkeypatch.setattr(Path, "read_bytes", counted)

    assert loader.maybe_reload(30) is False
    assert reads == 0


def test_decide_never_polls_policy_source(tmp_path, monkeypatch) -> None:
    path = tmp_path / "policy.py"
    path.write_bytes(POLICY)
    loader = HotReloadPolicy(path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("frame-critical decide polled the policy source")

    monkeypatch.setattr(loader, "maybe_reload", forbidden)
    decision = loader.decide(SimpleNamespace(
        baseline_action="stay",
        locally_admissible_actions=("stay",),
    ))
    assert decision.action == "stay"


def test_checkpoint_is_the_reload_poll_boundary(tmp_path, monkeypatch) -> None:
    path = tmp_path / "policy.py"
    path.write_bytes(POLICY)
    loader = HotReloadPolicy(path)
    polls = []

    monkeypatch.setattr(
        loader,
        "maybe_reload",
        lambda frame: polls.append(frame) or False,
    )
    assert loader.checkpoint() is False
    assert polls == [0]


def test_optional_failure_feedback_does_not_require_policy_callback(
    tmp_path,
) -> None:
    path = tmp_path / "policy.py"
    path.write_bytes(POLICY)
    loader = HotReloadPolicy(path)

    loader.observe_failure(PolicyFailureEvent(
        frame=101,
        scope=(2, 0, 0, 6),
        source_context="boss:sub33",
        kind="physical-hit",
    ))

    assert loader.last_error is None


def test_optional_certified_continuation_is_absent_for_legacy_policy(
    tmp_path,
) -> None:
    path = tmp_path / "policy.py"
    path.write_bytes(POLICY)
    loader = HotReloadPolicy(path)

    assert loader.continue_certified(SimpleNamespace(
        baseline_action="stay",
        locally_admissible_actions=("stay",),
    )) is None


def test_immutable_policy_disables_reload_feedback_and_checkpoint(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "policy.py"
    path.write_bytes(POLICY)
    loader = HotReloadPolicy(path, immutable=True)

    monkeypatch.setattr(
        Path,
        "stat",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("immutable policy polled its source")
        ),
    )
    assert loader.maybe_reload(0) is False
    assert loader.checkpoint() is False
    loader.observe(object())
    loader.observe_failure(object())
    assert loader.status()["immutable"] is True


def test_immutable_policy_still_receives_operational_publication_rejection(
    tmp_path,
) -> None:
    path = tmp_path / "option-policy.py"
    path.write_bytes(OPTION_POLICY)
    loader = HotReloadPolicy(path, immutable=True)
    decision = loader.decide(SimpleNamespace(
        baseline_action="stay",
        locally_admissible_actions=("stay",),
    ))

    loader.reject_publication(decision)

    assert loader.policy.rejected == 1


def test_identity_status_does_not_materialize_expensive_metrics(tmp_path) -> None:
    path = tmp_path / "metrics-policy.py"
    path.write_bytes(METRICS_POLICY)
    loader = HotReloadPolicy(path, immutable=True)

    identity = loader.status(include_metrics=False)

    assert "metrics" not in identity
    assert loader.policy.calls == 0
    assert loader.status()["metrics"] == {"calls": 1}
