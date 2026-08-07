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
