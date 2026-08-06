from __future__ import annotations

from pathlib import Path

from th06_rl.policy_loader import HotReloadPolicy


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
