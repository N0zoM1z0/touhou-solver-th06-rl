from __future__ import annotations

import hashlib
import json

import pytest

from scripts.authorize_autonomous_canary import authorize
from tests.test_autonomous_linear_q_policy import _state


def _write(path, value):
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_canary_authorization_binds_exact_clean_shadow(tmp_path) -> None:
    state_path = tmp_path / "shadow-state.json"
    shadow_path = tmp_path / "shadow-audit.json"
    state = _state()
    _write(state_path, state)
    shadow = {
        "schema": "autonomous-q-shadow-audit-v1",
        "shadow_eligible": True,
        "policy_state_sha256": hashlib.sha256(state_path.read_bytes()).hexdigest(),
        "heldout_episode_groups": state["fit_report"]["validation_groups"],
        "decisions": 1000,
        "policy_metrics": {"shadow_proposals": 20},
    }
    _write(shadow_path, shadow)
    active = authorize(state_path, shadow_path)
    assert active["mode"] == "active"
    assert active["selection"]["active_override_budget"] == 64
    assert len(
        active["authorization"]["active_canary"]["shadow_audit_sha256"]
    ) == 64


def test_canary_authorization_rejects_stale_shadow(tmp_path) -> None:
    state_path = tmp_path / "shadow-state.json"
    shadow_path = tmp_path / "shadow-audit.json"
    _write(state_path, _state())
    _write(shadow_path, {
        "schema": "autonomous-q-shadow-audit-v1",
        "shadow_eligible": True,
        "policy_state_sha256": "0" * 64,
    })
    with pytest.raises(ValueError, match="stale"):
        authorize(state_path, shadow_path)
