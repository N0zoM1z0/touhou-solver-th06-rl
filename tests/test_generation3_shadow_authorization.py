from __future__ import annotations

import hashlib
import json

import pytest

from scripts.authorize_option_advantage_canary import authorize


def _write(path, value) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _state() -> dict[str, object]:
    return {
        "schema": "autonomous-dr-option-advantage-policy-v1",
        "mode": "shadow",
        "fit_report": {
            "validation_groups": ["episode-a", "episode-b", "episode-c"],
            "validation_options": 41,
        },
        "authorization": {
            "fit_eligible": True,
            "active_canary": None,
        },
    }


def test_generation3_canary_authorization_binds_exact_native_shadow(
    tmp_path,
) -> None:
    state_path = tmp_path / "shadow.json"
    shadow_path = tmp_path / "audit.json"
    _write(state_path, _state())
    state_sha = hashlib.sha256(state_path.read_bytes()).hexdigest()
    _write(shadow_path, {
        "schema": "autonomous-generation-3-native-shadow-audit-v1",
        "shadow_eligible": True,
        "policy_state_sha256": state_sha,
        "heldout_episode_groups": ["episode-a", "episode-b", "episode-c"],
        "decisions": 41,
        "native_scorer_sha256": "a" * 64,
        "policy_metrics": {"shadow_proposals": 3},
        "latency": {"p95_ms": 0.7},
    })

    active = authorize(state_path, shadow_path)

    assert active["mode"] == "active"
    binding = active["authorization"]["active_canary"]
    assert binding["shadow_policy_state_sha256"] == state_sha
    assert binding["shadow_proposals"] == 3


def test_generation3_canary_authorization_rejects_stale_shadow(tmp_path) -> None:
    state_path = tmp_path / "shadow.json"
    shadow_path = tmp_path / "audit.json"
    _write(state_path, _state())
    _write(shadow_path, {
        "schema": "autonomous-generation-3-native-shadow-audit-v1",
        "shadow_eligible": True,
        "policy_state_sha256": "0" * 64,
        "heldout_episode_groups": ["episode-a", "episode-b", "episode-c"],
        "decisions": 41,
    })

    with pytest.raises(ValueError, match="stale"):
        authorize(state_path, shadow_path)
