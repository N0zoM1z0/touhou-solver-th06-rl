from __future__ import annotations

import base64
import hashlib
import json
import shutil
import subprocess
import zlib
from pathlib import Path

import pytest

from th06_rl.offline_learning import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
)
from th06_rl.policies.offline_ranker import (
    MODEL_CODEC,
    MODEL_SCHEMA,
    STATE_SCHEMA,
    OfflineRankerPolicy,
    NativeXGBoostRegressor,
    NativePrototypeSupport,
    PortablePrototypeSupport,
    PortableXGBoostRegressor,
)
from th06_rl.policy_api import PolicyContext


def _state(mode: str, *, selection: dict[str, object] | None = None) -> dict[str, object]:
    encoder = {name: ["unknown"] for name in CATEGORICAL_FEATURES}
    encoder.update({
        "action": ["stay", "right"],
        "baseline_action": ["stay", "right"],
        "current_action": ["stay", "right"],
        "source_context": ["boss:sub6"],
        "legal_mask": ["00011"],
        "hard_mask": ["00011"],
        "context_quality": ["exact-v5"],
        "transition_schema": ["th06-rl-transition-v5"],
    })
    # Feature 1 is the encoded candidate action: stay=0 takes leaf 1,
    # right=1 takes leaf 2.
    tree = [
        [1, 0.5, 1, 2, 2, 0.0],
        [-1, 0.0, -1, -1, -1, 1.0],
        [-1, 0.0, -1, -1, -1, 2.0],
    ]
    zero = [0.0] * len(FEATURE_NAMES)
    artifact = {
        "schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "categorical_features": list(CATEGORICAL_FEATURES),
        "encoder": encoder,
        "base_score": 0.25,
        "trees": [tree],
        "conformance": [{"features": zero, "prediction": 1.25}],
    }
    decoded = json.dumps(
        artifact,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return {
        "schema": STATE_SCHEMA,
        "mode": mode,
        "scope": [3, 0, 0, 6],
        "selection": selection or {"kind": "baseline-prior", "value": 0.18},
        "source_model_sha256": "a" * 64,
        "model_codec": MODEL_CODEC,
        "portable_model_sha256": hashlib.sha256(decoded).hexdigest(),
        "model_payload": base64.b64encode(zlib.compress(decoded)).decode(),
    }


def _context() -> PolicyContext:
    return PolicyContext(
        frame=100,
        scope=(3, 0, 0, 6),
        source_context="boss:sub6",
        baseline_action="stay",
        locally_admissible_actions=("stay", "right"),
        player_x=192.0,
        player_y=400.0,
        power=128,
        bullet_count=42,
        laser_count=0,
        hard_action_count=2,
        exploration_rate=0.0,
        current_action="stay",
        hard_admissible_actions=("stay", "right"),
        phase_elapsed_frames=300,
    )


def test_active_offline_ranker_can_only_select_from_native_set() -> None:
    policy = OfflineRankerPolicy()
    policy.import_state(_state("active"))

    decision = policy.decide(_context())

    assert decision.action == "right"
    assert decision.behavior_probability == 1.0
    assert policy.metrics()["active_overrides"] == 1


def test_shadow_offline_ranker_scores_but_returns_reactive_baseline() -> None:
    policy = OfflineRankerPolicy()
    policy.import_state(_state("shadow"))

    decision = policy.decide(_context())

    assert decision.action == "stay"
    assert policy.metrics()["shadow_disagreements"] == 1
    assert policy.metrics()["active_overrides"] == 0


def test_offline_ranker_rejects_scope_drift() -> None:
    policy = OfflineRankerPolicy()
    policy.import_state(_state("active"))
    context = _context()
    context = PolicyContext(**{**context.__dict__, "scope": (3, 0, 0, 5)})

    with pytest.raises(ValueError, match="scope mismatch"):
        policy.decide(context)


def test_support_margin_overrides_only_a_supported_improvement() -> None:
    supported = {
        "kind": "support-margin",
        "score_margin": 1.0,
        "minimum_support": 32,
        "supported_actions": [["boss:sub6", "right"]],
    }
    policy = OfflineRankerPolicy()
    policy.import_state(_state("active", selection=supported))
    assert policy.decide(_context()).action == "right"

    unsupported = {**supported, "supported_actions": []}
    policy = OfflineRankerPolicy()
    policy.import_state(_state("active", selection=unsupported))
    assert policy.decide(_context()).action == "stay"


def test_isolated_native_batch_scorer_matches_portable_model(tmp_path: Path) -> None:
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("C++ compiler is unavailable")
    library = tmp_path / "libth06_rl_ranker.so"
    subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(Path("native/include").resolve()),
            str(Path("native/src/th06_rl_ranker.cpp").resolve()),
            "-o",
            str(library),
        ],
        check=True,
    )
    state = _state("active")
    decoded = zlib.decompress(base64.b64decode(state["model_payload"]))
    artifact = json.loads(decoded)
    portable = PortableXGBoostRegressor(artifact)
    native = NativeXGBoostRegressor(
        library,
        expected_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        portable=portable,
    )
    rows = [
        [0.0] * len(FEATURE_NAMES),
        [0.0, 1.0, *([0.0] * (len(FEATURE_NAMES) - 2))],
    ]

    assert native.predict_many(rows) == pytest.approx(
        portable.predict_many(rows),
        abs=1e-6,
    )

    support_artifact = {
        "mean": [0.5, 1.0],
        "scale": [2.0, 4.0],
        "prototypes": [[[0.0, 0.0]]] * 18,
    }
    portable_support = PortablePrototypeSupport(
        support_artifact, feature_count=2
    )
    native_support = NativePrototypeSupport(
        library,
        expected_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),
        portable=portable_support,
    )
    queries = [[0.5, 1.0], [2.5, 5.0]]
    assert native_support.distances(queries, [0, 17]) == pytest.approx(
        portable_support.distances(queries, [0, 17]), abs=1e-6
    )
