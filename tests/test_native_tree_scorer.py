from __future__ import annotations

import hashlib
import shutil
import subprocess
from pathlib import Path

import pytest

from th06_rl.hazard_representation import (
    encode_hazard_set,
    hazard_codebook_feature_names,
    HAZARD_PRIMITIVE_FEATURE_NAMES,
    NativeHazardCodebookEncoder,
)
from th06_rl.g7_learner import ACTOR_FEATURE_NAMES
from th06_rl.learning_features import CAUSAL_TREE_FEATURE_SCHEMA
from th06_rl.policies.offline_ranker import (
    MODEL_SCHEMA,
    NativePrototypeSupport,
    NativeXGBoostPopulation,
    NativeXGBoostRegressor,
    PortablePrototypeSupport,
    PortableXGBoostRegressor,
)


FEATURE_NAMES = ACTOR_FEATURE_NAMES
FEATURE_SCHEMA = CAUSAL_TREE_FEATURE_SCHEMA


def _model_artifact() -> dict[str, object]:
    tree = [
        [1, 0.5, 1, 2, 2, 0.0],
        [-1, 0.0, -1, -1, -1, 1.0],
        [-1, 0.0, -1, -1, -1, 2.0],
    ]
    return {
        "schema": MODEL_SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "base_score": 0.25,
        "trees": [tree],
    }


def test_portable_tree_scorer_obeys_frozen_feature_schema() -> None:
    model = PortableXGBoostRegressor(
        _model_artifact(),
        expected_feature_schema=FEATURE_SCHEMA,
        expected_feature_names=FEATURE_NAMES,
    )
    rows = [
        [0.0] * len(FEATURE_NAMES),
        [0.0, 1.0, *([0.0] * (len(FEATURE_NAMES) - 2))],
    ]

    assert model.predict_many(rows) == pytest.approx((1.25, 2.25))


def test_native_scorers_match_portable_references(tmp_path: Path) -> None:
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
    digest = hashlib.sha256(library.read_bytes()).hexdigest()
    portable = PortableXGBoostRegressor(
        _model_artifact(),
        expected_feature_schema=FEATURE_SCHEMA,
        expected_feature_names=FEATURE_NAMES,
    )
    native = NativeXGBoostRegressor(
        library,
        expected_sha256=digest,
        portable=portable,
    )
    rows = [
        [0.0] * len(FEATURE_NAMES),
        [0.0, 1.0, *([0.0] * (len(FEATURE_NAMES) - 2))],
    ]

    assert native.predict_many(rows) == pytest.approx(
        portable.predict_many(rows), abs=1e-6
    )
    population = NativeXGBoostPopulation(
        library,
        expected_sha256=digest,
        portable=[portable, portable, portable],
    )
    assert population.predict_many(rows) == pytest.approx(
        [portable.predict_many(rows)] * 3,
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
        expected_sha256=digest,
        portable=portable_support,
    )
    queries = [[0.5, 1.0], [2.5, 5.0]]
    assert native_support.distances(queries, [0, 17]) == pytest.approx(
        portable_support.distances(queries, [0, 17]), abs=1e-6
    )

    codebook = {
        "schema": "game-neutral-hazard-codebook-v1",
        "primitive_feature_names": list(HAZARD_PRIMITIVE_FEATURE_NAMES),
        "prototype_count": 24,
        "mean": [0.5] * 14,
        "scale": [2.0] * 14,
        "prototypes": [[float(index) / 24.0] * 14 for index in range(24)],
    }
    primitives = (
        tuple(float(index) / 10.0 for index in range(14)),
        tuple(float(index) / 20.0 for index in range(14)),
    )
    native_codebook = NativeHazardCodebookEncoder(
        library,
        expected_sha256=digest,
        artifact=codebook,
        output_count=len(hazard_codebook_feature_names()),
    )
    assert native_codebook.encode(primitives) == pytest.approx(
        encode_hazard_set(primitives, codebook), abs=2e-6
    )
