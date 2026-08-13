"""Retained portable/native tree scoring and support primitives."""

from __future__ import annotations

import ctypes
import hashlib
import math
from pathlib import Path

from ..actions import ACTION_NAMES
from ..legacy_ranker_schema import FEATURE_NAMES, FEATURE_SCHEMA


MODEL_SCHEMA = "th06-rl-portable-xgboost-regressor-v1"
NATIVE_SCORER_ENV = "TH06_RL_OFFLINE_SCORER_LIBRARY"


class PortableXGBoostRegressor:
    """Small standard-library evaluator for exported scalar XGBoost trees."""

    def __init__(
        self,
        artifact: dict[str, object],
        *,
        expected_feature_schema: str = FEATURE_SCHEMA,
        expected_feature_names: tuple[str, ...] = FEATURE_NAMES,
    ) -> None:
        if artifact.get("schema") != MODEL_SCHEMA:
            raise ValueError("unsupported portable offline model schema")
        if tuple(artifact.get("feature_names", ())) != expected_feature_names:
            raise ValueError("portable offline model feature order mismatch")
        if artifact.get("feature_schema") != expected_feature_schema:
            raise ValueError("portable offline model feature schema mismatch")
        self.feature_names = expected_feature_names
        self.feature_count = len(expected_feature_names)
        self.base_score = float(artifact["base_score"])
        raw_trees = artifact.get("trees")
        if not isinstance(raw_trees, list) or not raw_trees:
            raise TypeError("portable offline model has no trees")
        trees = []
        for raw_tree in raw_trees:
            if not isinstance(raw_tree, list) or not raw_tree:
                raise TypeError("portable offline model tree is invalid")
            tree = []
            for raw_node in raw_tree:
                if not isinstance(raw_node, list) or len(raw_node) != 6:
                    raise TypeError("portable offline model node is invalid")
                feature, threshold, left, right, missing, leaf = raw_node
                feature = int(feature)
                if feature >= self.feature_count:
                    raise ValueError("portable tree feature is out of range")
                tree.append((
                    feature,
                    float(threshold),
                    int(left),
                    int(right),
                    int(missing),
                    float(leaf),
                ))
            for feature, _threshold, left, right, missing, _leaf in tree:
                if feature < 0:
                    continue
                if min(left, right, missing) < 0 or max(left, right, missing) >= len(tree):
                    raise ValueError("portable tree child is out of range")
            trees.append(tuple(tree))
        self.trees = tuple(trees)

    def predict(self, features: list[float]) -> float:
        if len(features) != self.feature_count:
            raise ValueError("portable model feature vector length mismatch")
        value = self.base_score
        for tree in self.trees:
            node_index = 0
            while True:
                feature, threshold, left, right, missing, leaf = tree[node_index]
                if feature < 0:
                    value += leaf
                    break
                candidate = features[feature]
                if math.isnan(candidate):
                    node_index = missing
                elif candidate < threshold:
                    node_index = left
                else:
                    node_index = right
        return value

    def predict_many(self, rows: list[list[float]]) -> tuple[float, ...]:
        return tuple(self.predict(row) for row in rows)


class _TreeNode(ctypes.Structure):
    _fields_ = [
        ("feature", ctypes.c_int32),
        ("threshold", ctypes.c_float),
        ("left", ctypes.c_int32),
        ("right", ctypes.c_int32),
        ("missing", ctypes.c_int32),
        ("leaf", ctypes.c_float),
    ]


class NativeXGBoostRegressor:
    """ctypes batch facade for the isolated tree-scoring DLL."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        portable: PortableXGBoostRegressor,
    ) -> None:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected_sha256:
            raise ValueError("native offline scorer SHA-256 mismatch")
        library = ctypes.CDLL(str(path))
        function = library.th06_rl_score_xgboost_v1
        function.argtypes = [
            ctypes.POINTER(_TreeNode),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_float),
        ]
        function.restype = ctypes.c_int
        raw_nodes = []
        offsets = [0]
        for tree in portable.trees:
            raw_nodes.extend(_TreeNode(*node) for node in tree)
            offsets.append(len(raw_nodes))
        self.library = library
        self.function = function
        self.nodes = (_TreeNode * len(raw_nodes))(*raw_nodes)
        self.offsets = (ctypes.c_int32 * len(offsets))(*offsets)
        self.tree_count = len(portable.trees)
        self.base_score = portable.base_score
        self.feature_count = portable.feature_count

    def predict_many(self, rows: list[list[float]]) -> tuple[float, ...]:
        if not rows:
            return ()
        if any(len(row) != self.feature_count for row in rows):
            raise ValueError("native scorer feature vector length mismatch")
        flat = (ctypes.c_float * (len(rows) * self.feature_count))(
            *(value for row in rows for value in row)
        )
        output = (ctypes.c_float * len(rows))()
        result = self.function(
            self.nodes,
            len(self.nodes),
            self.offsets,
            self.tree_count,
            flat,
            len(rows),
            self.feature_count,
            self.base_score,
            output,
        )
        if result != 0:
            raise RuntimeError(f"native offline scorer failed with status {result}")
        return tuple(float(value) for value in output)


class NativeXGBoostPopulation:
    """One-copy/one-call native facade for a complete model population."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        portable: list[PortableXGBoostRegressor],
    ) -> None:
        if not portable:
            raise ValueError("native scorer population cannot be empty")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("native population scorer SHA-256 mismatch")
        feature_count = portable[0].feature_count
        if any(model.feature_count != feature_count for model in portable):
            raise ValueError("native population feature schemas differ")
        library = ctypes.CDLL(str(path))
        function = library.th06_rl_score_xgboost_population_v1
        function.argtypes = [
            ctypes.POINTER(_TreeNode),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
        ]
        function.restype = ctypes.c_int
        raw_nodes = []
        tree_offsets = [0]
        model_tree_offsets = [0]
        for model in portable:
            for tree in model.trees:
                raw_nodes.extend(_TreeNode(*node) for node in tree)
                tree_offsets.append(len(raw_nodes))
            model_tree_offsets.append(len(tree_offsets) - 1)
        self.library = library
        self.function = function
        self.nodes = (_TreeNode * len(raw_nodes))(*raw_nodes)
        self.tree_offsets = (ctypes.c_int32 * len(tree_offsets))(*tree_offsets)
        self.model_tree_offsets = (
            ctypes.c_int32 * len(model_tree_offsets)
        )(*model_tree_offsets)
        self.base_scores = (ctypes.c_float * len(portable))(*(
            model.base_score for model in portable
        ))
        self.tree_count = len(tree_offsets) - 1
        self.model_count = len(portable)
        self.feature_count = feature_count

    def predict_many(
        self, rows: list[list[float]]
    ) -> tuple[tuple[float, ...], ...]:
        if not rows:
            return tuple(() for _index in range(self.model_count))
        if any(len(row) != self.feature_count for row in rows):
            raise ValueError("native population feature vector length mismatch")
        flat = (ctypes.c_float * (len(rows) * self.feature_count))(
            *(value for row in rows for value in row)
        )
        output = (ctypes.c_float * (self.model_count * len(rows)))()
        status = self.function(
            self.nodes,
            len(self.nodes),
            self.tree_offsets,
            self.tree_count,
            self.model_tree_offsets,
            self.model_count,
            flat,
            len(rows),
            self.feature_count,
            self.base_scores,
            output,
        )
        if status != 0:
            raise RuntimeError(
                f"native population scorer failed with status {status}"
            )
        return tuple(
            tuple(float(output[model * len(rows) + row]) for row in range(len(rows)))
            for model in range(self.model_count)
        )


class PortablePrototypeSupport:
    """Reference evaluator for per-action standardized local support."""

    def __init__(self, artifact: dict[str, object], *, feature_count: int) -> None:
        mean = artifact.get("mean")
        scale = artifact.get("scale")
        raw = artifact.get("prototypes")
        if (
            not isinstance(mean, list)
            or not isinstance(scale, list)
            or len(mean) != feature_count
            or len(scale) != feature_count
            or not isinstance(raw, list)
            or len(raw) != len(ACTION_NAMES)
        ):
            raise TypeError("prototype support artifact shape is invalid")
        self.mean = tuple(float(value) for value in mean)
        self.scale = tuple(float(value) for value in scale)
        if any(not math.isfinite(value) for value in self.mean) or any(
            not math.isfinite(value) or value <= 0.0 for value in self.scale
        ):
            raise ValueError("prototype support normalization is invalid")
        groups = []
        for rows in raw:
            if not isinstance(rows, list) or not rows:
                raise ValueError("every action needs at least one support prototype")
            converted = tuple(tuple(float(value) for value in row) for row in rows)
            if any(len(row) != feature_count for row in converted):
                raise ValueError("support prototype feature length mismatch")
            if any(not math.isfinite(value) for row in converted for value in row):
                raise ValueError("support prototype is non-finite")
            groups.append(converted)
        self.groups = tuple(groups)
        self.feature_count = feature_count

    def distances(
        self, rows: list[list[float]], action_indices: list[int]
    ) -> tuple[float, ...]:
        if len(rows) != len(action_indices) or any(
            len(row) != self.feature_count for row in rows
        ):
            raise ValueError("support query shape mismatch")
        result = []
        for row, action in zip(rows, action_indices, strict=True):
            normalized = tuple(
                (float(value) - center) / width
                for value, center, width in zip(
                    row, self.mean, self.scale, strict=True
                )
            )
            result.append(min(
                sum((left - right) ** 2 for left, right in zip(
                    normalized, prototype, strict=True
                )) / self.feature_count
                for prototype in self.groups[action]
            ))
        return tuple(result)


class NativePrototypeSupport:
    """Batch native facade for the same immutable support prototypes."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        portable: PortablePrototypeSupport,
    ) -> None:
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected_sha256:
            raise ValueError("native support scorer SHA-256 mismatch")
        library = ctypes.CDLL(str(path))
        function = library.th06_rl_min_support_distance_v1
        function.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_float),
        ]
        function.restype = ctypes.c_int
        flattened = [
            value for group in portable.groups for row in group for value in row
        ]
        offsets = [0]
        for group in portable.groups:
            offsets.append(offsets[-1] + len(group))
        self.library = library
        self.function = function
        self.mean = (ctypes.c_float * portable.feature_count)(*portable.mean)
        self.scale = (ctypes.c_float * portable.feature_count)(*portable.scale)
        self.prototypes = (ctypes.c_float * len(flattened))(*flattened)
        self.offsets = (ctypes.c_int32 * len(offsets))(*offsets)
        self.prototype_count = offsets[-1]
        self.action_count = len(portable.groups)
        self.feature_count = portable.feature_count

    def distances(
        self, rows: list[list[float]], action_indices: list[int]
    ) -> tuple[float, ...]:
        if not rows:
            return ()
        if len(rows) != len(action_indices) or any(
            len(row) != self.feature_count for row in rows
        ):
            raise ValueError("support query shape mismatch")
        flat = (ctypes.c_float * (len(rows) * self.feature_count))(
            *(value for row in rows for value in row)
        )
        actions = (ctypes.c_int32 * len(action_indices))(*action_indices)
        output = (ctypes.c_float * len(rows))()
        status = self.function(
            flat,
            len(rows),
            self.feature_count,
            self.mean,
            self.scale,
            self.prototypes,
            self.prototype_count,
            self.offsets,
            self.action_count,
            actions,
            output,
        )
        if status != 0:
            raise RuntimeError(f"native support scorer failed with status {status}")
        return tuple(float(value) for value in output)
