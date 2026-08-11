"""Immutable portable offline ranker above the native TH06 safety gate."""

from __future__ import annotations

import base64
from collections import Counter
import ctypes
import hashlib
import json
import math
import os
from pathlib import Path
import re
import zlib

from ..offline import ACTION_NAMES
from ..offline_learning import (
    CATEGORICAL_FEATURES,
    FEATURE_NAMES,
    FEATURE_SCHEMA,
)
from ..policy_api import POLICY_API_VERSION, PolicyDecision


STATE_SCHEMA = "th06-rl-offline-ranker-policy-v1"
MODEL_SCHEMA = "th06-rl-portable-xgboost-regressor-v1"
TRANSITION_SCHEMA = "th06-rl-transition-v5"
CONTEXT_QUALITY = "exact-v5"
MODEL_CODEC = "zlib-base64-json-v1"
NATIVE_SCORER_SCHEMA = "th06-rl-native-xgboost-scorer-v1"
NATIVE_SCORER_ENV = "TH06_RL_OFFLINE_SCORER_LIBRARY"
_ACTION_INDEX = {name: index for index, name in enumerate(ACTION_NAMES)}
_FEATURE_INDEX = {name: index for index, name in enumerate(FEATURE_NAMES)}
_CATEGORICAL_SET = frozenset(CATEGORICAL_FEATURES)
_NUMBER_RE = re.compile(r"-?\d+")


def _action_components(name: str) -> tuple[float, float, float]:
    if name not in _ACTION_INDEX:
        raise ValueError(f"unknown offline-ranker action {name!r}")
    core = name.removesuffix("_fast")
    return (
        float("right" in core) - float("left" in core),
        float("down" in core) - float("up" in core),
        float(not name.endswith("_fast")),
    )


def _mask(actions) -> str:
    value = 0
    for action in actions:
        try:
            value |= 1 << _ACTION_INDEX[action]
        except KeyError as error:
            raise ValueError(f"unknown offline-ranker action {action!r}") from error
    return f"{value:05x}"


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
            or len(raw) != len(_ACTION_INDEX)
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


class OfflineRankerPolicy:
    api_version = POLICY_API_VERSION
    name = "offline-ranker-uninitialized"

    def __init__(self) -> None:
        self.model: PortableXGBoostRegressor | None = None
        self.scorer: PortableXGBoostRegressor | NativeXGBoostRegressor | None = None
        self.scorer_backend = "uninitialized"
        self.encoder: dict[str, dict[str, int]] = {}
        self.scope: tuple[int, int, int, int] | None = None
        self.mode = "shadow"
        self.selection_kind = "baseline-prior"
        self.baseline_prior = 0.18
        self.score_margin = 0.0
        self.supported_actions: frozenset[tuple[str, str]] = frozenset()
        self.source_model_sha256 = ""
        self.portable_model_sha256 = ""
        self.decisions = 0
        self.shadow_disagreements = 0
        self.active_overrides = 0
        self.selected: Counter[str] = Counter()
        self.ranker_choices: Counter[str] = Counter()

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
            raise ValueError("unsupported offline-ranker policy state")
        mode = state.get("mode")
        if mode not in ("active", "shadow"):
            raise ValueError("offline-ranker mode must be active or shadow")
        raw_scope = state.get("scope")
        if not isinstance(raw_scope, list) or len(raw_scope) != 4:
            raise TypeError("offline-ranker scope must contain four integers")
        scope = tuple(int(value) for value in raw_scope)
        selection = state.get("selection")
        if not isinstance(selection, dict):
            raise ValueError("unsupported offline-ranker selection contract")
        selection_kind = selection.get("kind")
        if selection_kind not in ("baseline-prior", "support-margin"):
            raise ValueError("unsupported offline-ranker selection contract")
        baseline_prior = 0.0
        score_margin = 0.0
        supported_actions: frozenset[tuple[str, str]] = frozenset()
        if selection_kind == "baseline-prior":
            baseline_prior = float(selection.get("value", 0.0))
            if not math.isfinite(baseline_prior) or baseline_prior < 0.0:
                raise ValueError(
                    "offline-ranker baseline prior must be finite and nonnegative"
                )
        else:
            score_margin = float(selection.get("score_margin", 0.0))
            if not math.isfinite(score_margin) or score_margin < 0.0:
                raise ValueError(
                    "offline-ranker score margin must be finite and nonnegative"
                )
            raw_supported = selection.get("supported_actions")
            if not isinstance(raw_supported, list):
                raise TypeError("offline-ranker support table is absent")
            pairs = []
            for pair in raw_supported:
                if (
                    not isinstance(pair, list)
                    or len(pair) != 2
                    or not all(isinstance(value, str) for value in pair)
                    or pair[1] not in _ACTION_INDEX
                ):
                    raise TypeError("offline-ranker support row is invalid")
                pairs.append((pair[0], pair[1]))
            if len(pairs) != len(set(pairs)):
                raise ValueError("offline-ranker support table contains duplicates")
            supported_actions = frozenset(pairs)
        if state.get("model_codec") != MODEL_CODEC:
            raise ValueError("unsupported portable offline model codec")
        payload = state.get("model_payload")
        if not isinstance(payload, str):
            raise TypeError("portable offline model payload must be text")
        decoded = zlib.decompress(base64.b64decode(payload, validate=True))
        portable_sha = hashlib.sha256(decoded).hexdigest()
        if portable_sha != state.get("portable_model_sha256"):
            raise ValueError("portable offline model SHA-256 mismatch")
        artifact = json.loads(decoded.decode("utf-8"))
        if not isinstance(artifact, dict):
            raise TypeError("portable offline model root must be an object")
        raw_encoder = artifact.get("encoder")
        if not isinstance(raw_encoder, dict):
            raise TypeError("portable offline encoder is absent")
        encoder: dict[str, dict[str, int]] = {}
        for name in CATEGORICAL_FEATURES:
            values = raw_encoder.get(name)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise TypeError(f"portable offline encoder category {name!r} is invalid")
            if len(values) != len(set(values)):
                raise ValueError(f"portable offline encoder category {name!r} is duplicated")
            encoder[name] = {value: index for index, value in enumerate(values)}
        model = PortableXGBoostRegressor(artifact)
        conformance = artifact.get("conformance")
        if not isinstance(conformance, list) or not conformance:
            raise TypeError("portable offline model conformance vectors are absent")
        for row in conformance:
            if not isinstance(row, dict) or not isinstance(row.get("features"), list):
                raise TypeError("portable offline conformance vector is invalid")
            expected = float(row["prediction"])
            actual = model.predict([float(value) for value in row["features"]])
            if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-5):
                raise ValueError(
                    "portable offline model conformance failed: "
                    f"expected {expected}, observed {actual}"
                )
        scorer: PortableXGBoostRegressor | NativeXGBoostRegressor = model
        scorer_backend = "python-portable"
        native_contract = state.get("native_scorer")
        if native_contract is not None:
            if (
                not isinstance(native_contract, dict)
                or native_contract.get("schema") != NATIVE_SCORER_SCHEMA
            ):
                raise ValueError("unsupported native offline scorer contract")
            expected_scorer_sha = native_contract.get("sha256")
            if not isinstance(expected_scorer_sha, str) or len(expected_scorer_sha) != 64:
                raise ValueError("native offline scorer SHA-256 is invalid")
            scorer_path = os.environ.get(NATIVE_SCORER_ENV)
            if scorer_path:
                scorer = NativeXGBoostRegressor(
                    Path(scorer_path),
                    expected_sha256=expected_scorer_sha,
                    portable=model,
                )
                native_actual = scorer.predict_many([
                    [float(value) for value in row["features"]]
                    for row in conformance
                ])
                for actual, row in zip(native_actual, conformance, strict=True):
                    expected = float(row["prediction"])
                    if not math.isclose(actual, expected, rel_tol=2e-5, abs_tol=2e-5):
                        raise ValueError(
                            "native offline scorer conformance failed: "
                            f"expected {expected}, observed {actual}"
                        )
                scorer_backend = "native-batch"
            elif os.name == "nt":
                raise RuntimeError(
                    f"Windows offline ranker requires {NATIVE_SCORER_ENV}"
                )
        source_model_sha = state.get("source_model_sha256")
        if not isinstance(source_model_sha, str) or len(source_model_sha) != 64:
            raise ValueError("offline-ranker source model SHA-256 is invalid")
        self.model = model
        self.scorer = scorer
        self.scorer_backend = scorer_backend
        self.encoder = encoder
        self.scope = scope
        self.mode = mode
        self.selection_kind = selection_kind
        self.baseline_prior = baseline_prior
        self.score_margin = score_margin
        self.supported_actions = supported_actions
        self.source_model_sha256 = source_model_sha
        self.portable_model_sha256 = portable_sha
        self.name = f"offline-xgboost-{mode}-{source_model_sha[:12]}"

    def _candidate_rows(self, context, actions: tuple[str, ...]) -> list[list[float]]:
        legal = tuple(context.locally_admissible_actions)
        hard = tuple(context.hard_admissible_actions)
        baseline_dx, baseline_dy, baseline_focused = _action_components(
            context.baseline_action
        )
        phase_numbers = [
            float(value)
            for value in _NUMBER_RE.findall(str(context.source_context))[:6]
        ]
        phase_numbers.extend([-1.0] * (6 - len(phase_numbers)))
        edge_reserve = min(
            float(context.player_x) - 8.0,
            376.0 - float(context.player_x),
            float(context.player_y) - 16.0,
            432.0 - float(context.player_y),
        )
        common: dict[str, str | float] = {
            "source_context": str(context.source_context),
            "baseline_action": context.baseline_action,
            "current_action": context.current_action,
            "legal_mask": _mask(legal),
            "hard_mask": _mask(hard),
            "context_quality": CONTEXT_QUALITY,
            "transition_schema": TRANSITION_SCHEMA,
            "player_x": float(context.player_x),
            "player_y": float(context.player_y),
            "edge_reserve": edge_reserve,
            "power": float(context.power),
            "bullet_count": float(context.bullet_count),
            "laser_count": float(context.laser_count),
            "hard_action_count": float(context.hard_action_count),
            "legal_action_count": float(len(legal)),
            "phase_elapsed_frames": float(context.phase_elapsed_frames),
            "baseline_dx": baseline_dx,
            "baseline_dy": baseline_dy,
            "baseline_focused": baseline_focused,
        }
        common.update(
            {f"phase_number_{index}": value for index, value in enumerate(phase_numbers)}
        )
        base = [0.0] * len(FEATURE_NAMES)
        for name, value in common.items():
            index = _FEATURE_INDEX[name]
            if name in _CATEGORICAL_SET:
                base[index] = float(self.encoder[name].get(str(value), -1))
            else:
                base[index] = float(value)
        rows = []
        for action in actions:
            action_dx, action_dy, action_focused = _action_components(action)
            row = base.copy()
            row[_FEATURE_INDEX["action"]] = float(
                self.encoder["action"].get(action, -1)
            )
            row[_FEATURE_INDEX["action_dx"]] = action_dx
            row[_FEATURE_INDEX["action_dy"]] = action_dy
            row[_FEATURE_INDEX["action_focused"]] = action_focused
            row[_FEATURE_INDEX["action_stationary"]] = float(
                action_dx == 0.0 and action_dy == 0.0
            )
            row[_FEATURE_INDEX["action_diagonal"]] = float(
                action_dx != 0.0 and action_dy != 0.0
            )
            row[_FEATURE_INDEX["matches_baseline"]] = float(
                action == context.baseline_action
            )
            row[_FEATURE_INDEX["matches_current"]] = float(
                action == context.current_action
            )
            rows.append(row)
        return rows

    def decide(self, context) -> PolicyDecision:
        if self.model is None or self.scorer is None or self.scope is None:
            raise RuntimeError("offline-ranker state was not loaded")
        if tuple(context.scope) != self.scope:
            raise ValueError(
                f"offline-ranker scope mismatch: expected {self.scope}, got {context.scope}"
            )
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("offline-ranker received no native-safe actions")
        if context.baseline_action not in legal:
            raise ValueError("offline-ranker baseline is outside the native-safe set")
        rows = self._candidate_rows(context, legal)
        predictions = self.scorer.predict_many(rows)
        scores = dict(zip(legal, predictions, strict=True))
        if self.selection_kind == "baseline-prior":
            ranked = max(
                legal,
                key=lambda action: (
                    scores[action]
                    + (
                        self.baseline_prior
                        if action == context.baseline_action
                        else 0.0
                    ),
                    action,
                ),
            )
        else:
            supported = tuple(
                action
                for action in legal
                if action == context.baseline_action
                or (str(context.source_context), action) in self.supported_actions
            )
            candidate = max(
                supported,
                key=lambda action: (scores[action], action),
            )
            ranked = (
                candidate
                if scores[candidate]
                >= scores[context.baseline_action] + self.score_margin
                else context.baseline_action
            )
        selected = context.baseline_action if self.mode == "shadow" else ranked
        self.decisions += 1
        self.ranker_choices[ranked] += 1
        self.selected[selected] += 1
        if ranked != context.baseline_action:
            self.shadow_disagreements += 1
            if self.mode == "active":
                self.active_overrides += 1
        return PolicyDecision(selected, self.name, 1.0)

    def metrics(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "decisions": self.decisions,
            "shadow_disagreements": self.shadow_disagreements,
            "active_overrides": self.active_overrides,
            "baseline_prior": self.baseline_prior,
            "selection_kind": self.selection_kind,
            "score_margin": self.score_margin,
            "supported_action_pairs": len(self.supported_actions),
            "scope": list(self.scope) if self.scope is not None else None,
            "source_model_sha256": self.source_model_sha256,
            "portable_model_sha256": self.portable_model_sha256,
            "scorer_backend": self.scorer_backend,
            "selected": dict(self.selected),
            "ranker_choices": dict(self.ranker_choices),
        }


def create_policy() -> OfflineRankerPolicy:
    return OfflineRankerPolicy()
