"""Strict readers and labels for terminal original-retail Wine prefixes.

These prefixes are deliberately not complete-Stage imitation corpora.  They
are accepted only for factual failure-risk estimation: one model score for the
action that the frozen incumbent actually published.  Counterfactual actions
remain outside this data contract.
"""

from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path

from .offline import ACTION_NAMES, RunDescriptor, iter_run_transitions
from .offline_learning import LabeledTransition, label_transitions


FIRST_FAILURE_SCHEMA = "th06-rl-wine-first-failure-prefix-v1"
RISK_LABEL_SCHEMA = "th06-rl-factual-contiguous-failure-within-120-v2"
FAILURE_HORIZON_FRAMES = 120
TERMINAL_FAILURES = frozenset(("life-lost", "control-dead-end"))
RISK_FEATURE_SCHEMA = "th06-rl-wine-factual-risk-feature-v1"
RISK_FEATURE_SCHEMA_V2 = "th06-rl-wine-factual-risk-feature-v2"
FROZEN_INCUMBENT_POLICY_ID = "phase-local-hierarchical-ucb-v4"
RISK_CATEGORICAL_FEATURES = (
    "action",
    "baseline_action",
    "current_action",
)
RISK_NUMERIC_FEATURES = (
    "player_x",
    "player_y",
    "edge_reserve",
    "power",
    "bullet_count",
    "laser_count",
    "hard_action_count",
    "legal_action_count",
    "phase_elapsed_frames",
    "action_dx",
    "action_dy",
    "action_focused",
    "action_stationary",
    "action_diagonal",
    "baseline_dx",
    "baseline_dy",
    "baseline_focused",
    "current_dx",
    "current_dy",
    "current_focused",
    "matches_baseline",
    "matches_current",
    "incumbent_hard_clearance",
    "baseline_hard_clearance",
    "clearance_delta_baseline",
    "hard_clearance_min",
    "hard_clearance_max",
    "hard_clearance_span",
    "incumbent_clearance_rank",
    "incumbent_final_x",
    "incumbent_final_y",
    "incumbent_final_edge_reserve",
    "baseline_final_x",
    "baseline_final_y",
    "baseline_final_edge_reserve",
    *tuple(f"legal_{action}" for action in ACTION_NAMES),
    *tuple(f"hard_{action}" for action in ACTION_NAMES),
)
RISK_FEATURE_NAMES = (*RISK_CATEGORICAL_FEATURES, *RISK_NUMERIC_FEATURES)
RISK_CATEGORICAL_FEATURES_V2 = (
    "source_context",
    *RISK_CATEGORICAL_FEATURES,
)
RISK_NUMERIC_FEATURES_V2 = tuple(
    name for name in RISK_NUMERIC_FEATURES
    if name not in ("laser_count", "phase_elapsed_frames")
)
RISK_FEATURE_NAMES_V2 = (
    *RISK_CATEGORICAL_FEATURES_V2,
    *RISK_NUMERIC_FEATURES_V2,
)
_CLEARANCE_SENTINEL = 512.0


def risk_feature_contract(
    schema: str,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if schema == RISK_FEATURE_SCHEMA:
        return RISK_FEATURE_NAMES, RISK_CATEGORICAL_FEATURES
    if schema == RISK_FEATURE_SCHEMA_V2:
        return RISK_FEATURE_NAMES_V2, RISK_CATEGORICAL_FEATURES_V2
    raise ValueError(f"unsupported Wine risk feature schema: {schema!r}")


@dataclass(frozen=True)
class RiskExample:
    transition: LabeledTransition
    features: dict[str, str | float]
    failure_within_120: bool
    frames_to_failure: int | None

    @property
    def fallback_opportunity(self) -> bool:
        return self.transition.action != self.transition.baseline_action


@dataclass(frozen=True)
class FirstFailurePrefix:
    schema: str
    run_id: str
    run_dir: Path
    scope: tuple[int, int, int, int]
    executable_sha256: str
    native_kernel_sha256: str
    code_commit: str
    manifest_sha256: str
    run_sha256: str
    failure_kind: str
    failure_frame: int
    failure_context: str
    failure_segment_start_frame: int
    positive_window_start_frame: int
    transitions: int
    examples: tuple[RiskExample, ...]


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _stream_rows(run_dir: Path, manifest: dict[str, object], stream: str):
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        raise TypeError("first-failure manifest shard list is invalid")
    for raw in shards:
        if not isinstance(raw, dict) or raw.get("stream") != stream:
            continue
        name = str(raw.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError("first-failure shard path is unsafe")
        with gzip.open(run_dir / name, "rt", encoding="utf-8") as source:
            for line in source:
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise TypeError("first-failure stream row is not an object")
                yield value


def _frame(reference: object) -> int:
    marker = str(reference).rsplit(":f", 1)
    if len(marker) != 2:
        raise ValueError(f"snapshot reference has no frame: {reference!r}")
    return int(marker[1])


def _validate_clean_run_outcome(
    manifest: dict[str, object],
    *,
    failure_kind: str,
) -> None:
    """Require a failure prefix with no recorder/capture recovery history."""
    outcome = manifest.get("run_outcome")
    if not isinstance(outcome, dict):
        raise TypeError("first-failure run outcome is absent")
    for name in (
        "capture_failures",
        "infrastructure_failures",
        "corpus_failures",
        "trace_failures",
        "background_reactivations",
    ):
        if int(outcome.get(name, -1)) != 0:
            raise ValueError(f"first-failure prefix has nonzero {name}")
    if (
        outcome.get("corpus_failure") is not None
        or outcome.get("policy_transaction_failure") is not None
        or outcome.get("policy_state_recovered") is not False
        or outcome.get("policy_state_committed") is not False
        or outcome.get("policy_state_rolled_back") is not False
        or outcome.get("stage_completed") is not False
    ):
        raise ValueError("first-failure prefix has a contaminated run outcome")
    if failure_kind == "life-lost":
        expected = (10, 1, "authority-stop:physical HIT")
    else:
        expected = (
            12,
            0,
            str(outcome.get("termination_reason", "")),
        )
        if expected[2] not in (
            "authority-stop:Hard safe set empty",
            "authority-stop:local forecast has no safe continuation",
        ):
            raise ValueError("control-dead-end prefix has an invalid termination reason")
    observed = (
        int(outcome.get("controller_exit_code", -1)),
        int(outcome.get("physical_hits", -1)),
        str(outcome.get("termination_reason", "")),
    )
    if observed != expected:
        raise ValueError(
            f"first-failure outcome/terminal mismatch: {observed!r} != {expected!r}"
        )


def _validate_failure_window(
    raw_rows: list[dict[str, object]],
    *,
    failure_frame: int,
    failure_context: str,
    segment_start_frame: int,
    horizon: int = FAILURE_HORIZON_FRAMES,
) -> None:
    """Reject labels whose same-context causal window crosses a frame gap."""
    window_start = max(segment_start_frame, failure_frame - horizon)
    for index, row in enumerate(raw_rows):
        scope = row.get("scope")
        if not isinstance(scope, dict) or scope.get("phase_id") != failure_context:
            continue
        source_frame = _frame(row.get("snapshot_ref"))
        target_frame = _frame(row.get("next_snapshot_ref"))
        outcome = row.get("outcome_terms")
        if not isinstance(outcome, dict):
            raise TypeError("first-failure transition outcome is absent")
        elapsed = int(outcome.get("elapsed_frames", -1))
        if target_frame - source_frame != elapsed:
            raise ValueError(
                f"first-failure transition {index} has inconsistent frame evidence"
            )
        overlaps_window = source_frame <= failure_frame and target_frame >= window_start
        if overlaps_window and elapsed != 1:
            raise ValueError(
                "first-failure risk window crosses an observation gap at "
                f"transition {index}"
            )


def _terminal_segment_start(
    raw_rows: list[dict[str, object]],
    *,
    failure_sequence: int,
    failure_context: str,
) -> int:
    """Find the trailing contiguous occurrence of a reusable source context."""
    start = failure_sequence
    while start >= 0:
        row = raw_rows[start]
        scope = row.get("scope")
        if not isinstance(scope, dict) or scope.get("phase_id") != failure_context:
            break
        if start < failure_sequence:
            following = raw_rows[start + 1]
            if row.get("next_snapshot_ref") != following.get("snapshot_ref"):
                raise ValueError("first-failure transitions are not snapshot-contiguous")
            next_scope = row.get("next_scope")
            following_scope = following.get("scope")
            if not isinstance(next_scope, dict) or not isinstance(following_scope, dict):
                raise TypeError("first-failure transition scope linkage is absent")
            if next_scope.get("key") != following_scope.get("key"):
                raise ValueError("first-failure transitions are not scope-contiguous")
        start -= 1
    first = start + 1
    if first > failure_sequence:
        raise ValueError("terminal transition does not belong to its event context")
    return _frame(raw_rows[first].get("snapshot_ref"))


def _action_components(name: str) -> tuple[float, float, float]:
    core = name.removesuffix("_fast")
    return (
        float("right" in core) - float("left" in core),
        float("down" in core) - float("up" in core),
        float(not name.endswith("_fast")),
    )


def _edge(x: float, y: float) -> float:
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def risk_features(
    row: LabeledTransition,
    hard_action_evaluations,
) -> dict[str, str | float]:
    """Construct phase-agnostic features from factual native certificates."""
    evaluations: dict[str, tuple[float, float, float]] = {}
    for raw in hard_action_evaluations:
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            raise TypeError("hard action evaluation must contain action/clearance/x/y")
        action = str(raw[0])
        if action not in ACTION_NAMES or action in evaluations:
            raise ValueError("hard action evaluation identity is invalid or duplicated")
        clearance = _CLEARANCE_SENTINEL if raw[1] is None else float(raw[1])
        evaluations[action] = (clearance, float(raw[2]), float(raw[3]))
    if row.action not in evaluations or row.baseline_action not in evaluations:
        raise ValueError("factual/baseline action lacks a native Hard evaluation")
    incumbent_clearance, incumbent_x, incumbent_y = evaluations[row.action]
    baseline_clearance, baseline_x, baseline_y = evaluations[row.baseline_action]
    clearances = [value[0] for value in evaluations.values()]
    ordered = sorted(evaluations, key=lambda action: (-evaluations[action][0], action))
    rank = ordered.index(row.action) / max(1, len(ordered) - 1)
    current = str(row.features["current_action"])
    action_dx, action_dy, action_focused = _action_components(row.action)
    baseline_dx, baseline_dy, baseline_focused = _action_components(row.baseline_action)
    current_dx, current_dy, current_focused = _action_components(current)
    features: dict[str, str | float] = {
        "action": row.action,
        "baseline_action": row.baseline_action,
        "current_action": current,
        "player_x": float(row.features["player_x"]),
        "player_y": float(row.features["player_y"]),
        "edge_reserve": float(row.features["edge_reserve"]),
        "power": float(row.features["power"]),
        "bullet_count": float(row.features["bullet_count"]),
        "laser_count": float(row.features["laser_count"]),
        "hard_action_count": float(len(evaluations)),
        "legal_action_count": float(len(row.legal_actions)),
        "phase_elapsed_frames": float(row.features["phase_elapsed_frames"]),
        "action_dx": action_dx,
        "action_dy": action_dy,
        "action_focused": action_focused,
        "action_stationary": float(action_dx == 0.0 and action_dy == 0.0),
        "action_diagonal": float(action_dx != 0.0 and action_dy != 0.0),
        "baseline_dx": baseline_dx,
        "baseline_dy": baseline_dy,
        "baseline_focused": baseline_focused,
        "current_dx": current_dx,
        "current_dy": current_dy,
        "current_focused": current_focused,
        "matches_baseline": float(row.action == row.baseline_action),
        "matches_current": float(row.action == current),
        "incumbent_hard_clearance": incumbent_clearance,
        "baseline_hard_clearance": baseline_clearance,
        "clearance_delta_baseline": incumbent_clearance - baseline_clearance,
        "hard_clearance_min": min(clearances),
        "hard_clearance_max": max(clearances),
        "hard_clearance_span": max(clearances) - min(clearances),
        "incumbent_clearance_rank": rank,
        "incumbent_final_x": incumbent_x,
        "incumbent_final_y": incumbent_y,
        "incumbent_final_edge_reserve": _edge(incumbent_x, incumbent_y),
        "baseline_final_x": baseline_x,
        "baseline_final_y": baseline_y,
        "baseline_final_edge_reserve": _edge(baseline_x, baseline_y),
    }
    legal = set(row.legal_actions)
    hard = set(evaluations)
    features.update({f"legal_{action}": float(action in legal) for action in ACTION_NAMES})
    features.update({f"hard_{action}": float(action in hard) for action in ACTION_NAMES})
    if tuple(features) != RISK_FEATURE_NAMES:
        raise RuntimeError("risk feature construction order/schema mismatch")
    return features


def risk_features_for_context(
    context,
    action: str,
    *,
    feature_schema: str = RISK_FEATURE_SCHEMA,
) -> dict[str, str | float]:
    """Construct the identical factual feature row at the live policy boundary."""
    if action not in context.locally_admissible_actions:
        raise ValueError("risk-scored action is outside the native local set")
    row = LabeledTransition(
        run_id="live",
        sequence=int(context.frame),
        frame=int(context.frame),
        source_context=str(context.source_context),
        action=action,
        baseline_action=str(context.baseline_action),
        legal_actions=tuple(context.locally_admissible_actions),
        behavior_probability=1.0,
        features={
            "current_action": str(context.current_action),
            "player_x": float(context.player_x),
            "player_y": float(context.player_y),
            "edge_reserve": _edge(
                float(context.player_x), float(context.player_y),
            ),
            "power": float(context.power),
            "bullet_count": float(context.bullet_count),
            "laser_count": float(context.laser_count),
            "phase_elapsed_frames": float(context.phase_elapsed_frames),
        },
        reward=0.0,
    )
    features = risk_features(row, context.hard_action_evaluations)
    if feature_schema == RISK_FEATURE_SCHEMA:
        return features
    names, _categorical = risk_feature_contract(feature_schema)
    contextual: dict[str, str | float] = {
        "source_context": str(context.source_context),
    }
    contextual.update({name: features[name] for name in names[1:]})
    if tuple(contextual) != names:
        raise RuntimeError("risk v2 feature construction order/schema mismatch")
    return contextual


def label_failure_risk(
    rows: list[LabeledTransition],
    *,
    failure_frame: int,
    failure_context: str,
    segment_start_frame: int | None = None,
    horizon: int = FAILURE_HORIZON_FRAMES,
    features_by_sequence: dict[int, dict[str, str | float]] | None = None,
) -> tuple[RiskExample, ...]:
    if horizon <= 0:
        raise ValueError("failure-risk horizon must be positive")
    examples = []
    positive_start = max(
        failure_frame - horizon,
        segment_start_frame if segment_start_frame is not None else failure_frame - horizon,
    )
    for row in rows:
        lag = failure_frame - row.frame
        positive = (
            row.source_context == failure_context
            and positive_start <= row.frame <= failure_frame
        )
        examples.append(RiskExample(
            transition=row,
            features=(
                dict(row.features)
                if features_by_sequence is None
                else features_by_sequence[row.sequence]
            ),
            failure_within_120=positive,
            frames_to_failure=lag if positive else None,
        ))
    return tuple(examples)


def load_first_failure_prefix(
    run_dir: Path,
    *,
    expected_scope: tuple[int, int, int, int],
    expected_executable_sha256: str,
    expected_native_kernel_sha256: str,
    expected_policy_id: str | None = FROZEN_INCUMBENT_POLICY_ID,
) -> FirstFailurePrefix:
    """Validate and load one terminal prefix; reject benchmark continuation."""
    run_dir = run_dir.resolve()
    manifest_path = run_dir / "manifest.json"
    run_path = run_dir / "run.json"
    manifest = _object(manifest_path)
    run = _object(run_path)
    run_id = str(run.get("run_id", ""))
    if not run_id or run_id != run_dir.name or manifest.get("run_id") != run_id:
        raise ValueError("first-failure run identity mismatch")
    if run.get("schema_version") != "th06-rl-run-v1":
        raise ValueError("unsupported first-failure run schema")
    if manifest.get("schema_version") != "th06-rl-manifest-v2":
        raise ValueError("unsupported first-failure manifest schema")
    if (
        manifest.get("complete") is not True
        or int(manifest.get("dropped_records", -1)) != 0
        or manifest.get("stage_trajectory_complete") is not False
    ):
        raise ValueError("first-failure prefix is incomplete or is a complete Stage")
    episode = manifest.get("episode")
    if (
        not isinstance(episode, dict)
        or episode.get("unit") != "practice-stage"
        or episode.get("complete") is not False
    ):
        raise ValueError("first-failure prefix is not an incomplete Practice episode")
    schemas = run.get("schemas")
    if not isinstance(schemas, dict) or schemas.get("transition") != "th06-rl-transition-v5":
        raise ValueError("first-failure prefix lacks exact-v5 transitions")
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        raise TypeError("first-failure run metadata is absent")
    scope = tuple(int(metadata[name]) for name in (
        "difficulty", "character", "shot_type", "stage",
    ))
    if scope != expected_scope:
        raise ValueError(f"first-failure scope mismatch: {scope} != {expected_scope}")
    executable_sha256 = str(metadata.get("executable_sha256", ""))
    native_kernel_sha256 = str(metadata.get("native_kernel_sha256", ""))
    if executable_sha256 != expected_executable_sha256:
        raise ValueError("first-failure retail executable SHA-256 mismatch")
    if native_kernel_sha256 != expected_native_kernel_sha256:
        raise ValueError("first-failure native kernel SHA-256 mismatch")
    planner = metadata.get("planner")
    if (
        metadata.get("input_backend") != "supervisor-callsite-v2"
        or
        not isinstance(planner, dict)
        or planner.get("algorithm") != "observed-native-gate-v1"
        or float(planner.get("exploration_rate", -1.0)) != 0.0
    ):
        raise ValueError("first-failure prefix was not collected by a frozen incumbent")

    events = list(_stream_rows(run_dir, manifest, "events"))
    if len(events) != 2 or events[-1].get("event") != "run-end":
        raise ValueError("first-failure prefix must contain one terminal event then run-end")
    failure_kind = str(events[0].get("event", ""))
    if failure_kind not in TERMINAL_FAILURES:
        raise ValueError(f"unsupported first-failure terminal event: {failure_kind!r}")
    failure_sequence = int(events[0].get("sequence", -1))
    failure_frame = _frame(events[0].get("snapshot_ref"))
    failure_scope = events[0].get("scope")
    if not isinstance(failure_scope, dict):
        raise TypeError("first-failure terminal scope is absent")
    failure_context = str(failure_scope.get("phase_id", ""))
    if not failure_context:
        raise ValueError("first-failure terminal context is absent")
    _validate_clean_run_outcome(manifest, failure_kind=failure_kind)

    transitions = int((manifest.get("records") or {}).get("transitions", -1))
    if transitions <= 0 or failure_sequence != transitions - 1:
        raise ValueError("first-failure event is not the final transition")
    descriptor = RunDescriptor(
        run_id=run_id,
        remote_path=run_id,
        scope=scope,  # type: ignore[arg-type]
        transition_schema="th06-rl-transition-v5",
        transitions=transitions,
        storage_complete=True,
        stage_complete=False,
        # This explicit local override is safe only after all terminal-prefix
        # checks above.  It allows the common factual feature constructor to
        # retain eligible rows without claiming complete-Stage training data.
        training_eligible=True,
        code_commit=str(metadata.get("code_commit", "")),
        native_kernel_sha256=native_kernel_sha256,
        physical_hits=1 if failure_kind == "life-lost" else 0,
        manifest_sha256=_sha256(manifest_path),
        run_sha256=_sha256(run_path),
    )
    raw_rows = list(iter_run_transitions(
        run_dir.parent,
        descriptor,
        verify_sha256=True,
    ))
    if len(raw_rows) != transitions:
        raise ValueError("first-failure transition count does not match its manifest")
    if _frame(raw_rows[failure_sequence].get("next_snapshot_ref")) != failure_frame:
        raise ValueError("first-failure event does not identify the terminal target frame")
    failure_segment_start_frame = _terminal_segment_start(
        raw_rows,
        failure_sequence=failure_sequence,
        failure_context=failure_context,
    )
    _validate_failure_window(
        raw_rows,
        failure_frame=failure_frame,
        failure_context=failure_context,
        segment_start_frame=failure_segment_start_frame,
    )
    for index, row in enumerate(raw_rows):
        scope_row = row.get("scope")
        if not isinstance(scope_row, dict) or tuple(
            int(scope_row[name]) for name in (
                "difficulty", "character", "shot_type", "stage",
            )
        ) != expected_scope:
            raise ValueError(f"scope pollution in first-failure transition {index}")
        outcome = row.get("outcome_terms")
        if not isinstance(outcome, dict):
            raise TypeError("first-failure transition outcome is absent")
        if outcome.get("bomb_used"):
            raise ValueError("Bomb transition is forbidden in first-failure data")
        observed_terminal = (
            outcome.get("life_lost") is True
            if failure_kind == "life-lost"
            else outcome.get("control_dead_end") is True
        )
        if observed_terminal != (index == failure_sequence):
            raise ValueError("first-failure terminal outcome/event mismatch")
        if index == failure_sequence and failure_kind == "life-lost":
            # A HIT observed across a capture/staleness gap is infrastructure
            # evidence, not an attributable factual action label.
            if (
                row.get("learning_eligible") is not True
                or int(outcome.get("elapsed_frames", 0)) != 1
            ):
                raise ValueError(
                    "first-failure HIT is not attributable across an exact one-frame transition"
                )
    labeled = label_transitions(raw_rows, descriptor, exact_context_only=True)
    decisions: dict[int, dict[str, object]] = {}
    for frame_row in _stream_rows(run_dir, manifest, "frames"):
        sequence = int(frame_row.get("sequence", -1))
        if sequence >= transitions:
            continue
        decision = frame_row.get("decision")
        if not isinstance(decision, dict):
            raise TypeError("first-failure frame decision is absent")
        if sequence in decisions:
            raise ValueError("duplicate first-failure frame sequence")
        decisions[sequence] = decision
    features_by_sequence = {}
    policy_rows = []
    for row in labeled:
        decision = decisions.get(row.sequence)
        if decision is None:
            raise ValueError("eligible transition has no matching frame decision")
        # Input-lease frames publish a previously requested action without
        # calling the policy and consequently have no reactive baseline.
        # They are valid delivery evidence but cannot be guard decisions.
        if decision.get("reason") != "ok":
            continue
        if (
            expected_policy_id is not None
            and decision.get("policy_id") != expected_policy_id
        ):
            raise ValueError(
                "first-failure policy identity does not match the frozen incumbent"
            )
        features_by_sequence[row.sequence] = risk_features(
            row,
            decision.get("hard_actions", ()),
        )
        policy_rows.append(row)
    examples = label_failure_risk(
        policy_rows,
        failure_frame=failure_frame,
        failure_context=failure_context,
        segment_start_frame=failure_segment_start_frame,
        features_by_sequence=features_by_sequence,
    )
    if not examples or not any(example.failure_within_120 for example in examples):
        raise ValueError("first-failure prefix has no eligible positive risk window")
    return FirstFailurePrefix(
        schema=FIRST_FAILURE_SCHEMA,
        run_id=run_id,
        run_dir=run_dir,
        scope=scope,  # type: ignore[arg-type]
        executable_sha256=executable_sha256,
        native_kernel_sha256=native_kernel_sha256,
        code_commit=str(metadata.get("code_commit", "")),
        manifest_sha256=descriptor.manifest_sha256,
        run_sha256=descriptor.run_sha256,
        failure_kind=failure_kind,
        failure_frame=failure_frame,
        failure_context=failure_context,
        failure_segment_start_frame=failure_segment_start_frame,
        positive_window_start_frame=max(
            failure_segment_start_frame,
            failure_frame - FAILURE_HORIZON_FRAMES,
        ),
        transitions=transitions,
        examples=examples,
    )
