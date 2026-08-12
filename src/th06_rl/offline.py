"""Standalone readers and audits for immutable TH06-RL corpus snapshots.

This module deliberately does not import the live donor adapter.  CPU training
hosts can validate compact transition shards without a game process, a game
binary, or the older runtime checkout.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import re
from typing import Iterator


DATASET_SCHEMA = "th06-rl-hf-dataset-v1"
TRANSITION_SCHEMAS = tuple(f"th06-rl-transition-v{version}" for version in range(1, 9))
ACTION_NAMES = (
    "stay", "up", "down", "left", "right",
    "up_left", "up_right", "down_left", "down_right",
    "stay_fast", "up_fast", "down_fast", "left_fast", "right_fast",
    "up_left_fast", "up_right_fast", "down_left_fast", "down_right_fast",
)
ACTION_SET = frozenset(ACTION_NAMES)
HIT_HORIZONS = (30, 60, 120)
_FRAME_RE = re.compile(r":f(\d+)$")


@dataclass(frozen=True)
class RunDescriptor:
    run_id: str
    remote_path: str
    scope: tuple[int, int, int, int]
    transition_schema: str
    transitions: int
    storage_complete: bool
    stage_complete: bool
    training_eligible: bool
    code_commit: str
    native_kernel_sha256: str | None
    physical_hits: int | None
    manifest_sha256: str
    run_sha256: str


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


def load_dataset_index(root: Path) -> tuple[dict[str, object], tuple[RunDescriptor, ...]]:
    manifest = _object(root / "dataset_manifest.json")
    if manifest.get("schema") != DATASET_SCHEMA:
        raise ValueError("unsupported TH06-RL dataset manifest")
    raw_runs = manifest.get("runs")
    if not isinstance(raw_runs, list):
        raise TypeError("dataset runs must be a list")
    runs: list[RunDescriptor] = []
    seen: set[str] = set()
    for raw in raw_runs:
        if not isinstance(raw, dict):
            raise TypeError("dataset run row must be an object")
        run_id = str(raw.get("run_id", ""))
        remote_path = str(raw.get("remote_path", ""))
        if not run_id or run_id in seen or remote_path != f"runs/{run_id}":
            raise ValueError(f"invalid or duplicate run identity: {run_id!r}")
        seen.add(run_id)
        schemas = raw.get("schemas")
        records = raw.get("records")
        if not isinstance(schemas, dict) or not isinstance(records, dict):
            raise TypeError(f"run schema/record metadata is invalid: {run_id}")
        transition_schema = str(schemas.get("transition", ""))
        if transition_schema not in TRANSITION_SCHEMAS:
            raise ValueError(f"unsupported transition schema in {run_id}: {transition_schema}")
        scope = tuple(int(raw[name]) for name in ("difficulty", "character", "shot_type", "stage"))
        runs.append(RunDescriptor(
            run_id=run_id,
            remote_path=remote_path,
            scope=scope,  # type: ignore[arg-type]
            transition_schema=transition_schema,
            transitions=int(records.get("transitions", 0)),
            storage_complete=raw.get("storage_complete") is True,
            stage_complete=raw.get("stage_trajectory_complete") is True,
            training_eligible=raw.get("training_eligible_complete_stage") is True,
            code_commit=str(raw.get("code_commit", "")),
            native_kernel_sha256=(
                str(raw["native_kernel_sha256"])
                if raw.get("native_kernel_sha256") is not None
                else None
            ),
            physical_hits=(
                int(raw["physical_hits"])
                if raw.get("physical_hits") is not None
                else None
            ),
            manifest_sha256=str(raw.get("manifest_sha256", "")),
            run_sha256=str(raw.get("run_sha256", "")),
        ))
    return manifest, tuple(runs)


def _transition_shards(
    root: Path,
    run: RunDescriptor,
    *,
    verify_sha256: bool,
) -> tuple[Path, ...]:
    run_dir = root / run.remote_path
    manifest_path = run_dir / "manifest.json"
    run_path = run_dir / "run.json"
    if verify_sha256:
        for path, expected in (
            (manifest_path, run.manifest_sha256),
            (run_path, run.run_sha256),
        ):
            if len(expected) != 64 or _sha256(path) != expected:
                raise ValueError(f"metadata digest mismatch: {path}")
    local_manifest = _object(manifest_path)
    if local_manifest.get("run_id") != run.run_id:
        raise ValueError(f"local run identity mismatch: {run.run_id}")
    shards = local_manifest.get("shards")
    if not isinstance(shards, list):
        raise TypeError(f"run shard list is invalid: {run.run_id}")
    selected: list[Path] = []
    expected_records = 0
    for raw in shards:
        if not isinstance(raw, dict) or raw.get("stream") != "transitions":
            continue
        name = str(raw.get("path", ""))
        if not name or Path(name).name != name:
            raise ValueError(f"unsafe transition shard name in {run.run_id}: {name!r}")
        path = run_dir / name
        if not path.is_file() or path.is_symlink():
            raise FileNotFoundError(path)
        size = int(raw.get("compressed_bytes", -1))
        if size < 0 or path.stat().st_size != size:
            raise ValueError(f"transition shard size mismatch: {path}")
        digest = str(raw.get("sha256", ""))
        if verify_sha256 and (len(digest) != 64 or _sha256(path) != digest):
            raise ValueError(f"transition shard digest mismatch: {path}")
        expected_records += int(raw.get("records", 0))
        selected.append(path)
    if expected_records != run.transitions:
        raise ValueError(f"transition record total mismatch: {run.run_id}")
    return tuple(selected)


def iter_run_transitions(
    root: Path,
    run: RunDescriptor,
    *,
    verify_sha256: bool = True,
) -> Iterator[dict[str, object]]:
    expected_sequence = 0
    for shard in _transition_shards(root, run, verify_sha256=verify_sha256):
        with gzip.open(shard, "rt", encoding="utf-8") as source:
            for line_number, raw in enumerate(source, 1):
                try:
                    row = json.loads(raw)
                except json.JSONDecodeError as error:
                    raise ValueError(f"malformed transition: {shard}:{line_number}") from error
                if not isinstance(row, dict):
                    raise TypeError(f"transition is not an object: {shard}:{line_number}")
                if row.get("schema_version") != run.transition_schema:
                    raise ValueError(f"transition schema mismatch: {shard}:{line_number}")
                if int(row.get("sequence", -1)) != expected_sequence:
                    raise ValueError(
                        f"transition sequence gap in {run.run_id}: "
                        f"expected {expected_sequence}, got {row.get('sequence')}"
                    )
                expected_sequence += 1
                yield row
    if expected_sequence != run.transitions:
        raise ValueError(
            f"transition row count mismatch in {run.run_id}: "
            f"{expected_sequence} != {run.transitions}"
        )


def _frame(reference: object, fallback: int) -> int:
    match = _FRAME_RE.search(str(reference))
    return int(match.group(1)) if match else fallback


def _scope_name(scope: tuple[int, int, int, int]) -> str:
    return "/".join(map(str, scope))


class _AuditAccumulator:
    def __init__(self) -> None:
        self.counts: Counter[str] = Counter()
        self.schemas: Counter[str] = Counter()
        self.commits: Counter[str] = Counter()
        self.exclusions: Counter[str] = Counter()
        self.propensity_buckets: Counter[str] = Counter()
        self.legal_sizes: Counter[int] = Counter()
        self.legal_opportunities: Counter[str] = Counter()
        self.selected: Counter[str] = Counter()
        self.weight_sum: Counter[str] = Counter()
        self.weight_square_sum: Counter[str] = Counter()
        self.source_contexts: set[str] = set()
        self.legal_masks: set[tuple[str, ...]] = set()
        self.hit_exposures: Counter[int] = Counter()

    def add_run(self, run: RunDescriptor) -> None:
        self.counts["runs"] += 1
        self.counts["indexed_transitions"] += run.transitions
        self.counts["storage_complete_runs"] += run.storage_complete
        self.counts["stage_complete_runs"] += run.stage_complete
        self.counts["training_eligible_runs"] += run.training_eligible
        self.counts["training_eligible_indexed_transitions"] += (
            run.transitions if run.training_eligible else 0
        )
        self.schemas[run.transition_schema] += 1
        self.commits[run.code_commit] += 1

    @staticmethod
    def _bucket(probability: float) -> str:
        if probability < 0.01:
            return "[0,0.01)"
        if probability < 0.05:
            return "[0.01,0.05)"
        if probability < 0.20:
            return "[0.05,0.20)"
        if probability < 0.80:
            return "[0.20,0.80)"
        return "[0.80,1]"

    def audit_run(self, root: Path, run: RunDescriptor, *, verify_sha256: bool) -> None:
        recent: deque[tuple[int, str, str]] = deque()
        for row in iter_run_transitions(root, run, verify_sha256=verify_sha256):
            self.counts["rows"] += 1
            legal_raw = row.get("legal_actions")
            legal = tuple(str(action) for action in legal_raw) if isinstance(legal_raw, list) else ()
            action = row.get("published_action")
            action = str(action) if action is not None else None
            proposal = row.get("proposed_action")
            proposal = str(proposal) if proposal is not None else None
            outcome = row.get("outcome_terms")
            outcome = outcome if isinstance(outcome, dict) else {}
            policy_context = row.get("policy_context")
            phase = str(
                (row.get("scope") or {}).get("key", "")
                if isinstance(row.get("scope"), dict)
                else ""
            )
            self.source_contexts.add(phase)
            self.legal_masks.add(tuple(sorted(legal)))
            self.legal_sizes[len(legal)] += 1
            self.legal_opportunities.update(legal)
            self.counts["published_rows"] += action is not None
            self.counts["transition_learning_eligible_rows"] += row.get("learning_eligible") is True
            self.counts["exact_policy_context_rows"] += isinstance(policy_context, dict)
            self.counts["life_lost_rows"] += outcome.get("life_lost") is True
            self.counts["bomb_rows"] += outcome.get("bomb_used") is True
            self.counts["control_dead_end_rows"] += outcome.get("control_dead_end") is True
            self.counts["authority_lost_rows"] += outcome.get("authority_lost") is True
            self.counts["observation_gap_rows"] += int(outcome.get("elapsed_frames", 0)) != 1
            if proposal != action:
                self.counts["proposal_publication_mismatch_rows"] += 1
            if any(candidate not in ACTION_SET for candidate in legal):
                self.counts["unknown_legal_action_rows"] += 1
            if action is not None and action not in legal:
                self.counts["published_outside_legal_rows"] += 1
            if action is not None and action not in ACTION_SET:
                self.counts["unknown_published_action_rows"] += 1
            reasons = row.get("learning_exclusion_reasons")
            if isinstance(reasons, list):
                self.exclusions.update(str(reason) for reason in reasons)

            try:
                probability = float(row.get("behavior_probability", 0.0))
            except (TypeError, ValueError):
                probability = 0.0
            if 0.0 < probability <= 1.0:
                self.counts["valid_propensity_rows"] += 1
                self.propensity_buckets[self._bucket(probability)] += 1
            else:
                self.counts["invalid_propensity_rows"] += 1

            trainable = bool(
                run.training_eligible
                and row.get("learning_eligible") is True
                and action in ACTION_SET
                and action in legal
                and proposal == action
                and not outcome.get("bomb_used")
                and 0.0 < probability <= 1.0
            )
            frame = _frame(row.get("snapshot_ref"), int(row.get("sequence", 0)))
            hit_frame = _frame(row.get("next_snapshot_ref"), frame + 1)
            while recent and frame - recent[0][0] > max(HIT_HORIZONS):
                recent.popleft()
            if trainable:
                self.counts["trainable_rows"] += 1
                self.counts["trainable_exact_policy_context_rows"] += isinstance(policy_context, dict)
                self.selected[action] += 1
                weight = min(20.0, 1.0 / probability)
                self.weight_sum[action] += weight
                self.weight_square_sum[action] += weight * weight
                recent.append((frame, phase, action))
            if outcome.get("life_lost") is True:
                self.counts["physical_hit_events"] += 1
                for action_frame, action_phase, _action in recent:
                    lag = hit_frame - action_frame
                    if action_phase != phase or lag < 0:
                        continue
                    for horizon in HIT_HORIZONS:
                        if lag <= horizon:
                            self.hit_exposures[horizon] += 1
                recent.clear()
            elif outcome.get("phase_changed") is True:
                recent.clear()

    def result(self) -> dict[str, object]:
        rows = self.counts["rows"]
        trainable = self.counts["trainable_rows"]
        indexed = self.counts["indexed_transitions"]
        eligible_indexed = self.counts["training_eligible_indexed_transitions"]
        action_coverage = {}
        for action in ACTION_NAMES:
            opportunities = self.legal_opportunities[action]
            selected = self.selected[action]
            sum_weight = self.weight_sum[action]
            square_weight = self.weight_square_sum[action]
            action_coverage[action] = {
                "legal_opportunities": opportunities,
                "selected_trainable": selected,
                "selection_per_legal_opportunity": selected / opportunities if opportunities else None,
                "clipped_ipw_ess": (
                    sum_weight * sum_weight / square_weight if square_weight else 0.0
                ),
            }
        return {
            "counts": dict(sorted(self.counts.items())),
            "ratios": {
                "eligible_complete_stage_transitions_per_indexed": (
                    eligible_indexed / indexed if indexed else None
                ),
                "transition_learning_eligible_per_processed": (
                    self.counts["transition_learning_eligible_rows"] / rows if rows else None
                ),
                "factual_trainable_per_processed": trainable / rows if rows else None,
                "exact_policy_context_per_trainable": (
                    self.counts["trainable_exact_policy_context_rows"] / trainable
                    if trainable else None
                ),
            },
            "run_schemas": dict(sorted(self.schemas.items())),
            "code_commits": dict(sorted(self.commits.items())),
            "learning_exclusions": dict(self.exclusions.most_common()),
            "behavior_probability_buckets": dict(sorted(self.propensity_buckets.items())),
            "legal_action_set_sizes": {str(k): v for k, v in sorted(self.legal_sizes.items())},
            "unique_source_contexts": len(self.source_contexts),
            "unique_legal_action_sets": len(self.legal_masks),
            "hit_window_exposures": {
                str(horizon): {
                    "positive_rows": self.hit_exposures[horizon],
                    "positive_rate_per_trainable": (
                        self.hit_exposures[horizon] / trainable if trainable else None
                    ),
                }
                for horizon in HIT_HORIZONS
            },
            "action_coverage": action_coverage,
        }


def audit_dataset(
    root: Path,
    *,
    verify_sha256: bool = True,
) -> dict[str, object]:
    manifest, runs = load_dataset_index(root)
    overall = _AuditAccumulator()
    by_scope: dict[tuple[int, int, int, int], _AuditAccumulator] = defaultdict(_AuditAccumulator)
    for run in runs:
        overall.add_run(run)
        by_scope[run.scope].add_run(run)
        overall.audit_run(root, run, verify_sha256=verify_sha256)
        by_scope[run.scope].audit_run(root, run, verify_sha256=False)
    return {
        "schema": "th06-rl-offline-corpus-audit-v1",
        "dataset_schema": manifest.get("schema"),
        "overall": overall.result(),
        "scopes": {
            _scope_name(scope): accumulator.result()
            for scope, accumulator in sorted(by_scope.items())
        },
    }
