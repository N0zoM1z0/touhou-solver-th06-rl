#!/usr/bin/env python3
"""Build factual, episode-grouped labels from one-shot Wine interventions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.audit_wine_intervention_pair import (
    POLICY_PREFIX,
    _event_identity,
    _root_hash,
)
from th06_rl.corpus import FRAME_BUDGET_MS
from th06_rl.wine_intervention_learning import (
    FEATURE_NAMES,
    FEATURE_SCHEMA,
    action_relative_features,
    edge_reserve,
)


SCHEMA = "th06-rl-wine-intervention-dataset-v1"
OUTCOME_HORIZON = 120


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _rows(run_dir: Path, manifest: dict[str, object], stream: str):
    for shard in manifest.get("shards", ()):  # type: ignore[union-attr]
        if not isinstance(shard, dict) or shard.get("stream") != stream:
            continue
        with gzip.open(run_dir / str(shard["path"]), "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _dataset_row(run_dir: Path) -> tuple[dict[str, object] | None, str | None]:
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    if manifest.get("complete") is not True or int(manifest.get("dropped_records", 0)):
        raise ValueError(f"incomplete intervention corpus: {run_dir}")
    outcome = manifest.get("run_outcome")
    metadata = run.get("metadata")
    if not isinstance(outcome, dict) or not isinstance(metadata, dict):
        raise TypeError("Wine intervention provenance/outcome is absent")
    forbidden = (
        "capture_failures",
        "infrastructure_failures",
        "corpus_failures",
        "trace_failures",
        "background_reactivations",
    )
    if any(int(outcome.get(name, -1)) != 0 for name in forbidden):
        raise ValueError(f"intervention run has infrastructure failures: {run_dir}")
    frames = list(_rows(run_dir, manifest, "frames"))
    events = []
    for index, row in enumerate(frames):
        decision = row.get("decision")
        if not isinstance(decision, dict):
            continue
        policy_id = decision.get("policy_id")
        if (
            isinstance(policy_id, str)
            and policy_id.startswith(POLICY_PREFIX)
            and float(decision.get("behavior_probability", 1.0)) < 1.0
        ):
            events.append((index, row, decision))
    if not events:
        return None, "no-eligible-intervention"
    if len(events) != 1:
        raise ValueError(f"run has {len(events)} intervention events: {run_dir}")
    index, event, decision = events[0]
    snapshot = event.get("snapshot")
    if not isinstance(snapshot, dict):
        raise TypeError("intervention event has no snapshot")
    if (
        decision.get("reason") != "ok"
        or decision.get("published_action") is None
        or int(decision.get("observation_gap", 0)) != 1
        or float(decision.get("capture_ms", float("inf"))) > FRAME_BUDGET_MS
    ):
        return None, "intervention-publication-not-learning-eligible"
    identity = _event_identity(decision)
    selected = str(decision["published_action"])
    incumbent_action = identity["incumbent_action"]
    local = tuple(str(value) for value in decision["locally_admissible_actions"])
    if selected not in local or incumbent_action not in local:
        raise ValueError("intervention action is outside the recorded local set")

    event_frame = int(snapshot["frame"])
    last_frame = int((manifest.get("summary") or {}).get("last_frame", event_frame))
    outcome_end = min(event_frame + OUTCOME_HORIZON, last_frame)
    window = [
        row
        for row in frames[index:]
        if event_frame <= int(row["snapshot"]["frame"]) <= outcome_end
    ]
    observed_frames = [int(row["snapshot"]["frame"]) for row in window]
    if observed_frames != list(range(event_frame, outcome_end + 1)):
        return None, "outcome-window-has-observation-gap"
    hard_counts = [len((row.get("decision") or {}).get("hard_actions", ())) for row in window]
    local_counts = [
        len((row.get("decision") or {}).get("locally_admissible_actions", ()))
        for row in window
    ]
    edges = [
        edge_reserve(float(row["snapshot"]["x"]), float(row["snapshot"]["y"]))
        for row in window
    ]
    survival_frames = max(0, outcome_end - event_frame)
    terminal_within_horizon = last_frame <= event_frame + OUTCOME_HORIZON
    physical_hit = bool(int(outcome.get("physical_hits", 0)))
    survived_fraction = survival_frames / OUTCOME_HORIZON
    score = (
        survived_fraction
        + 0.10 * min(hard_counts) / 18.0
        + 0.05 * max(0.0, min(1.0, min(edges) / 32.0))
        - 0.50 * float(physical_hit)
    )
    features = action_relative_features(
        player_x=float(snapshot["x"]),
        player_y=float(snapshot["y"]),
        bullet_count=int(snapshot["live_bullet_count"]),
        hard_action_count=len(decision["hard_actions"]),
        local_action_count=len(local),
        effort_horizon=int(decision["effort_horizon"]),
        current_action=str(decision["current_action"]),
        baseline_action=str(decision["baseline_action"]),
        action=selected,
        incumbent_action=incumbent_action,
        evaluations=decision["hard_actions"],
    )
    return {
        "run_id": run["run_id"],
        "run_dir": str(run_dir.resolve()),
        "pair_id": identity["pair_id"],
        "arm": identity["arm"],
        "frame": event_frame,
        "root_hash": _root_hash(snapshot),
        "behavior_probability": float(decision["behavior_probability"]),
        "incumbent_action": incumbent_action,
        "alternative_action": identity["alternative_action"],
        "selected_action": selected,
        "features": features,
        "outcome": {
            "horizon_frames": OUTCOME_HORIZON,
            "survival_frames": survival_frames,
            "survived_horizon": not terminal_within_horizon,
            "terminal_within_horizon": terminal_within_horizon,
            "physical_hit": physical_hit,
            "termination_reason": outcome.get("termination_reason"),
            "min_hard_action_count": min(hard_counts),
            "min_local_action_count": min(local_counts),
            "min_edge_reserve": min(edges),
            "score": score,
        },
        "provenance": {
            key: metadata.get(key)
            for key in (
                "code_commit",
                "executable_sha256",
                "native_kernel_sha256",
                "difficulty",
                "character",
                "shot_type",
                "stage",
            )
        },
    }, None


def build(run_dirs: list[Path]) -> dict[str, object]:
    rows = []
    excluded = []
    for run_dir in run_dirs:
        row, reason = _dataset_row(run_dir.resolve())
        if row is None:
            excluded.append({"run_dir": str(run_dir.resolve()), "reason": reason})
        else:
            rows.append(row)
    provenance = {
        (
            row["provenance"]["executable_sha256"],
            row["provenance"]["native_kernel_sha256"],
            row["provenance"]["difficulty"],
            row["provenance"]["character"],
            row["provenance"]["shot_type"],
            row["provenance"]["stage"],
        )
        for row in rows
    }
    if len(provenance) > 1:
        raise ValueError("intervention dataset mixes physical provenance/scope")
    return {
        "schema": SCHEMA,
        "feature_schema": FEATURE_SCHEMA,
        "feature_names": list(FEATURE_NAMES),
        "outcome_horizon_frames": OUTCOME_HORIZON,
        "grouping_unit": "physical-episode-and-pair-id",
        "rows": rows,
        "excluded": excluded,
        "summary": {
            "requested_runs": len(run_dirs),
            "eligible_rows": len(rows),
            "excluded_runs": len(excluded),
            "incumbent_rows": sum(row["arm"] == "incumbent" for row in rows),
            "alternative_rows": sum(row["arm"] == "alternative" for row in rows),
            "unique_pairs": len({row["pair_id"] for row in rows}),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    dataset = build(args.runs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(dataset, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(dataset["summary"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
