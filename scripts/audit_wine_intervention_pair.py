#!/usr/bin/env python3
"""Accept a Wine intervention pair only when its physical branch root matches."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path


SCHEMA = "th06-rl-wine-intervention-pair-audit-v1"
POLICY_PREFIX = "wine-one-shot-intervention-v1:"
NON_PHYSICAL_CAPTURE_FIELDS = ("capture_attempts", "bullet_read_retries")


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return value


def _rows(run_dir: Path, manifest: dict[str, object], stream: str):
    for shard in manifest.get("shards", ()):  # type: ignore[union-attr]
        if not isinstance(shard, dict) or shard.get("stream") != stream:
            continue
        with gzip.open(run_dir / str(shard["path"]), "rt", encoding="utf-8") as source:
            for line in source:
                yield json.loads(line)


def _root_projection(snapshot: dict[str, object]) -> dict[str, object]:
    """Remove capture mechanics while retaining every physical source field."""
    return {
        key: value
        for key, value in snapshot.items()
        if key not in NON_PHYSICAL_CAPTURE_FIELDS
    }


def _root_hash(snapshot: dict[str, object]) -> str:
    payload = json.dumps(
        _root_projection(snapshot),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _intervention_event(
    run_dir: Path,
    manifest: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    events = []
    for row in _rows(run_dir, manifest, "frames"):
        decision = row.get("decision")
        if not isinstance(decision, dict):
            continue
        policy_id = decision.get("policy_id")
        if (
            isinstance(policy_id, str)
            and policy_id.startswith(POLICY_PREFIX)
            and float(decision.get("behavior_probability", 1.0)) < 1.0
        ):
            events.append((row, decision))
    if len(events) != 1:
        raise ValueError(
            f"expected exactly one propensity-recorded intervention in {run_dir}, "
            f"found {len(events)}"
        )
    return events[0]


def _event_identity(decision: dict[str, object]) -> dict[str, str]:
    policy_id = str(decision["policy_id"])
    parts = policy_id.split(":")
    if len(parts) != 5:
        raise ValueError(f"invalid intervention policy id: {policy_id}")
    _name, pair_id, arm, incumbent_action, alternative_action = parts
    return {
        "pair_id": pair_id,
        "arm": arm,
        "incumbent_action": incumbent_action,
        "alternative_action": alternative_action,
    }


def _run_summary(run_dir: Path) -> dict[str, object]:
    run = _object(run_dir / "run.json")
    manifest = _object(run_dir / "manifest.json")
    if manifest.get("complete") is not True or int(manifest.get("dropped_records", 0)):
        raise ValueError(f"incomplete or dropped intervention corpus: {run_dir}")
    row, decision = _intervention_event(run_dir, manifest)
    snapshot = row.get("snapshot")
    if not isinstance(snapshot, dict):
        raise ValueError("intervention frame has no compact snapshot")
    identity = _event_identity(decision)
    outcome = manifest.get("run_outcome")
    if not isinstance(outcome, dict):
        raise ValueError("manifest has no run outcome")
    last_frame = int((manifest.get("summary") or {}).get("last_frame", -1))
    event_frame = int(snapshot["frame"])
    metadata = run.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError("run has no metadata")
    planner = metadata.get("planner")
    if not isinstance(planner, dict):
        raise ValueError("run has no planner provenance")
    return {
        "run_dir": str(run_dir.resolve()),
        "run_id": run.get("run_id"),
        "identity": identity,
        "sequence": int(row["sequence"]),
        "frame": event_frame,
        "root_hash": _root_hash(snapshot),
        "rng_seed_at_root": int(snapshot["rng_seed"]),
        "rng_generation_at_root": int(snapshot["rng_generation"]),
        "diagnostic_rng_seed": planner.get("diagnostic_rng_seed"),
        "scope": {
            key: metadata.get(key)
            for key in ("difficulty", "character", "shot_type", "stage")
        },
        "executable_sha256": metadata.get("executable_sha256"),
        "native_kernel_sha256": metadata.get("native_kernel_sha256"),
        "policy_id": decision.get("policy_id"),
        "policy_sha256": decision.get("policy_sha256"),
        "behavior_probability": decision.get("behavior_probability"),
        "published_action": decision.get("published_action"),
        "hard_actions": decision.get("hard_actions"),
        "locally_admissible_actions": decision.get("locally_admissible_actions"),
        "survival_frames_after_intervention": max(0, last_frame - event_frame),
        "last_frame": last_frame,
        "physical_hits": int(outcome.get("physical_hits", 0)),
        "stage_completed": bool(outcome.get("stage_completed", False)),
        "termination_reason": outcome.get("termination_reason"),
    }


def audit(incumbent_run: Path, alternative_run: Path) -> dict[str, object]:
    incumbent = _run_summary(incumbent_run)
    alternative = _run_summary(alternative_run)
    left_identity = incumbent["identity"]
    right_identity = alternative["identity"]
    assert isinstance(left_identity, dict) and isinstance(right_identity, dict)
    contract_errors = []
    if left_identity.get("arm") != "incumbent":
        contract_errors.append("left-arm-is-not-incumbent")
    if right_identity.get("arm") != "alternative":
        contract_errors.append("right-arm-is-not-alternative")
    for field in (
        "pair_id",
        "incumbent_action",
        "alternative_action",
    ):
        if left_identity.get(field) != right_identity.get(field):
            contract_errors.append(f"identity-mismatch:{field}")
    for field in (
        "diagnostic_rng_seed",
        "scope",
        "executable_sha256",
        "native_kernel_sha256",
        "policy_sha256",
    ):
        if incumbent.get(field) != alternative.get(field):
            contract_errors.append(f"provenance-mismatch:{field}")
    if incumbent.get("behavior_probability") != 0.5:
        contract_errors.append("incumbent-propensity-is-not-0.5")
    if alternative.get("behavior_probability") != 0.5:
        contract_errors.append("alternative-propensity-is-not-0.5")
    root_match = incumbent["root_hash"] == alternative["root_hash"]
    pair_accepted = not contract_errors and root_match
    if pair_accepted:
        survival_delta = (
            int(alternative["survival_frames_after_intervention"])
            - int(incumbent["survival_frames_after_intervention"])
        )
        hit_delta = int(alternative["physical_hits"]) - int(incumbent["physical_hits"])
    else:
        survival_delta = None
        hit_delta = None
    return {
        "schema": SCHEMA,
        "pair_accepted": pair_accepted,
        "root_match": root_match,
        "contract_errors": contract_errors,
        "causal_effect_available": pair_accepted,
        "survival_frame_delta_alternative_minus_incumbent": survival_delta,
        "physical_hit_delta_alternative_minus_incumbent": hit_delta,
        "incumbent": incumbent,
        "alternative": alternative,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-run", type=Path, required=True)
    parser.add_argument("--alternative-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.incumbent_run, args.alternative_run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["pair_accepted"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
