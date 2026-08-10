#!/usr/bin/env python3
"""Create one evidence-gated active consensus experiment state."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from scripts.export_wine_risk_guard_policy import (
    _canonical,
    _object,
    _precision_lower_bound,
    _sha256,
)
from th06_rl.policies.offline_risk_consensus import (
    ACTIVE_STATE_SCHEMA,
    STATE_SCHEMA,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shadow-state", type=Path, required=True)
    parser.add_argument("--shadow-validation-replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    shadow = _object(args.shadow_state)
    if shadow.get("schema") != STATE_SCHEMA or shadow.get("mode") != "shadow":
        raise SystemExit("consensus source state is not shadow-only")
    acceptance = shadow.get("acceptance")
    consensus = shadow.get("consensus")
    native = shadow.get("native_scorer")
    if (
        not isinstance(acceptance, dict)
        or acceptance.get("active_authorized") is not False
        or not isinstance(consensus, dict)
        or consensus.get("aggregation") != "minimum"
        or consensus.get("publication") != "incumbent-action-only"
        or not isinstance(native, dict)
    ):
        raise SystemExit("consensus shadow source contract is incomplete")

    replay = _object(args.shadow_validation_replay)
    totals = replay.get("totals")
    runs = replay.get("runs")
    if (
        replay.get("schema") != "th06-rl-wine-risk-consensus-replay-v1"
        or replay.get("mode") != "shadow"
        or replay.get("passed") is not True
        or replay.get("state_sha256") != _sha256(args.shadow_state)
        or replay.get("production_native_scorer_sha256") != native.get("sha256")
        or not isinstance(totals, dict)
        or not isinstance(runs, list)
    ):
        raise SystemExit("consensus physical-shadow validation did not pass")
    replay_state_path = Path(str(replay.get("state", "")))
    if (
        not replay_state_path.is_file()
        or _sha256(replay_state_path) != replay.get("state_sha256")
    ):
        raise SystemExit("validated consensus shadow state is absent or changed")
    validation_run_ids = {
        str(row.get("run_id")) for row in runs if isinstance(row, dict)
    }
    calibration_run_ids = {
        str(value) for value in acceptance.get("external_run_ids", ())
    }
    positive = int(totals.get("candidate_positive", 0))
    negative = int(totals.get("candidate_negative", 0))
    unlabeled = int(totals.get("candidate_unlabeled", 0))
    labeled = positive + negative
    candidates = int(totals.get("candidates", 0))
    calls = int(totals.get("policy_calls", 0))
    lower_bound = _precision_lower_bound(positive, labeled)
    if (
        len(validation_run_ids) < 2
        or validation_run_ids & calibration_run_ids
        or int(totals.get("recorded_incumbent_mismatches", -1)) != 0
        or int(totals.get("shadow_action_contract_violations", -1)) != 0
        or positive <= 0
        or negative != 0
        or lower_bound < 0.90
        or not calls
        or candidates / calls > 0.02
        or unlabeled > max(2, candidates // 20)
    ):
        raise SystemExit("consensus active-experiment validation is insufficient")

    active = json.loads(json.dumps(shadow))
    active["schema"] = ACTIVE_STATE_SCHEMA
    active["mode"] = "active"
    active["consensus"]["publication"] = "native-reactive-baseline-only"
    active["consensus"]["intervention_gate"] = "rising-edge"
    active["acceptance"]["active_authorized"] = True
    active["active_authorization"] = {
        "kind": "single-first-failure-physical-causal-experiment",
        "publication": "native-reactive-baseline-only",
        "intervention_gate": "rising-edge",
        "shadow_state": str(args.shadow_state.resolve()),
        "shadow_state_sha256": _sha256(args.shadow_state),
        "shadow_validation_replay": str(args.shadow_validation_replay.resolve()),
        "shadow_validation_replay_sha256": _sha256(args.shadow_validation_replay),
        "validation_run_ids": sorted(validation_run_ids),
        "validation_runs": len(validation_run_ids),
        "policy_calls": calls,
        "candidates": candidates,
        "candidate_positive": positive,
        "candidate_negative": negative,
        "candidate_unlabeled": unlabeled,
        "precision": positive / labeled,
        "precision_lower_bound_95_one_sided": lower_bound,
        "activation_rate": candidates / calls,
        "not_a_promotion": True,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(_canonical(active) + b"\n")
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "mode": "active",
        "validation_runs": len(validation_run_ids),
        "candidate_positive": positive,
        "candidate_negative": negative,
        "precision_lower_bound_95_one_sided": lower_bound,
        "activation_rate": candidates / calls,
        "not_a_promotion": True,
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
