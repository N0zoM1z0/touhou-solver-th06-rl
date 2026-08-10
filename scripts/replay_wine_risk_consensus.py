#!/usr/bin/env python3
"""Replay a shadow Wine risk consensus against strict physical prefixes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

from scripts.replay_wine_risk_guard import _context, _object
from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policies.offline_ranker import NATIVE_SCORER_ENV
from th06_rl.policies.offline_risk_consensus import OfflineRiskConsensusPolicy
from th06_rl.wine_risk import _stream_rows, load_first_failure_prefix


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--retail-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--scope", default="3/0/0/6")
    parser.add_argument(
        "--expect-recorded-actions",
        choices=("state", "incumbent"),
        default="state",
        help="whether the corpus was produced by this state or its incumbent",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain four integers")
    if not args.native_scorer.is_file():
        parser.error("native replay scorer is absent")

    source_state = _object(args.state)
    mode = source_state.get("mode")
    if mode not in ("shadow", "active"):
        raise SystemExit("risk-consensus replay state mode is invalid")
    incumbent_state = source_state.get("incumbent_state")
    native_contract = source_state.get("native_scorer")
    if not isinstance(incumbent_state, dict) or not isinstance(native_contract, dict):
        raise SystemExit("risk-consensus replay state contract is incomplete")
    runtime_state = json.loads(json.dumps(source_state))
    runtime_state["native_scorer"]["sha256"] = _sha256(args.native_scorer)
    os.environ[NATIVE_SCORER_ENV] = str(args.native_scorer.resolve())

    reports = []
    total_calls = total_mismatches = total_policy_mismatches = 0
    total_contract_violations = 0
    total_candidates = total_positive = total_negative = total_unlabeled = 0
    started = time.perf_counter()
    for run_dir in args.runs:
        prefix = load_first_failure_prefix(
            run_dir,
            expected_scope=scope,  # type: ignore[arg-type]
            expected_executable_sha256=args.retail_sha256,
            expected_native_kernel_sha256=args.native_sha256,
            expected_policy_id=None,
        )
        manifest = _object(run_dir / "manifest.json")
        decisions = {
            int(row["sequence"]): row["decision"]
            for row in _stream_rows(run_dir, manifest, "frames")
            if isinstance(row.get("decision"), dict)
        }
        examples = {
            example.transition.sequence: example for example in prefix.examples
        }
        incumbent = AdaptivePolicy()
        incumbent.import_state(incumbent_state)
        consensus = OfflineRiskConsensusPolicy()
        consensus.import_state(runtime_state)
        calls = 0
        mismatches = []
        policy_mismatches = []
        contract_violations = []
        candidates = []
        for row in _stream_rows(run_dir, manifest, "transitions"):
            sequence = int(row.get("sequence", -1))
            decision = decisions.get(sequence)
            if not isinstance(decision, dict) or decision.get("reason") != "ok":
                continue
            context = _context(row, decision)
            direct = incumbent.decide(context)
            shadow = consensus.decide(context)
            recorded = row.get("proposed_action")
            calls += 1
            if direct.action != recorded and len(mismatches) < 20:
                mismatches.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "recorded": recorded,
                    "incumbent": direct.action,
                })
            candidate = shadow.policy_id.endswith("-candidate")
            expected_action = (
                context.baseline_action
                if mode == "active" and candidate else direct.action
            )
            if shadow.action != expected_action and len(contract_violations) < 20:
                contract_violations.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "incumbent": direct.action,
                    "expected": expected_action,
                    "consensus": shadow.action,
                })
            expected_recorded = (
                shadow.action
                if args.expect_recorded_actions == "state" else direct.action
            )
            if expected_recorded != recorded and len(policy_mismatches) < 20:
                policy_mismatches.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "expected": expected_recorded,
                    "recorded": recorded,
                })
            if candidate:
                example = examples.get(sequence)
                candidates.append({
                    "sequence": sequence,
                    "frame": context.frame,
                    "source_context": context.source_context,
                    "incumbent_action": direct.action,
                    "baseline_action": context.baseline_action,
                    "label": (
                        None if example is None else example.failure_within_120
                    ),
                    "frames_to_failure": (
                        None if example is None else example.frames_to_failure
                    ),
                })
        positive = sum(row["label"] is True for row in candidates)
        negative = sum(row["label"] is False for row in candidates)
        unlabeled = sum(row["label"] is None for row in candidates)
        metrics = consensus.metrics()
        total_calls += calls
        total_mismatches += len(mismatches)
        total_policy_mismatches += len(policy_mismatches)
        total_contract_violations += len(contract_violations)
        total_candidates += len(candidates)
        total_positive += positive
        total_negative += negative
        total_unlabeled += unlabeled
        reports.append({
            "run_id": prefix.run_id,
            "manifest_sha256": prefix.manifest_sha256,
            "run_sha256": prefix.run_sha256,
            "policy_calls": calls,
            "recorded_incumbent_mismatches": mismatches,
            "recorded_policy_mismatches": policy_mismatches,
            "shadow_action_contract_violations": contract_violations,
            "candidates": len(candidates),
            "candidate_positive": positive,
            "candidate_negative": negative,
            "candidate_unlabeled": unlabeled,
            "candidate_examples": candidates[:40],
            "consensus_metrics": metrics,
        })

    passed = (
        total_policy_mismatches == 0
        and total_contract_violations == 0
        and all(
            set(run["consensus_metrics"]["scorer_backends"])
            == {"native-batch"}
            for run in reports
        )
    )
    report = {
        "schema": "th06-rl-wine-risk-consensus-replay-v1",
        "state": str(args.state.resolve()),
        "state_sha256": _sha256(args.state),
        "production_native_scorer_sha256": native_contract.get("sha256"),
        "replay_native_scorer": str(args.native_scorer.resolve()),
        "replay_native_scorer_sha256": _sha256(args.native_scorer),
        "scope": list(scope),
        "mode": mode,
        "expect_recorded_actions": args.expect_recorded_actions,
        "runs": reports,
        "totals": {
            "runs": len(reports),
            "policy_calls": total_calls,
            "recorded_incumbent_mismatches": total_mismatches,
            "recorded_policy_mismatches": total_policy_mismatches,
            "shadow_action_contract_violations": total_contract_violations,
            "candidates": total_candidates,
            "candidate_positive": total_positive,
            "candidate_negative": total_negative,
            "candidate_unlabeled": total_unlabeled,
            "replay_seconds": time.perf_counter() - started,
        },
        "passed": passed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "output_sha256": _sha256(args.output),
        "passed": passed,
        "totals": report["totals"],
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
