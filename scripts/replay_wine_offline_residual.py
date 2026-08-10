#!/usr/bin/env python3
"""Audit an offline-ranker proposal above frozen UCB on factual Wine prefixes.

This script never publishes the ranker's action.  It reconstructs the frozen
incumbent, asks the existing active-form ranker what it *would* have selected,
and measures proposal enrichment against factual terminal-window labels.  The
labels say only that the incumbent was near a later failure; they do not prove
that the counterfactual ranker action would have prevented that failure.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

from scripts.replay_wine_risk_guard import _context, _object
from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policies.offline_ranker import (
    NATIVE_SCORER_ENV,
    OfflineRankerPolicy,
)
from th06_rl.policies.offline_risk_guard import OfflineRiskGuardPolicy
from th06_rl.wine_risk import (
    _stream_rows,
    load_first_failure_prefix,
)


SCHEMA = "th06-rl-wine-offline-residual-replay-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_counts(rows: list[dict[str, object]]) -> dict[str, int]:
    return {
        "candidates": len(rows),
        "candidate_positive": sum(row["label"] is True for row in rows),
        "candidate_negative": sum(row["label"] is False for row in rows),
        "candidate_unlabeled": sum(row["label"] is None for row in rows),
    }


def _append_limited(
    rows: list[dict[str, object]], row: dict[str, object], *, limit: int = 40,
) -> None:
    if len(rows) < limit:
        rows.append(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--incumbent-state", type=Path, required=True)
    parser.add_argument("--ranker-state", type=Path, required=True)
    parser.add_argument("--risk-state", type=Path, required=True)
    parser.add_argument("--native-scorer", type=Path, required=True)
    parser.add_argument("--retail-sha256", required=True)
    parser.add_argument("--native-sha256", required=True)
    parser.add_argument("--scope", default="3/0/0/6")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    scope = tuple(int(value) for value in args.scope.split("/"))
    if len(scope) != 4:
        parser.error("scope must contain four integers")
    if not args.native_scorer.is_file():
        parser.error("native replay scorer is absent")

    incumbent_state = _object(args.incumbent_state)
    source_ranker_state = _object(args.ranker_state)
    source_risk_state = _object(args.risk_state)
    if source_ranker_state.get("mode") != "active":
        raise SystemExit("residual audit requires an active-form ranker proposal")
    if source_risk_state.get("mode") != "shadow":
        raise SystemExit("residual audit requires a shadow risk state")
    embedded_incumbent = source_risk_state.get("incumbent_state")
    if embedded_incumbent != incumbent_state:
        raise SystemExit("risk state does not embed the exact audited incumbent")
    if tuple(source_ranker_state.get("scope", ())) != scope:
        raise SystemExit("ranker state scope does not match the replay scope")
    if tuple(source_risk_state.get("scope", ())) != scope:
        raise SystemExit("risk state scope does not match the replay scope")

    scorer_sha = _sha256(args.native_scorer)
    ranker_state = json.loads(json.dumps(source_ranker_state))
    risk_state = json.loads(json.dumps(source_risk_state))
    for state in (ranker_state, risk_state):
        native_contract = state.get("native_scorer")
        if not isinstance(native_contract, dict):
            raise SystemExit("proposal state has no production scorer contract")
        native_contract["sha256"] = scorer_sha
    os.environ[NATIVE_SCORER_ENV] = str(args.native_scorer.resolve())

    run_reports = []
    total_calls = 0
    total_recorded_mismatches = 0
    total_action_contract_violations = 0
    total_ranker_supported = 0
    total_ranker_residual = 0
    total_risk = 0
    total_joint = 0
    total_ranker_positive = total_ranker_negative = total_ranker_unlabeled = 0
    total_risk_positive = total_risk_negative = total_risk_unlabeled = 0
    total_joint_positive = total_joint_negative = total_joint_unlabeled = 0
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
        ranker = OfflineRankerPolicy()
        ranker.import_state(ranker_state)
        risk = OfflineRiskGuardPolicy()
        risk.import_state(risk_state)

        calls = 0
        recorded_mismatches = []
        action_contract_violations = []
        recorded_mismatch_count = 0
        action_contract_violation_count = 0
        ranker_supported = 0
        ranker_rows: list[dict[str, object]] = []
        risk_rows: list[dict[str, object]] = []
        joint_rows: list[dict[str, object]] = []
        ranker_examples: list[dict[str, object]] = []
        joint_examples: list[dict[str, object]] = []
        residual_guard_eligible_sequences: list[int] = []

        for row in _stream_rows(run_dir, manifest, "transitions"):
            sequence = int(row.get("sequence", -1))
            decision = decisions.get(sequence)
            if not isinstance(decision, dict) or decision.get("reason") != "ok":
                continue
            context = _context(row, decision)
            incumbent_decision = incumbent.decide(context)
            ranker_decision = ranker.decide(context)
            recorded = row.get("proposed_action")
            calls += 1

            if incumbent_decision.action != recorded:
                recorded_mismatch_count += 1
                _append_limited(recorded_mismatches, {
                    "sequence": sequence,
                    "frame": context.frame,
                    "recorded": recorded,
                    "incumbent": incumbent_decision.action,
                }, limit=20)
            if ranker_decision.action not in context.locally_admissible_actions:
                action_contract_violation_count += 1
                _append_limited(action_contract_violations, {
                    "sequence": sequence,
                    "frame": context.frame,
                    "ranker_action": ranker_decision.action,
                    "locally_admissible_actions": list(
                        context.locally_admissible_actions
                    ),
                }, limit=20)

            example = examples.get(sequence)
            label = None if example is None else example.failure_within_120
            frames_to_failure = (
                None if example is None else example.frames_to_failure
            )
            risk_score = (
                risk.score_action(context, incumbent_decision.action)
                if incumbent_decision.action != context.baseline_action
                else None
            )
            risk_candidate = (
                risk_score is not None and risk_score >= risk.threshold
            )
            supported_ranker_override = (
                ranker_decision.action != context.baseline_action
            )
            residual_proposal = (
                supported_ranker_override
                and ranker_decision.action != incumbent_decision.action
            )
            joint_candidate = residual_proposal and risk_candidate
            ranker_supported += int(supported_ranker_override)

            evidence = {
                "run_id": prefix.run_id,
                "sequence": sequence,
                "frame": context.frame,
                "source_context": context.source_context,
                "incumbent_action": incumbent_decision.action,
                "baseline_action": context.baseline_action,
                "ranker_action": ranker_decision.action,
                "risk_score": risk_score,
                "label": label,
                "frames_to_failure": frames_to_failure,
            }
            if residual_proposal:
                ranker_rows.append(evidence)
                _append_limited(ranker_examples, evidence)
                # The residual trainer accepts only the same exact-context,
                # learning-eligible factual examples used by
                # load_first_failure_prefix().  A decision can still be
                # replayable (and therefore useful to the broad audit) while
                # being excluded from factual risk training.
                if (
                    risk_score is not None
                    and example is not None
                    and example.fallback_opportunity
                ):
                    residual_guard_eligible_sequences.append(sequence)
            if risk_candidate:
                risk_rows.append(evidence)
            if joint_candidate:
                joint_rows.append(evidence)
                _append_limited(joint_examples, evidence)

        ranker_counts = _label_counts(ranker_rows)
        risk_counts = _label_counts(risk_rows)
        joint_counts = _label_counts(joint_rows)
        total_calls += calls
        total_recorded_mismatches += recorded_mismatch_count
        total_action_contract_violations += action_contract_violation_count
        total_ranker_supported += ranker_supported
        total_ranker_residual += ranker_counts["candidates"]
        total_risk += risk_counts["candidates"]
        total_joint += joint_counts["candidates"]
        total_ranker_positive += ranker_counts["candidate_positive"]
        total_ranker_negative += ranker_counts["candidate_negative"]
        total_ranker_unlabeled += ranker_counts["candidate_unlabeled"]
        total_risk_positive += risk_counts["candidate_positive"]
        total_risk_negative += risk_counts["candidate_negative"]
        total_risk_unlabeled += risk_counts["candidate_unlabeled"]
        total_joint_positive += joint_counts["candidate_positive"]
        total_joint_negative += joint_counts["candidate_negative"]
        total_joint_unlabeled += joint_counts["candidate_unlabeled"]
        run_reports.append({
            "run_id": prefix.run_id,
            "manifest_sha256": prefix.manifest_sha256,
            "run_sha256": prefix.run_sha256,
            "policy_calls": calls,
            "recorded_incumbent_mismatch_count": recorded_mismatch_count,
            "recorded_incumbent_mismatches": recorded_mismatches,
            "action_contract_violation_count": action_contract_violation_count,
            "action_contract_violations": action_contract_violations,
            "ranker_supported_overrides": ranker_supported,
            "ranker_residual": ranker_counts,
            "residual_guard_eligible": len(residual_guard_eligible_sequences),
            "residual_guard_eligible_sequences": (
                residual_guard_eligible_sequences
            ),
            "risk_only": risk_counts,
            "joint": joint_counts,
            "ranker_examples": ranker_examples,
            "joint_examples": joint_examples,
            "ranker_metrics": ranker.metrics(),
            "risk_metrics": risk.metrics(),
        })

    totals = {
        "runs": len(run_reports),
        "policy_calls": total_calls,
        "recorded_incumbent_mismatches": total_recorded_mismatches,
        "action_contract_violations": total_action_contract_violations,
        "ranker_supported_overrides": total_ranker_supported,
        "ranker_residual": {
            "candidates": total_ranker_residual,
            "candidate_positive": total_ranker_positive,
            "candidate_negative": total_ranker_negative,
            "candidate_unlabeled": total_ranker_unlabeled,
        },
        "risk_only": {
            "candidates": total_risk,
            "candidate_positive": total_risk_positive,
            "candidate_negative": total_risk_negative,
            "candidate_unlabeled": total_risk_unlabeled,
        },
        "joint": {
            "candidates": total_joint,
            "candidate_positive": total_joint_positive,
            "candidate_negative": total_joint_negative,
            "candidate_unlabeled": total_joint_unlabeled,
        },
        "ranker_residual_activation_rate": (
            total_ranker_residual / total_calls if total_calls else math.nan
        ),
        "risk_activation_rate": total_risk / total_calls if total_calls else math.nan,
        "joint_activation_rate": (
            total_joint / total_calls if total_calls else math.nan
        ),
        "replay_seconds": time.perf_counter() - started,
    }
    passed = (
        total_recorded_mismatches == 0
        and total_action_contract_violations == 0
        and all(
            run["ranker_metrics"]["scorer_backend"] == "native-batch"
            and run["risk_metrics"]["scorer_backend"] == "native-batch"
            for run in run_reports
        )
    )
    report = {
        "schema": SCHEMA,
        "scope": list(scope),
        "incumbent_state": str(args.incumbent_state.resolve()),
        "incumbent_state_sha256": _sha256(args.incumbent_state),
        "ranker_state": str(args.ranker_state.resolve()),
        "ranker_state_sha256": _sha256(args.ranker_state),
        "risk_state": str(args.risk_state.resolve()),
        "risk_state_sha256": _sha256(args.risk_state),
        "replay_native_scorer": str(args.native_scorer.resolve()),
        "replay_native_scorer_sha256": scorer_sha,
        "semantics": {
            "published_action": "none-offline-replay-only",
            "recorded_action_authority": "frozen-incumbent",
            "ranker_action": "counterfactual-proposal-not-causal-evidence",
            "label": "factual-incumbent-terminal-window-only",
            "residual_guard_eligibility": (
                "exact-factual-risk-example-and-fallback-opportunity"
            ),
        },
        "runs": run_reports,
        "totals": totals,
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
        "totals": totals,
    }, indent=2, sort_keys=True))
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
