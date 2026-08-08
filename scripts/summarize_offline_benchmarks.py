#!/usr/bin/env python3
"""Consolidate corpus, native, recorded-frame, and learning benchmarks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _learning_summary(path: Path, expected_revision: str) -> dict[str, object]:
    manifest = _object(path)
    dataset = manifest.get("dataset")
    if not isinstance(dataset, dict) or dataset.get("revision") != expected_revision:
        raise ValueError(f"policy manifest revision mismatch: {path}")
    raw_results = manifest.get("results")
    if not isinstance(raw_results, dict) or not raw_results:
        raise ValueError(f"policy manifest has no model results: {path}")
    rows = []
    boundary_passes = True
    for algorithm, raw in sorted(raw_results.items()):
        if not isinstance(raw, dict):
            raise TypeError(f"invalid model result in {path}: {algorithm}")
        factual = raw.get("factual_validation")
        hit = raw.get("hit_120_ranking")
        policy = raw.get("policy_evaluation")
        if not all(isinstance(value, dict) for value in (factual, hit, policy)):
            raise TypeError(f"incomplete model result in {path}: {algorithm}")
        assert isinstance(factual, dict) and isinstance(hit, dict) and isinstance(policy, dict)
        variants = policy.get("policy_variants")
        if not isinstance(variants, dict):
            raise TypeError(f"missing policy variants in {path}: {algorithm}")
        cautious = variants.get("support32_margin_0_5")
        baseline = policy.get("reactive_baseline")
        if not isinstance(cautious, dict) or not isinstance(baseline, dict):
            raise TypeError(f"missing benchmark policy in {path}: {algorithm}")
        safe = (
            policy.get("any_variant_selected_outside_native_safe_set") == 0
            and policy.get("bomb_selections_across_variants") == 0
        )
        boundary_passes = boundary_passes and safe
        rows.append({
            "algorithm": algorithm,
            "mae": factual.get("mae"),
            "rmse": factual.get("rmse"),
            "hit_120_average_precision": hit.get("average_precision_from_negative_q"),
            "hit_120_roc_auc": hit.get("roc_auc_from_negative_q"),
            "reactive_baseline": {
                name: baseline.get(name) for name in ("clipped_dr", "clipped_wis", "clipped_ess")
            },
            "support32_margin_0_5": {
                name: cautious.get(name)
                for name in (
                    "clipped_dr",
                    "clipped_wis",
                    "clipped_ess",
                    "vs_baseline_disagreements",
                )
            },
            "action_boundary_passes": safe,
            "peak_rss_mib": raw.get("process_peak_rss_mib"),
        })
    split = manifest.get("split")
    if not isinstance(split, dict):
        raise TypeError(f"missing split in {path}")
    return {
        "manifest": str(path),
        "scope": manifest.get("scope"),
        "view": manifest.get("view"),
        "split": split,
        "models": rows,
        "best_factual_mae": min(rows, key=lambda row: float(row["mae"])),
        "best_hit_120_roc_auc": max(
            rows,
            key=lambda row: float(row["hit_120_roc_auc"] or float("-inf")),
        ),
        "all_action_boundaries_pass": boundary_passes,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--recorded", type=Path, required=True)
    parser.add_argument("--policy", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    audit = _object(args.audit)
    native = _object(args.native)
    recorded = _object(args.recorded)
    revision = str(audit.get("dataset_revision", ""))
    if not revision or recorded.get("dataset_revision") != revision:
        raise ValueError("corpus audit and recorded-frame revisions differ")
    overall = audit.get("overall")
    scopes = audit.get("scopes")
    if not isinstance(overall, dict) or not isinstance(scopes, dict):
        raise TypeError("corpus audit is incomplete")
    learning = [_learning_summary(path, revision) for path in args.policy]
    result = {
        "schema": "th06-rl-offline-benchmark-suite-v1",
        "dataset_revision": revision,
        "corpus": {"overall": overall, "scopes": scopes},
        "geometry_and_planning": native,
        "recorded_decision_path": recorded,
        "learning": learning,
        "passes": {
            "native": native.get("passes") is True,
            "recorded_frame_coherence": recorded.get("passes") is True,
            "learned_action_boundary": all(
                row["all_action_boundaries_pass"] for row in learning
            ),
        },
    }
    payload = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return 0 if all(result["passes"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
