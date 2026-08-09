#!/usr/bin/env python3
"""Build a model-linked, evidence-tiered Pareto population for headless TH06.

The archive never promotes one offline winner.  It links immutable model hashes
to offline reports and closed-loop manifests, keeps non-dominated candidates
per exact learning scope, and explicitly distinguishes first-failure evidence
from HIT-continuation evidence.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping

try:
    from summarize_headless_continuation import summarize_transition_file
except ModuleNotFoundError:  # Imported as scripts.build_headless_policy_population in tests.
    from scripts.summarize_headless_continuation import summarize_transition_file


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_source(binary: Path) -> dict[str, Any]:
    source = binary.resolve().parent
    commit = subprocess.run(
        ["git", "-C", str(source), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "-C", str(source), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {"commit": commit, "clean": not dirty, "binary_sha256": _sha256(binary)}


def _source_key(source: Mapping[str, Any]) -> tuple[str, str]:
    return str(source.get("commit", "")), str(source.get("binary_sha256", ""))


def _compatible_sources(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    raw = report.get("compatible_headless_sources")
    if not isinstance(raw, list):
        single = report.get("headless_source") or report.get("factual_source")
        raw = [single] if isinstance(single, Mapping) else []
    return [dict(source) for source in raw if isinstance(source, Mapping)]


def _scope_key(scope: Mapping[str, Any]) -> tuple[int, int, int, int]:
    return tuple(int(scope[name]) for name in ("difficulty", "character", "shot_type", "stage"))


def _offline_primary(holdout: Mapping[str, Any]) -> tuple[str | None, float | None]:
    for name in (
        "acceptable_top1_accuracy",
        "counterfactual_best_top1_accuracy",
        "teacher_top1_accuracy",
        "generic_teacher_top1_accuracy",
    ):
        value = holdout.get(name)
        if isinstance(value, (int, float)):
            return name, float(value)
    return None, None


def _action_entropy(holdout: Mapping[str, Any]) -> float | None:
    counts = holdout.get("selected_action_counts")
    if not isinstance(counts, Mapping):
        return None
    values = [int(value) for value in counts.values() if int(value) > 0]
    total = sum(values)
    if total <= 0 or len(values) <= 1:
        return 0.0
    entropy = -sum((value / total) * math.log(value / total) for value in values)
    return entropy / math.log(len(values))


def _manifest_run(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    hits_value = manifest.get("physical_hits")
    hits = (
        int(hits_value)
        if isinstance(hits_value, (int, float))
        else int(manifest.get("physical_hit") is True)
    )
    ticks = int(manifest.get("transition_count", 0))
    termination = str(manifest.get("termination_reason", "unknown"))
    authority_events = int(manifest.get("authority_failure_events", 0))
    forced_actions = int(manifest.get("benchmark_forced_actions", 0))
    strict_nmnb_clear = bool(
        manifest.get("nmnb_stage_clear") is True
        and termination in {"chain-exit-success", "stage-clear-success"}
        and hits == 0
        and authority_events == 0
        and forced_actions == 0
    )
    return {
        "manifest": str(path),
        "status": "complete",
        "seed": manifest.get("initial_seed"),
        "ticks": ticks,
        "termination_reason": termination,
        "physical_hits": hits,
        "physical_hit_ticks": manifest.get("physical_hit_ticks", []),
        "continue_after_hit": manifest.get("continue_after_hit") is True,
        "authority_failure": termination == "authority-failure",
        "authority_failure_events": authority_events,
        "benchmark_forced_actions": forced_actions,
        "nmnb_stage_clear": strict_nmnb_clear,
    }


def _partial_run(path: Path, summary: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifest": str(path),
        "status": "interrupted-partial",
        "seed": summary.get("seed"),
        "ticks": int(summary.get("observed_ticks", 0)),
        "termination_reason": "interrupted-partial",
        "physical_hits": int(summary.get("physical_hits", 0)),
        "physical_hit_ticks": summary.get("physical_hit_ticks", []),
        "continue_after_hit": summary.get("continue_after_hit") is True,
        "authority_failure": False,
        "authority_failure_events": int(summary.get("benchmark_forced_rows", 0)),
        "benchmark_forced_actions": int(summary.get("benchmark_forced_rows", 0)),
        "nmnb_stage_clear": False,
    }


def _closed_loop_metrics(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ticks = sum(run["ticks"] for run in runs)
    forced_actions = sum(run["benchmark_forced_actions"] for run in runs)
    continuation = [run for run in runs if run["continue_after_hit"]]
    continuation_ticks = sum(run["ticks"] for run in continuation)
    continuation_hits = sum(run["physical_hits"] for run in continuation)
    continuation_forced_actions = sum(
        run["benchmark_forced_actions"] for run in continuation
    )
    continuation_seeds = sorted({
        run["seed"] for run in continuation if isinstance(run["seed"], int)
    })
    complete_continuation = [run for run in continuation if run["status"] == "complete"]
    natural_stage_clears = [
        run for run in complete_continuation
        if run["termination_reason"] in {"chain-exit-success", "stage-clear-success"}
    ]
    nmnb_stage_clears = [
        run for run in natural_stage_clears
        if run["physical_hits"] == 0 and run["nmnb_stage_clear"]
    ]
    return {
        "runs": len(runs),
        "seeds": sorted({run["seed"] for run in runs if isinstance(run["seed"], int)}),
        "total_ticks": ticks,
        "benchmark_forced_actions": forced_actions,
        "benchmark_forced_actions_per_1000_ticks": (
            forced_actions * 1000.0 / ticks if ticks else None
        ),
        "minimum_ticks_before_stop": min((run["ticks"] for run in runs), default=0),
        "mean_ticks_before_stop": ticks / len(runs) if runs else 0.0,
        "authority_failure_rate": (
            sum(run["authority_failure"] for run in runs) / len(runs) if runs else None
        ),
        "tick_limit_rate": (
            sum(run["termination_reason"] == "tick-limit" for run in runs) / len(runs)
            if runs else None
        ),
        "nmnb_stage_clears": sum(run["nmnb_stage_clear"] for run in runs),
        "continuation_runs": len(continuation),
        "continuation_complete_runs": len(complete_continuation),
        "continuation_seeds": continuation_seeds,
        "continuation_natural_stage_clears": len(natural_stage_clears),
        "continuation_nmnb_stage_clears": len(nmnb_stage_clears),
        "continuation_ticks": continuation_ticks,
        "continuation_hits": continuation_hits,
        "continuation_hits_per_1000_ticks": (
            continuation_hits * 1000.0 / continuation_ticks if continuation_ticks else None
        ),
        "continuation_forced_actions": continuation_forced_actions,
        "continuation_forced_actions_per_1000_ticks": (
            continuation_forced_actions * 1000.0 / continuation_ticks
            if continuation_ticks else None
        ),
        "hit_rate_evidence_complete": bool(continuation),
        "runs_detail": runs,
    }


def _dominates(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    """Compare only commensurate closed-loop evidence, never offline objectives."""

    left_metrics = left["closed_loop"]
    right_metrics = right["closed_loop"]
    if left_metrics["hit_rate_evidence_complete"] != right_metrics["hit_rate_evidence_complete"]:
        return False
    maximize = ("minimum_ticks_before_stop", "tick_limit_rate")
    minimize = ("authority_failure_rate",)
    if left_metrics["hit_rate_evidence_complete"]:
        minimize += (
            "continuation_hits_per_1000_ticks",
            "continuation_forced_actions_per_1000_ticks",
        )
    weak = all(left_metrics[name] >= right_metrics[name] for name in maximize) and all(
        left_metrics[name] <= right_metrics[name] for name in minimize
    )
    strict = any(left_metrics[name] > right_metrics[name] for name in maximize) or any(
        left_metrics[name] < right_metrics[name] for name in minimize
    )
    return weak and strict


def build_population(
    model_roots: Iterable[Path],
    rollout_roots: Iterable[Path],
    *,
    runtime_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_files = sorted({
        path.resolve()
        for root in model_roots
        for path in (root.rglob("*.joblib") if root.is_dir() else [root])
        if path.is_file() and path.suffix == ".joblib"
    })
    manifests_by_ranker: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    partials_by_ranker: dict[str, list[tuple[Path, Mapping[str, Any]]]] = defaultdict(list)
    for root in rollout_roots:
        paths = root.rglob("manifest.json") if root.is_dir() else [root]
        for path in paths:
            if not path.is_file():
                continue
            raw = json.loads(path.read_text(encoding="utf-8"))
            ranker = raw.get("ranker") if isinstance(raw, Mapping) else None
            sha = ranker.get("sha256") if isinstance(ranker, Mapping) else None
            if isinstance(sha, str):
                manifests_by_ranker[sha].append((path.resolve(), raw))
        partial_paths = root.rglob("transitions.jsonl.gz.partial") if root.is_dir() else []
        for path in partial_paths:
            summary = summarize_transition_file(path)
            sha = summary.get("ranker_sha256")
            if isinstance(sha, str):
                partials_by_ranker[sha].append((path.resolve(), summary))

    candidates: list[dict[str, Any]] = []
    for model in model_files:
        sha = _sha256(model)
        report_path = model.parent / "report.json"
        report: Mapping[str, Any] = {}
        if report_path.is_file():
            raw_report = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(raw_report, Mapping):
                report = raw_report
        scope = report.get("scope")
        if not isinstance(scope, Mapping) or not all(
            name in scope for name in ("difficulty", "character", "shot_type", "stage")
        ):
            continue
        holdout = report.get("holdout")
        holdout = holdout if isinstance(holdout, Mapping) else {}
        metric_name, metric_value = _offline_primary(holdout)
        runs = [
            _manifest_run(path, manifest)
            for path, manifest in manifests_by_ranker.get(sha, [])
            if isinstance(manifest.get("scope"), Mapping)
            and _scope_key(manifest["scope"]) == _scope_key(scope)
        ]
        runs.extend(
            _partial_run(path, summary)
            for path, summary in partials_by_ranker.get(sha, [])
            if isinstance(summary.get("scope"), Mapping)
            and _scope_key(summary["scope"]) == _scope_key(scope)
        )
        closed_loop = _closed_loop_metrics(runs)
        labels = report.get("counterfactual_labels")
        target = labels.get("target") if isinstance(labels, Mapping) else None
        compatible_sources = _compatible_sources(report)
        runtime_compatible = (
            None
            if runtime_source is None
            else runtime_source.get("clean") is True
            and any(
                source.get("clean") is True
                and _source_key(source) == _source_key(runtime_source)
                for source in compatible_sources
            )
        )
        evidence_tier = (
            "continuation-evidenced"
            if closed_loop["continuation_runs"] > 0
            else "first-failure-only"
            if closed_loop["runs"] > 0
            else "offline-only"
        )
        candidates.append({
            "model_sha256": sha,
            "model_path": str(model),
            "report_path": str(report_path) if report_path.is_file() else None,
            "scope": dict(scope),
            "algorithm": report.get("algorithm"),
            "objective_family": target or report.get("schema") or "unknown",
            "compatible_headless_sources": compatible_sources,
            "runtime_compatible": runtime_compatible,
            "offline_primary_metric": {"name": metric_name, "value": metric_value},
            "holdout_action_entropy": _action_entropy(holdout),
            "native_legal_action_ratio": holdout.get("native_legal_action_ratio"),
            "holdout_bomb_actions": holdout.get("bomb_actions"),
            "evidence_tier": evidence_tier,
            "closed_loop": closed_loop,
            "promotion_allowed": False,
            "promotion_blocker": (
                "requires per-stage multi-seed HIT-continuation evidence and paired Windows shadow/canary"
            ),
            "pareto_member": False,
        })

    groups: dict[
        tuple[tuple[int, int, int, int], str, tuple[tuple[str, str], ...]],
        list[dict[str, Any]],
    ] = defaultdict(list)
    for candidate in candidates:
        if candidate["closed_loop"]["runs"]:
            source_group = tuple(sorted(
                _source_key(source) for source in candidate["compatible_headless_sources"]
            ))
            groups[(
                _scope_key(candidate["scope"]),
                candidate["evidence_tier"],
                source_group,
            )].append(candidate)
    for group in groups.values():
        for candidate in group:
            candidate["pareto_member"] = not any(
                other is not candidate and _dominates(other, candidate) for other in group
            )

    historical = [
        candidate["model_sha256"]
        for candidate in candidates
        if candidate["pareto_member"]
    ]
    active = lambda candidate: runtime_source is None or candidate["runtime_compatible"] is True
    research = [
        candidate["model_sha256"]
        for candidate in candidates
        if candidate["pareto_member"]
        and candidate["evidence_tier"] == "first-failure-only"
        and active(candidate)
    ]
    high_quality = [
        candidate["model_sha256"]
        for candidate in candidates
        if candidate["pareto_member"]
        and candidate["evidence_tier"] == "continuation-evidenced"
        and candidate["closed_loop"]["continuation_runs"] >= 2
        and len(candidate["closed_loop"]["continuation_seeds"]) >= 2
        and candidate["closed_loop"]["continuation_complete_runs"]
        == candidate["closed_loop"]["continuation_runs"]
        and candidate["closed_loop"]["continuation_nmnb_stage_clears"]
        == candidate["closed_loop"]["continuation_runs"]
        and active(candidate)
    ]
    high_quality_set = set(high_quality)
    queue = [
        candidate["model_sha256"]
        for candidate in candidates
        if candidate["pareto_member"]
        and active(candidate)
        and candidate["model_sha256"] not in high_quality_set
    ]
    return {
        "schema": "th06-rl-headless-policy-population-v1",
        "selection_contract": {
            "unit": "exact difficulty-character-shot-stage scope",
            "authority": "models rank the native safe action set only",
            "pareto_axes": [
                "minimum_ticks_before_stop:max",
                "tick_limit_rate:max",
                "authority_failure_rate:min",
                "continuation_hits_per_1000_ticks:min when observed",
            ],
            "offline_metrics": "diagnostic and diversity metadata, never promotion evidence",
            "high_quality_gate": (
                "at least two unique-seed complete continuation runs, all natural NMNB stage clears"
            ),
            "windows_gate": "paired shadow/canary required before incumbent replacement",
        },
        "candidate_count": len(candidates),
        "active_runtime_source": dict(runtime_source) if runtime_source is not None else None,
        "historical_pareto_population": historical,
        "research_population": research,
        "high_quality_population": high_quality,
        "continuation_evaluation_queue": queue,
        "candidates": sorted(candidates, key=lambda item: (_scope_key(item["scope"]), item["model_sha256"])),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-root", type=Path, action="append", required=True)
    parser.add_argument("--rollouts-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--binary",
        type=Path,
        default=root / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"headless binary not found: {args.binary}")
    result = build_population(
        args.models_root,
        args.rollouts_root,
        runtime_source=_runtime_source(args.binary),
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
