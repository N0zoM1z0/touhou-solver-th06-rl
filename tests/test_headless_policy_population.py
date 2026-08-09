from __future__ import annotations

import hashlib
import gzip
import json
from pathlib import Path

from scripts.build_headless_policy_population import build_population


SCOPE = {"difficulty": 3, "character": 0, "shot_type": 0, "stage": 3}


def _candidate(root: Path, name: str, payload: bytes, accuracy: float) -> str:
    directory = root / name
    directory.mkdir(parents=True)
    model = directory / "teacher-ranker.joblib"
    model.write_bytes(payload)
    (directory / "report.json").write_text(json.dumps({
        "schema": "test-ranker",
        "algorithm": "test",
        "scope": SCOPE,
        "holdout": {
            "acceptable_top1_accuracy": accuracy,
            "native_legal_action_ratio": 1.0,
            "bomb_actions": 0,
            "selected_action_counts": {"left": 2, "right": 2},
        },
    }), encoding="utf-8")
    return hashlib.sha256(payload).hexdigest()


def _rollout(
    root: Path,
    name: str,
    sha: str,
    ticks: int,
    *,
    continuation: bool,
    physical_hits: int | None = None,
    nmnb_stage_clear: bool = False,
    forced_actions: int = 0,
) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "ranker": {"sha256": sha},
        "scope": SCOPE,
        "initial_seed": ticks,
        "transition_count": ticks,
        "termination_reason": "stage-clear-success" if nmnb_stage_clear else "tick-limit",
        "continue_after_hit": continuation,
        "physical_hits": (
            physical_hits if physical_hits is not None else 1 if continuation else 0
        ),
        "physical_hit_ticks": [ticks // 2] if physical_hits else [],
        "authority_failure_events": forced_actions,
        "benchmark_forced_actions": forced_actions,
        "nmnb_stage_clear": nmnb_stage_clear,
    }), encoding="utf-8")


def test_population_keeps_evidence_tiers_and_does_not_use_offline_score_to_promote(
    tmp_path: Path,
) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    stronger_sha = _candidate(models, "stronger", b"stronger", 0.4)
    weaker_sha = _candidate(models, "weaker", b"weaker", 0.99)
    _rollout(rollouts, "stronger", stronger_sha, 3000, continuation=False)
    _rollout(rollouts, "weaker", weaker_sha, 1000, continuation=False)

    result = build_population([models], [rollouts])

    assert result["candidate_count"] == 2
    assert result["research_population"] == [stronger_sha]
    assert result["high_quality_population"] == []
    assert result["continuation_evaluation_queue"] == [stronger_sha]
    candidates = {row["model_sha256"]: row for row in result["candidates"]}
    assert candidates[weaker_sha]["offline_primary_metric"]["value"] == 0.99
    assert candidates[weaker_sha]["pareto_member"] is False


def test_hit_continuations_are_evidence_but_not_high_quality(tmp_path: Path) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    sha = _candidate(models, "candidate", b"candidate", 0.8)
    _rollout(rollouts, "seed1", sha, 2000, continuation=True)
    _rollout(rollouts, "seed2", sha, 3000, continuation=True)

    result = build_population([models], [rollouts])

    assert result["high_quality_population"] == []
    candidate = result["candidates"][0]
    assert candidate["closed_loop"]["continuation_runs"] == 2
    assert candidate["closed_loop"]["continuation_hits"] == 2
    assert candidate["closed_loop"]["continuation_hits_per_1000_ticks"] == 0.4


def test_high_quality_requires_multi_seed_natural_nmnb_stage_clears(tmp_path: Path) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    sha = _candidate(models, "candidate", b"candidate", 0.8)
    _rollout(
        rollouts,
        "seed1",
        sha,
        2000,
        continuation=True,
        physical_hits=0,
        nmnb_stage_clear=True,
    )
    _rollout(
        rollouts,
        "seed2",
        sha,
        3000,
        continuation=True,
        physical_hits=0,
        nmnb_stage_clear=True,
    )

    result = build_population([models], [rollouts])

    assert result["high_quality_population"] == [sha]
    metrics = result["candidates"][0]["closed_loop"]
    assert metrics["continuation_seeds"] == [2000, 3000]
    assert metrics["continuation_complete_runs"] == 2
    assert metrics["continuation_natural_stage_clears"] == 2
    assert metrics["continuation_nmnb_stage_clears"] == 2


def test_manifest_nmnb_flag_with_forced_release_is_not_high_quality(tmp_path: Path) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    sha = _candidate(models, "candidate", b"candidate", 0.8)
    for index, ticks in enumerate((2000, 3000), 1):
        _rollout(
            rollouts,
            f"seed{index}",
            sha,
            ticks,
            continuation=True,
            physical_hits=0,
            nmnb_stage_clear=True,
            forced_actions=1,
        )

    result = build_population([models], [rollouts])

    assert result["high_quality_population"] == []
    assert result["candidates"][0]["closed_loop"]["continuation_nmnb_stage_clears"] == 0


def test_active_population_requires_exact_runtime_source(tmp_path: Path) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    sha = _candidate(models, "candidate", b"candidate", 0.8)
    report = models / "candidate" / "report.json"
    raw = json.loads(report.read_text(encoding="utf-8"))
    raw["compatible_headless_sources"] = [{
        "commit": "old", "binary_sha256": "old-bin", "clean": True,
    }]
    report.write_text(json.dumps(raw), encoding="utf-8")
    _rollout(rollouts, "seed1", sha, 2000, continuation=False)

    result = build_population([models], [rollouts], runtime_source={
        "commit": "current", "binary_sha256": "current-bin", "clean": True,
    })

    assert result["historical_pareto_population"] == [sha]
    assert result["research_population"] == []
    assert result["continuation_evaluation_queue"] == []
    assert result["candidates"][0]["runtime_compatible"] is False


def test_interrupted_continuation_is_linked_by_run_intent(tmp_path: Path) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    sha = _candidate(models, "candidate", b"candidate", 0.8)
    run = rollouts / "partial"
    run.mkdir(parents=True)
    (run / "run-intent.json").write_text(json.dumps({
        "scope": SCOPE,
        "initial_seed": 23,
        "continue_after_hit": True,
        "ranker": {"sha256": sha},
    }), encoding="utf-8")
    row = {
        "sequence": 0,
        "tick": 1,
        "next_tick": 2,
        "scope": SCOPE,
        "source_context": "timeline:1",
        "behavior": {"policy": "ranker", "selected_action": "stay_fast"},
        "benchmark_forced_action": False,
        "outcome_terms": {"deaths_delta": 1, "bombs_used_delta": 0},
    }
    (run / "transitions.jsonl.gz.partial").write_bytes(
        gzip.compress(json.dumps(row).encode() + b"\n")[:-8]
    )

    result = build_population([models], [rollouts])

    candidate = result["candidates"][0]
    assert candidate["evidence_tier"] == "continuation-evidenced"
    assert candidate["closed_loop"]["continuation_runs"] == 1
    assert candidate["closed_loop"]["continuation_hits"] == 1
    assert candidate["closed_loop"]["runs_detail"][0]["status"] == "interrupted-partial"
    assert result["high_quality_population"] == []
    assert result["continuation_evaluation_queue"] == [sha]
