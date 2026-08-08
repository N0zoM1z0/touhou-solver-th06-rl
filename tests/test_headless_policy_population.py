from __future__ import annotations

import hashlib
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


def _rollout(root: Path, name: str, sha: str, ticks: int, *, continuation: bool) -> None:
    directory = root / name
    directory.mkdir(parents=True)
    (directory / "manifest.json").write_text(json.dumps({
        "ranker": {"sha256": sha},
        "scope": SCOPE,
        "initial_seed": ticks,
        "transition_count": ticks,
        "termination_reason": "tick-limit",
        "continue_after_hit": continuation,
        "physical_hits": 1 if continuation else 0,
        "physical_hit_ticks": [ticks // 2] if continuation else [],
        "nmnb_stage_clear": False,
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


def test_high_quality_population_requires_multiple_continuation_runs(tmp_path: Path) -> None:
    models = tmp_path / "models"
    rollouts = tmp_path / "rollouts"
    sha = _candidate(models, "candidate", b"candidate", 0.8)
    _rollout(rollouts, "seed1", sha, 2000, continuation=True)
    _rollout(rollouts, "seed2", sha, 3000, continuation=True)

    result = build_population([models], [rollouts])

    assert result["high_quality_population"] == [sha]
    candidate = result["candidates"][0]
    assert candidate["closed_loop"]["continuation_runs"] == 2
    assert candidate["closed_loop"]["continuation_hits"] == 2
    assert candidate["closed_loop"]["continuation_hits_per_1000_ticks"] == 0.4


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
