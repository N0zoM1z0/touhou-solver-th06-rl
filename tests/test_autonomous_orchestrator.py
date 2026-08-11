from __future__ import annotations

import pytest

from scripts.run_autonomous_learning import (
    _seed_schedule,
    _validate_retail_report,
    _verdict,
    parse_args,
)


def test_seed_schedule_is_deterministic_and_uses_unique_game_rng() -> None:
    first = _seed_schedule(6006, 12)
    assert first == _seed_schedule(6006, 12)
    assert len({row["game_rng_seed"] for row in first}) == 12
    assert all(0 <= row["policy_seed"] < 2**64 for row in first)


def test_retail_report_requires_cleanup_immutability_and_full_hit_accounting() -> None:
    report = {
        "error": None,
        "evaluation_mode": "hit-continuation-benchmark",
        "diagnostic_rng_seed": None,
        "immutable_policy_state_equal": True,
        "leftover_prefix_processes": [],
        "controller_returncode": 0,
        "controller_completion": {
            "practice_stage_completed": True,
            "practice_stage": 6,
            "physical_hits": 4,
        },
        "trace": {"physical_hits_in_run": 4},
    }
    _validate_retail_report(
        report,
        mode="hit-continuation-benchmark",
        diagnostic_rng_seed=None,
        full_stage=6,
    )
    report["trace"]["physical_hits_in_run"] = 3
    with pytest.raises(RuntimeError, match="accounting"):
        _validate_retail_report(
            report,
            mode="hit-continuation-benchmark",
            diagnostic_rng_seed=None,
            full_stage=6,
        )


def test_final_verdict_uses_only_strict_aggregate_hit_improvement() -> None:
    assert _verdict([
        {"arm": "baseline", "physical_hits": 5},
        {"arm": "candidate", "physical_hits": 4},
        {"arm": "baseline", "physical_hits": 6},
        {"arm": "candidate", "physical_hits": 5},
    ])["verdict"] == "effective"
    assert _verdict([
        {"arm": "baseline", "physical_hits": 5},
        {"arm": "candidate", "physical_hits": 5},
    ])["verdict"] == "ineffective"


def test_generation_defaults_predeclare_two_learning_rounds(tmp_path) -> None:
    args = parse_args(["--output-root", str(tmp_path / "generation")])
    assert args.collection_episodes == 10
    assert args.round_size == 5
    assert args.minimum_rounds_before_canary == 2
    assert args.canary_episodes == 2
    assert args.full_stage_pairs == 2
