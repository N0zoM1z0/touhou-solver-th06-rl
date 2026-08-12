from __future__ import annotations

import json

import pytest

from scripts.run_autonomous_learning_v3 import (
    CANARY_PAIRS,
    COLLECTION_EPISODES,
    FINAL_PAIRS,
    FIT_BOUNDARIES,
    GENERATION_SEED,
    _rate_ratio,
    _round_seeds,
    parse_args,
)


def test_generation3_cli_locks_evidence_contract(tmp_path) -> None:
    args = parse_args(["--output-root", str(tmp_path / "generation-3")])
    assert GENERATION_SEED == 260812
    assert COLLECTION_EPISODES == 24
    assert FIT_BOUNDARIES == (12, 16, 20, 24)
    assert CANARY_PAIRS == 3
    assert FINAL_PAIRS == 12
    with pytest.raises(SystemExit):
        parse_args([
            "--output-root", str(tmp_path / "generation-3"),
            "--collection-episodes", "4",
        ])
    assert args.threads == 12


def test_generation3_uses_exact_three_precommitted_canary_seeds() -> None:
    schedule = json.loads(
        (parse_args([]).output_root.parents[1]
         / "config/autonomous_generation3_seeds.json").read_text()
    )
    seeds = _round_seeds(schedule, 3)
    assert seeds == [22621, 23181, 17021]


def test_generation3_rate_ratio_reports_finite_approximate_interval() -> None:
    report = _rate_ratio(candidate=4, baseline=8)
    assert 0.0 < report["estimate"] < 1.0
    assert report["approximate_95_percent_lower"] < report["estimate"]
    assert report["approximate_95_percent_upper"] > report["estimate"]
