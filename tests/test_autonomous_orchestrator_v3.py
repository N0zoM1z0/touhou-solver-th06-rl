from __future__ import annotations

import json

import pytest

import scripts.run_generation3_preflight as preflight
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


def test_changed_preflight_contract_archives_cache_before_rerun(
    tmp_path, monkeypatch,
) -> None:
    root = tmp_path / "preflight"
    root.mkdir()
    (root / "preflight.json").write_text("{}", encoding="utf-8")
    archived = []
    monkeypatch.setattr(
        preflight,
        "_archive_incomplete",
        lambda path: archived.append(path) or path.rename(path.with_name("old")),
    )
    monkeypatch.setattr(
        preflight,
        "_validate_seed_contract",
        lambda: {"generation_seed": 260812, "smoke_game_rng_seed": 1},
    )
    monkeypatch.setattr(
        preflight,
        "run_causal_recovery_smoke",
        lambda threads: {"passed": False},
    )

    with pytest.raises(RuntimeError, match="causal"):
        preflight.run(root, threads=1, seconds=45.0)

    assert archived == [root]
    assert (tmp_path / "old/preflight.json").is_file()
