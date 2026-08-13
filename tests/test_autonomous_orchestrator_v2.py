from __future__ import annotations

import json

import scripts.run_autonomous_learning_v2 as orchestrator
from scripts.run_autonomous_learning_v2 import (
    _archive_incomplete,
    _complete_run,
    _fit_boundaries,
    _seed_schedule,
    parse_args,
)


def test_generation_2_defaults_lock_complete_stage_conservative_learning(
    tmp_path,
) -> None:
    args = parse_args(["--output-root", str(tmp_path / "generation-2")])
    assert args.collection_episodes == 8
    assert args.initial_fit_episodes == 6
    assert args.validation_episodes == 2
    assert args.ensemble_members == 5
    assert args.bellman_iterations == 6
    assert args.n_step_frames == 60
    assert args.canary_pairs == 1
    assert args.full_stage_pairs == 2
    assert _fit_boundaries(args) == (6, 8)


def test_generation_2_rng_schedule_is_deterministic_and_unique() -> None:
    first = _seed_schedule(260811, 12)
    assert first == _seed_schedule(260811, 12)
    assert len({row["game_rng_seed"] for row in first}) == 12


def test_generation_2_preserves_partial_artifact_before_retry(tmp_path) -> None:
    partial = tmp_path / "episode-000"
    partial.mkdir()
    (partial / "controller.log").write_text("partial", encoding="utf-8")

    _archive_incomplete(partial)

    assert not partial.exists()
    assert (tmp_path / "episode-000.incomplete-001/controller.log").is_file()


def test_generation_2_reuses_validated_complete_wine_run(
    tmp_path, monkeypatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    corpus = tmp_path / "corpus"
    run_dir = corpus / "run-a"
    run_dir.mkdir(parents=True)
    report = {
        "controller_completion": {"physical_hits": 3},
        "trace": {"corpus_run_ids": ["run-a"]},
    }
    (artifact / "report.json").write_text(
        json.dumps(report), encoding="utf-8"
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_retail_report",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        orchestrator,
        "_validate_run",
        lambda path: ({}, {
            "stage_trajectory_complete": True,
            "run_outcome": {"stage_completed": True, "physical_hits": 3},
        }),
    )
    monkeypatch.setattr(
        orchestrator.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("cached Wine run must not launch a process")
        ),
    )

    loaded, bound = _complete_run(
        artifact_dir=artifact,
        policy_plugin=tmp_path / "policy.py",
        policy_state=tmp_path / "policy.json",
        scorer=None,
        difficulty="lunatic",
        stage=6,
        rng_seed=123,
        corpus_root=corpus,
    )

    assert loaded == report
    assert bound == run_dir
