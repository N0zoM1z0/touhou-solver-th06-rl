from __future__ import annotations

from scripts.run_autonomous_learning_v2 import _seed_schedule, parse_args


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


def test_generation_2_rng_schedule_is_deterministic_and_unique() -> None:
    first = _seed_schedule(260811, 12)
    assert first == _seed_schedule(260811, 12)
    assert len({row["game_rng_seed"] for row in first}) == 12
