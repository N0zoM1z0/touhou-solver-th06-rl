from th06_rl.generation7.effect_diagnostics import (
    binary_ipw_effect,
    binary_ipw_exposure_diagnostics,
    bootstrap_sign_stability,
    permutation_null,
    synthetic_delayed_effect,
)


def test_synthetic_delayed_effect_recovers_known_positive_cost() -> None:
    report = synthetic_delayed_effect(
        episodes=80,
        rows_per_episode=200,
        delay=4,
        seed=7,
    )
    assert report["passes"] is True
    assert report["estimated_effect"] > 0.5


def test_null_and_bootstrap_gates_are_episode_grouped() -> None:
    from th06_rl.generation7.effect_diagnostics import (
        EffectEpisode,
        EffectRow,
    )

    groups = tuple(
        EffectEpisode(
            f"episode-{episode}",
            "fixture",
            6,
            tuple(
                EffectRow(
                    action="up" if index % 2 else "stay",
                    baseline_action="stay",
                    legal_actions=("stay", "up"),
                    behavior_probabilities=(0.5, 0.5),
                    hit_cost=int((index + episode) % 11 == 0),
                )
                for index in range(100)
            ),
        )
        for episode in range(20)
    )
    assert abs(float(binary_ipw_effect(groups, horizon=4)["effect"])) < 0.1
    assert permutation_null(
        groups, horizon=4, replicates=20, seed=3, kind="reward-suffix"
    )["passes"] is True
    stability = bootstrap_sign_stability(
        groups, horizon=4, replicates=20, seed=3
    )
    assert stability["replicates"] == 20


def test_binary_effect_excludes_singleton_safe_sets_without_positivity() -> None:
    from th06_rl.generation7.effect_diagnostics import EffectEpisode, EffectRow

    episode = EffectEpisode("episode", "fixture", 6, (
        EffectRow("stay", "stay", ("stay",), (1.0,), 1),
        EffectRow("up", "stay", ("stay", "up"), (0.5, 0.5), 0),
    ))
    report = binary_ipw_effect((episode,), horizon=1)
    assert report["rows"] == 1


def test_exposure_diagnostic_reports_post_assignment_interval_change() -> None:
    from th06_rl.generation7.effect_diagnostics import EffectEpisode, EffectRow

    rows = (
        EffectRow(
            "stay", "stay", ("stay", "up"), (0.5, 0.5), 0,
            duration_frames=4, complied=True,
        ),
        EffectRow(
            "up", "stay", ("stay", "up"), (0.5, 0.5), 0,
            duration_frames=1, complied=False,
        ),
    )
    report = binary_ipw_exposure_diagnostics((
        EffectEpisode("one", "fixture", 6, rows),
        EffectEpisode("two", "fixture", 6, rows),
    ))
    assert report["duration_frames"]["difference"] == -3.0
    assert report["compliance_probability"]["difference"] == -1.0
