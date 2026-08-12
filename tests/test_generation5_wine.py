from dataclasses import replace

from scripts.run_generation5_wine import normalized_option_sha256
from th06_rl.implicit_learning import delayed_effect_episodes


def test_normalized_wine_digest_ignores_only_episode_and_option_ids(
    monkeypatch,
) -> None:
    rows = delayed_effect_episodes(count=1, options=12, delay=3)
    renamed = [
        replace(row, episode_id="other", option_id=f"other-{index}")
        for index, row in enumerate(rows)
    ]
    monkeypatch.setattr(
        "scripts.run_generation5_wine.load_audited_option_episode",
        lambda path: (rows if str(path) == "first" else renamed, {}),
    )

    assert normalized_option_sha256("first") == normalized_option_sha256("second")


def test_normalized_wine_digest_detects_factual_change(monkeypatch) -> None:
    rows = delayed_effect_episodes(count=1, options=12, delay=3)
    changed = [*rows]
    changed[3] = replace(changed[3], option_hit_cost=changed[3].option_hit_cost + 1.0)
    monkeypatch.setattr(
        "scripts.run_generation5_wine.load_audited_option_episode",
        lambda path: (rows if str(path) == "first" else changed, {}),
    )

    assert normalized_option_sha256("first") != normalized_option_sha256("second")
