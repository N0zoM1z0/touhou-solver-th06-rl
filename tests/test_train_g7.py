from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import scripts.train_g7 as train_g7


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_training_entry_keeps_holdout_disjoint_and_cannot_authorize_wine(
    tmp_path: Path, monkeypatch,
) -> None:
    dataset = tmp_path / "datasets/d0.json"
    config_path = tmp_path / "config/g7.json"
    _write(dataset, {"schema": "fixture"})
    config = json.loads(
        (train_g7.REPOSITORY / "config/g7_training_v2.json").read_text()
    )
    config["minimum_validation_episodes"] = 2
    config["validation_fraction"] = 0.5
    _write(config_path, config)
    episodes = tuple(
        (SimpleNamespace(episode_id=f"episode-{index}"),)
        for index in range(6)
    )
    fitted_ids = set()
    evaluated_ids = set()

    def fit(training, **_kwargs):
        fitted_ids.update(episode[0].episode_id for episode in training)
        return {"schema": "candidate"}

    def evaluate(_candidate, validation, **_kwargs):
        evaluated_ids.update(episode[0].episode_id for episode in validation)
        return {"passed": True}

    monkeypatch.setattr(train_g7, "REPOSITORY", tmp_path)
    monkeypatch.setattr(train_g7, "load_admitted_episodes", lambda *_args, **_kwargs: episodes)
    monkeypatch.setattr(train_g7, "fit_g7_candidate", fit)
    monkeypatch.setattr(train_g7, "evaluate_candidate", evaluate)

    result = train_g7.train(dataset, config_path)

    assert fitted_ids
    assert evaluated_ids
    assert not fitted_ids & evaluated_ids
    assert fitted_ids | evaluated_ids == {
        episode[0].episode_id for episode in episodes
    }
    assert result["heldout_evaluation"]["passed"] is True
    assert result["authorization"] == "wine-canary-forbidden"
