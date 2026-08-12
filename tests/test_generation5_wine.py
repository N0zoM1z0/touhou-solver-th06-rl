from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.run_generation5_wine import complete_run, normalized_option_sha256
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


def test_complete_run_retries_same_frozen_row_after_strict_audit_failure(
    monkeypatch, tmp_path,
) -> None:
    artifact = tmp_path / "episode-002"
    corpus = tmp_path / "corpus"
    validations = iter([
        ValueError("physical corpus has infrastructure failure"),
        ({"controller_returncode": 0}, corpus / "good"),
    ])
    corpus_snapshots = iter([set(), {"bad"}, {"bad", "good"}])
    archived = []
    commands = []

    def validate(**_arguments):
        result = next(validations)
        if isinstance(result, BaseException):
            raise result
        return result

    monkeypatch.setattr(
        "scripts.run_generation5_wine._validate_complete_run", validate
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine._corpus_runs",
        lambda _root: next(corpus_snapshots),
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine._archive_incomplete",
        lambda path: archived.append(path),
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine.subprocess.run",
        lambda command, **_kwargs: commands.append(command)
        or SimpleNamespace(returncode=0),
    )

    report, run_dir = complete_run(
        artifact_dir=artifact,
        worker={
            "game_dir": tmp_path / "game",
            "wine_prefix": tmp_path / "prefix",
            "display": ":105",
        },
        stage=4,
        policy_plugin=tmp_path / "policy.py",
        policy_state=tmp_path / "state.json",
        scorer=None,
        rng_seed=27158,
        corpus_root=corpus,
    )

    assert report["controller_returncode"] == 0
    assert run_dir == corpus / "good"
    assert len(commands) == 2
    assert commands[0] == commands[1]
    assert archived == [artifact]


def test_complete_run_stops_after_bounded_infra_attempts(
    monkeypatch, tmp_path,
) -> None:
    attempts = []
    monkeypatch.setattr(
        "scripts.run_generation5_wine._validate_complete_run",
        lambda **_arguments: (_ for _ in ()).throw(ValueError("invalid corpus")),
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine._corpus_runs", lambda _root: set()
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine._archive_incomplete",
        lambda path: attempts.append(path),
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0),
    )

    with pytest.raises(ValueError, match="invalid corpus"):
        complete_run(
            artifact_dir=tmp_path / "episode",
            worker={
                "game_dir": tmp_path / "game",
                "wine_prefix": tmp_path / "prefix",
                "display": ":105",
            },
            stage=4,
            policy_plugin=Path("policy.py"),
            policy_state=Path("state.json"),
            scorer=None,
            rng_seed=1,
            corpus_root=tmp_path / "corpus",
        )

    assert len(attempts) == 2


def test_complete_run_passes_frozen_disjoint_cpu_partitions(
    monkeypatch, tmp_path,
) -> None:
    commands = []
    monkeypatch.setattr(
        "scripts.run_generation5_wine._validate_complete_run",
        lambda **_arguments: ({"controller_returncode": 0}, None),
    )
    monkeypatch.setattr(
        "scripts.run_generation5_wine.subprocess.run",
        lambda command, **_kwargs: commands.append(command)
        or SimpleNamespace(returncode=0),
    )

    complete_run(
        artifact_dir=tmp_path / "episode",
        worker={
            "game_dir": tmp_path / "game",
            "wine_prefix": tmp_path / "prefix",
            "display": ":108",
        },
        stage=4,
        policy_plugin=Path("policy.py"),
        policy_state=Path("state.json"),
        scorer=Path("scorer.dll"),
        rng_seed=None,
        corpus_root=None,
        game_cpu_list="0-7",
        controller_cpu_list="8-31",
    )

    command = commands[0]
    assert command[command.index("--game-cpu-list") + 1] == "0-7"
    assert command[command.index("--controller-cpu-list") + 1] == "8-31"
