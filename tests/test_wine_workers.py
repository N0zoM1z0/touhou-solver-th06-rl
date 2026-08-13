from pathlib import Path

import pytest

from th06_rl.wine_workers import prepare_wine_worker


def test_wine_worker_has_independent_game_prefix_and_display(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "東方紅魔郷.exe").write_bytes(b"retail")
    (source / "score.dat").write_bytes(b"score")

    worker = prepare_wine_worker(
        root=tmp_path / "workers",
        source_game_dir=source,
        worker=1,
        directory="wine-1",
        display=":98",
    )
    repeated = prepare_wine_worker(
        root=tmp_path / "workers",
        source_game_dir=source,
        worker=1,
        directory="wine-1",
        display=":98",
    )

    assert repeated == worker
    assert Path(worker["game_dir"]) != source
    assert Path(worker["wine_prefix"]).is_dir()
    assert worker["display"] == ":98"


def test_wine_worker_rejects_contract_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "東方紅魔郷.exe").write_bytes(b"retail")
    root = tmp_path / "workers"
    prepare_wine_worker(
        root=root,
        source_game_dir=source,
        worker=0,
        directory="wine-0",
        display=":97",
    )

    with pytest.raises(ValueError, match="contract differs"):
        prepare_wine_worker(
            root=root,
            source_game_dir=source,
            worker=0,
            directory="wine-0",
            display=":101",
        )


def test_wine_worker_rejects_frozen_source_inventory_drift(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "東方紅魔郷.exe").write_bytes(b"retail")
    with pytest.raises(ValueError, match="source-game inventory differs"):
        prepare_wine_worker(
            root=tmp_path / "workers",
            source_game_dir=source,
            worker=0,
            directory="wine-0",
            display=":91",
            source_inventory_sha256="0" * 64,
        )
