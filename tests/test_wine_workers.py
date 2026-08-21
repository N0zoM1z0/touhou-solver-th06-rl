import hashlib
from pathlib import Path

import pytest

from th06_rl.wine_workers import (
    allocate_worker_specifications,
    attest_retail_template,
    prepare_wine_worker,
    validate_wine_worker,
    validate_worker_specifications,
)


def _attest(source: Path) -> None:
    attest_retail_template(
        source,
        archive_sha256="a" * 64,
        executable_sha256=hashlib.sha256(
            (source / "東方紅魔郷.exe").read_bytes()
        ).hexdigest(),
    )


def test_wine_worker_has_independent_game_prefix_and_display(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "東方紅魔郷.exe").write_bytes(b"retail")
    (source / "score.dat").write_bytes(b"score")
    _attest(source)

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
    assert validate_wine_worker(worker) == worker
    assert Path(worker["game_dir"]) != source
    assert Path(worker["wine_prefix"]).is_dir()
    assert worker["display"] == ":98"


def test_wine_worker_rejects_contract_drift(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "東方紅魔郷.exe").write_bytes(b"retail")
    _attest(source)
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
    _attest(source)
    with pytest.raises(ValueError, match="source-game inventory differs"):
        prepare_wine_worker(
            root=tmp_path / "workers",
            source_game_dir=source,
            worker=0,
            directory="wine-0",
            display=":91",
            source_inventory_sha256="0" * 64,
        )


def test_wine_worker_rejects_mutated_template(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "東方紅魔郷.exe").write_bytes(b"retail")
    (source / "score.dat").write_bytes(b"score")
    _attest(source)
    (source / "score.dat").write_bytes(b"mutated")

    with pytest.raises(ValueError, match="template integrity differs"):
        prepare_wine_worker(
            root=tmp_path / "workers",
            source_game_dir=source,
            worker=0,
            directory="wine-0",
            display=":97",
        )


def test_multi_worker_allocation_is_disjoint_and_resource_bounded() -> None:
    rows = allocate_worker_specifications(
        available_cpus=tuple(range(64)), workers=8, cpus_per_worker=8,
    )

    assert len(rows) == 8
    assert rows[0] == {
        "worker": 0,
        "directory": "wine-0",
        "display": ":107",
        "game_cpu_list": "0,1,2,3",
        "controller_cpu_list": "4,5,6,7",
    }
    assert rows[-1] == {
        "worker": 7,
        "directory": "wine-7",
        "display": ":114",
        "game_cpu_list": "56,57,58,59",
        "controller_cpu_list": "60,61,62,63",
    }


def test_worker_contract_rejects_cross_worker_cpu_and_display_overlap() -> None:
    rows = allocate_worker_specifications(
        available_cpus=tuple(range(16)), workers=2, cpus_per_worker=8,
    )
    rows[1]["game_cpu_list"] = "0,8,9,10"
    with pytest.raises(ValueError, match="overlap"):
        validate_worker_specifications(rows)

    rows = allocate_worker_specifications(
        available_cpus=tuple(range(16)), workers=2, cpus_per_worker=8,
    )
    rows[1]["display"] = rows[0]["display"]
    with pytest.raises(ValueError, match="overlap"):
        validate_worker_specifications(rows)


def test_worker_contract_rejects_path_escape_and_insufficient_cpu() -> None:
    row = allocate_worker_specifications(
        available_cpus=tuple(range(8)), workers=1, cpus_per_worker=8,
    )[0]
    row["directory"] = "../shared"
    with pytest.raises(ValueError, match="overlap"):
        validate_worker_specifications([row])

    with pytest.raises(ValueError, match="bounded"):
        allocate_worker_specifications(
            available_cpus=tuple(range(64)), workers=4, cpus_per_worker=17,
        )


def test_worker_contract_rejects_noncanonical_identity() -> None:
    rows = allocate_worker_specifications(
        available_cpus=tuple(range(16)), workers=2, cpus_per_worker=8,
    )
    rows[1]["worker"] = 2
    with pytest.raises(ValueError, match="contiguous"):
        validate_worker_specifications(rows)
