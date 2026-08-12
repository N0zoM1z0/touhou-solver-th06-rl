from __future__ import annotations

from dataclasses import replace

import pytest

from th06_rl.implicit_learning import delayed_effect_episodes
from th06_rl.audited_option_loader import AUDITED_OPTION_LOADER_CONTRACT
from th06_rl.option_cache import load_cached_option_episode


def test_option_cache_is_bound_to_manifest_and_loader_contract(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    manifest = run / "manifest.json"
    manifest.write_text('{"version":1}\n', encoding="utf-8")
    contract = tmp_path / "loader.py"
    contract.write_text("version = 1\n", encoding="utf-8")
    cache = tmp_path / "cache"
    calls = []
    rows = delayed_effect_episodes(count=1, options=24, delay=5)

    def loader(path):
        calls.append(path)
        return rows, {"physical_hits": sum(row.option_hit_cost for row in rows)}

    first_rows, first_report, first_hit = load_cached_option_episode(
        run, loader=loader, cache_root=cache, contract_files=(contract,)
    )
    second_rows, second_report, second_hit = load_cached_option_episode(
        run, loader=loader, cache_root=cache, contract_files=(contract,)
    )

    assert first_hit is False
    assert second_hit is True
    assert len(calls) == 1
    assert second_rows == first_rows
    assert second_report == first_report

    contract.write_text("version = 2\n", encoding="utf-8")
    _third_rows, _third_report, third_hit = load_cached_option_episode(
        run, loader=loader, cache_root=cache, contract_files=(contract,)
    )
    assert third_hit is False
    assert len(calls) == 2


def test_option_cache_rejects_tampered_payload(tmp_path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    (run / "manifest.json").write_text("{}\n", encoding="utf-8")
    contract = tmp_path / "loader.py"
    contract.write_text("version = 1\n", encoding="utf-8")
    cache = tmp_path / "cache"
    sample = delayed_effect_episodes(count=1, options=24, delay=5)[0]
    loader = lambda _path: ([replace(sample, episode_id="cache")], {})
    load_cached_option_episode(
        run, loader=loader, cache_root=cache, contract_files=(contract,)
    )
    payload = next(cache.glob("*.pickle"))
    payload.write_bytes(payload.read_bytes() + b"tamper")

    with pytest.raises(ValueError, match="metadata"):
        load_cached_option_episode(
            run, loader=loader, cache_root=cache, contract_files=(contract,)
        )


def test_production_cache_contract_excludes_training_cli_orchestration() -> None:
    names = {path.name for path in AUDITED_OPTION_LOADER_CONTRACT}

    assert "audited_option_loader.py" in names
    assert "fit_supported_implicit_q.py" not in names
    assert "smoke_supported_implicit_q_wine.py" not in names
