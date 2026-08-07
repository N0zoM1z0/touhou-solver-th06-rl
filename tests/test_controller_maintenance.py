from __future__ import annotations

from types import SimpleNamespace

from th06_rl.th06.controller import _reload_poll_window


def snapshot(*, player_state=0, in_menu=False, time_stopped=False):
    return SimpleNamespace(
        player_state=player_state,
        in_menu=in_menu,
        time_stopped=time_stopped,
    )


def test_reload_poll_is_deferred_during_active_play() -> None:
    assert _reload_poll_window(snapshot()) is False
    assert _reload_poll_window(snapshot(player_state=3)) is False


def test_reload_poll_is_admitted_outside_active_control() -> None:
    assert _reload_poll_window(snapshot(player_state=2)) is True
    assert _reload_poll_window(snapshot(in_menu=True)) is True
    assert _reload_poll_window(snapshot(time_stopped=True)) is True
