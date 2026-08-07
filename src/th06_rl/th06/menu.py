"""Background-tolerant native menu navigation for Reimu-A Practice."""

from __future__ import annotations

import time
from collections.abc import Callable

from .donor import enable_donor_imports

enable_donor_imports()
from th06.native import read_menu_state  # noqa: E402


STATE_PRE_INPUT = 1
STATE_MAIN_MENU = 2
STATE_DIFFICULTY_SELECT = 7
STATE_CHARACTER_SELECT = 9
STATE_SHOT_SELECT = 11
STATE_PRACTICE_LVL_SELECT = 17
BACKGROUND_MENU_TIMEOUT = 15.0
# A background TH06 instance can be throttled to roughly four updates/second.
# The donor's foreground-oriented 50 ms tap may then begin and end between two
# game updates. The authoritative Supervisor starts key repeat after 30 equal
# input frames; 350 ms is only 21 frames at 60 Hz but spans a throttled update.
BACKGROUND_TAP_SECONDS = 0.35


class MenuNavigationError(RuntimeError):
    """A pre-Stage menu attempt that is safe to retry with a fresh process."""


def _maintain_activity(maintain: Callable[[], object] | None) -> None:
    if maintain is not None:
        maintain()


def _tap(
    keyboard,
    key: str,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    if maintain_activity is None:
        keyboard.tap(key, hold_seconds=BACKGROUND_TAP_SECONDS)
        return

    # Keep the source calc chain alive throughout both edges of a background
    # tap. A focus change during Keyboard.tap's blocking sleep could otherwise
    # clear GameWindow::lastActiveAppValue after the one pre-tap maintenance.
    keyboard.set_auxiliary(key, True)
    try:
        deadline = time.monotonic() + BACKGROUND_TAP_SECONDS
        while time.monotonic() < deadline:
            _maintain_activity(maintain_activity)
            time.sleep(0.02)
    finally:
        keyboard.set_auxiliary(key, False)
    deadline = time.monotonic() + 0.12
    while time.monotonic() < deadline:
        _maintain_activity(maintain_activity)
        time.sleep(0.02)


def _wait_release_tick(
    process,
    timeout: float = BACKGROUND_MENU_TIMEOUT,
    *,
    maintain_activity: Callable[[], object] | None = None,
):
    """Wait until TH06 has sampled one menu tick after the bridge release."""
    initial = read_menu_state(process)
    deadline = time.monotonic() + timeout
    last = initial
    while time.monotonic() < deadline:
        _maintain_activity(maintain_activity)
        last = read_menu_state(process)
        if last[0] != initial[0] or last[2] != initial[2]:
            return last
        time.sleep(0.02)
    raise MenuNavigationError(
        f"menu did not sample released input; last={last}"
    )


def _settled_tap(
    process,
    keyboard,
    key: str,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    _tap(keyboard, key, maintain_activity=maintain_activity)
    # MainMenu uses WAS_PRESSED, which compares the current and previous
    # sampled masks (authoritative utils.hpp).  A background-throttled game
    # can otherwise miss the short zero-mask interval between two taps and
    # treat the next Select as one continuous hold.  A state/timer tick proves
    # that the bridge's release was sampled before another key is published.
    _wait_release_tick(process, maintain_activity=maintain_activity)


def _wait_state(
    process,
    wanted: int,
    timeout: float = BACKGROUND_MENU_TIMEOUT,
    *,
    maintain_activity: Callable[[], object] | None = None,
):
    deadline = time.monotonic() + timeout
    last = (-1, -1, -1)
    while time.monotonic() < deadline:
        _maintain_activity(maintain_activity)
        last = read_menu_state(process)
        if last[0] == wanted:
            return last
        time.sleep(0.02)
    raise MenuNavigationError(f"menu state {wanted} not reached; last={last}")


def _wait_timer(
    process,
    state: int,
    minimum: int,
    timeout: float = BACKGROUND_MENU_TIMEOUT,
    *,
    maintain_activity: Callable[[], object] | None = None,
):
    deadline = time.monotonic() + timeout
    last = (-1, -1, -1)
    while time.monotonic() < deadline:
        _maintain_activity(maintain_activity)
        last = read_menu_state(process)
        if last[0] == state and last[2] >= minimum:
            return last
        time.sleep(0.02)
    raise MenuNavigationError(
        f"menu timer not ready for state {state}; last={last}"
    )


def _set_cursor(
    process,
    keyboard,
    state: int,
    target: int,
    length: int,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    for _ in range(length + 1):
        _maintain_activity(maintain_activity)
        current_state, cursor, _timer = read_menu_state(process)
        if current_state != state:
            raise MenuNavigationError(
                f"menu left state {state} while selecting cursor"
            )
        if cursor == target:
            return
        downward = (target - cursor) % length
        upward = (cursor - target) % length
        _settled_tap(
            process,
            keyboard,
            "down" if downward <= upward else "up",
            maintain_activity=maintain_activity,
        )
    raise MenuNavigationError(
        f"could not select cursor {target} in state {state}"
    )


def _enter_main_menu(
    process,
    keyboard,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    deadline = time.monotonic() + BACKGROUND_MENU_TIMEOUT
    last = (-1, -1, -1)
    while time.monotonic() < deadline:
        _maintain_activity(maintain_activity)
        last = read_menu_state(process)
        state, _cursor, timer = last
        if state == STATE_MAIN_MENU:
            break
        if state == STATE_PRE_INPUT and timer >= 30:
            _settled_tap(
                process,
                keyboard,
                "shoot",
                maintain_activity=maintain_activity,
            )
        time.sleep(0.02)
    else:
        raise MenuNavigationError(f"main menu not reached; last={last}")
    _wait_timer(
        process,
        STATE_MAIN_MENU,
        20,
        maintain_activity=maintain_activity,
    )


def _select_reimu_a(
    process,
    keyboard,
    difficulty: int,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    if difficulty not in range(4):
        raise ValueError("menu difficulty must be Easy/Normal/Hard/Lunatic")
    _set_cursor(
        process,
        keyboard,
        STATE_MAIN_MENU,
        target=2,
        length=8,
        maintain_activity=maintain_activity,
    )
    _settled_tap(
        process,
        keyboard,
        "shoot",
        maintain_activity=maintain_activity,
    )
    _wait_state(
        process,
        STATE_DIFFICULTY_SELECT,
        maintain_activity=maintain_activity,
    )
    _set_cursor(
        process,
        keyboard,
        STATE_DIFFICULTY_SELECT,
        target=difficulty,
        length=4,
        maintain_activity=maintain_activity,
    )
    _settled_tap(
        process,
        keyboard,
        "shoot",
        maintain_activity=maintain_activity,
    )
    _wait_timer(
        process,
        STATE_CHARACTER_SELECT,
        30,
        maintain_activity=maintain_activity,
    )
    _set_cursor(
        process,
        keyboard,
        STATE_CHARACTER_SELECT,
        target=0,
        length=2,
        maintain_activity=maintain_activity,
    )
    _settled_tap(
        process,
        keyboard,
        "shoot",
        maintain_activity=maintain_activity,
    )
    _wait_timer(
        process,
        STATE_SHOT_SELECT,
        30,
        maintain_activity=maintain_activity,
    )
    _set_cursor(
        process,
        keyboard,
        STATE_SHOT_SELECT,
        target=0,
        length=2,
        maintain_activity=maintain_activity,
    )
    _settled_tap(
        process,
        keyboard,
        "shoot",
        maintain_activity=maintain_activity,
    )


def start_reimu_a_practice(
    process,
    keyboard,
    stage: int,
    difficulty: int,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    if not 1 <= stage <= 6:
        raise ValueError("Practice stage must be in 1..6")
    _enter_main_menu(
        process,
        keyboard,
        maintain_activity=maintain_activity,
    )
    _select_reimu_a(
        process,
        keyboard,
        difficulty,
        maintain_activity=maintain_activity,
    )
    _wait_timer(
        process,
        STATE_PRACTICE_LVL_SELECT,
        30,
        maintain_activity=maintain_activity,
    )
    target = stage - 1
    seen = set()
    for _ in range(7):
        _maintain_activity(maintain_activity)
        state, cursor, _timer = read_menu_state(process)
        if state != STATE_PRACTICE_LVL_SELECT:
            raise MenuNavigationError(
                "menu left Practice stage selection unexpectedly"
            )
        if cursor == target:
            _tap(
                keyboard,
                "shoot",
                maintain_activity=maintain_activity,
            )
            time.sleep(1.0)
            return
        if cursor in seen:
            raise MenuNavigationError(f"Practice stage {stage} is not unlocked")
        seen.add(cursor)
        _settled_tap(
            process,
            keyboard,
            "down",
            maintain_activity=maintain_activity,
        )
    raise MenuNavigationError(f"could not select Practice stage {stage}")
