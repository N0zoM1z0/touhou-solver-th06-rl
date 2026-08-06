"""Background-tolerant native menu navigation for Reimu-A Practice."""

from __future__ import annotations

import time

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


def _wait_state(process, wanted: int, timeout: float = BACKGROUND_MENU_TIMEOUT):
    deadline = time.monotonic() + timeout
    last = (-1, -1, -1)
    while time.monotonic() < deadline:
        last = read_menu_state(process)
        if last[0] == wanted:
            return last
        time.sleep(0.02)
    raise RuntimeError(f"menu state {wanted} not reached; last={last}")


def _wait_timer(
    process,
    state: int,
    minimum: int,
    timeout: float = BACKGROUND_MENU_TIMEOUT,
):
    deadline = time.monotonic() + timeout
    last = (-1, -1, -1)
    while time.monotonic() < deadline:
        last = read_menu_state(process)
        if last[0] == state and last[2] >= minimum:
            return last
        time.sleep(0.02)
    raise RuntimeError(f"menu timer not ready for state {state}; last={last}")


def _set_cursor(process, keyboard, state: int, target: int, length: int) -> None:
    for _ in range(length + 1):
        current_state, cursor, _timer = read_menu_state(process)
        if current_state != state:
            raise RuntimeError(f"menu left state {state} while selecting cursor")
        if cursor == target:
            return
        downward = (target - cursor) % length
        upward = (cursor - target) % length
        keyboard.tap("down" if downward <= upward else "up")
    raise RuntimeError(f"could not select cursor {target} in state {state}")


def _enter_main_menu(process, keyboard) -> None:
    deadline = time.monotonic() + BACKGROUND_MENU_TIMEOUT
    last = (-1, -1, -1)
    while time.monotonic() < deadline:
        last = read_menu_state(process)
        state, _cursor, timer = last
        if state == STATE_MAIN_MENU:
            break
        if state == STATE_PRE_INPUT and timer >= 30:
            keyboard.tap("shoot")
        time.sleep(0.02)
    else:
        raise RuntimeError(f"main menu not reached; last={last}")
    _wait_timer(process, STATE_MAIN_MENU, 20)


def _select_reimu_a(process, keyboard, difficulty: int) -> None:
    if difficulty not in range(4):
        raise ValueError("menu difficulty must be Easy/Normal/Hard/Lunatic")
    _set_cursor(process, keyboard, STATE_MAIN_MENU, target=2, length=8)
    keyboard.tap("shoot")
    _wait_state(process, STATE_DIFFICULTY_SELECT)
    _set_cursor(
        process,
        keyboard,
        STATE_DIFFICULTY_SELECT,
        target=difficulty,
        length=4,
    )
    keyboard.tap("shoot")
    _wait_timer(process, STATE_CHARACTER_SELECT, 30)
    _set_cursor(process, keyboard, STATE_CHARACTER_SELECT, target=0, length=2)
    keyboard.tap("shoot")
    _wait_timer(process, STATE_SHOT_SELECT, 30)
    _set_cursor(process, keyboard, STATE_SHOT_SELECT, target=0, length=2)
    keyboard.tap("shoot")


def start_reimu_a_practice(
    process,
    keyboard,
    stage: int,
    difficulty: int,
) -> None:
    if not 1 <= stage <= 6:
        raise ValueError("Practice stage must be in 1..6")
    _enter_main_menu(process, keyboard)
    _select_reimu_a(process, keyboard, difficulty)
    _wait_timer(process, STATE_PRACTICE_LVL_SELECT, 30)
    target = stage - 1
    seen = set()
    for _ in range(7):
        state, cursor, _timer = read_menu_state(process)
        if state != STATE_PRACTICE_LVL_SELECT:
            raise RuntimeError("menu left Practice stage selection unexpectedly")
        if cursor == target:
            keyboard.tap("shoot")
            time.sleep(1.0)
            return
        if cursor in seen:
            raise RuntimeError(f"Practice stage {stage} is not unlocked")
        seen.add(cursor)
        keyboard.tap("down")
    raise RuntimeError(f"could not select Practice stage {stage}")
