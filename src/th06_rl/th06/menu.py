"""Background-tolerant native menu navigation for Reimu-A Practice."""

from __future__ import annotations

import struct
import time
from collections.abc import Callable

from .background_input import KEY_MASKS
from ..retail.native import read_menu_state


STATE_PRE_INPUT = 1
STATE_MAIN_MENU = 2
STATE_DIFFICULTY_SELECT = 7
STATE_CHARACTER_SELECT = 9
STATE_SHOT_SELECT = 11
STATE_PRACTICE_LVL_SELECT = 17
BACKGROUND_MENU_TIMEOUT = 15.0
# Maximum wait for the authoritative Supervisor to sample a published edge. A
# background TH06 instance can be throttled to roughly four updates/second, so
# a foreground-oriented 50 ms tap is insufficient. We release as soon
# as the real current/last pair proves the edge, rather than holding for all
# 350 ms and risking source key repeat when Wine runs faster than expected.
BACKGROUND_TAP_SECONDS = 0.35
ADDR_CURRENT_INPUT = 0x0069D904
ADDR_LAST_INPUT = 0x0069D908
ADDR_HELD_REPEAT = 0x0069D90C
ADDR_HELD_FRAMES = 0x0069D910


class MenuNavigationError(RuntimeError):
    """A pre-Stage menu attempt that is safe to retry with a fresh process."""


def _maintain_activity(maintain: Callable[[], object] | None) -> None:
    if maintain is not None:
        maintain()


def _read_input_state(process) -> tuple[int, int, int, int]:
    block = process.read(
        ADDR_CURRENT_INPUT,
        ADDR_HELD_FRAMES + 2 - ADDR_CURRENT_INPUT,
    )
    return tuple(
        struct.unpack_from("<H", block, address - ADDR_CURRENT_INPUT)[0]
        for address in (
            ADDR_CURRENT_INPUT,
            ADDR_LAST_INPUT,
            ADDR_HELD_REPEAT,
            ADDR_HELD_FRAMES,
        )
    )


def _tap(
    process,
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
    expected_mask = getattr(keyboard, "published_input_mask", None)
    key_mask = KEY_MASKS[key]
    input_sampled = expected_mask is None
    last_input_state = None
    try:
        deadline = time.monotonic() + BACKGROUND_TAP_SECONDS
        while time.monotonic() < deadline:
            _maintain_activity(maintain_activity)
            if expected_mask is not None:
                last_input_state = _read_input_state(process)
                current, _previous, _repeat, _held = last_input_state
                # The preceding settled tap proved that TH06 sampled the
                # released mask. Once current equals this new publication, a
                # Supervisor transition (and therefore a WAS_PRESSED edge)
                # necessarily occurred even if a cross-process read missed
                # the one frame where ``previous`` still held the release.
                input_sampled = current == expected_mask and current & key_mask != 0
                if input_sampled:
                    break
            time.sleep(0.02)
    finally:
        keyboard.set_auxiliary(key, False)
    if not input_sampled:
        raise MenuNavigationError(
            "retail menu did not sample the exact background input mask "
            f"0x{expected_mask:04x}; last_input_state={last_input_state}"
        )


def _wait_release_tick(
    process,
    timeout: float = BACKGROUND_MENU_TIMEOUT,
    *,
    maintain_activity: Callable[[], object] | None = None,
    expected_mask: int | None = None,
):
    """Wait until TH06 has sampled the release, with a legacy timer fallback."""
    initial = read_menu_state(process)
    deadline = time.monotonic() + timeout
    last = initial
    last_input_state = None
    while time.monotonic() < deadline:
        _maintain_activity(maintain_activity)
        last = read_menu_state(process)
        if expected_mask is not None:
            last_input_state = _read_input_state(process)
            if last_input_state[0] == expected_mask:
                return last
        elif last[0] != initial[0] or last[2] != initial[2]:
            return last
        time.sleep(0.02)
    raise MenuNavigationError(
        "menu did not sample released input; "
        f"expected_mask={expected_mask}, last_input_state={last_input_state}, "
        f"last_menu_state={last}"
    )


def _settled_tap(
    process,
    keyboard,
    key: str,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    _tap(process, keyboard, key, maintain_activity=maintain_activity)
    # MainMenu uses WAS_PRESSED, which compares the current and previous
    # sampled masks (authoritative utils.hpp). Require the source input global
    # itself to equal the released bridge mask before publishing another key.
    # Keyboard-like test implementations retain the old timer fallback.
    _wait_release_tick(
        process,
        maintain_activity=maintain_activity,
        expected_mask=getattr(keyboard, "published_input_mask", None),
    )


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
    main_menu_cursor: int,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    if difficulty not in range(4):
        raise ValueError("menu difficulty must be Easy/Normal/Hard/Lunatic")
    _set_cursor(
        process,
        keyboard,
        STATE_MAIN_MENU,
        target=main_menu_cursor,
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
        main_menu_cursor=2,
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
                process,
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


def start_reimu_a_route(
    process,
    keyboard,
    difficulty: int,
    *,
    maintain_activity: Callable[[], object] | None = None,
) -> None:
    """Enter ordinary Start with Reimu-A; battle movement remains generic."""
    _enter_main_menu(
        process,
        keyboard,
        maintain_activity=maintain_activity,
    )
    _select_reimu_a(
        process,
        keyboard,
        difficulty,
        main_menu_cursor=0,
        maintain_activity=maintain_activity,
    )
    time.sleep(1.0)
