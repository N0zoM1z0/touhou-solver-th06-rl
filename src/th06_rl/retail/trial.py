"""Small, source-grounded lifecycle checks for physical trials."""

from __future__ import annotations

from dataclasses import dataclass

from .model import PLAYER_ALIVE, PLAYER_DEAD


SUPERVISOR_MAIN_MENU = 1
SUPERVISOR_GAMEPLAY = 2
SUPERVISOR_RESULT_FROM_GAME = 7


def physical_hit(previous_player_state: int | None, player_state: int) -> bool:
    """Player::Die changes ALIVE directly to DEAD in the authoritative source."""
    return previous_player_state == PLAYER_ALIVE and player_state == PLAYER_DEAD


def stop_trial_now(process, keyboard, terminate_process: bool) -> None:
    """Release input and synchronously stop the exact trial before artifact I/O."""
    try:
        if keyboard is not None:
            keyboard.release_all()
    finally:
        if terminate_process:
            process.terminate()


@dataclass
class PracticeTrial:
    gameplay_seen: bool = False

    def observe_supervisor(self, current_state: int) -> bool:
        """Return true after a played Practice stage reaches its result/menu path."""
        if current_state == SUPERVISOR_GAMEPLAY:
            self.gameplay_seen = True
        return self.gameplay_seen and current_state in (
            SUPERVISOR_MAIN_MENU,
            SUPERVISOR_RESULT_FROM_GAME,
        )
