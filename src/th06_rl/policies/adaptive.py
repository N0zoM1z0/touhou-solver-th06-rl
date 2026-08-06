"""Phase-isolated online bandit above native/local safety gates."""

from __future__ import annotations

import base64
from collections import Counter
import json
import math
import random
import zlib

from ..policy_api import POLICY_API_VERSION, PolicyDecision


STATE_SCHEMA = "th06-rl-online-hierarchical-ucb-v3"
LEGACY_STATE_SCHEMAS = (
    "th06-rl-online-hierarchical-ucb-v2",
    "th06-rl-online-ucb-v1",
)
PACKED_STATE_SCHEMA = "th06-rl-online-ucb-packed-v1"
REWARD_VERSION = "survival-reserve-v1"
MIDDLE_BACKOFF_WEIGHT = 4.0
FINE_BACKOFF_WEIGHT = 4.0
PHASE_CLOCK_BIN_FRAMES = 30
ACTION_NAMES = (
    "stay",
    "up",
    "down",
    "left",
    "right",
    "up_left",
    "up_right",
    "down_left",
    "down_right",
    "stay_fast",
    "up_fast",
    "down_fast",
    "left_fast",
    "right_fast",
    "up_left_fast",
    "up_right_fast",
    "down_left_fast",
    "down_right_fast",
)
ACTION_BITS = {name: 1 << index for index, name in enumerate(ACTION_NAMES)}


def unpack_state(state: dict[str, object]) -> dict[str, object]:
    """Decode a compact restart checkpoint while retaining legacy support."""
    if state.get("schema") != PACKED_STATE_SCHEMA:
        return state
    if state.get("codec") != "zlib-base64-v1":
        raise ValueError("unsupported packed policy checkpoint codec")
    encoded = state.get("payload")
    if not isinstance(encoded, str):
        raise TypeError("packed policy checkpoint payload must be text")
    decoded = json.loads(
        zlib.decompress(base64.b64decode(encoded, validate=True)).decode("utf-8")
    )
    if not isinstance(decoded, dict):
        raise TypeError("decoded policy checkpoint root must be an object")
    return decoded


class AdaptivePolicy:
    api_version = POLICY_API_VERSION
    name = "phase-local-hierarchical-ucb-v3"

    def __init__(self) -> None:
        self.random = random.Random(6004)
        self.decisions = 0
        self.exploratory_decisions = 0
        self.selected: Counter[str] = Counter()
        self.opportunities: Counter[str] = Counter()
        self.trials: Counter[str] = Counter()
        self.reward_sum: Counter[str] = Counter()
        self.middle_trials: Counter[str] = Counter()
        self.middle_reward_sum: Counter[str] = Counter()
        self.fine_selected: Counter[str] = Counter()
        self.fine_trials: Counter[str] = Counter()
        self.fine_reward_sum: Counter[str] = Counter()
        self.pending_keys: dict[
            tuple[int, str],
            tuple[str, str, str],
        ] = {}

    @staticmethod
    def _threat_bin(bullets: int, lasers: int) -> str:
        if lasers:
            return "laser"
        if bullets == 0:
            return "clear"
        if bullets < 16:
            return "sparse"
        if bullets < 48:
            return "medium"
        return "dense"

    @staticmethod
    def _position_bin(value: float, low: float, high: float) -> int:
        normalized = max(0.0, min(0.999999, (value - low) / (high - low)))
        return int(normalized * 4)

    def _context_key(self, context) -> str:
        scope = "/".join(map(str, context.scope))
        x_bin = self._position_bin(context.player_x, 8.0, 376.0)
        y_bin = self._position_bin(context.player_y, 16.0, 432.0)
        reserve = min(4, context.hard_action_count // 4)
        threat = self._threat_bin(context.bullet_count, context.laser_count)
        return (
            f"{scope}|{context.source_context}|x{x_bin}|y{y_bin}|"
            f"threat:{threat}|reserve:{reserve}"
        )

    @staticmethod
    def _action_mask(actions: tuple[str, ...]) -> int:
        mask = 0
        for action in actions:
            try:
                mask |= ACTION_BITS[action]
            except KeyError as error:
                raise ValueError(f"unknown policy action {action!r}") from error
        return mask

    def _fine_context_key(self, context) -> str:
        middle = self._middle_context_key(context)
        hard_mask = self._action_mask(context.hard_admissible_actions)
        legal_mask = self._action_mask(context.locally_admissible_actions)
        return (
            f"{middle}|hard:{hard_mask:05x}|legal:{legal_mask:05x}"
        )

    def _middle_context_key(self, context) -> str:
        coarse = self._context_key(context)
        clock = min(
            127,
            max(0, int(context.phase_elapsed_frames))
            // PHASE_CLOCK_BIN_FRAMES,
        )
        return (
            f"{coarse}|clock:{clock}|current:{context.current_action}|"
            f"baseline:{context.baseline_action}"
        )

    @staticmethod
    def _action_key(context_key: str, action: str) -> str:
        return f"{context_key}|action:{action}"

    @staticmethod
    def _middle_action_key_from_fine(fine_key: str) -> str | None:
        try:
            context_key, action = fine_key.rsplit("|action:", 1)
            middle_key, _frontier = context_key.rsplit("|hard:", 1)
        except ValueError:
            return None
        return f"{middle_key}|action:{action}"

    def decide(self, context):
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("policy received no locally admissible actions")
        if context.baseline_action not in legal:
            raise ValueError("reactive baseline is outside the local safe set")
        context_key = self._context_key(context)
        middle_context_key = self._middle_context_key(context)
        fine_context_key = self._fine_context_key(context)
        total_trials = sum(
            self.trials[self._action_key(context_key, action)] for action in legal
        )
        fine_total_trials = sum(
            self.fine_trials[self._action_key(fine_context_key, action)]
            for action in legal
        )
        middle_total_trials = sum(
            self.middle_trials[
                self._action_key(middle_context_key, action)
            ]
            for action in legal
        )
        scores = {}
        for action in legal:
            key = self._action_key(context_key, action)
            middle_key = self._action_key(middle_context_key, action)
            fine_key = self._action_key(fine_context_key, action)
            trials = self.trials[key]
            coarse_empirical = self.reward_sum[key] / trials if trials else 0.0
            middle_trials = self.middle_trials[middle_key]
            middle_empirical = (
                (
                    self.middle_reward_sum[middle_key]
                    + MIDDLE_BACKOFF_WEIGHT * coarse_empirical
                )
                / (middle_trials + MIDDLE_BACKOFF_WEIGHT)
                if middle_trials
                else coarse_empirical
            )
            fine_trials = self.fine_trials[fine_key]
            empirical = (
                (
                    self.fine_reward_sum[fine_key]
                    + FINE_BACKOFF_WEIGHT * middle_empirical
                )
                / (fine_trials + FINE_BACKOFF_WEIGHT)
                if fine_trials
                else middle_empirical
            )
            optimism = (
                0.12
                * math.sqrt(
                    math.log(max(
                        2,
                        fine_total_trials
                        + middle_total_trials
                        + total_trials
                        + 2,
                    ))
                    / fine_trials
                )
                if fine_trials
                else 0.12
            )
            baseline_prior = 0.18 if action == context.baseline_action else 0.0
            scores[action] = empirical + optimism + baseline_prior
            self.opportunities[key] += 1
        greedy = max(legal, key=lambda action: (scores[action], action))
        exploration = max(0.0, min(1.0, float(context.exploration_rate)))
        if len(legal) == 1 or exploration == 0.0:
            probabilities = {action: 0.0 for action in legal}
            probabilities[greedy] = 1.0
        else:
            weights = {
                action: 1.0
                / math.sqrt(
                    1.0
                    + self.fine_selected[
                        self._action_key(fine_context_key, action)
                    ]
                )
                for action in legal
            }
            total = sum(weights.values())
            probabilities = {
                action: exploration * weights[action] / total for action in legal
            }
            probabilities[greedy] += 1.0 - exploration
        draw = self.random.random()
        cumulative = 0.0
        chosen = legal[-1]
        for action in legal:
            cumulative += probabilities[action]
            if draw <= cumulative:
                chosen = action
                break
        key = self._action_key(context_key, chosen)
        fine_key = self._action_key(fine_context_key, chosen)
        self.selected[key] += 1
        self.fine_selected[fine_key] += 1
        middle_key = self._action_key(middle_context_key, chosen)
        self.pending_keys[(context.frame, chosen)] = (
            key,
            middle_key,
            fine_key,
        )
        self.decisions += 1
        if chosen != greedy:
            self.exploratory_decisions += 1
        return PolicyDecision(chosen, self.name, max(1e-12, probabilities[chosen]))

    def observe(self, outcome) -> None:
        keys = self.pending_keys.pop((outcome.frame, outcome.action), None)
        # A resident controller may span a policy hot reload.  Outcomes made
        # by the pre-latency-filter API have no field yet and retain legacy
        # eligibility; newly constructed outcomes carry the explicit bit.
        if (
            keys is None
            or not outcome.published
            or not getattr(outcome, "learning_eligible", True)
        ):
            return
        key, middle_key, fine_key = keys
        reward = 1.0
        if outcome.life_lost:
            reward -= 100.0
        if outcome.bomb_used:
            reward -= 100.0
        if outcome.control_dead_end:
            reward -= 25.0
        if outcome.authority_lost:
            reward -= 25.0
        if outcome.next_hard_action_count >= 0:
            reward += 0.25 * min(1.0, outcome.next_hard_action_count / 18.0)
        edge = min(
            outcome.next_player_x - 8.0,
            376.0 - outcome.next_player_x,
            outcome.next_player_y - 16.0,
            432.0 - outcome.next_player_y,
        )
        reward += 0.10 * max(0.0, min(1.0, edge / 32.0))
        if outcome.phase_changed and not (
            outcome.life_lost
            or outcome.bomb_used
            or outcome.control_dead_end
            or outcome.authority_lost
        ):
            reward += 5.0
        self.trials[key] += 1
        self.reward_sum[key] += reward
        self.middle_trials[middle_key] += 1
        self.middle_reward_sum[middle_key] += reward
        self.fine_trials[fine_key] += 1
        self.fine_reward_sum[fine_key] += reward

    def export_state(self) -> dict[str, object]:
        state = {
            "schema": STATE_SCHEMA,
            "reward_version": REWARD_VERSION,
            "decisions": self.decisions,
            "exploratory_decisions": self.exploratory_decisions,
            "selected": dict(self.selected),
            "opportunities": dict(self.opportunities),
            "trials": dict(self.trials),
            "reward_sum": dict(self.reward_sum),
            "middle_trials": dict(self.middle_trials),
            "middle_reward_sum": dict(self.middle_reward_sum),
            "fine_selected": dict(self.fine_selected),
            "fine_trials": dict(self.fine_trials),
            "fine_reward_sum": dict(self.fine_reward_sum),
        }
        payload = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            "schema": PACKED_STATE_SCHEMA,
            "codec": "zlib-base64-v1",
            "payload": base64.b64encode(zlib.compress(payload, level=1)).decode(
                "ascii"
            ),
        }

    def metrics(self) -> dict[str, object]:
        return {
            "reward_version": REWARD_VERSION,
            "decisions": self.decisions,
            "exploratory_decisions": self.exploratory_decisions,
            "trained_scope_actions": len(self.trials),
            "trained_middle_scope_actions": len(self.middle_trials),
            "trained_fine_scope_actions": len(self.fine_trials),
            "observed_trials": sum(self.trials.values()),
            "pending": len(self.pending_keys),
        }

    def import_state(self, state: dict[str, object]) -> None:
        state = unpack_state(state)
        if state.get("schema") not in (STATE_SCHEMA, *LEGACY_STATE_SCHEMAS):
            return
        if state.get("reward_version") != REWARD_VERSION:
            return
        self.decisions = max(0, int(state.get("decisions", 0)))
        self.exploratory_decisions = max(
            0, int(state.get("exploratory_decisions", 0))
        )
        for counter, field, cast in (
            (self.selected, "selected", int),
            (self.opportunities, "opportunities", int),
            (self.trials, "trials", int),
            (self.reward_sum, "reward_sum", float),
            (self.middle_trials, "middle_trials", int),
            (self.middle_reward_sum, "middle_reward_sum", float),
            (self.fine_selected, "fine_selected", int),
            (self.fine_trials, "fine_trials", int),
            (self.fine_reward_sum, "fine_reward_sum", float),
        ):
            values = state.get(field, {})
            if isinstance(values, dict):
                counter.update({str(key): cast(value) for key, value in values.items()})
        # V2 exact keys contain the entire middle-key prefix, so their
        # observed statistics can be losslessly aggregated into the new
        # backoff without replaying corpus or resetting any learned value.
        if not self.middle_trials and self.fine_trials:
            for fine_key, trials in self.fine_trials.items():
                middle_key = self._middle_action_key_from_fine(fine_key)
                if middle_key is None:
                    continue
                self.middle_trials[middle_key] += trials
                self.middle_reward_sum[middle_key] += self.fine_reward_sum[
                    fine_key
                ]
        self.random = random.Random(6004)
        for _ in range(self.decisions):
            self.random.random()


def create_policy() -> AdaptivePolicy:
    return AdaptivePolicy()
