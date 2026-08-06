"""Phase-isolated online bandit above native/local safety gates."""

from __future__ import annotations

from collections import Counter
import math
import random

from ..policy_api import POLICY_API_VERSION, PolicyDecision


STATE_SCHEMA = "th06-rl-online-ucb-v1"
REWARD_VERSION = "survival-reserve-v1"


class AdaptivePolicy:
    api_version = POLICY_API_VERSION
    name = "phase-local-ucb-v1"

    def __init__(self) -> None:
        self.random = random.Random(6004)
        self.decisions = 0
        self.exploratory_decisions = 0
        self.selected: Counter[str] = Counter()
        self.opportunities: Counter[str] = Counter()
        self.trials: Counter[str] = Counter()
        self.reward_sum: Counter[str] = Counter()
        self.pending_keys: dict[tuple[int, str], str] = {}

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
    def _action_key(context_key: str, action: str) -> str:
        return f"{context_key}|action:{action}"

    def decide(self, context):
        legal = tuple(sorted(set(context.locally_admissible_actions)))
        if not legal:
            raise ValueError("policy received no locally admissible actions")
        if context.baseline_action not in legal:
            raise ValueError("reactive baseline is outside the local safe set")
        context_key = self._context_key(context)
        total_trials = sum(
            self.trials[self._action_key(context_key, action)] for action in legal
        )
        scores = {}
        for action in legal:
            key = self._action_key(context_key, action)
            trials = self.trials[key]
            empirical = self.reward_sum[key] / trials if trials else 0.0
            optimism = (
                0.12 * math.sqrt(math.log(max(2, total_trials + 2)) / trials)
                if trials
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
                    1.0 + self.selected[self._action_key(context_key, action)]
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
        self.selected[key] += 1
        self.pending_keys[(context.frame, chosen)] = key
        self.decisions += 1
        if chosen != greedy:
            self.exploratory_decisions += 1
        return PolicyDecision(chosen, self.name, max(1e-12, probabilities[chosen]))

    def observe(self, outcome) -> None:
        key = self.pending_keys.pop((outcome.frame, outcome.action), None)
        if key is None or not outcome.published:
            return
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

    def export_state(self) -> dict[str, object]:
        return {
            "schema": STATE_SCHEMA,
            "reward_version": REWARD_VERSION,
            "decisions": self.decisions,
            "exploratory_decisions": self.exploratory_decisions,
            "selected": dict(self.selected),
            "opportunities": dict(self.opportunities),
            "trials": dict(self.trials),
            "reward_sum": dict(self.reward_sum),
        }

    def metrics(self) -> dict[str, object]:
        return {
            "reward_version": REWARD_VERSION,
            "decisions": self.decisions,
            "exploratory_decisions": self.exploratory_decisions,
            "trained_scope_actions": len(self.trials),
            "observed_trials": sum(self.trials.values()),
            "pending": len(self.pending_keys),
        }

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") != STATE_SCHEMA:
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
        ):
            values = state.get(field, {})
            if isinstance(values, dict):
                counter.update({str(key): cast(value) for key, value in values.items()})
        self.random = random.Random(6004)
        for _ in range(self.decisions):
            self.random.random()


def create_policy() -> AdaptivePolicy:
    return AdaptivePolicy()
