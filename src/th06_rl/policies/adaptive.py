"""Cold-start policy: preserve the local baseline while collecting evidence."""

from __future__ import annotations

from ..policy_api import POLICY_API_VERSION, PolicyDecision


class AdaptivePolicy:
    api_version = POLICY_API_VERSION
    name = "reactive-coldstart-v1"

    def __init__(self) -> None:
        self.decisions = 0

    def decide(self, context):
        self.decisions += 1
        return PolicyDecision(
            context.baseline_action,
            self.name,
            1.0,
        )

    def export_state(self) -> dict[str, object]:
        return {"schema": "th06-rl-online-v1", "decisions": self.decisions}

    def import_state(self, state: dict[str, object]) -> None:
        if state.get("schema") == "th06-rl-online-v1":
            self.decisions = int(state.get("decisions", 0))


def create_policy() -> AdaptivePolicy:
    return AdaptivePolicy()

