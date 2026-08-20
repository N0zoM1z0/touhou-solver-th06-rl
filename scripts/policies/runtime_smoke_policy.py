"""Infrastructure-only immutable policy that delegates to the native baseline."""

from th06_rl.policy_api import POLICY_API_VERSION, PolicyDecision


class RuntimeSmokePolicy:
    api_version = POLICY_API_VERSION
    name = "runtime-smoke-reactive-baseline-v1"

    def __init__(self) -> None:
        self.decisions = 0

    def import_state(self, state: dict[str, object]) -> None:
        if state != {
            "schema": "th06-rl-runtime-smoke-policy-v1",
            "policy_id": self.name,
            "frozen": True,
        }:
            raise ValueError("runtime smoke policy state identity mismatch")

    def decide(self, context) -> PolicyDecision:
        self.decisions += 1
        return PolicyDecision(context.baseline_action, self.name)

    def metrics(self) -> dict[str, object]:
        return {
            "decisions": self.decisions,
            "purpose": "infrastructure-smoke-only",
        }


def create_policy() -> RuntimeSmokePolicy:
    return RuntimeSmokePolicy()
