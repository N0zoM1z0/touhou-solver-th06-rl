"""Load one hash-identifiable policy and keep it immutable for the run."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import types

from .policy_api import POLICY_API_VERSION, PolicyContext, PolicyDecision


class ImmutablePolicy:
    """One-shot policy loader with no online feedback or reload surface."""

    def __init__(self, path: Path, *, state_path: Path) -> None:
        self.path = path.resolve()
        self.state_path = state_path.resolve()
        source = self.path.read_bytes()
        self.digest = hashlib.sha256(source).hexdigest()
        self.policy_failures = 0
        self.last_error: str | None = None
        self.policy = self._load(source, self._load_state())

    def _load_state(self) -> dict[str, object]:
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("policy state root must be an object")
        return value

    def _load(self, source: bytes, state: dict[str, object]):
        module = types.ModuleType(f"th06_rl.policies._immutable_{self.path.stem}")
        module.__file__ = str(self.path)
        module.__package__ = "th06_rl.policies"
        exec(compile(source, str(self.path), "exec"), module.__dict__)
        factory = getattr(module, "create_policy", None)
        if not callable(factory):
            raise TypeError("policy must export create_policy()")
        policy = factory()
        if getattr(policy, "api_version", None) != POLICY_API_VERSION:
            raise RuntimeError("policy API version mismatch")
        if not callable(getattr(policy, "decide", None)):
            raise TypeError("policy must implement decide(context)")
        restore = getattr(policy, "import_state", None)
        if not callable(restore):
            raise TypeError("immutable policy must implement import_state(state)")
        restore(state)
        return policy

    def decide(self, context: PolicyContext) -> PolicyDecision:
        try:
            decision = self.policy.decide(context)
            if decision.action not in context.locally_admissible_actions:
                raise ValueError(
                    f"policy proposed non-local action {decision.action!r}"
                )
            if set(dict(decision.behavior_probabilities)) != set(
                context.locally_admissible_actions
            ):
                raise ValueError(
                    "policy did not report its complete behavior distribution"
                )
            return decision
        except Exception as error:
            self.policy_failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            return PolicyDecision(
                context.baseline_action,
                "reactive-baseline-policy-error",
                1.0,
                tuple(
                    (action, float(action == context.baseline_action))
                    for action in context.locally_admissible_actions
                ),
            )

    def status(self, *, include_metrics: bool = True) -> dict[str, object]:
        result = {
            "immutable": True,
            "policy_failures": self.policy_failures,
            # Retained report aliases let frozen generation audits read new
            # controller output without restoring a reload mechanism.
            "generation": 1,
            "reloads": 0,
            "reload_failures": self.policy_failures,
            "last_error": self.last_error,
            "sha256": self.digest,
            "policy_id": getattr(self.policy, "name", None),
        }
        if not include_metrics:
            return result
        try:
            metrics = (
                self.policy.metrics()
                if callable(getattr(self.policy, "metrics", None))
                else {}
            )
        except Exception as error:
            metrics = {"error": f"{type(error).__name__}: {error}"}
        return {**result, "metrics": metrics}
