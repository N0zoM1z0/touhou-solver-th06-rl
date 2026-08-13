"""TH105-style last-known-good hot reload with restart checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
import types

from .policy_api import (
    POLICY_API_VERSION,
    PolicyContext,
    PolicyDecision,
    PolicyFailureEvent,
    PolicyOutcome,
)


class HotReloadPolicy:
    def __init__(
        self,
        path: Path,
        *,
        state_path: Path | None = None,
        check_interval_frames: int = 30,
        immutable: bool = False,
    ) -> None:
        if check_interval_frames <= 0:
            raise ValueError("check interval must be positive")
        self.path = path.resolve()
        self.state_path = state_path.resolve() if state_path else None
        self.check_interval_frames = check_interval_frames
        self.immutable = immutable
        self.policy = None
        self.digest: str | None = None
        self.signature: tuple[int, int] | None = None
        self.failed_signature: tuple[int, int] | None = None
        self.failed_digest: str | None = None
        self.generation = 0
        self.reloads = 0
        self.reload_failures = 0
        self.last_error: str | None = None
        self._rollback: tuple[
            object,
            str | None,
            tuple[int, int] | None,
        ] | None = None
        self._restart_state = self._load_restart_state()
        self.maybe_reload(0, force=True)

    def _load_restart_state(self) -> dict[str, object]:
        if self.state_path is None or not self.state_path.is_file():
            return {}
        value = json.loads(self.state_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("policy checkpoint root must be an object")
        return value

    def _load(self, source: bytes):
        self.generation += 1
        module = types.ModuleType(
            f"th06_rl.policies._hot_{self.path.stem}_{self.generation}"
        )
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
        state = self._restart_state
        if self.policy is not None:
            export = getattr(self.policy, "export_state", None)
            if callable(export):
                state = export()
        restore = getattr(policy, "import_state", None)
        if state and callable(restore):
            restore(state)
        return policy

    def maybe_reload(self, frame: int, *, force: bool = False) -> bool:
        if self.immutable and not force:
            return False
        if not force and frame % self.check_interval_frames:
            return False
        try:
            stat = self.path.stat()
            signature = (stat.st_mtime_ns, stat.st_size)
            if not force and signature in (
                self.signature,
                self.failed_signature,
            ):
                return False
            source = self.path.read_bytes()
            digest = hashlib.sha256(source).hexdigest()
            if not force and digest == self.digest:
                self.signature = signature
                return False
            if not force and digest == self.failed_digest:
                self.failed_signature = signature
                return False
            policy = self._load(source)
        except Exception as error:
            self.reload_failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            if "digest" in locals():
                self.failed_digest = digest
            if "signature" in locals():
                self.failed_signature = signature
            if self.policy is None:
                raise RuntimeError(
                    f"initial policy load failed: {self.last_error}"
                ) from error
            return False
        self._rollback = (
            (self.policy, self.digest, self.signature)
            if self.policy is not None else None
        )
        self.policy = policy
        self.digest = digest
        self.signature = signature
        self.failed_digest = None
        self.failed_signature = None
        self.reloads += 1
        self.last_error = None
        return True

    def decide(self, context: PolicyContext) -> PolicyDecision:
        # Never touch the policy source path in the frame-critical decision
        # call.  In the Windows deployment the repository is reached through
        # WSL's UNC provider, where even an unchanged stat can cost most of a
        # TH06 frame.  checkpoint() polls and atomically activates changes at
        # a low-frequency boundary instead.
        assert self.policy is not None
        try:
            decision = self.policy.decide(context)
            if decision.action not in context.locally_admissible_actions:
                raise ValueError(
                    f"policy proposed non-local action {decision.action!r}"
                )
            return decision
        except Exception as error:
            self.reload_failures += 1
            self.last_error = f"{type(error).__name__}: {error}"
            self.failed_digest = self.digest
            if self._rollback is not None:
                self.policy, self.digest, self.signature = self._rollback
                self._rollback = None
                try:
                    fallback = self.policy.decide(context)
                    if fallback.action not in context.locally_admissible_actions:
                        raise ValueError(
                            "rollback proposed a non-local action"
                        )
                    return fallback
                except Exception as fallback_error:
                    self.last_error += (
                        "; rollback "
                        f"{type(fallback_error).__name__}: {fallback_error}"
                    )
            return PolicyDecision(
                context.baseline_action,
                "reactive-baseline-policy-error",
            )

    def continue_certified(
        self, context: PolicyContext
    ) -> PolicyDecision | None:
        """Let an interested policy trace a freshly certified input lease."""
        assert self.policy is not None
        callback = getattr(self.policy, "continue_certified", None)
        if not callable(callback):
            return None
        try:
            decision = callback(context)
            if (
                not isinstance(decision, PolicyDecision)
                or decision.action not in context.locally_admissible_actions
            ):
                raise ValueError("policy continued outside the certified lease")
            return decision
        except Exception as error:
            self.reload_failures += 1
            self.last_error = (
                f"continue_certified {type(error).__name__}: {error}"
            )
            return None

    def reject_publication(self, decision: PolicyDecision) -> None:
        """Abort tentative operational policy state after no input was issued.

        This callback remains available for immutable policies because it is
        action-delivery bookkeeping, not learner feedback or a checkpoint
        mutation.
        """
        if self.policy is None:
            return
        callback = getattr(self.policy, "reject_publication", None)
        if callable(callback):
            callback(decision)

    def checkpoint(self) -> bool:
        if self.immutable:
            return False
        # Hot reload is deliberately coupled to this low-frequency durability
        # boundary, not to game-frame modulo checks in decide().
        self.maybe_reload(0)
        if self.state_path is None or self.policy is None:
            return False
        export = getattr(self.policy, "export_state", None)
        if not callable(export):
            return False
        state = export()
        if not isinstance(state, dict):
            raise TypeError("policy export_state() must return a dict")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(
            state,
            sort_keys=True,
            separators=(",", ":"),
        ) + "\n"
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.state_path.name}.",
            suffix=".tmp",
            dir=self.state_path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, self.state_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return True

    def observe(self, outcome: PolicyOutcome) -> None:
        if self.immutable:
            return
        if self.policy is None:
            return
        callback = getattr(self.policy, "observe", None)
        if not callable(callback):
            return
        try:
            callback(outcome)
        except Exception as error:
            # Learning must not take input authority away from the already
            # certified reactive action. Keep the last-good runtime alive and
            # surface the failure in status/checkpoint telemetry.
            self.reload_failures += 1
            self.last_error = f"observe {type(error).__name__}: {error}"

    def observe_failure(self, event: PolicyFailureEvent) -> None:
        """Deliver sparse physical feedback outside the action publish path."""
        if self.immutable:
            return
        if self.policy is None:
            return
        callback = getattr(self.policy, "observe_failure", None)
        if not callable(callback):
            return
        try:
            callback(event)
        except Exception as error:
            # Failure credit is advisory learning state. It must never take
            # input authority from the native gate or stop a continuous run.
            self.reload_failures += 1
            self.last_error = (
                f"observe_failure {type(error).__name__}: {error}"
            )

    def status(self, *, include_metrics: bool = True) -> dict[str, object]:
        result = {
            "immutable": self.immutable,
            "generation": self.generation,
            "reloads": self.reloads,
            "reload_failures": self.reload_failures,
            "last_error": self.last_error,
            "sha256": self.digest,
            "policy_id": getattr(self.policy, "name", None),
        }
        if not include_metrics:
            return result
        try:
            metrics = (
                self.policy.metrics()
                if self.policy is not None
                and callable(getattr(self.policy, "metrics", None))
                else {}
            )
        except Exception as error:
            metrics = {"error": f"{type(error).__name__}: {error}"}
        return {**result, "metrics": metrics}
