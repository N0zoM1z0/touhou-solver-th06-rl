#!/usr/bin/env python3
"""Replay one recorded oracle closure and differential-test native authority.

This is an offline source-runtime diagnostic.  It never enlarges the resident
safe set: all ordinary actions are tried directly in isolated COW children so
the result can distinguish a source/model mismatch from a deliberately
conservative collision margin or a tested constant-action dead end.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping

try:
    from audit_headless_feasibility_oracle import audit_file
    from label_headless_feasibility_oracle import (
        COMPLETED_TERMINATIONS,
        _load_run,
        _runtime_provenance,
        _write_prefix,
        action_trace_sha256,
    )
except ModuleNotFoundError:
    from scripts.audit_headless_feasibility_oracle import audit_file
    from scripts.label_headless_feasibility_oracle import (
        COMPLETED_TERMINATIONS,
        _load_run,
        _runtime_provenance,
        _write_prefix,
        action_trace_sha256,
    )

from th06_rl.headless import HeadlessScope
from th06_rl.headless_corpus import canonical_observation_sha256
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import (
    COLLISION_MARGIN,
    HARD_HORIZON,
    HEADLESS_DELIVERY_CONTRACT,
    HEADLESS_DELIVERY_DELAYS,
    KINEMATICS,
    HeadlessAuthorityUnavailable,
    action_from_input,
    certify_lowered_headless_actions,
    lower_headless_hard_hazards,
)
from th06_rl.native import ACTIONS, NativeKernel


SCHEMA = "th06-rl-headless-authority-failure-differential-v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expand_action_trace(branch: Mapping[str, Any]) -> list[str]:
    trace = []
    runs = branch.get("action_trace_rle")
    if not isinstance(runs, list):
        raise ValueError("selected branch has no reproducible action trace")
    for run in runs:
        if not isinstance(run, Mapping):
            raise ValueError("selected branch action trace is malformed")
        action = str(run.get("action"))
        ticks = int(run.get("ticks", 0))
        if action not in {candidate.name for candidate in ACTIONS} or ticks <= 0:
            raise ValueError("selected branch action trace is forbidden")
        trace.extend([action] * ticks)
    if len(trace) != int(branch.get("actions_issued", -1)):
        raise ValueError("selected branch action trace length differs from outcome")
    if action_trace_sha256(trace) != branch.get("action_trace_sha256"):
        raise ValueError("selected branch action trace SHA-256 differs")
    return trace


def classify_differential(
    *,
    configured: set[str],
    margin_zero: set[str],
    source_safe: set[str],
    authority_error: str | None = None,
) -> str:
    if authority_error is not None:
        return (
            "source-safe-but-native-observation-incomplete"
            if source_safe
            else "source-immediate-dead-end-under-constant-actions"
        )
    if source_safe - margin_zero or margin_zero - source_safe:
        return "geometry-model-mismatch"
    if source_safe and not configured:
        return "conservative-margin-closure"
    if not source_safe:
        return "source-immediate-dead-end-under-constant-actions"
    return "configured-authority-consistent"


def _find_branch(
    document: Mapping[str, Any],
    *,
    sequence: int,
    first_action: str,
    continuation: str,
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    checkpoints = [
        item for item in document.get("checkpoints", ())
        if isinstance(item, Mapping) and int(item.get("sequence", -1)) == sequence
    ]
    if len(checkpoints) != 1:
        raise ValueError("oracle artifact does not contain exactly one requested checkpoint")
    checkpoint = checkpoints[0]
    branches = [
        item for item in checkpoint.get("branches", ())
        if isinstance(item, Mapping)
        and item.get("first_action") == first_action
        and item.get("continuation") == continuation
    ]
    if len(branches) != 1:
        raise ValueError("oracle artifact does not contain exactly one requested branch")
    return checkpoint, branches[0]


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("oracle", type=Path)
    parser.add_argument("--checkpoint-sequence", type=int, required=True)
    parser.add_argument("--first-action", required=True)
    parser.add_argument("--continuation", required=True)
    parser.add_argument("--horizon", type=int, default=HARD_HORIZON)
    parser.add_argument(
        "--binary",
        type=Path,
        default=root / "reference/GensokyoClub-th06-portable/th06",
    )
    parser.add_argument(
        "--game-directory",
        type=Path,
        default=root / "reference/th06-game-original/th06",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.horizon != HARD_HORIZON:
        parser.error(f"failure differential horizon must equal Hard ({HARD_HORIZON})")

    oracle = args.oracle.resolve()
    audited = audit_file(oracle)
    if not audited["valid"]:
        parser.error("oracle artifact fails structural audit: " + "; ".join(audited["errors"]))
    document = json.loads(oracle.read_text(encoding="utf-8"))
    checkpoint, branch = _find_branch(
        document,
        sequence=args.checkpoint_sequence,
        first_action=args.first_action,
        continuation=args.continuation,
    )
    if branch.get("termination_reason") != "authority-failure":
        parser.error("selected branch did not terminate on authority failure")
    trace = expand_action_trace(branch)

    run = Path(str(document["input_run"])).resolve()
    manifest, rows = _load_run(run)
    row = rows[args.checkpoint_sequence]
    if row.get("observation_sha256") != checkpoint.get("observation_sha256"):
        parser.error("oracle checkpoint and input corpus observation differ")
    scope_data = document["scope"]
    scope = HeadlessScope(
        int(scope_data["difficulty"]),
        int(scope_data["character"]),
        int(scope_data["shot_type"]),
        int(scope_data["stage"]),
    )
    binary = args.binary.resolve()
    runtime_source = _runtime_provenance(binary)
    for name in ("commit", "binary_sha256"):
        if runtime_source.get(name) != document.get("runtime_source", {}).get(name):
            parser.error("current authoritative runtime differs from oracle runtime")

    kernel = NativeKernel()
    source_outcomes = []
    with tempfile.TemporaryDirectory(prefix="th06-authority-differential-") as raw:
        workspace = Path(raw)
        actions_path = workspace / "terminal-prefix.txt"
        _write_prefix(actions_path, rows, args.checkpoint_sequence)
        with actions_path.open("a", encoding="utf-8") as stream:
            stream.write("".join(f"{action}\n" for action in trace))
        terminal_tick = int(branch["terminal_tick"])
        server = HeadlessForkserver(
            binary=binary,
            game_directory=args.game_directory.resolve(),
            scope=scope,
            seed=int(manifest["initial_seed"]),
        )
        try:
            if server.start() != 1:
                raise ValueError("unexpected stage-entry checkpoint tick")
            server.enter_checkpoint(terminal_tick=terminal_tick, actions_path=actions_path)
            observation = server.begin_step_session(terminal_tick=terminal_tick + 1)
            observed_digest = canonical_observation_sha256(observation)
            if observed_digest != branch.get("terminal_observation_sha256"):
                server.abort_step_session()
                raise ValueError("authority-failure terminal fingerprint is not reproducible")
            authority_error = None
            try:
                hazards = lower_headless_hard_hazards(observation, HARD_HORIZON)
                prepared = kernel.prepare_hazards(hazards)
                configured = tuple(
                    item.action.name
                    for item in certify_lowered_headless_actions(
                        observation, prepared, kernel=kernel
                    )
                )
                player = observation["player"]
                if not isinstance(player, Mapping):
                    raise ValueError("authority-failure player is incoherent")
                margin_zero = tuple(
                    item.action.name
                    for item in kernel.certify_actions(
                        x=float(player["x"]),
                        y=float(player["y"]),
                        half_width=float(player["half_width"]),
                        half_height=float(player["half_height"]),
                        kinematics=KINEMATICS,
                        current_action=action_from_input(int(observation["input"])),
                        hazards=prepared,
                        delivery_delays=HEADLESS_DELIVERY_DELAYS,
                        collision_margin=0.0,
                    )
                )
            except HeadlessAuthorityUnavailable as error:
                authority_error = str(error)
                configured = ()
                margin_zero = ()
            # Close the fingerprint child without treating its deliberate
            # diagnostic read as an authority failure.
            server.step_session(ACTIONS[0].name)
            server.finish_step_session()

            initial_deaths = int(observation["deaths"])
            initial_bombs = int(observation["bombs_used"])
            for action in ACTIONS:
                trial = server.begin_step_session(
                    terminal_tick=terminal_tick + HARD_HORIZON
                )
                issued = 0
                while trial.get("terminal_reason") is None:
                    trial = server.step_session(action.name)
                    issued += 1
                result = server.finish_step_session()
                terminal = result.terminal_observation
                deaths_delta = int(terminal["deaths"]) - initial_deaths
                bombs_delta = int(terminal["bombs_used"]) - initial_bombs
                termination = str(terminal["terminal_reason"])
                source_safe = (
                    termination in COMPLETED_TERMINATIONS
                    and deaths_delta == 0
                    and bombs_delta == 0
                )
                source_outcomes.append({
                    "action": action.name,
                    "source_safe": source_safe,
                    "termination_reason": termination,
                    "ticks": int(terminal["tick"]) - terminal_tick,
                    "actions_issued": issued,
                    "physical_deaths_delta": deaths_delta,
                    "bombs_used_delta": bombs_delta,
                })
            server.leave_checkpoint()
        finally:
            server.close()

    source_safe = tuple(
        item["action"] for item in source_outcomes if item["source_safe"]
    )
    classification = classify_differential(
        configured=set(configured),
        margin_zero=set(margin_zero),
        source_safe=set(source_safe),
        authority_error=authority_error,
    )
    result = {
        "schema": SCHEMA,
        "interpretation": (
            "offline isolated COW source differential; results do not enlarge "
            "resident collision authority"
        ),
        "oracle": {"path": str(oracle), "sha256": _sha256(oracle)},
        "runtime_source": runtime_source,
        "scope": scope_data,
        "initial_seed": manifest["initial_seed"],
        "checkpoint_sequence": args.checkpoint_sequence,
        "branch": {
            "first_action": args.first_action,
            "continuation": args.continuation,
            "action_trace_sha256": branch["action_trace_sha256"],
            "terminal_tick": branch["terminal_tick"],
            "terminal_observation_sha256": branch["terminal_observation_sha256"],
        },
        "hard_horizon": HARD_HORIZON,
        "runtime_delivery_contract": HEADLESS_DELIVERY_CONTRACT,
        "runtime_delivery_delays": list(HEADLESS_DELIVERY_DELAYS),
        "configured_collision_margin": COLLISION_MARGIN,
        "native_authority_error": authority_error,
        "native_comparison_available": authority_error is None,
        "configured_native_actions": list(configured),
        "margin_zero_native_actions": list(margin_zero),
        "source_safe_constant_actions": list(source_safe),
        "configured_false_negatives": (
            sorted(set(source_safe) - set(configured)) if authority_error is None else []
        ),
        "configured_false_positives": sorted(set(configured) - set(source_safe)),
        "margin_zero_false_negatives": (
            sorted(set(source_safe) - set(margin_zero)) if authority_error is None else []
        ),
        "margin_zero_false_positives": sorted(set(margin_zero) - set(source_safe)),
        "classification": classification,
        "source_outcomes": source_outcomes,
    }
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
