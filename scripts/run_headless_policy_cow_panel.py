#!/usr/bin/env python3
"""Run a reject-only, policy-faithful source COW panel for one Wine region."""

from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from th06_rl.headless import HeadlessClient, HeadlessScope
from th06_rl.headless_corpus import (
    canonical_observation_sha256,
    compact_state_features,
)
from th06_rl.headless_forkserver import HeadlessForkserver
from th06_rl.headless_geometry import HeadlessAuthorityUnavailable
from th06_rl.native import NativeKernel
from th06_rl.policies.adaptive import AdaptivePolicy
from th06_rl.policy_api import PolicyContext

try:
    from audit_retail_policy_continuation import (
        RETAIL_DELIVERY_DELAYS,
        _policy_keys,
        _sha256,
        _source_policy_context,
    )
    from audit_targeted_headless_cow import robust_outcome_rank
    from label_headless_cow_counterfactuals import _runtime_provenance
except ModuleNotFoundError:  # Imported as scripts.run_headless_policy_cow_panel.
    from scripts.audit_retail_policy_continuation import (
        RETAIL_DELIVERY_DELAYS,
        _policy_keys,
        _sha256,
        _source_policy_context,
    )
    from scripts.audit_targeted_headless_cow import robust_outcome_rank
    from scripts.label_headless_cow_counterfactuals import _runtime_provenance


SCHEMA = "th06-rl-headless-policy-cow-panel-v1"
TARGET_CONTEXT = "boss:0:sub10:life_cb14:timer_cb13:nonspell"


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def _action_sha256(actions: Sequence[str]) -> str:
    return hashlib.sha256(
        "".join(f"{action}\n" for action in actions).encode("ascii")
    ).hexdigest()


def _boundary_reserve(observation: Mapping[str, Any]) -> float:
    player = observation.get("player")
    if not isinstance(player, Mapping):
        raise HeadlessAuthorityUnavailable("headless player is incoherent")
    x = float(player["x"])
    y = float(player["y"])
    return min(x - 8.0, 376.0 - x, y - 16.0, 432.0 - y)


def is_target_checkpoint(context: PolicyContext) -> bool:
    """Use only automatic context and generic physical bins."""
    reserve = min(
        context.player_x - 8.0,
        376.0 - context.player_x,
        context.player_y - 16.0,
        432.0 - context.player_y,
    )
    return (
        context.source_context == TARGET_CONTEXT
        and 0.0 <= reserve <= 16.0
        and context.bullet_count >= 384
        and context.laser_count == 0
        and len(context.locally_admissible_actions) >= 5
    )


def unique_robust_winner(outcomes: Sequence[Mapping[str, Any]]) -> str | None:
    if not outcomes:
        return None
    best = max(robust_outcome_rank(outcome) for outcome in outcomes)
    winners = sorted(
        str(outcome["first_action"])
        for outcome in outcomes
        if robust_outcome_rank(outcome) == best
    )
    return winners[0] if len(winners) == 1 else None


def _new_policy(state: Mapping[str, Any]) -> AdaptivePolicy:
    policy = AdaptivePolicy()
    policy.import_state(dict(state))
    return policy


def _checkpoint_record(
    *,
    sequence: int,
    observation: Mapping[str, Any],
    context: PolicyContext,
    diagnostics: Mapping[str, Any],
    factual_action: str,
    policy: AdaptivePolicy,
    phase_state: Mapping[str, int | str],
) -> dict[str, Any]:
    evaluations = {
        action: {
            "min_clearance": clearance,
            "final_x": final_x,
            "final_y": final_y,
        }
        for action, clearance, final_x, final_y in context.hard_action_evaluations
        if action in context.locally_admissible_actions
    }
    return {
        "sequence": sequence,
        "tick": int(observation["tick"]),
        "game_frame": int(observation["game_frame"]),
        "observation_sha256": canonical_observation_sha256(observation),
        "source_context": context.source_context,
        "phase_start_frame": int(phase_state["start_frame"]),
        "phase_elapsed_frames": context.phase_elapsed_frames,
        "state": compact_state_features(observation),
        "factual_action": factual_action,
        "baseline_action": context.baseline_action,
        "current_action": context.current_action,
        "hard_actions": list(context.hard_admissible_actions),
        "local_actions": list(context.locally_admissible_actions),
        "local_action_evaluations": evaluations,
        "policy_keys": _policy_keys(policy, context),
        "diagnostics": dict(diagnostics),
    }


def _run_root(
    *,
    binary: Path,
    game_directory: Path,
    scope: HeadlessScope,
    seed: int,
    max_tick: int,
    horizon: int,
    policy_state: Mapping[str, Any],
) -> tuple[dict[str, Any], list[str], list[PolicyContext], dict[str, Any] | None]:
    policy = _new_policy(policy_state)
    kernel = NativeKernel()
    phase_state: dict[str, int | str] = {"context": "", "start_frame": 0}
    actions: list[str] = []
    contexts: list[PolicyContext] = []
    latest: dict[str, Any] | None = None
    termination_reason = "root-error"
    authority_reason = None
    initial_deaths = 0
    terminal_observation: Mapping[str, Any] | None = None
    client = HeadlessClient(
        binary=binary,
        game_directory=game_directory,
        scope=scope,
        seed=seed,
        max_ticks=max_tick,
    )
    try:
        observation = client.start()
        initial_deaths = int(observation["deaths"])
        while observation.get("terminal_reason") is None:
            try:
                context, diagnostics = _source_policy_context(
                    observation,
                    scope=(
                        scope.difficulty,
                        scope.character,
                        scope.shot_type,
                        scope.stage,
                    ),
                    kernel=kernel,
                    phase_state=phase_state,
                    horizon=horizon,
                )
            except HeadlessAuthorityUnavailable as error:
                termination_reason = "authority-failure"
                authority_reason = str(error)
                terminal_observation = observation
                break
            selected = policy.decide(context)
            if selected.action not in context.locally_admissible_actions:
                raise HeadlessAuthorityUnavailable(
                    "frozen policy escaped the local native-safe set"
                )
            contexts.append(context)
            if is_target_checkpoint(context):
                latest = _checkpoint_record(
                    sequence=len(actions),
                    observation=observation,
                    context=context,
                    diagnostics=diagnostics,
                    factual_action=selected.action,
                    policy=policy,
                    phase_state=phase_state,
                )
            actions.append(selected.action)
            observation = client.step(selected.action)
        else:
            terminal_observation = observation
            termination_reason = str(observation["terminal_reason"])
    finally:
        client.close()
    if terminal_observation is None:
        raise RuntimeError("source root produced no terminal observation")
    if len(actions) != len(contexts):
        raise RuntimeError("source root policy/action sequence is inconsistent")
    terminal_tick = int(terminal_observation["tick"])
    root = {
        "seed": seed,
        "split": "development-odd-seeds" if seed % 2 else "confirmation-even-seeds",
        "actions": actions,
        "action_count": len(actions),
        "action_sha256": _action_sha256(actions),
        "termination_reason": termination_reason,
        "authority_reason": authority_reason,
        "terminal_tick": terminal_tick,
        "physical_deaths_delta": int(terminal_observation["deaths"])
        - initial_deaths,
    }
    if latest is not None:
        sequence = int(latest["sequence"])
        tick = int(latest["tick"])
        if sequence != tick - 1:
            raise RuntimeError(
                f"source checkpoint prefix length {sequence} does not match tick {tick}"
            )
    return root, actions, contexts, latest


def _restore_policy_before_checkpoint(
    *,
    policy_state: Mapping[str, Any],
    contexts: Sequence[PolicyContext],
    actions: Sequence[str],
    checkpoint_sequence: int,
) -> tuple[AdaptivePolicy, dict[str, Any]]:
    if checkpoint_sequence >= len(contexts):
        raise ValueError("checkpoint is outside the source policy calls")
    policy = _new_policy(policy_state)
    mismatches = []
    for sequence in range(checkpoint_sequence):
        selected = policy.decide(contexts[sequence])
        if selected.action != actions[sequence]:
            mismatches.append(
                {
                    "sequence": sequence,
                    "recorded": actions[sequence],
                    "replayed": selected.action,
                }
            )
            break
    if mismatches:
        raise ValueError(f"source frozen-policy replay mismatch: {mismatches[0]}")
    return policy, {
        "calls_before_checkpoint": checkpoint_sequence,
        "action_mismatches": mismatches,
    }


def _branch_checkpoint(
    *,
    server: HeadlessForkserver,
    scope: HeadlessScope,
    checkpoint: Mapping[str, Any],
    policy_before_checkpoint: AdaptivePolicy,
    branch_frames: int,
    horizon: int,
    root: Mapping[str, Any],
    root_actions: Sequence[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    kernel = NativeKernel()
    checkpoint_tick = int(checkpoint["tick"])
    checkpoint_sequence = int(checkpoint["sequence"])
    outcomes = []
    for first_action in checkpoint["local_actions"]:
        policy = deepcopy(policy_before_checkpoint)
        phase_state: dict[str, int | str] = {
            "context": str(checkpoint["source_context"]),
            "start_frame": int(checkpoint["phase_start_frame"]),
        }
        session_active = False
        observation = server.begin_step_session(
            terminal_tick=checkpoint_tick + branch_frames
        )
        session_active = True
        try:
            if canonical_observation_sha256(observation) != checkpoint[
                "observation_sha256"
            ]:
                raise ValueError("source COW checkpoint observation mismatch")
            context, _diagnostics = _source_policy_context(
                observation,
                scope=(
                    scope.difficulty,
                    scope.character,
                    scope.shot_type,
                    scope.stage,
                ),
                kernel=kernel,
                phase_state=phase_state,
                horizon=horizon,
            )
            if list(context.locally_admissible_actions) != checkpoint["local_actions"]:
                raise ValueError("source COW checkpoint local action set mismatch")
            direct = policy.decide(context)
            if direct.action != checkpoint["factual_action"]:
                raise ValueError("restored source checkpoint policy action mismatch")
            if str(first_action) not in context.locally_admissible_actions:
                raise ValueError("requested first action escaped the local safe set")
            issued = [str(first_action)]
            checkpoint_deaths = int(observation["deaths"])
            minimum_width = len(context.hard_admissible_actions)
            observation = server.step_session(str(first_action))
            termination_reason = None
            authority_reason = None
            while observation.get("terminal_reason") is None:
                try:
                    context, _diagnostics = _source_policy_context(
                        observation,
                        scope=(
                            scope.difficulty,
                            scope.character,
                            scope.shot_type,
                            scope.stage,
                        ),
                        kernel=kernel,
                        phase_state=phase_state,
                        horizon=horizon,
                    )
                except HeadlessAuthorityUnavailable as error:
                    termination_reason = "authority-failure"
                    authority_reason = str(error)
                    result = server.abort_step_session()
                    session_active = False
                    terminal_observation = result.terminal_observation
                    terminal_tick = int(observation["tick"])
                    terminal_reserve = _boundary_reserve(observation)
                    break
                minimum_width = min(
                    minimum_width, len(context.hard_admissible_actions)
                )
                selected = policy.decide(context)
                issued.append(selected.action)
                observation = server.step_session(selected.action)
            else:
                result = server.finish_step_session()
                session_active = False
                terminal_observation = result.terminal_observation
                termination_reason = str(terminal_observation["terminal_reason"])
                terminal_tick = int(terminal_observation["tick"])
                terminal_reserve = _boundary_reserve(terminal_observation)
            outcomes.append(
                {
                    "first_action": str(first_action),
                    "termination_reason": termination_reason,
                    "authority_reason": authority_reason,
                    "terminal_tick": terminal_tick,
                    "survival_ticks": terminal_tick - checkpoint_tick,
                    "actions_issued": len(issued),
                    "action_sha256": _action_sha256(issued),
                    "first_32_actions": issued[:32],
                    "action_counts": dict(sorted(Counter(issued).items())),
                    "minimum_native_legal_actions": minimum_width,
                    "terminal_boundary_reserve": terminal_reserve,
                    "physical_deaths_delta": int(terminal_observation["deaths"])
                    - checkpoint_deaths,
                    "factual_suffix_matches": (
                        str(first_action) != checkpoint["factual_action"]
                        or issued
                        == list(
                            root_actions[
                                checkpoint_sequence : checkpoint_sequence
                                + len(issued)
                            ]
                        )
                    ),
                }
            )
        finally:
            if session_active:
                server.abort_step_session()

    winner = unique_robust_winner(outcomes)
    factual_outcome = next(
        outcome
        for outcome in outcomes
        if outcome["first_action"] == checkpoint["factual_action"]
    )
    expected_within_horizon = (
        int(root["terminal_tick"]) <= checkpoint_tick + branch_frames
    )
    factual_regression = {
        "action_suffix_matches": factual_outcome["factual_suffix_matches"],
        "root_terminal_within_branch": expected_within_horizon,
        "terminal_matches": (
            not expected_within_horizon
            or (
                factual_outcome["termination_reason"]
                == root["termination_reason"]
                and factual_outcome["terminal_tick"] == root["terminal_tick"]
            )
        ),
    }
    factual_regression["passed"] = all(
        (
            factual_regression["action_suffix_matches"],
            factual_regression["terminal_matches"],
        )
    )
    if not factual_regression["passed"]:
        raise ValueError("source factual COW regression failed")
    return outcomes, {
        "unique_robust_winner": winner,
        "unique_non_incumbent_winner": (
            winner is not None and winner != checkpoint["factual_action"]
        ),
        "factual_regression": factual_regression,
    }


def run_seed(
    *,
    binary: Path,
    game_directory: Path,
    scope: HeadlessScope,
    seed: int,
    max_tick: int,
    branch_frames: int,
    horizon: int,
    policy_state: Mapping[str, Any],
) -> dict[str, Any]:
    root, actions, contexts, checkpoint = _run_root(
        binary=binary,
        game_directory=game_directory,
        scope=scope,
        seed=seed,
        max_tick=max_tick,
        horizon=horizon,
        policy_state=policy_state,
    )
    if checkpoint is None:
        return {"root": root, "checkpoint": None, "outcomes": []}
    sequence = int(checkpoint["sequence"])
    policy, replay = _restore_policy_before_checkpoint(
        policy_state=policy_state,
        contexts=contexts,
        actions=actions,
        checkpoint_sequence=sequence,
    )
    with tempfile.TemporaryDirectory(prefix=f"th06-source-cow-{seed}-") as raw:
        prefix = Path(raw) / "prefix-actions.txt"
        prefix.write_text(
            "".join(f"{action}\n" for action in actions[:sequence]),
            encoding="ascii",
        )
        server = HeadlessForkserver(
            binary=binary,
            game_directory=game_directory,
            scope=scope,
            seed=seed,
        )
        try:
            if server.start() != 1:
                raise ValueError("unexpected source root checkpoint tick")
            server.enter_checkpoint(
                terminal_tick=int(checkpoint["tick"]),
                actions_path=prefix,
            )
            try:
                outcomes, summary = _branch_checkpoint(
                    server=server,
                    scope=scope,
                    checkpoint=checkpoint,
                    policy_before_checkpoint=policy,
                    branch_frames=branch_frames,
                    horizon=horizon,
                    root=root,
                    root_actions=actions,
                )
            finally:
                server.leave_checkpoint()
        finally:
            server.close()
    checkpoint = dict(checkpoint)
    checkpoint["prefix_action_count"] = sequence
    checkpoint["prefix_action_sha256"] = _action_sha256(actions[:sequence])
    checkpoint["policy_restore"] = replay
    checkpoint.update(summary)
    return {"root": root, "checkpoint": checkpoint, "outcomes": outcomes}


def summarize_gate(seeds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    selected = [seed for seed in seeds if seed["checkpoint"] is not None]
    non_incumbent = [
        seed
        for seed in selected
        if seed["checkpoint"]["unique_non_incumbent_winner"] is True
    ]
    selected_by_split = Counter(seed["root"]["split"] for seed in selected)
    winners_by_split = Counter(seed["root"]["split"] for seed in non_incumbent)
    support_passed = (
        selected_by_split["development-odd-seeds"] >= 3
        and selected_by_split["confirmation-even-seeds"] >= 3
        and winners_by_split["development-odd-seeds"] >= 2
        and winners_by_split["confirmation-even-seeds"] >= 2
    )
    return {
        "valid_checkpoints_by_split": dict(sorted(selected_by_split.items())),
        "unique_non_incumbent_winners_by_split": dict(
            sorted(winners_by_split.items())
        ),
        "source_support_gate_passed": support_passed,
        "candidate_fit_unlocked": support_passed,
        "candidate_count": 0,
        "remaining_gate": (
            "fit at most three generic candidates on Wine plus odd seeds, then "
            "require improvement on at least two even seeds and no regression"
            if support_passed
            else "source support insufficient; keep the frozen incumbent"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--policy-state", type=Path, required=True)
    parser.add_argument("--expected-policy-state-sha256", required=True)
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-source-binary-sha256", required=True)
    parser.add_argument("--max-tick", type=int, default=8000)
    parser.add_argument("--branch-frames", type=int, default=600)
    parser.add_argument("--horizon", type=int, default=12)
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
    args = parser.parse_args(argv)
    seeds = tuple(args.seed)
    if (
        not seeds
        or len(seeds) != len(set(seeds))
        or any(seed not in range(1 << 16) for seed in seeds)
        or args.max_tick <= 1
        or args.branch_frames <= 0
        or args.horizon < 4
    ):
        parser.error("seeds or horizon bounds are invalid")
    if args.output.exists():
        parser.error(f"output already exists: {args.output}")
    binary = args.binary.resolve()
    policy_path = args.policy_state.resolve()
    if _sha256(policy_path) != args.expected_policy_state_sha256:
        parser.error("frozen policy state hash mismatch")
    source = _runtime_provenance(binary)
    if (
        source.get("clean") is not True
        or source.get("commit") != args.expected_source_commit
        or source.get("binary_sha256") != args.expected_source_binary_sha256
    ):
        parser.error("source runtime identity mismatch")
    scope = HeadlessScope(3, 0, 0, 6)
    policy_state = _object(policy_path)
    results = []
    for seed in seeds:
        results.append(
            run_seed(
                binary=binary,
                game_directory=args.game_directory.resolve(),
                scope=scope,
                seed=seed,
                max_tick=args.max_tick,
                branch_frames=args.branch_frames,
                horizon=args.horizon,
                policy_state=policy_state,
            )
        )
    report = {
        "schema": SCHEMA,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "authority": "headless-reject-only-original-retail-Wine-promotes",
        "scope": {
            "difficulty": 3,
            "character": 0,
            "shot_type": 0,
            "stage": 6,
        },
        "seeds": list(seeds),
        "split_contract": {
            "development": "odd seeds",
            "confirmation": "even seeds",
        },
        "policy": {
            "id": AdaptivePolicy.name,
            "state": str(policy_path),
            "state_sha256": _sha256(policy_path),
            "exploration_rate": 0.0,
            "observe_suppressed": True,
        },
        "source": source,
        "delivery_delays": list(RETAIL_DELIVERY_DELAYS),
        "max_tick": args.max_tick,
        "branch_frames": args.branch_frames,
        "horizon": args.horizon,
        "selection": {
            "source_context": TARGET_CONTEXT,
            "latest_eligible_per_seed": True,
            "boundary_reserve_max": 16.0,
            "bullet_count_min": 384,
            "laser_count": 0,
            "local_action_count_min": 5,
        },
        "seed_results": results,
        "gate": summarize_gate(results),
        "evidence_boundary": {
            "training_corpus": False,
            "promotion_authority": False,
            "native_gate_unchanged": True,
            "bomb_forbidden": True,
            "wine_shadow_required": True,
            "complete_natural_wine_stage_hit_count_required": True,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "schema": SCHEMA,
                "seeds": list(seeds),
                "valid_checkpoints": sum(
                    result["checkpoint"] is not None for result in results
                ),
                "source_support_gate_passed": report["gate"][
                    "source_support_gate_passed"
                ],
                "output": str(args.output.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
