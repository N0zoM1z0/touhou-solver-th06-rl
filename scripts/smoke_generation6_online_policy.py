#!/usr/bin/env python3
"""Audit the complete Generation-6 policy on factual contexts and Wine."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.advantage_learning import (  # noqa: E402
    _object,
    _rows,
    encode_hazard_set,
    rich_candidate_vector_from_encoding,
    rich_feature_names,
)
from th06_rl.iql_actor_learning import (  # noqa: E402
    actor_population_choice,
    iql_actor_model_from_artifact,
)
from th06_rl.learning_features import tree_candidate_vector  # noqa: E402
from th06_rl.offline import ACTION_NAMES  # noqa: E402
from th06_rl.policy_api import PolicyContext  # noqa: E402
from th06_rl.policies.autonomous_iql_actor import (  # noqa: E402
    AutonomousIqlActorPolicy,
)
from th06_rl.policies.offline_ranker import (  # noqa: E402
    NATIVE_SCORER_ENV,
    PortablePrototypeSupport,
)
from th06_rl.th06.learning_adapter import (  # noqa: E402
    ACTION_FEATURE_NAMES,
    OBSERVATION_FEATURE_NAMES,
)
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _pairs(raw: object) -> tuple[tuple[str, float], ...]:
    if not isinstance(raw, list):
        raise TypeError("online preflight feature vector is absent")
    return tuple((str(name), float(value)) for name, value in raw)


def _context(row: dict[str, object]) -> PolicyContext:
    replay = row.get("policy_context")
    legal_raw = row.get("legal_actions")
    if not isinstance(replay, dict) or not isinstance(legal_raw, list):
        raise TypeError("online preflight boundary lacks adapter context")
    legal = tuple(map(str, legal_raw))
    baseline = str(row.get("baseline_action", ""))
    if not legal or baseline not in legal:
        raise ValueError("online preflight native-safe set is invalid")
    action_raw = replay.get("action_features")
    hazard_raw = replay.get("hazard_primitives")
    if not isinstance(action_raw, list) or not isinstance(hazard_raw, list):
        raise TypeError("online preflight action/hazard input is absent")
    scope = row.get("scope")
    if not isinstance(scope, list) or len(scope) != 4:
        scope = [0, 0, 0, 0]
    return PolicyContext(
        frame=int(row["sequence"]),
        scope=tuple(map(int, scope)),
        source_context="adapter-hidden-from-learner",
        baseline_action=baseline,
        locally_admissible_actions=legal,
        player_x=float(replay.get("player_x", 0.0)),
        player_y=float(replay.get("player_y", 0.0)),
        power=int(replay.get("power", 0)),
        bullet_count=int(replay.get("bullet_count", 0)),
        laser_count=int(replay.get("laser_count", 0)),
        hard_action_count=int(replay.get("hard_action_count", len(legal))),
        exploration_rate=0.0,
        current_action=str(replay.get("current_action", "stay")),
        hard_admissible_actions=tuple(map(
            str, replay.get("hard_admissible_actions", ())
        )),
        phase_elapsed_frames=int(replay.get("phase_elapsed_frames", 0)),
        effort_horizon=int(replay.get("effort_horizon", 0)),
        observation_features=_pairs(replay.get("observation_features")),
        action_features=tuple(
            (str(action), _pairs(features)) for action, features in action_raw
        ),
        hazard_primitives=tuple(
            tuple(map(float, primitive)) for primitive in hazard_raw
        ),
        history_features=_pairs(replay.get("history_features")),
    )


def _factual_contexts(count: int) -> list[PolicyContext]:
    _registry, entries = load_wine_corpus_registry(
        REPOSITORY / "config/wine_corpus_registry.json", repository=REPOSITORY
    )
    eligible = select_wine_corpora(
        entries,
        required_capabilities=frozenset(("sequential_offline_rl",)),
    )
    if not eligible:
        raise RuntimeError("no reusable Wine context source passed capabilities")
    entry = eligible[0]
    manifest = _object(entry.path / "manifest.json")
    result = []
    for row in _rows(
        entry.path, manifest, transition_schema=entry.transition_schema
    ):
        option = row.get("option")
        if isinstance(option, dict) and option.get("boundary") is True:
            result.append(_context(row))
    if len(result) < count:
        raise RuntimeError("factual Wine source has too few option boundaries")
    # This set is a computational-width stress fixture, not statistical
    # gameplay evidence. Select only by input sizes known to dominate runtime;
    # never by action, HIT, phase, RNG, learner score, or outcome.
    result.sort(key=lambda context: (
        len(context.hazard_primitives),
        len(context.locally_admissible_actions),
        context.frame,
    ), reverse=True)
    return sorted(result[:count], key=lambda context: context.frame)


def _portable_choices(
    contexts: list[PolicyContext], candidate: dict[str, object]
) -> list[str]:
    actors = [iql_actor_model_from_artifact(row) for row in candidate["actors"]]
    support = PortablePrototypeSupport(
        candidate["support"], feature_count=len(rich_feature_names())
    )
    threshold = float(candidate["support"]["threshold"])
    factual = frozenset(candidate["support"]["factual_supported_actions"])
    action_index = {action: index for index, action in enumerate(ACTION_NAMES)}
    result = []
    for context in contexts:
        hazard = encode_hazard_set(
            context.hazard_primitives, candidate["representation"]
        )
        history = tuple(value for _name, value in context.history_features)
        legal = tuple(context.locally_admissible_actions)
        rows = [list(rich_candidate_vector_from_encoding(
            tree_candidate_vector(
                observation_features=context.observation_features,
                action_features=context.action_features,
                action=action,
                baseline_action=context.baseline_action,
                current_action=context.current_action,
                observation_names=OBSERVATION_FEATURE_NAMES,
                action_names=ACTION_FEATURE_NAMES,
            ), hazard, history,
        )) for action in legal]
        distances = support.distances(rows, [action_index[action] for action in legal])
        mask = [
            action in factual and distance <= threshold
            for action, distance in zip(legal, distances, strict=True)
        ]
        scores = [actor.predict(rows) for actor in actors]
        mean = [[
            sum(member[index] for member in scores) / len(scores)
            for index in range(len(legal))
        ]]
        result.append(actor_population_choice(
            mean,
            SimpleNamespace(
                legal_actions=legal, baseline_action=context.baseline_action
            ),
            supported=mask,
        ))
    return result


def _native_choices(
    contexts: list[PolicyContext], state: dict[str, object], library: Path,
    *, repetitions: int,
) -> tuple[list[str], dict[str, float | int]]:
    prior = os.environ.get(NATIVE_SCORER_ENV)
    try:
        os.environ[NATIVE_SCORER_ENV] = str(library.resolve())
        policy = AutonomousIqlActorPolicy()
        policy.import_state(state)
    finally:
        if prior is None:
            os.environ.pop(NATIVE_SCORER_ENV, None)
        else:
            os.environ[NATIVE_SCORER_ENV] = prior
    choices = [policy._proposal(
        context, tuple(context.locally_admissible_actions), context.baseline_action
    ) for context in contexts]
    for index in range(repetitions):
        context = contexts[index % len(contexts)]
        policy._proposal(
            context, tuple(context.locally_admissible_actions),
            context.baseline_action,
        )
    metrics = policy.metrics()
    return choices, {
        "latency_p95_ms": float(metrics["latency_p95_ms"]),
        "over_four_ms": int(metrics["over_four_ms"]),
        "deadline_misses": int(metrics["deadline_misses"]),
        "samples": len(policy.timing_ms),
    }


def _windows_path(path: Path) -> str:
    return subprocess.check_output(
        ["winepath", "-w", str(path.resolve())], text=True
    ).strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--linux-state", type=Path, required=True)
    parser.add_argument("--windows-state", type=Path, required=True)
    parser.add_argument("--linux-library", type=Path, required=True)
    parser.add_argument("--windows-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", type=int, default=64)
    parser.add_argument("--repetitions", type=int, default=1200)
    parser.add_argument(
        "--windows-python", type=Path,
        default=REPOSITORY / "reference/tools/windows-python-3.11.9-embed-win32/python.exe",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace online preflight: {args.output}")
    candidate = _object(args.candidate)
    linux_state = _object(args.linux_state)
    contexts = _factual_contexts(args.contexts)
    expected = _portable_choices(contexts, candidate)
    linux, linux_timing = _native_choices(
        contexts, linux_state, args.linux_library,
        repetitions=args.repetitions,
    )
    with tempfile.TemporaryDirectory(prefix="th06-g6-online-") as temporary:
        fixture = Path(temporary) / "fixture.json"
        fixture.write_text(json.dumps({
            "contexts": [asdict(context) for context in contexts]
        }), encoding="utf-8")
        command = [
            "taskset", "-c", "0-31", "wine", str(args.windows_python),
            _windows_path(REPOSITORY / "scripts/windows_generation6_online_policy.py"),
            "--fixture", _windows_path(fixture),
            "--state", _windows_path(args.windows_state),
            "--library", _windows_path(args.windows_library),
            "--repetitions", str(args.repetitions),
        ]
        completed = subprocess.run(
            command, check=True, text=True, capture_output=True,
            env={**os.environ, "WINEDEBUG": "-all"},
        )
    windows = json.loads(completed.stdout.strip().splitlines()[-1])
    gates = {
        "linux_exact_portable_actions": linux == expected,
        "windows_exact_portable_actions": windows["choices"] == expected,
        "linux_p95_below_4_ms": linux_timing["latency_p95_ms"] < 4.0,
        "windows_p95_below_4_ms": windows["latency_p95_ms"] < 4.0,
        "linux_zero_deadline_misses": linux_timing["deadline_misses"] == 0,
        "windows_zero_deadline_misses": windows["deadline_misses"] == 0,
    }
    report = {
        "schema": "autonomous-generation-6-online-policy-preflight-v1",
        "evidence_eligible": False,
        "candidate_sha256": _sha256(args.candidate),
        "linux_state_sha256": _sha256(args.linux_state),
        "windows_state_sha256": _sha256(args.windows_state),
        "linux_library_sha256": _sha256(args.linux_library),
        "windows_library_sha256": _sha256(args.windows_library),
        "factual_contexts": len(contexts),
        "portable_proposals": sum(
            choice != context.baseline_action
            for choice, context in zip(expected, contexts, strict=True)
        ),
        "linux": linux_timing,
        "windows": {key: value for key, value in windows.items() if key != "choices"},
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, sort_keys=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
