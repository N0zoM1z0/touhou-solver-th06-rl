"""Portable sign-consensus forecast gate for a Generation-7 actor ensemble."""

from __future__ import annotations

import math

from th06_rl.g7_learner import LINEAR_ACTOR_SCHEMA, linear_actor_scores
from th06_rl.offline_options import ActorState


FORECAST_SCHEMA = "th06-rl-g7-linear-sign-consensus-v1"


def build_forecast_artifact(
    actors,
    *,
    minimum_score_advantage: float = 0.0,
    required_vote_fraction: float = 1.0,
) -> dict[str, object]:
    """Freeze independently fitted actors into a bounded online gate."""
    rows = tuple(actors)
    if (
        len(rows) < 3
        or any(
            not isinstance(actor, dict)
            or actor.get("schema") != LINEAR_ACTOR_SCHEMA
            for actor in rows
        )
        or not math.isfinite(minimum_score_advantage)
        or minimum_score_advantage < 0.0
        or not math.isfinite(required_vote_fraction)
        or not 0.5 < required_vote_fraction <= 1.0
    ):
        raise ValueError("forecast ensemble contract is invalid")
    return {
        "schema": FORECAST_SCHEMA,
        "actors": list(rows),
        "members": len(rows),
        "minimum_score_advantage": minimum_score_advantage,
        "required_vote_fraction": required_vote_fraction,
    }


def forecast_accepted_actions(
    artifact: dict[str, object],
    state: ActorState,
    *,
    supported_actions,
) -> tuple[str, ...]:
    """Accept a deviation only when the declared actor quorum beats baseline."""
    actors = artifact.get("actors")
    try:
        minimum = float(artifact.get("minimum_score_advantage"))
        required = float(artifact.get("required_vote_fraction"))
        members = int(artifact.get("members"))
    except (TypeError, ValueError) as error:
        raise ValueError("forecast numeric artifact is invalid") from error
    if (
        artifact.get("schema") != FORECAST_SCHEMA
        or not isinstance(actors, list)
        or members != len(actors)
        or members < 3
        or not math.isfinite(minimum)
        or minimum < 0.0
        or not math.isfinite(required)
        or not 0.5 < required <= 1.0
    ):
        raise ValueError("forecast artifact contract mismatch")
    supported = set(map(str, supported_actions)) & set(state.legal_actions)
    baseline = state.baseline_action
    if baseline not in supported:
        return ()
    score_rows = []
    for actor in actors:
        if not isinstance(actor, dict):
            raise ValueError("forecast actor artifact is malformed")
        scores = dict(linear_actor_scores(actor, state))
        if set(scores) != set(state.legal_actions):
            raise ValueError("forecast actor does not score the exact legal set")
        score_rows.append(scores)
    accepted = [baseline]
    required_votes = math.ceil(required * members - 1e-12)
    for action in sorted(supported - {baseline}):
        votes = sum(
            scores[action] - scores[baseline] >= minimum
            for scores in score_rows
        )
        if votes >= required_votes:
            accepted.append(action)
    return tuple(sorted(accepted))
