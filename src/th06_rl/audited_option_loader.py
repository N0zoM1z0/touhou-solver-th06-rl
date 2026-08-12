"""Stable source contract for loading factual Generation-3/4 Wine options."""

from __future__ import annotations

from pathlib import Path

from .advantage_learning import (
    BEHAVIOR_POLICY as GENERATION3_POLICY,
    TRANSITION_SCHEMA as GENERATION3_TRANSITION,
    _object,
    load_option_episode,
)
from .sequential_learning import (
    BEHAVIOR_POLICY as GENERATION4_POLICY,
    TRANSITION_SCHEMA as GENERATION4_TRANSITION,
)


_PACKAGE = Path(__file__).resolve().parent
AUDITED_OPTION_LOADER_CONTRACT = (
    _PACKAGE / "advantage_learning.py",
    _PACKAGE / "autonomous_learning.py",
    _PACKAGE / "corpus.py",
    Path(__file__).resolve(),
)


def load_audited_option_episode(run_dir: Path):
    """Load only factual option schemas accepted by Generation 5."""
    run = _object(run_dir / "run.json")
    schemas = run.get("schemas")
    transition = schemas.get("transition") if isinstance(schemas, dict) else None
    if transition == GENERATION3_TRANSITION:
        return load_option_episode(
            run_dir,
            exploration_probability=0.10,
            behavior_policy=GENERATION3_POLICY,
            transition_schema=GENERATION3_TRANSITION,
        )
    if transition == GENERATION4_TRANSITION:
        return load_option_episode(
            run_dir,
            exploration_probability=None,
            behavior_policy=GENERATION4_POLICY,
            transition_schema=GENERATION4_TRANSITION,
        )
    raise ValueError(f"unsupported implicit-Q Wine corpus schema: {transition}")
