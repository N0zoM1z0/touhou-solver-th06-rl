from __future__ import annotations

from types import SimpleNamespace

from scripts.audit_generation6_native_decisions import (
    DEFAULT_AUDIT_MAXIMUM_SECONDS,
    DEFAULT_AUDIT_WORKERS,
    _Panel,
)


def _case(index: int):
    raw = SimpleNamespace(hazard_primitives=tuple((0.0,) for _ in range(index)))
    sample = SimpleNamespace(
        episode_id=f"episode-{index // 3}",
        option_id=f"option-{index}",
        legal_actions=("stay", "left") if index % 2 else ("stay",),
    )
    rows = ((float(index), -float(index)),)
    return raw, sample, rows, 1.0 / (index + 1)


def _identities(panel: _Panel) -> list[tuple[str, str, tuple[str, ...]]]:
    return [
        (
            selected["case"][1].episode_id,
            selected["case"][1].option_id,
            tuple(sorted(selected["reasons"])),
        )
        for selected in panel.cases()
    ]


def test_parallel_panel_reduction_is_input_order_invariant() -> None:
    cases = [_case(index) for index in range(12)]
    forward = _Panel(3)
    reverse = _Panel(3)
    for target, rows in ((forward, cases), (reverse, reversed(cases))):
        for raw, sample, vectors, margin in rows:
            target.add(
                raw=raw, sample=sample, rows=vectors, margin=margin
            )

    assert _identities(forward) == _identities(reverse)


def test_full_corpus_audit_keeps_bounded_parallelism_as_default() -> None:
    assert DEFAULT_AUDIT_WORKERS == 16
    assert DEFAULT_AUDIT_WORKERS <= 32
    assert DEFAULT_AUDIT_MAXIMUM_SECONDS == 180.0
