from __future__ import annotations

from scripts.audit_retail_replay_cow import summarize_pair_results


def test_retail_replay_pair_gate_requires_unanimous_independent_support() -> None:
    rejected = summarize_pair_results(["tie", "right-better", "left-better"])
    supported = summarize_pair_results(["left-better", "left-better"])

    assert rejected["residual_candidates"] == 0
    assert rejected["conclusion"] == "left-alternative-rejected"
    assert supported["residual_candidates"] == 1
    assert supported["left_alternative_unanimous"] is True
