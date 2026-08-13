from __future__ import annotations

from scripts.run_generation6_decision_successor import (
    CONTRACT_SCHEMA,
    SCHEMA,
)


def test_decision_gameplay_uses_separate_frozen_contract_and_ledger() -> None:
    assert CONTRACT_SCHEMA == "autonomous-generation-6-decision-gameplay-contract-v1"
    assert SCHEMA == "autonomous-generation-6-decision-gameplay-ledger-v1"
