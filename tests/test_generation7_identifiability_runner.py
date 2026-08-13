from pathlib import Path


def test_generation7_contract_forbids_wine_and_new_collection() -> None:
    import json

    contract = json.loads(
        (Path(__file__).parents[1] / "config/generation7_offline_contract.json")
        .read_text(encoding="utf-8")
    )
    assert contract["wine_outcome_facing_authorized"] is False
    assert contract["new_collection_authorized"] is False
    assert contract["concurrent_wine_collection_authorized"] is False
    assert (
        contract["treatment_unit"]
        == "randomized-proposal-assignment-intention-to-treat"
    )
    assert (
        contract["post_assignment_native_revalidation"]
        == "factual-deployment-kernel-not-a-filter"
    )
    assert "fixed-physical-time-or-semi-markov-value" in contract["required_gates"]
    assert contract["cost"] == {
        "name": "physical_hit_count",
        "gamma": 1.0,
        "terminal_value": 0.0,
        "reward_shaping": False,
    }
