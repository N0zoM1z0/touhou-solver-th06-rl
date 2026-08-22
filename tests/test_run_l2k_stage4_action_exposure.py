from pathlib import Path

from scripts.run_l2k_stage4_action_exposure import (
    derive_episode_policy_state,
    load_prereg,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_l2k_preregistration_and_episode_states_are_frozen() -> None:
    prereg = load_prereg(
        REPOSITORY / "experiments/l2k-stage4-action-exposure-v1.json"
    )

    assert prereg["collection"]["exposure_roots"] == 4
    assert prereg["collection"]["serial_wine_workers"] == 1
    assert prereg["gate"]["h16_support_is_retry_or_contract_gate"] is False
    assert [
        derive_episode_policy_state(prereg, episode=index)["policy_seed"]
        for index in range(2)
    ] == [13326741063985372124, 5175935992713083027]
