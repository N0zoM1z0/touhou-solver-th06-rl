from pathlib import Path

from scripts.run_l2l_stage4_action_exposure import (
    derive_episode_policy_state,
    load_prereg,
)


REPOSITORY = Path(__file__).resolve().parents[1]


def test_l2l_preregistration_and_episode_states_are_frozen() -> None:
    prereg = load_prereg(
        REPOSITORY / "experiments/l2l-stage4-action-exposure-v1.json"
    )

    assert prereg["collection"]["exposure_roots"] == 12
    assert prereg["target"]["horizon_unit_frames"] == 12
    assert prereg["collection"]["serial_wine_workers"] == 1
    assert prereg["gate"]["parallel_collection_admitted"] is False
    states = [derive_episode_policy_state(prereg, episode=index) for index in range(2)]
    assert [state["exposure_roots"] for state in states] == [12, 12]
    assert [state["policy_seed"] for state in states] == [
        12754757775496269706,
        17509614397864923417,
    ]
