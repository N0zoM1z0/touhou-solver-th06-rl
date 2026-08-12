from pathlib import Path

from th06_rl.curriculum_contract import load_curriculum_schedule


def test_generation5_curriculum_schedule_is_frozen_and_valid() -> None:
    repository = Path(__file__).resolve().parents[1]

    schedule = load_curriculum_schedule(
        repository / "config/autonomous_generation5_curriculum_seeds.json"
    )

    assert [row["stage"] for row in schedule["stages"]] == [4, 5, 6]
    assert schedule["resource_contract"]["maximum_training_threads"] == 32
