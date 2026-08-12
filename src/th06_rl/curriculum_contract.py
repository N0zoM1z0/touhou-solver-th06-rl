"""Validation for the outcome-independent Generation-5 curriculum schedule."""

from __future__ import annotations

import json
from pathlib import Path


CURRICULUM_SCHEMA = "autonomous-generation-5-curriculum-seeds-v1"
EXPECTED_STAGES = {
    4: {"episodes": 20, "fits": (10, 15, 20), "canaries": (20,)},
    5: {"episodes": 12, "fits": (8, 12), "canaries": (8, 12)},
    6: {"episodes": 16, "fits": (8, 12, 16), "canaries": (8, 12, 16)},
}


def load_curriculum_schedule(path: Path) -> dict[str, object]:
    schedule = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(schedule, dict):
        raise TypeError("curriculum schedule root is not an object")
    resources = schedule.get("resource_contract")
    stages = schedule.get("stages")
    differential = schedule.get("parallelism_differential")
    final = schedule.get("final_evaluation")
    if (
        schedule.get("schema") != CURRICULUM_SCHEMA
        or schedule.get("generation_seed") != 260_813
        or not isinstance(resources, dict)
        or resources.get("maximum_training_threads") != 32
        or resources.get("collection_parallelism") != 4
        or resources.get("canonical_evaluation_parallelism") != 1
        or resources.get("fit_and_collection_may_overlap") is not False
        or not isinstance(stages, list)
        or [row.get("stage") for row in stages if isinstance(row, dict)] != [4, 5, 6]
        or not isinstance(differential, dict)
        or differential.get("evidence_eligible") is not False
        or differential.get("stage") != 4
        or differential.get("serial_worker") != 0
        or differential.get("concurrent_workers") != [0, 1]
        or not isinstance(final, dict)
        or final.get("stage") != 6
        or final.get("fixed_rng") is not False
        or final.get("complete_stage_hit_continuation") is not True
        or final.get("pairs") != 12
        or final.get("trial_order") != ["baseline", "candidate"] * 12
    ):
        raise ValueError("Generation-5 curriculum top-level contract differs")
    workers = resources.get("workers")
    expected_workers = [
        {"worker": index, "display": f":{97 + index}", "directory": f"wine-{index}"}
        for index in range(4)
    ]
    if not isinstance(workers, list) or workers != expected_workers:
        raise ValueError("Generation-5 Wine worker assignment differs")

    game_seeds = [int(differential["game_rng_seed"])]
    policy_seeds = [int(differential["policy_seed"])]
    for row in stages:
        if not isinstance(row, dict):
            raise TypeError("curriculum stage row is not an object")
        stage = int(row["stage"])
        expected = EXPECTED_STAGES[stage]
        collection = row.get("collection")
        fits = row.get("fits")
        canary = row.get("canary")
        if (
            tuple(row.get("fit_boundaries", ())) != expected["fits"]
            or tuple(row.get("canary_boundaries", ())) != expected["canaries"]
            or not isinstance(collection, list)
            or len(collection) != expected["episodes"]
            or [item.get("episode") for item in collection if isinstance(item, dict)]
            != list(range(expected["episodes"]))
            or [item.get("worker") for item in collection if isinstance(item, dict)]
            != [index % 4 for index in range(expected["episodes"])]
            or not isinstance(fits, list)
            or tuple(item.get("boundary") for item in fits if isinstance(item, dict))
            != expected["fits"]
            or not isinstance(canary, list)
            or [
                (item.get("boundary"), item.get("pair"))
                for item in canary if isinstance(item, dict)
            ]
            != [
                (boundary, pair)
                for boundary in expected["canaries"] for pair in range(3)
            ]
        ):
            raise ValueError(f"Stage-{stage} curriculum schedule differs")
        modes = [item.get("mode") for item in fits]
        if stage == 4:
            if modes != [
                "non-authorizing-smoke", "non-authorizing-smoke", "production"
            ]:
                raise ValueError("Stage-4 smoke/production boundary modes differ")
            for item in fits[:2]:
                parameters = (
                    item.get("iterations"), item.get("q_trees"), item.get("value_trees")
                )
                if parameters != (2, 8, 8):
                    raise ValueError("Stage-4 smoke model budget differs")
        elif modes != ["production"] * len(fits):
            raise ValueError(f"Stage-{stage} requires production fits")
        game_seeds.extend(int(item["game_rng_seed"]) for item in collection)
        game_seeds.extend(int(item["game_rng_seed"]) for item in canary)
        policy_seeds.extend(int(item["policy_seed"]) for item in collection)
    all_seeds = [*game_seeds, *policy_seeds]
    if (
        len(game_seeds) != len(set(game_seeds))
        or len(policy_seeds) != len(set(policy_seeds))
        or any(not 0 < value < 2**16 for value in all_seeds)
    ):
        raise ValueError("Generation-5 curriculum seeds are not unique and bounded")
    return schedule
