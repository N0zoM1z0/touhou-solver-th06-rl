#!/usr/bin/env python3
"""Re-audit immutable L2k episodes after the declared lifecycle-check erratum."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import copy
import json
import multiprocessing
from pathlib import Path
import sys
from typing import Any


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))
if str(REPOSITORY / "src") not in sys.path:
    sys.path.insert(0, str(REPOSITORY / "src"))

from scripts.gate_parallel_wine import _atomic_json, _sha256  # noqa: E402
from th06_rl.action_exposure_audit_erratum import (  # noqa: E402
    audit_episode,
    summarize_action_exposure_audits,
)


SCHEMA = "th06-rl-l2k-action-exposure-audit-erratum-v1"
PREREGISTRATION = REPOSITORY / "experiments/l2k-stage4-action-exposure-v1.json"
ORIGINAL_RESULT = (
    REPOSITORY / "artifacts/l2k-stage4-action-exposure-v1/experiment-result.json"
)
COLLECTION_LEDGER = (
    REPOSITORY / "artifacts/l2k-stage4-action-exposure-v1/collection-ledger.json"
)
OUTPUT = (
    REPOSITORY / "artifacts/l2k-stage4-action-exposure-v1/audit-erratum-v1.json"
)
EXPECTED_PREREGISTRATION_SHA256 = (
    "9e2b2adc540b7e9d8493850ec43948d62c193b66d8f7e57e14403affbdaba94b"
)
EXPECTED_ORIGINAL_RESULT_SHA256 = (
    "f40ec816faff8340ba029f500c6de239239215695e02113909bda481733891fb"
)
EXPECTED_COLLECTION_LEDGER_SHA256 = (
    "6a858752955432bc54368990563185f9fd4ab8b6172babd8082132725cf3999f"
)


def _object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return value


def _audit_kwargs(preregistration: dict[str, Any]) -> dict[str, object]:
    collection = preregistration["collection"]
    gate = preregistration["gate"]
    if "control-dead-end" not in collection["lifecycle_interruptions"]:
        raise ValueError("L2k did not declare control-dead-end interruption")
    return {
        "exposure_roots": int(collection["exposure_roots"]),
        "minimum_complete_groups_per_episode": int(
            gate["minimum_complete_groups_per_episode"]
        ),
        "minimum_assignments_per_action": int(
            gate["minimum_assignments_per_action"]
        ),
        "minimum_no_override_fraction": float(
            gate["minimum_no_override_fraction"]
        ),
        "minimum_four_execution_fraction": float(
            gate["minimum_four_execution_fraction"]
        ),
        "maximum_control_dead_end_rate": float(
            gate["maximum_control_dead_end_rate"]
        ),
        "h16_support_diagnostic_minimum": int(
            gate["h16_support_diagnostic_minimum"]
        ),
    }


def _without_known_erratum_fields(audit: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(audit)
    result.pop("decision", None)
    result.pop("gates", None)
    for episode in result["episodes"]:
        episode.pop("contract_violation_count", None)
        episode.pop("contract_violation_sample", None)
    return result


def run() -> dict[str, object]:
    expected_hashes = {
        PREREGISTRATION: EXPECTED_PREREGISTRATION_SHA256,
        ORIGINAL_RESULT: EXPECTED_ORIGINAL_RESULT_SHA256,
        COLLECTION_LEDGER: EXPECTED_COLLECTION_LEDGER_SHA256,
    }
    for path, expected in expected_hashes.items():
        if _sha256(path) != expected:
            raise ValueError(f"immutable L2k input changed: {path}")

    preregistration = _object(PREREGISTRATION)
    original = _object(ORIGINAL_RESULT)
    ledger = _object(COLLECTION_LEDGER)
    original_audit = original["action_exposure_audit"]
    original_gates = original_audit["gates"]
    if (
        original["decision"] != "reject-action-exposure-collection-contract"
        or original_gates.get("exact_exposure_contract") is not False
        or not all(
            value for name, value in original_gates.items()
            if name != "exact_exposure_contract"
        )
        or [
            int(row["contract_violation_count"])
            for row in original_audit["episodes"]
        ] != [3, 2]
        or any(
            not str(item).endswith(":unexplained-interruption")
            for row in original_audit["episodes"]
            for item in row["contract_violation_sample"]
        )
    ):
        raise ValueError("original L2k failure is not the bounded erratum case")

    run_dirs = [REPOSITORY / row["run_dir"] for row in ledger["episodes"]]
    exposure_roots = int(preregistration["collection"]["exposure_roots"])
    with ProcessPoolExecutor(
        max_workers=len(run_dirs),
        mp_context=multiprocessing.get_context("spawn"),
    ) as executor:
        futures = [
            executor.submit(audit_episode, path, exposure_roots=exposure_roots)
            for path in run_dirs
        ]
        episodes = [future.result() for future in futures]
    corrected = summarize_action_exposure_audits(
        episodes,
        **_audit_kwargs(preregistration),
    )
    if (
        _without_known_erratum_fields(corrected)
        != _without_known_erratum_fields(original_audit)
        or any(int(row["contract_violation_count"]) != 0 for row in episodes)
        or not all(corrected["gates"].values())
        or corrected["decision"] != "proceed-action-exposure-training-collection"
    ):
        raise ValueError("corrected audit changed facts outside the bounded erratum")

    result: dict[str, object] = {
        "schema": SCHEMA,
        "status": "post-hoc-contract-alignment-erratum",
        "original_result_path": ORIGINAL_RESULT.relative_to(REPOSITORY).as_posix(),
        "original_result_sha256": EXPECTED_ORIGINAL_RESULT_SHA256,
        "original_decision": original["decision"],
        "original_result_unchanged": True,
        "collection_ledger_path": COLLECTION_LEDGER.relative_to(REPOSITORY).as_posix(),
        "collection_ledger_sha256": EXPECTED_COLLECTION_LEDGER_SHA256,
        "preregistration_path": PREREGISTRATION.relative_to(REPOSITORY).as_posix(),
        "preregistration_sha256": EXPECTED_PREREGISTRATION_SHA256,
        "auditor_path": "src/th06_rl/action_exposure_audit_erratum.py",
        "auditor_sha256": _sha256(
            REPOSITORY / "src/th06_rl/action_exposure_audit_erratum.py"
        ),
        "episode_loader_sha256": _sha256(
            REPOSITORY / "src/th06_rl/episode_dataset.py"
        ),
        "correction": (
            "accept an incomplete group only when its last collapsed decision "
            "epoch contains a factual outcome.control_dead_end transition, as "
            "already declared by the frozen lifecycle contract"
        ),
        "thresholds_changed": False,
        "episodes_recollected": False,
        "model_fitted": False,
        "learned_policy_run": False,
        "corrected_audit": corrected,
        "corrected_decision": corrected["decision"],
        "pilot_train_data_admitted": True,
        "next_admitted_step": "preregister-stateful-serial-versus-parallel-differential",
    }
    if OUTPUT.is_file():
        if _object(OUTPUT) != result:
            raise ValueError("existing L2k erratum artifact differs")
    else:
        _atomic_json(OUTPUT, result)
    return result


def main() -> int:
    print(json.dumps(run(), indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
