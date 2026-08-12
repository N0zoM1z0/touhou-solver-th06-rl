#!/usr/bin/env python3
"""Export the frozen Generation-6 actor into an immutable online state."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import sys
import zlib

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.policies.autonomous_iql_actor import (  # noqa: E402
    ALLOWED_NATIVE_SCORER_SHA256,
    CANDIDATE_SCHEMA,
    DENSITY_RATIO_CAP,
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_CANARY_CONTRACT_SHA256,
    EXPECTED_DEPLOYABLE_AUDIT_SHA256,
    EXPECTED_QUALIFICATION_SHA256,
    INTERVENTION_CAP,
    MINIMUM_UNIFORM_MASS,
    MODEL_CODEC,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA,
)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def export_state(
    *,
    candidate_path: Path,
    qualification_path: Path,
    deployable_audit_path: Path,
    native_scorer_path: Path,
    canary_contract_path: Path,
    mode: str,
    policy_seed: int,
) -> dict[str, object]:
    if mode not in ("shadow", "active"):
        raise ValueError("Generation-6 mode must be shadow or active")
    if not 0 <= policy_seed < 2**64:
        raise ValueError("Generation-6 policy seed must be an unsigned uint64")
    candidate_bytes = candidate_path.read_bytes()
    if _sha256_bytes(candidate_bytes) != EXPECTED_CANDIDATE_SHA256:
        raise ValueError("frozen Generation-6 candidate drifted")
    candidate = json.loads(candidate_bytes.decode("utf-8"))
    if (
        not isinstance(candidate, dict)
        or candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("passed") is not True
        or candidate.get("qualification_samples_loaded") is not False
    ):
        raise ValueError("Generation-6 candidate did not pass frozen preflight")

    qualification = _object(qualification_path)
    if (
        _sha256(qualification_path) != EXPECTED_QUALIFICATION_SHA256
        or qualification.get("schema")
        != "autonomous-generation-6-qualification-result-v1"
        or qualification.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA256
        or qualification.get("passed") is not True
        or qualification.get("qualification_disclosed") is not True
    ):
        raise ValueError("Generation-6 qualification evidence is invalid")
    audit = _object(deployable_audit_path)
    if (
        _sha256(deployable_audit_path) != EXPECTED_DEPLOYABLE_AUDIT_SHA256
        or audit.get("schema")
        != "autonomous-generation-6-deployable-target-audit-v1"
        or audit.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA256
        or audit.get("qualification_result_sha256")
        != EXPECTED_QUALIFICATION_SHA256
        or audit.get("passed") is not True
        or audit.get("target", {}).get("online_reconstructible") is not True
    ):
        raise ValueError("Generation-6 deployable-target audit is invalid")
    native_sha256 = _sha256(native_scorer_path)
    if native_sha256 not in ALLOWED_NATIVE_SCORER_SHA256:
        raise ValueError("Generation-6 native scorer is not preflighted")

    canary = _object(canary_contract_path)
    canary_sha256 = _sha256(canary_contract_path)
    if (
        canary_sha256 != EXPECTED_CANARY_CONTRACT_SHA256
        or
        canary.get("schema") != "autonomous-generation-6-wine-canary-v1"
        or canary.get("candidate_sha256") != EXPECTED_CANDIDATE_SHA256
        or canary.get("qualification_result_sha256")
        != EXPECTED_QUALIFICATION_SHA256
        or canary.get("deployable_target_audit_sha256")
        != EXPECTED_DEPLOYABLE_AUDIT_SHA256
        or canary.get("normal_speed") is not True
        or canary.get("natural_rng") is not True
        or canary.get("complete_stage_hit_continuation") is not True
        or canary.get("bomb") != "forbidden"
        or int(canary.get("maximum_interventions", -1)) not in range(1, 65)
    ):
        raise ValueError("Generation-6 Wine canary contract is invalid")

    authorization = {
        "offline_qualification_passed": True,
        "qualification_result_sha256": EXPECTED_QUALIFICATION_SHA256,
        "deployable_target_audit_passed": True,
        "deployable_target_audit_sha256": EXPECTED_DEPLOYABLE_AUDIT_SHA256,
        "frozen_wine_canary": None,
    }
    if mode == "active":
        authorization["frozen_wine_canary"] = {
            "schema": "autonomous-generation-6-wine-canary-authorization-v1",
            "contract_sha256": canary_sha256,
            "normal_speed": True,
            "natural_rng": True,
            "complete_stage_hit_continuation": True,
            "maximum_interventions": int(canary["maximum_interventions"]),
            "bomb": "forbidden",
        }
    return {
        "schema": STATE_SCHEMA,
        "mode": mode,
        "candidate_codec": MODEL_CODEC,
        "candidate_sha256": EXPECTED_CANDIDATE_SHA256,
        "candidate_payload": base64.b64encode(
            zlib.compress(candidate_bytes, level=9)
        ).decode("ascii"),
        "native_scorer": {"sha256": native_sha256},
        "policy_seed": policy_seed,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
        "intervention": {
            "probability_cap": INTERVENTION_CAP,
            "minimum_uniform_mass": MINIMUM_UNIFORM_MASS,
            "density_ratio_cap": DENSITY_RATIO_CAP,
            "formula": "min(cap, density_ratio_cap * minimum_uniform_mass / safe_set_size)",
        },
        "authorization": authorization,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "active"), required=True)
    parser.add_argument("--policy-seed", type=lambda value: int(value, 0), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--candidate", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-candidate/candidate-v1.json",
    )
    parser.add_argument(
        "--qualification", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-qualification/qualification-v1.json",
    )
    parser.add_argument(
        "--deployable-audit", type=Path,
        default=REPOSITORY / "artifacts/autonomous-generation-6-qualification/deployable-target-audit-v1.json",
    )
    parser.add_argument(
        "--native-scorer", type=Path,
        default=REPOSITORY / "build/native-win32-fully-static/libth06_rl_ranker.dll",
    )
    parser.add_argument(
        "--canary-contract", type=Path,
        default=REPOSITORY / "config/autonomous_generation6_wine_canary.json",
    )
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(f"refusing to replace policy state: {args.output}")
    state = export_state(
        candidate_path=args.candidate,
        qualification_path=args.qualification,
        deployable_audit_path=args.deployable_audit,
        native_scorer_path=args.native_scorer,
        canary_contract_path=args.canary_contract,
        mode=args.mode,
        policy_seed=args.policy_seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":"), allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "mode": args.mode,
        "policy_seed": args.policy_seed,
        "bytes": args.output.stat().st_size,
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
