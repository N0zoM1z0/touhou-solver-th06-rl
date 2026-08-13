#!/usr/bin/env python3
"""Export a dynamically fitted G6 autonomous-round actor state."""

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
    ALLOWED_AUTONOMOUS_ROUND_CONTRACT_SHA256,
    ALLOWED_NATIVE_SCORER_SHA256,
    ALLOWED_PREFLIGHT_NATIVE_SCORER_SHA256,
    CANDIDATE_SCHEMA,
    DENSITY_RATIO_CAP,
    INTERVENTION_CAP,
    MINIMUM_UNIFORM_MASS,
    MODEL_CODEC,
    OPTION_HORIZON_FRAMES,
    STATE_SCHEMA,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _valid_contract(contract: dict[str, object]) -> bool:
    return bool(
        contract.get("schema") in (
            "autonomous-generation-6-round-contract-v1",
            "autonomous-generation-6-decision-successor-contract-v2",
        )
        and contract.get("normal_speed") is True
        and contract.get("natural_rng") is True
        and contract.get("complete_stage_hit_continuation") is True
        and contract.get("bomb") == "forbidden"
    )


def export_round_state(
    *, candidate_path: Path, smoke_path: Path, registry_path: Path,
    scorer_path: Path, contract_path: Path, mode: str, policy_seed: int,
) -> dict[str, object]:
    if mode not in ("shadow", "active") or not 0 <= policy_seed < 2**64:
        raise ValueError("invalid Generation-6 round policy mode or seed")
    contract_sha = _sha256(contract_path)
    if contract_sha not in ALLOWED_AUTONOMOUS_ROUND_CONTRACT_SHA256:
        raise ValueError("Generation-6 autonomous round is not allowlisted")
    contract = _object(contract_path)
    if (
        not _valid_contract(contract)
    ):
        raise ValueError("Generation-6 autonomous round contract is invalid")
    candidate_bytes = candidate_path.read_bytes()
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    candidate = json.loads(candidate_bytes)
    smoke = _object(smoke_path)
    registry_sha = _sha256(registry_path)
    scorer_sha = _sha256(scorer_path)
    if (
        not isinstance(candidate, dict)
        or candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("passed") is not True
        or candidate.get("autonomous_round_contract_sha256") != contract_sha
        or candidate.get("training_identity", {}).get("sha256") != registry_sha
        or smoke.get("schema") not in (
            "autonomous-generation-6-round-offline-smoke-v1",
            "autonomous-generation-6-decision-offline-smoke-v2",
        )
        or smoke.get("passed") is not True
        or smoke.get("contract_sha256") != contract_sha
        or smoke.get("training_registry_sha256") != registry_sha
        or smoke.get("candidate_sha256") != candidate_sha
        or (
            contract.get("schema")
            == "autonomous-generation-6-decision-successor-contract-v2"
            and (
                smoke.get("schema")
                != "autonomous-generation-6-decision-offline-smoke-v2"
                or len(str(smoke.get("decision_audit_sha256", ""))) != 64
            )
        )
        or scorer_sha not in ALLOWED_NATIVE_SCORER_SHA256
    ):
        raise ValueError("Generation-6 autonomous round evidence is invalid")
    authorization = {
        "offline_qualification_passed": False,
        "deployable_target_audit_passed": False,
        "autonomous_round": {
            "contract_sha256": contract_sha,
            "training_registry_sha256": registry_sha,
            "offline_smoke_passed": True,
            "offline_smoke_sha256": _sha256(smoke_path),
            "candidate_sha256": candidate_sha,
        },
        "frozen_wine_canary": None,
    }
    if mode == "active":
        authorization["frozen_wine_canary"] = {
            "schema": "autonomous-generation-6-round-evidence-authorization-v1",
            "contract_sha256": contract_sha,
            "candidate_sha256": candidate_sha,
            "normal_speed": True,
            "natural_rng": True,
            "complete_stage_hit_continuation": True,
            "maximum_interventions": int(contract["maximum_interventions"]),
            "bomb": "forbidden",
        }
    return {
        "schema": STATE_SCHEMA,
        "mode": mode,
        "candidate_codec": MODEL_CODEC,
        "candidate_sha256": candidate_sha,
        "candidate_payload": base64.b64encode(
            zlib.compress(candidate_bytes, level=9)
        ).decode("ascii"),
        "native_scorer": {"sha256": scorer_sha},
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


def export_round_preflight_state(
    *, candidate_path: Path, registry_path: Path, scorer_path: Path,
    contract_path: Path, policy_seed: int,
) -> dict[str, object]:
    """Export a shadow-only state for exact native path conformance.

    This state cannot intervene and deliberately does not claim that the
    offline audit has passed.  It breaks the otherwise circular dependency in
    which the audit must bind the online preflight but policy import would
    require that audit before allowing the preflight to execute.
    """
    if not 0 <= policy_seed < 2**64:
        raise ValueError("invalid Generation-6 round preflight seed")
    contract_sha = _sha256(contract_path)
    if contract_sha not in ALLOWED_AUTONOMOUS_ROUND_CONTRACT_SHA256:
        raise ValueError("Generation-6 autonomous round is not allowlisted")
    contract = _object(contract_path)
    candidate_bytes = candidate_path.read_bytes()
    candidate = json.loads(candidate_bytes)
    candidate_sha = hashlib.sha256(candidate_bytes).hexdigest()
    registry_sha = _sha256(registry_path)
    scorer_sha = _sha256(scorer_path)
    if (
        not _valid_contract(contract)
        or not isinstance(candidate, dict)
        or candidate.get("schema") != CANDIDATE_SCHEMA
        or candidate.get("passed") is not True
        or candidate.get("autonomous_round_contract_sha256") != contract_sha
        or candidate.get("training_identity", {}).get("sha256") != registry_sha
        or scorer_sha not in ALLOWED_PREFLIGHT_NATIVE_SCORER_SHA256
    ):
        raise ValueError("Generation-6 round preflight inputs are invalid")
    return {
        "schema": STATE_SCHEMA,
        "mode": "shadow",
        "candidate_codec": MODEL_CODEC,
        "candidate_sha256": candidate_sha,
        "candidate_payload": base64.b64encode(
            zlib.compress(candidate_bytes, level=9)
        ).decode("ascii"),
        "native_scorer": {"sha256": scorer_sha},
        "policy_seed": policy_seed,
        "option_horizon_frames": OPTION_HORIZON_FRAMES,
        "intervention": {
            "probability_cap": INTERVENTION_CAP,
            "minimum_uniform_mass": MINIMUM_UNIFORM_MASS,
            "density_ratio_cap": DENSITY_RATIO_CAP,
            "formula": (
                "min(cap, density_ratio_cap * minimum_uniform_mass / "
                "safe_set_size)"
            ),
        },
        "authorization": {
            "offline_qualification_passed": False,
            "deployable_target_audit_passed": False,
            "autonomous_round": {
                "contract_sha256": contract_sha,
                "training_registry_sha256": registry_sha,
                "candidate_sha256": candidate_sha,
                "preflight_only": True,
            },
            "frozen_wine_canary": None,
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--smoke", type=Path)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--scorer", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument(
        "--mode", choices=("preflight", "shadow", "active"), required=True
    )
    parser.add_argument("--policy-seed", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.mode == "preflight":
        if args.smoke is not None:
            parser.error("--smoke is forbidden for a preflight-only state")
        state = export_round_preflight_state(
            candidate_path=args.candidate, registry_path=args.registry,
            scorer_path=args.scorer, contract_path=args.contract,
            policy_seed=args.policy_seed,
        )
    else:
        if args.smoke is None:
            parser.error("--smoke is required for shadow/active evidence states")
        state = export_round_state(
            candidate_path=args.candidate, smoke_path=args.smoke,
            registry_path=args.registry, scorer_path=args.scorer,
            contract_path=args.contract, mode=args.mode,
            policy_seed=args.policy_seed,
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(state, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(json.dumps({"output": str(args.output), "sha256": _sha256(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
