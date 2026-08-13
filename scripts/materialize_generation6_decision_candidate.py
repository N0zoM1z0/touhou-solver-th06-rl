#!/usr/bin/env python3
"""Materialize the frozen G6 fit only after decision-level conformance passes."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.advantage_learning import rich_feature_names  # noqa: E402


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _validate_frozen_inputs(contract: dict[str, object]) -> None:
    frozen = contract.get("frozen_inputs")
    if not isinstance(frozen, dict) or not frozen:
        raise ValueError("decision successor frozen inputs are absent")
    for name, expected in frozen.items():
        path = Path(str(name))
        if not path.is_absolute():
            path = REPOSITORY / path
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"decision successor frozen input drifted: {name}")
    implementation = str(contract.get("implementation_commit", ""))
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", implementation, "HEAD"],
        cwd=REPOSITORY,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError("decision successor implementation is not an ancestor")


def materialize_candidate(
    *, contract: dict[str, object], contract_sha256: str,
    fit: dict[str, object], fit_sha256: str,
    audit: dict[str, object], audit_sha256: str,
    registry_sha256: str, linux_library_sha256: str,
) -> dict[str, object]:
    reused = contract.get("reused_training")
    numeric = contract.get("numeric_conformance")
    performance = contract.get("audit_performance")
    full = audit.get("full_linux")
    panel = audit.get("wide_panel")
    timing = audit.get("timing")
    gates = {
        "contract_is_frozen_v2_successor": (
            contract.get("schema")
            == "autonomous-generation-6-decision-successor-contract-v2"
            and contract.get("authorization_eligible") is False
        ),
        "fit_checkpoint_identity_exact": bool(
            isinstance(reused, dict)
            and reused.get("fit_checkpoint_sha256") == fit_sha256
            and fit.get("schema")
            == "autonomous-generation-6-fit-checkpoint-v1"
            and fit.get("training_identity", {}).get("sha256")
            == registry_sha256 == reused.get("registry_sha256")
        ),
        "no_collection_or_refit": bool(
            isinstance(reused, dict)
            and reused.get("new_wine_collection") is False
            and reused.get("refit") is False
            and reused.get(
                "manual_stage_phase_frame_rng_hit_or_failure_targeting"
            ) is False
        ),
        "formal_audit_identity_exact": bool(
            audit.get("schema")
            == "autonomous-generation-6-native-decision-conformance-v1"
            and audit.get("contract_sha256") == contract_sha256
            and audit.get("fit_checkpoint_sha256") == fit_sha256
            and audit.get("training_registry_sha256") == registry_sha256
            and audit.get("linux_library_sha256") == linux_library_sha256
            and audit.get("passed") is True
        ),
        "float64_serving_contract_exact": bool(
            isinstance(numeric, dict)
            and numeric.get("reference")
            == audit.get("reference", {}).get("kind")
            == "native-order-centered-float64-v1"
            and numeric.get("serialized_parameter_precision") == "float32"
            and numeric.get("serving_intermediate_precision") == "float64"
        ),
        "full_corpus_decisions_and_support_exact": bool(
            isinstance(full, dict) and isinstance(reused, dict)
            and full.get("options") == reused.get("factual_options") == 167250
            and full.get("exact_choices") == full.get("options")
            and full.get("exact_support_masks") == full.get("options")
            and full.get("finite_fixed_width_rows") == full.get("options")
        ),
        "wide_panel_fully_certified": bool(
            isinstance(panel, dict) and panel.get("cases") == 320
            and panel.get("exact_choices") == panel.get("cases")
            and panel.get("covered_target_errors") == panel.get("cases")
            and panel.get("certified_decisions") == panel.get("cases")
        ),
        "bounded_audit_performance_exact": bool(
            isinstance(performance, dict) and isinstance(timing, dict)
            and audit.get("worker_processes")
            == performance.get("worker_processes") == 16
            and audit.get("math_library_threads_per_worker")
            == performance.get("math_library_threads_per_worker") == 1
            and float(timing.get("total_seconds", float("inf")))
            <= float(performance.get(
                "full_corpus_and_panel_maximum_seconds", float("-inf")
            ))
        ),
        "bomb_forbidden": contract.get("bomb") == "forbidden",
    }
    if not all(gates.values()):
        failed = [name for name, value in gates.items() if not value]
        raise ValueError(f"decision successor candidate gates failed: {failed}")
    actors = fit.get("actors")
    if not isinstance(actors, list) or len(actors) != 7:
        raise ValueError("decision successor fit lacks seven actors")
    result = {
        "schema": "autonomous-generation-6-candidate-v1",
        "evidence_eligible": False,
        "authorization_eligible": False,
        "autonomous_round_contract_sha256": contract_sha256,
        "qualification_samples_loaded": True,
        "training_identity": fit["training_identity"],
        "training_identity_sha256": fit["training_identity_sha256"],
        "development_episode_groups": int(reused["episodes"]),
        "development_options": int(reused["factual_options"]),
        "source_fit_checkpoint_sha256": fit_sha256,
        "selection": {
            "kind": "baseline-centered-float64-complete-population-mean-supported-intervention",
            "physical_safety": "native-safe-set-only",
            "bomb": "forbidden",
        },
        "numeric_serving": {
            "serialized_parameter_precision": "float32",
            "intermediate_precision": "float64",
            "normalization_precision": "float32",
            "ffp_contract": "off",
        },
        "feature_names": list(rich_feature_names()),
        "representation": fit["representation"],
        "support": fit["support"],
        "support_report": fit["support_report"],
        "actors": actors,
        "actor_diagnostics": fit["actor_diagnostics"],
        "actor_bootstrap": fit["actor_bootstrap"],
        "advantage_crossfit": fit["advantage_crossfit"],
        "native_decision_conformance": {
            "sha256": audit_sha256,
            "linux_library_sha256": linux_library_sha256,
            "full_factual_options": full["options"],
            "exact_choices": full["exact_choices"],
            "exact_support_masks": full["exact_support_masks"],
            "wide_panel_cases": panel["cases"],
            "minimum_margin_ratio": panel["minimum_margin_ratio"],
            "timing": timing,
        },
        "gates": gates,
        "passed": True,
    }
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--fit-checkpoint", type=Path, required=True)
    parser.add_argument("--decision-audit", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--linux-library", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    contract = _object(args.contract)
    _validate_frozen_inputs(contract)
    result = materialize_candidate(
        contract=contract,
        contract_sha256=_sha256(args.contract),
        fit=_object(args.fit_checkpoint),
        fit_sha256=_sha256(args.fit_checkpoint),
        audit=_object(args.decision_audit),
        audit_sha256=_sha256(args.decision_audit),
        registry_sha256=_sha256(args.registry),
        linux_library_sha256=_sha256(args.linux_library),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "sha256": _sha256(args.output),
        "passed": result["passed"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
