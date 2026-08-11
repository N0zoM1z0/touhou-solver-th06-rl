#!/usr/bin/env python3
"""Run a resumable sequential batch of immutable Wine intervention arms."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from scripts.audit_wine_intervention_pair import audit
from scripts.make_wine_intervention_pair import write_pair


SCHEMA = "th06-rl-wine-intervention-batch-v1"
ACCEPTED_TRIAL_RETURN_CODES = (0, 10, 12)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _parse_seeds(value: str) -> tuple[int, ...]:
    seeds = tuple(int(item.strip(), 0) for item in value.split(",") if item.strip())
    if not seeds or len(set(seeds)) != len(seeds):
        raise argparse.ArgumentTypeError("seeds must be a non-empty unique list")
    if any(not 0 <= seed <= 0xFFFF for seed in seeds):
        raise argparse.ArgumentTypeError("every seed must fit source u16")
    return seeds


def _arm_order(index: int) -> tuple[str, str]:
    return (
        ("incumbent", "alternative")
        if index % 2 == 0
        else ("alternative", "incumbent")
    )


def _write_json(path: Path, value: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _trace_run_id(path: Path) -> str:
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            row = json.loads(line)
            run_id = row.get("run_id")
            if isinstance(run_id, str) and run_id:
                return run_id
    raise ValueError(f"trace contains no corpus run id: {path}")


def _verify_report(path: Path, seed: int) -> dict[str, object]:
    report = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise ValueError(f"run report root is not an object: {path}")
    errors = []
    if report.get("error") is not None:
        errors.append(f"runner-error:{report.get('error')}")
    if report.get("diagnostic_rng_seed") != seed:
        errors.append("diagnostic-seed-mismatch")
    if report.get("evaluation_mode") != "fixed-rng-first-failure-intervention":
        errors.append("evaluation-mode-mismatch")
    if report.get("immutable_policy_state_equal") is not True:
        errors.append("policy-state-mutated")
    if report.get("leftover_prefix_processes") != []:
        errors.append("leftover-prefix-processes")
    if int(report.get("controller_returncode", -1)) not in ACCEPTED_TRIAL_RETURN_CODES:
        errors.append(f"controller-returncode:{report.get('controller_returncode')}")
    if errors:
        raise ValueError(f"invalid Wine trial report {path}: {errors}")
    return report


def main() -> int:
    repository = REPOSITORY
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--incumbent-state", type=Path, required=True)
    parser.add_argument("--seeds", type=_parse_seeds, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--stage", type=int, choices=range(1, 7), default=6)
    parser.add_argument(
        "--difficulty", choices=("normal", "hard", "lunatic"), default="lunatic"
    )
    args = parser.parse_args()

    incumbent_path = args.incumbent_state.resolve()
    incumbent = json.loads(incumbent_path.read_text(encoding="utf-8"))
    if not isinstance(incumbent, dict):
        parser.error("incumbent state root must be an object")
    output_root = args.output_root.resolve()
    corpus_root = args.corpus_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "batch.json"
    if manifest_path.is_file():
        batch = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected = {
            "schema": SCHEMA,
            "seeds": list(args.seeds),
            "stage": args.stage,
            "difficulty": args.difficulty,
            "incumbent_state_sha256": _sha256(incumbent_path),
        }
        if any(batch.get(key) != value for key, value in expected.items()):
            raise SystemExit("existing batch manifest does not match requested contract")
    else:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        batch = {
            "schema": SCHEMA,
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "repository_commit": commit,
            "incumbent_state": str(incumbent_path),
            "incumbent_state_sha256": _sha256(incumbent_path),
            "seeds": list(args.seeds),
            "stage": args.stage,
            "difficulty": args.difficulty,
            "pairs": [],
            "complete": False,
        }
        _write_json(manifest_path, batch)

    pairs_by_seed = {int(row["seed"]): row for row in batch["pairs"]}
    for index, seed in enumerate(args.seeds):
        pair_id = f"stage{args.stage}-seed{seed:04x}-batch"
        pair_dir = output_root / "states" / pair_id
        if not pair_dir.exists():
            write_pair(
                incumbent=incumbent,
                output_dir=pair_dir,
                pair_id=pair_id,
            )
        pair = pairs_by_seed.get(seed)
        if pair is None:
            pair = {
                "seed": seed,
                "pair_id": pair_id,
                "arm_order": list(_arm_order(index)),
                "runs": {},
                "audit": None,
            }
            batch["pairs"].append(pair)
            pairs_by_seed[seed] = pair
            _write_json(manifest_path, batch)

        for arm in pair["arm_order"]:
            runs = pair["runs"]
            artifact_dir = output_root / "runs" / f"{pair_id}-{arm}"
            report_path = artifact_dir / "report.json"
            if report_path.is_file():
                report = _verify_report(report_path, seed)
            else:
                command = [
                    sys.executable,
                    str(repository / "scripts/run_wine_retail.py"),
                    "--practice-stage",
                    str(args.stage),
                    "--difficulty",
                    args.difficulty,
                    "--seconds",
                    "0",
                    "--exploration-rate",
                    "0",
                    "--immutable-policy",
                    "--diagnostic-rng-seed",
                    hex(seed),
                    "--policy-plugin",
                    str(repository / "src/th06_rl/policies/wine_intervention.py"),
                    "--policy-state",
                    str(pair_dir / f"{arm}.json"),
                    "--first-failure-corpus-root",
                    str(corpus_root),
                    "--artifact-dir",
                    str(artifact_dir),
                ]
                print(f"START seed=0x{seed:04x} arm={arm}", flush=True)
                completed = subprocess.run(command, cwd=repository, check=False)
                if completed.returncode not in ACCEPTED_TRIAL_RETURN_CODES:
                    raise SystemExit(
                        f"Wine trial failed seed=0x{seed:04x} arm={arm} "
                        f"rc={completed.returncode}"
                    )
                report = _verify_report(report_path, seed)
            run_id = _trace_run_id(artifact_dir / "trace.jsonl")
            run_dir = corpus_root / run_id
            if not (run_dir / "manifest.json").is_file():
                raise SystemExit(f"completed corpus is absent: {run_dir}")
            runs[arm] = {
                "artifact_dir": str(artifact_dir),
                "run_dir": str(run_dir),
                "run_id": run_id,
                "controller_returncode": report["controller_returncode"],
                "physical_hits": report["trace"]["physical_hits_in_run"],
            }
            _write_json(manifest_path, batch)
            print(f"DONE seed=0x{seed:04x} arm={arm} run={run_id}", flush=True)

        try:
            pair_report = audit(
                Path(pair["runs"]["incumbent"]["run_dir"]),
                Path(pair["runs"]["alternative"]["run_dir"]),
            )
        except ValueError as error:
            # A physical run can jump from a long advisory frontier directly
            # to Hard-empty without ever offering the predeclared urgent
            # intervention. That is an ineligible training episode, not an
            # infrastructure or safety failure. Retain it and continue; the
            # grouped dataset builder will exclude the missing event.
            if "expected exactly one propensity-recorded intervention" not in str(error):
                raise
            pair_report = {
                "schema": "th06-rl-wine-intervention-pair-audit-v1",
                "pair_accepted": False,
                "root_match": False,
                "contract_errors": ["no-eligible-intervention-in-one-or-more-arms"],
                "causal_effect_available": False,
                "survival_frame_delta_alternative_minus_incumbent": None,
                "physical_hit_delta_alternative_minus_incumbent": None,
                "ineligible_reason": str(error),
            }
        audit_path = output_root / "audits" / f"{pair_id}.json"
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(audit_path, pair_report)
        pair["audit"] = {
            "path": str(audit_path),
            "pair_accepted": pair_report["pair_accepted"],
            "root_match": pair_report["root_match"],
            "contract_errors": pair_report["contract_errors"],
        }
        _write_json(manifest_path, batch)

    batch["complete"] = True
    batch["finished_utc"] = datetime.now(timezone.utc).isoformat()
    batch["summary"] = {
        "pairs": len(batch["pairs"]),
        "episodes": sum(len(pair["runs"]) for pair in batch["pairs"]),
        "exact_root_pairs": sum(
            bool((pair.get("audit") or {}).get("pair_accepted"))
            for pair in batch["pairs"]
        ),
    }
    _write_json(manifest_path, batch)
    print(json.dumps(batch["summary"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
