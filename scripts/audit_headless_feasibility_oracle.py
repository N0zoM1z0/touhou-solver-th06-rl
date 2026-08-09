#!/usr/bin/env python3
"""Audit exact-state feasibility artifacts and probe the bottleneck axes."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

try:
    from label_headless_feasibility_oracle import (
        AUTHORITY,
        COMPLETED_TERMINATIONS,
        SCHEMA,
        checkpoint_verdict,
        outcome_rank,
    )
except ModuleNotFoundError:
    from scripts.label_headless_feasibility_oracle import (
        AUTHORITY,
        COMPLETED_TERMINATIONS,
        SCHEMA,
        checkpoint_verdict,
        outcome_rank,
    )


def _files(paths: Iterable[Path]) -> tuple[Path, ...]:
    result = []
    for path in paths:
        if path.is_file():
            result.append(path)
        elif path.is_dir():
            result.extend(sorted(path.rglob("*.json")))
    return tuple(dict.fromkeys(item.resolve() for item in result))


def _branch_feasible(branch: Mapping[str, Any]) -> bool:
    return (
        str(branch.get("termination_reason")) in COMPLETED_TERMINATIONS
        and int(branch.get("physical_deaths_delta", -1)) == 0
        and int(branch.get("bombs_used_delta", -1)) == 0
    )


def audit_file(path: Path) -> dict[str, Any]:
    errors = []
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("schema") != SCHEMA:
        errors.append("unsupported schema")
    if document.get("authority") != AUTHORITY:
        errors.append("invalid authority contract")
    if document.get("runtime_source", {}).get("clean") is not True:
        errors.append("oracle runtime source was dirty")
    input_source = document.get("input_source", {})
    runtime_source = document.get("runtime_source", {})
    if not all(
        input_source.get(name) == runtime_source.get(name)
        for name in ("commit", "binary_sha256")
    ):
        errors.append("input corpus and oracle runtime source differ")
    code_source = document.get("code_source")
    if not isinstance(code_source, Mapping) or not isinstance(code_source.get("commit"), str):
        errors.append("oracle implementation provenance missing")
    input_corpus = document.get("input_corpus")
    if (
        not isinstance(input_corpus, Mapping)
        or not isinstance(input_corpus.get("manifest_sha256"), str)
        or not isinstance(input_corpus.get("transitions"), Mapping)
    ):
        errors.append("input corpus artifact provenance missing")
    branch_frames = int(document.get("branch_frames", 0))
    continuations = document.get("continuations")
    continuation_names = (
        [str(item.get("name")) for item in continuations if isinstance(item, Mapping)]
        if isinstance(continuations, list)
        else []
    )
    if (
        branch_frames <= 0
        or not continuation_names
        or len(continuation_names) != len(set(continuation_names))
    ):
        errors.append("invalid branch bound or continuation declaration")
    checkpoints = document.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        errors.append("missing checkpoints")
        checkpoints = []
    verdicts: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    continuation_witnesses: Counter[str] = Counter()
    branch_count = 0
    native_actions = 0
    discriminative = 0
    native_set_revisions = 0
    for checkpoint_index, checkpoint in enumerate(checkpoints):
        prefix = f"checkpoint {checkpoint_index}"
        legal = checkpoint.get("native_legal_actions")
        branches = checkpoint.get("branches")
        if (
            not isinstance(legal, list)
            or not legal
            or "bomb" in legal
            or len(legal) != len(set(legal))
        ):
            errors.append(f"{prefix}: invalid native legal set")
            continue
        input_legal = checkpoint.get("input_native_legal_actions", legal)
        if not isinstance(input_legal, list) or "bomb" in input_legal:
            errors.append(f"{prefix}: invalid input native legal set")
            continue
        revised = input_legal != legal
        if checkpoint.get("native_set_revised", revised) != revised:
            errors.append(f"{prefix}: native set revision flag mismatch")
        if revised and document.get("native_set_revision_allowed") is not True:
            errors.append(f"{prefix}: undeclared native set revision")
        native_set_revisions += int(revised)
        if not isinstance(branches, list):
            errors.append(f"{prefix}: branches missing")
            continue
        expected_pairs = {(action, continuation) for action in legal for continuation in continuation_names}
        actual_pairs = [
            (branch.get("first_action"), branch.get("continuation"))
            for branch in branches
            if isinstance(branch, Mapping)
        ]
        if len(actual_pairs) != len(set(actual_pairs)) or set(actual_pairs) != expected_pairs:
            errors.append(f"{prefix}: branches do not cover action-continuation product")
            continue
        native_actions += len(legal)
        branch_count += len(branches)
        recomputed_feasible = []
        for branch in branches:
            terminal = str(branch.get("termination_reason"))
            terminations[terminal] += 1
            survival = int(branch.get("survival_ticks", -1))
            issued = int(branch.get("actions_issued", -1))
            if not 0 <= survival <= branch_frames or not 0 <= issued <= branch_frames:
                errors.append(f"{prefix}: branch exceeds declared bound")
            if int(branch.get("bombs_used_delta", -1)) != 0:
                errors.append(f"{prefix}: branch observed Bomb use")
            expected_feasible = _branch_feasible(branch)
            if branch.get("feasible") is not expected_feasible:
                errors.append(f"{prefix}: branch feasibility summary mismatch")
            if expected_feasible:
                recomputed_feasible.append(str(branch["first_action"]))
                continuation_witnesses[str(branch["continuation"])] += 1
        feasible_actions = sorted(set(recomputed_feasible))
        if checkpoint.get("feasible_actions") != feasible_actions:
            errors.append(f"{prefix}: feasible action summary mismatch")
        expected_rank = max(outcome_rank(branch) for branch in branches)
        expected_best = sorted({
            str(branch["first_action"])
            for branch in branches
            if outcome_rank(branch) == expected_rank
        })
        if checkpoint.get("best_actions") != expected_best:
            errors.append(f"{prefix}: best action summary mismatch")
        factual = str(checkpoint.get("factual_action"))
        local = str(checkpoint.get("local_teacher_action"))
        if checkpoint.get("factual_action_has_witness") is not (factual in feasible_actions):
            errors.append(f"{prefix}: factual witness flag mismatch")
        if checkpoint.get("local_teacher_action_has_witness") is not (local in feasible_actions):
            errors.append(f"{prefix}: local witness flag mismatch")
        expected_verdict = checkpoint_verdict(
            feasible_actions=tuple(feasible_actions),
            factual_action=factual,
        )
        if checkpoint.get("verdict") != expected_verdict:
            errors.append(f"{prefix}: verdict mismatch")
        verdicts[expected_verdict] += 1
        discriminative += 0 < len(feasible_actions) < len(legal)
        exact_features = checkpoint.get("exact_snapshot_features")
        if not isinstance(exact_features, Mapping) or not exact_features:
            errors.append(f"{prefix}: exact snapshot feature probe missing")
        elif any(
            not isinstance(value, (int, float)) or not math.isfinite(float(value))
            for value in exact_features.values()
        ):
            errors.append(f"{prefix}: exact snapshot feature probe is non-finite")
    return {
        "path": str(path),
        "valid": not errors,
        "errors": errors,
        "scope": document.get("scope"),
        "initial_seed": document.get("initial_seed"),
        "input_source": document.get("input_source"),
        "runtime_source": document.get("runtime_source"),
        "checkpoints": len(checkpoints),
        "native_actions": native_actions,
        "branches": branch_count,
        "discriminative_checkpoints": discriminative,
        "native_set_revisions": native_set_revisions,
        "verdicts": dict(sorted(verdicts.items())),
        "terminations": dict(sorted(terminations.items())),
        "continuation_witnesses": dict(sorted(continuation_witnesses.items())),
        "document": document,
    }


def _candidate_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        document = result["document"]
        seed = int(document["initial_seed"])
        for checkpoint in document["checkpoints"]:
            legal = tuple(str(action) for action in checkpoint["native_legal_actions"])
            feasible = frozenset(str(action) for action in checkpoint["feasible_actions"])
            if not 0 < len(feasible) < len(legal):
                continue
            candidates = {
                str(item["action"]): item
                for item in checkpoint["action_candidates"]
            }
            if set(candidates) != set(legal):
                continue
            checkpoint_key = f"{seed}:{checkpoint['observation_sha256']}"
            for action in legal:
                compact = {
                    f"state_{name}": value
                    for name, value in checkpoint["compact_state"].items()
                    if isinstance(value, (str, int, float, bool))
                }
                compact.update({
                    f"candidate_{name}": value
                    for name, value in candidates[action].items()
                    if name not in {"selected", "teacher"}
                    and isinstance(value, (str, int, float, bool))
                })
                compact["source_context"] = str(checkpoint["source_context"])
                compact["legal_set"] = ",".join(legal)
                exact = dict(compact)
                exact.update({
                    f"exact_{name}": float(value)
                    for name, value in checkpoint["exact_snapshot_features"].items()
                })
                rows.append({
                    "seed": seed,
                    "checkpoint": checkpoint_key,
                    "action": action,
                    "label": int(action in feasible),
                    "factual": action == checkpoint["factual_action"],
                    "local": action == checkpoint["local_teacher_action"],
                    "compact": compact,
                    "exact": exact,
                })
    return rows


def _probe_view(
    rows: list[dict[str, Any]],
    *,
    view: str,
    threads: int,
) -> dict[str, Any]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.feature_extraction import DictVectorizer

    seeds = sorted({int(row["seed"]) for row in rows})
    selected = 0
    correct = 0
    heldout_seeds = []
    for seed in seeds:
        train = [row for row in rows if row["seed"] != seed]
        test = [row for row in rows if row["seed"] == seed]
        if not train or not test or len({row["label"] for row in train}) < 2:
            continue
        vectorizer = DictVectorizer(sparse=False)
        x_train = vectorizer.fit_transform([row[view] for row in train])
        x_test = vectorizer.transform([row[view] for row in test])
        model = ExtraTreesClassifier(
            n_estimators=160,
            min_samples_leaf=1,
            class_weight="balanced",
            random_state=6006,
            n_jobs=threads,
        )
        model.fit(x_train, [row["label"] for row in train])
        positive_index = list(model.classes_).index(1)
        probabilities = model.predict_proba(x_test)[:, positive_index]
        grouped: dict[str, list[tuple[dict[str, Any], float]]] = defaultdict(list)
        for row, probability in zip(test, probabilities, strict=True):
            grouped[str(row["checkpoint"])].append((row, float(probability)))
        for candidates in grouped.values():
            winner, _ = max(
                candidates,
                key=lambda item: (item[1], str(item[0]["action"])),
            )
            selected += 1
            correct += int(winner["label"] == 1)
        heldout_seeds.append(seed)
    return {
        "view": view,
        "heldout_seeds": heldout_seeds,
        "heldout_checkpoints": selected,
        "top1_feasible_rate": correct / selected if selected else None,
    }


def representation_probe(
    results: list[dict[str, Any]],
    *,
    threads: int,
) -> dict[str, Any]:
    rows = _candidate_rows(results)
    checkpoints = sorted({str(row["checkpoint"]) for row in rows})
    seeds = sorted({int(row["seed"]) for row in rows})
    factual_checkpoints = {
        str(row["checkpoint"])
        for row in rows
        if row["factual"] and row["label"] == 1
    }
    local_checkpoints = {
        str(row["checkpoint"])
        for row in rows
        if row["local"] and row["label"] == 1
    }
    base = {
        "schema": "th06-rl-headless-feasibility-representation-probe-v1",
        "interpretation": (
            "leave-one-seed-out diagnostic; exact-derived gain is evidence, "
            "not proof, of compact representation loss"
        ),
        "seeds": seeds,
        "discriminative_checkpoints": len(checkpoints),
        "candidate_rows": len(rows),
        "factual_feasible_rate": (
            len(factual_checkpoints) / len(checkpoints) if checkpoints else None
        ),
        "local_teacher_feasible_rate": (
            len(local_checkpoints) / len(checkpoints) if checkpoints else None
        ),
    }
    if len(seeds) < 2 or len(checkpoints) < 4:
        return {
            **base,
            "status": "insufficient-cross-seed-evidence",
            "compact": None,
            "exact_derived": None,
        }
    compact = _probe_view(rows, view="compact", threads=threads)
    exact = _probe_view(rows, view="exact", threads=threads)
    if not compact["heldout_checkpoints"] or not exact["heldout_checkpoints"]:
        status = "insufficient-train-class-support"
    else:
        status = "complete"
    return {
        **base,
        "status": status,
        "compact": compact,
        "exact_derived": exact,
    }


def summarize(results: list[dict[str, Any]], *, threads: int) -> dict[str, Any]:
    valid = [result for result in results if result["valid"]]
    scopes = {json.dumps(result["scope"], sort_keys=True) for result in valid}
    sources = {json.dumps(result["runtime_source"], sort_keys=True) for result in valid}
    mixed_scope = len(scopes) > 1
    mixed_source = len(sources) > 1
    total_checkpoints = sum(int(result["checkpoints"]) for result in valid)
    verdicts: Counter[str] = Counter()
    terminations: Counter[str] = Counter()
    continuation_witnesses: Counter[str] = Counter()
    for result in valid:
        verdicts.update(result["verdicts"])
        terminations.update(result["terminations"])
        continuation_witnesses.update(result["continuation_witnesses"])
    no_witness = verdicts["oracle-no-witness"]
    policy_selection = verdicts["policy-selection-witness"]
    probe = (
        representation_probe(valid, threads=threads)
        if valid and not mixed_scope and not mixed_source
        else {
            "status": "mixed-scope-or-source",
            "compact": None,
            "exact_derived": None,
        }
    )
    signals = {
        "geometry_authority_or_search_ceiling": (
            no_witness / total_checkpoints if total_checkpoints else None
        ),
        "observed_policy_selection": (
            policy_selection / total_checkpoints if total_checkpoints else None
        ),
        "compact_representation_loss": None,
        "learner_gap_within_compact_view": None,
    }
    compact = probe.get("compact")
    exact = probe.get("exact_derived")
    factual_rate = probe.get("factual_feasible_rate")
    if (
        probe.get("status") == "complete"
        and isinstance(compact, Mapping)
        and isinstance(exact, Mapping)
        and compact.get("top1_feasible_rate") is not None
        and exact.get("top1_feasible_rate") is not None
    ):
        compact_rate = float(compact["top1_feasible_rate"])
        exact_rate = float(exact["top1_feasible_rate"])
        signals["compact_representation_loss"] = exact_rate - compact_rate
        if factual_rate is not None:
            signals["learner_gap_within_compact_view"] = compact_rate - float(factual_rate)
    return {
        "schema": "th06-rl-headless-feasibility-oracle-audit-v1",
        "interpretation": (
            "witnesses are constructive; oracle-no-witness is not an "
            "infeasibility proof"
        ),
        "files": len(results),
        "valid_files": len(valid),
        "mixed_scope": mixed_scope,
        "mixed_runtime_source": mixed_source,
        "checkpoints": total_checkpoints,
        "native_actions": sum(int(result["native_actions"]) for result in valid),
        "branches": sum(int(result["branches"]) for result in valid),
        "discriminative_checkpoints": sum(
            int(result["discriminative_checkpoints"]) for result in valid
        ),
        "native_set_revisions": sum(
            int(result["native_set_revisions"]) for result in valid
        ),
        "verdicts": dict(sorted(verdicts.items())),
        "terminations": dict(sorted(terminations.items())),
        "continuation_witnesses": dict(sorted(continuation_witnesses.items())),
        "signals": signals,
        "representation_probe": probe,
        "file_results": [
            {key: value for key, value in result.items() if key != "document"}
            for result in results
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.threads <= 12:
        parser.error("threads must be in 1..12")
    files = _files(args.paths)
    if not files:
        parser.error("no feasibility JSON files found")
    result = summarize([audit_file(path) for path in files], threads=args.threads)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if result["valid_files"] == result["files"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
