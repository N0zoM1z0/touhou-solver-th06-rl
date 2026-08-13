#!/usr/bin/env python3
"""Audit a frozen G6 fit at the deployed decision boundary, without Wine play."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import heapq
import json
import math
from pathlib import Path
import statistics
import sys
import time

import numpy as np

REPOSITORY = Path(__file__).resolve().parents[1]
for path in (REPOSITORY, REPOSITORY / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from th06_rl.advantage_learning import (  # noqa: E402
    _augment_steps,
    rich_candidate_vector_from_encoding,
    rich_feature_names,
)
from th06_rl.audited_option_loader import (  # noqa: E402
    AUDITED_OPTION_LOADER_CONTRACT,
    load_audited_option_episode,
)
from th06_rl.hazard_representation import (  # noqa: E402
    NativeHazardCodebookEncoder,
)
from th06_rl.iql_actor_learning import (  # noqa: E402
    NativeIqlActorPopulation,
    iql_actor_model_from_artifact,
)
from th06_rl.native_decision_conformance import (  # noqa: E402
    certify_mean_population_decision,
    native_order_centered_portability_reference,
)
from th06_rl.offline import ACTION_NAMES  # noqa: E402
from th06_rl.option_cache import load_cached_option_episode  # noqa: E402
from th06_rl.policies.offline_ranker import (  # noqa: E402
    NativePrototypeSupport,
    PortablePrototypeSupport,
)
from th06_rl.resource_control import enforce_training_cpu_affinity  # noqa: E402
from th06_rl.wine_corpus_registry import (  # noqa: E402
    load_wine_corpus_registry,
    select_wine_corpora,
)


SCHEMA = "autonomous-generation-6-native-decision-conformance-v1"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"JSON root is not an object: {path}")
    return value


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[math.ceil(quantile * len(ordered)) - 1]


def _zero_error_choice(scores, sample, supported) -> object:
    return certify_mean_population_decision(
        scores,
        np.zeros_like(scores, dtype=np.float64),
        sample.legal_actions,
        sample.baseline_action,
        supported,
    )


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


class _Panel:
    """Keep deterministic numerical stress cases without outcome selection."""

    def __init__(self, per_dimension: int) -> None:
        self.limit = per_dimension
        self.heaps: dict[str, list[tuple[float, int, object]]] = {
            "near_tie": [],
            "maximum_hazard_count": [],
            "maximum_candidate_width": [],
            "maximum_feature_magnitude": [],
            "identity_hash_stratified": [],
        }
        self.serial = 0

    def _keep(self, name: str, priority: float, value: object) -> None:
        heap = self.heaps[name]
        row = (priority, self.serial, value)
        self.serial += 1
        if len(heap) < self.limit:
            heapq.heappush(heap, row)
        elif priority > heap[0][0]:
            heapq.heapreplace(heap, row)

    def add(self, *, raw, sample, rows, margin: float) -> None:
        identity = f"{sample.episode_id}\0{sample.option_id}".encode()
        digest = int.from_bytes(hashlib.sha256(identity).digest(), "big")
        maximum = max(abs(float(value)) for row in rows for value in row)
        case = (raw, sample, rows)
        self._keep("near_tie", -margin, case)
        self._keep(
            "maximum_hazard_count", float(len(raw.hazard_primitives)), case
        )
        self._keep(
            "maximum_candidate_width", float(len(sample.legal_actions)), case
        )
        self._keep("maximum_feature_magnitude", maximum, case)
        # Smaller hashes form an outcome-independent deterministic sample.
        self._keep("identity_hash_stratified", -float(digest), case)

    def cases(self):
        selected: dict[tuple[str, str], dict[str, object]] = {}
        for reason, heap in self.heaps.items():
            for _priority, _serial, case in heap:
                _raw, sample, _rows = case
                key = (sample.episode_id, sample.option_id)
                selected.setdefault(key, {
                    "case": case, "reasons": [],
                })["reasons"].append(reason)
        return [selected[key] for key in sorted(selected)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--registry", type=Path,
        default=REPOSITORY / "config/wine_corpus_registry.json",
    )
    parser.add_argument(
        "--cache-dir", type=Path,
        default=REPOSITORY / "artifacts/cache/audited-option-episodes",
    )
    parser.add_argument(
        "--linux-library", type=Path,
        default=REPOSITORY / "build/native/libth06_rl_ranker.so",
    )
    parser.add_argument("--contract-sha256", required=True)
    parser.add_argument("--expected-options", type=int, required=True)
    parser.add_argument("--panel-per-dimension", type=int, default=64)
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError(args.output)
    if len(args.contract_sha256) != 64 or args.expected_options < 1:
        parser.error("contract SHA and expected option count are required")
    started = time.perf_counter()
    affinity = enforce_training_cpu_affinity(args.threads)
    fit = _object(args.fit_checkpoint)
    registry, entries = load_wine_corpus_registry(
        args.registry, repository=REPOSITORY
    )
    development = select_wine_corpora(
        entries, required_capabilities=frozenset({"sequential_offline_rl"})
    )
    if (
        fit.get("schema") != "autonomous-generation-6-fit-checkpoint-v1"
        or fit.get("training_identity", {}).get("sha256") != _sha256(args.registry)
        or not development
    ):
        raise ValueError("native-decision audit input identity differs")
    actors = [iql_actor_model_from_artifact(row) for row in fit["actors"]]
    if len(actors) != 7:
        raise ValueError("native-decision audit requires seven actors")
    native_sha = _sha256(args.linux_library)
    native_actor = NativeIqlActorPopulation(
        args.linux_library, expected_sha256=native_sha, models=actors
    )
    portable_support = PortablePrototypeSupport(
        fit["support"], feature_count=len(rich_feature_names())
    )
    native_support = NativePrototypeSupport(
        args.linux_library,
        expected_sha256=native_sha,
        portable=portable_support,
    )
    base_width = len(rich_feature_names())
    native_encoder = None
    action_index = {name: index for index, name in enumerate(ACTION_NAMES)}
    threshold = float(fit["support"]["threshold"])
    factual = frozenset(fit["support"]["factual_supported_actions"])
    panel = _Panel(args.panel_per_dimension)
    options = candidates = exact_choices = exact_support_masks = 0
    exact_hazard_rows = finite_rows = 0
    proposal_count = 0
    native_margins: list[float] = []
    maximum_hazard_error = maximum_support_error = 0.0
    mismatches: list[dict[str, object]] = []
    episode_rows: list[dict[str, object]] = []

    for episode_index, entry in enumerate(development, 1):
        raw, _report, cache_hit = load_cached_option_episode(
            entry.path,
            loader=load_audited_option_episode,
            cache_root=args.cache_dir,
            contract_files=AUDITED_OPTION_LOADER_CONTRACT,
        )
        samples = _augment_steps(raw, fit["representation"])
        if native_encoder is None:
            native_encoder = NativeHazardCodebookEncoder(
                args.linux_library,
                expected_sha256=native_sha,
                artifact=fit["representation"],
                output_count=(
                    base_width - len(raw[0].vector)
                    - len(raw[0].history_features)
                ),
            )
        episode_exact = 0
        for source, sample in zip(raw, samples, strict=True):
            expected_hazard = tuple(sample.candidate_vectors[0][
                len(source.vector):
                len(source.vector) + native_encoder.output_count
            ])
            actual_hazard = native_encoder.encode(source.hazard_primitives)
            hazard_error = max(abs(left - right) for left, right in zip(
                expected_hazard, actual_hazard, strict=True
            ))
            maximum_hazard_error = max(maximum_hazard_error, hazard_error)
            exact_hazard_rows += int(hazard_error <= 2e-5)
            native_rows = tuple(tuple(rich_candidate_vector_from_encoding(
                vector, actual_hazard, source.history_features
            )) for vector in source.candidate_vectors)
            finite_rows += int(all(
                len(row) == base_width
                and all(math.isfinite(float(value)) for value in row)
                for row in native_rows
            ))
            actions = [action_index[action] for action in sample.legal_actions]
            portable_distances = portable_support.distances(
                sample.candidate_vectors, actions
            )
            native_distances = native_support.distances(native_rows, actions)
            support_error = max(abs(left - right) for left, right in zip(
                portable_distances, native_distances, strict=True
            ))
            maximum_support_error = max(maximum_support_error, support_error)
            portable_mask = [
                action in factual and distance <= threshold
                for action, distance in zip(
                    sample.legal_actions, portable_distances, strict=True
                )
            ]
            native_mask = [
                action in factual and distance <= threshold
                for action, distance in zip(
                    sample.legal_actions, native_distances, strict=True
                )
            ]
            mask_exact = portable_mask == native_mask
            exact_support_masks += int(mask_exact)
            portable_scores = np.asarray([
                actor.predict(sample.candidate_vectors) for actor in actors
            ])
            portable_choice = _zero_error_choice(
                portable_scores, sample, portable_mask
            )
            baseline = sample.legal_actions.index(sample.baseline_action)
            native_scores = np.asarray(native_actor.predict_centered(
                native_rows, baseline_index=baseline
            ))
            native_choice = _zero_error_choice(
                native_scores, sample, native_mask
            )
            choice_exact = portable_choice.choice == native_choice.choice
            exact_choices += int(choice_exact)
            episode_exact += int(choice_exact and mask_exact)
            proposal_count += native_choice.choice != sample.baseline_action
            native_margins.append(float(native_choice.decision_margin))
            if len(mismatches) < 20 and (not choice_exact or not mask_exact):
                mismatches.append({
                    "episode_id": sample.episode_id,
                    "option_id": sample.option_id,
                    "portable_choice": portable_choice.choice,
                    "native_choice": native_choice.choice,
                    "support_mask_exact": mask_exact,
                })
            panel.add(
                raw=source, sample=sample, rows=native_rows,
                margin=float(native_choice.decision_margin),
            )
            options += 1
            candidates += len(sample.legal_actions)
        episode_rows.append({
            "run_id": entry.run_id,
            "stage": entry.stage,
            "options": len(samples),
            "exact_choices_and_support": episode_exact,
            "cache_hit": bool(cache_hit),
        })
        print(json.dumps({
            "episode": episode_index,
            "episodes": len(development),
            "run_id": entry.run_id,
            "options_total": options,
            "elapsed_seconds": time.perf_counter() - started,
        }, sort_keys=True), flush=True)

    panel_rows = []
    panel_exact = panel_covered = panel_certified = 0
    minimum_panel_ratio = math.inf
    maximum_panel_target_error = 0.0
    for selected in panel.cases():
        source, sample, rows = selected["case"]
        actions = [action_index[action] for action in sample.legal_actions]
        distances = native_support.distances(rows, actions)
        mask = [
            action in factual and distance <= threshold
            for action, distance in zip(
                sample.legal_actions, distances, strict=True
            )
        ]
        baseline = sample.legal_actions.index(sample.baseline_action)
        references = [
            native_order_centered_portability_reference(
                actor, rows, baseline_index=baseline
            )
            for actor in actors
        ]
        scores = np.stack([row.scores for row in references])
        errors = np.stack([row.error_bounds for row in references])
        certificate = certify_mean_population_decision(
            scores, errors, sample.legal_actions,
            sample.baseline_action, mask,
        )
        actual = np.asarray(native_actor.predict_centered(
            rows, baseline_index=baseline
        ))
        actual_choice = _zero_error_choice(actual, sample, mask)
        target_error = float(np.max(np.abs(actual - scores)))
        covered = bool(np.all(np.abs(actual - scores) <= errors))
        exact = actual_choice.choice == certificate.choice
        panel_exact += int(exact)
        panel_covered += int(covered)
        panel_certified += int(certificate.certified)
        minimum_panel_ratio = min(
            minimum_panel_ratio, certificate.margin_ratio
        )
        maximum_panel_target_error = max(
            maximum_panel_target_error, target_error
        )
        panel_rows.append({
            "episode_id": sample.episode_id,
            "option_id": sample.option_id,
            "reasons": sorted(selected["reasons"]),
            "hazard_count": len(source.hazard_primitives),
            "candidate_count": len(sample.legal_actions),
            "baseline_action": sample.baseline_action,
            "canonical_choice": certificate.choice,
            "linux_choice": actual_choice.choice,
            "decision_margin": _finite_or_none(certificate.decision_margin),
            "error_envelope": _finite_or_none(certificate.error_envelope),
            "margin_ratio": _finite_or_none(certificate.margin_ratio),
            "target_error": target_error,
            "target_error_covered": covered,
            "decision_certified": certificate.certified,
        })
    gates = {
        "exact_expected_full_corpus_option_count": options == args.expected_options,
        "all_rows_finite_and_fixed_width": finite_rows == options,
        "all_hazard_encodings_within_2e_5": exact_hazard_rows == options,
        "all_portable_linux_support_masks_exact": exact_support_masks == options,
        "all_portable_linux_mean_population_choices_exact": (
            exact_choices == options
        ),
        "wide_panel_nonempty": bool(panel_rows),
        "wide_panel_all_target_errors_inside_envelope": (
            panel_covered == len(panel_rows)
        ),
        "wide_panel_all_selected_margins_above_envelope": (
            panel_certified == len(panel_rows)
        ),
        "wide_panel_all_canonical_linux_choices_exact": (
            panel_exact == len(panel_rows)
        ),
        "bomb_absent_from_action_space": "bomb" not in ACTION_NAMES,
    }
    completed = time.perf_counter()
    result = {
        "schema": SCHEMA,
        "evidence_eligible": False,
        "authorization_eligible": False,
        "contract_sha256": args.contract_sha256,
        "fit_checkpoint_sha256": _sha256(args.fit_checkpoint),
        "training_registry_sha256": _sha256(args.registry),
        "training_sources": sorted({entry.source for entry in development}),
        "linux_library_sha256": native_sha,
        "reference": {
            "kind": "native-order-centered-float32-v1",
            "tanh_absolute_allowance_float32_unit_roundoffs": 8,
            "ffp_contract": "off",
            "policy_quantity": "baseline-centered-mean-population-advantage",
        },
        "full_linux": {
            "episodes": len(development),
            "options": options,
            "candidate_rows": candidates,
            "exact_choices": exact_choices,
            "exact_support_masks": exact_support_masks,
            "hazard_rows_within_tolerance": exact_hazard_rows,
            "finite_fixed_width_rows": finite_rows,
            "proposals": proposal_count,
            "proposal_rate": proposal_count / options,
            "maximum_hazard_encoding_error": maximum_hazard_error,
            "maximum_support_distance_error": maximum_support_error,
            "decision_margin_minimum": min(native_margins),
            "decision_margin_p01": _percentile(native_margins, 0.01),
            "decision_margin_p50": statistics.median(native_margins),
            "mismatches": mismatches,
            "episodes_detail": episode_rows,
        },
        "wide_panel": {
            "selection": {
                "per_dimension": args.panel_per_dimension,
                "dimensions": sorted(panel.heaps),
                "manual_stage_phase_frame_rng_hit_or_failure_targeting": False,
            },
            "cases": len(panel_rows),
            "exact_choices": panel_exact,
            "covered_target_errors": panel_covered,
            "certified_decisions": panel_certified,
            "minimum_margin_ratio": _finite_or_none(minimum_panel_ratio),
            "maximum_target_error": maximum_panel_target_error,
            "rows": panel_rows,
        },
        "resource_contract": affinity.as_dict(),
        "timing_seconds": completed - started,
        "gates": gates,
        "passed": all(gates.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output": str(args.output),
        "passed": result["passed"],
        "full_linux": {
            key: result["full_linux"][key] for key in (
                "episodes", "options", "candidate_rows", "exact_choices",
                "exact_support_masks", "proposals",
            )
        },
        "wide_panel": {
            key: result["wide_panel"][key] for key in (
                "cases", "exact_choices", "covered_target_errors",
                "certified_decisions", "minimum_margin_ratio",
            )
        },
        "failed_gates": [name for name, value in gates.items() if not value],
    }, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
