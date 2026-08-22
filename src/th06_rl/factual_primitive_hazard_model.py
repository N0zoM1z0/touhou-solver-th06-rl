"""Direct-Brier sigmoid DeepSets over recorded observed-hazard primitives."""

from __future__ import annotations

import copy
import math
import time

import numpy as np
import torch
from torch import nn

from .factual_hazard_model import hazard_predictions
from .factual_primitive_dataset import (
    PRIMITIVE_FEATURE_SCHEMA,
    TOKEN_FEATURE_NAMES,
    PrimitiveProbeDataset,
)
from .factual_probe_boundary_diagnostics import (
    _calibration_summary,
    _stratum_result,
)
from .factual_probes import (
    PROBE_FEATURE_NAMES,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


PRIMITIVE_HAZARD_FIT_SCHEMA = "th06-rl-l2i-primitive-set-hazard-fit-v1"
PRIMITIVE_HAZARD_EVALUATION_SCHEMA = (
    "th06-rl-l2i-primitive-set-hazard-evaluation-v1"
)
MODEL_KIND = "shared-sigmoid-deepsets-brier-regressor"
MODEL_NAMES = (
    "object_full",
    "scalar_only",
    "object_current_action_ablated",
)
ACTION_CONTEXT_INDICES = (11, 12, 13, 14)
ACTION_RELATIVE_START = 6


class PrimitiveSetHazard(nn.Module):
    def __init__(
        self,
        *,
        token_width: int,
        token_hidden: int,
        head_hidden_1: int,
        head_hidden_2: int,
        token_cap: int,
    ) -> None:
        super().__init__()
        self.token_cap = token_cap
        self.token_encoder = nn.Sequential(
            nn.Linear(token_width + len(ACTION_CONTEXT_INDICES), token_hidden),
            nn.ReLU(),
            nn.Linear(token_hidden, token_hidden),
            nn.ReLU(),
        )
        head_width = len(PROBE_FEATURE_NAMES) + token_hidden * 2 + 1
        self.head = nn.Sequential(
            nn.Linear(head_width, head_hidden_1),
            nn.ReLU(),
            nn.Linear(head_hidden_1, head_hidden_2),
            nn.ReLU(),
            nn.Linear(head_hidden_2, 1),
        )

    def forward(
        self,
        scalar: torch.Tensor,
        tokens: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action = scalar[:, ACTION_CONTEXT_INDICES]
        action = action[:, None, :].expand(-1, tokens.shape[1], -1)
        encoded = self.token_encoder(torch.cat((tokens, action), dim=2))
        active = mask[:, :, None]
        active_float = active.to(encoded.dtype)
        count = active_float.sum(dim=1)
        mean = (encoded * active_float).sum(dim=1) / count.clamp_min(1.0)
        masked = encoded.masked_fill(~active, -torch.inf)
        maximum = masked.max(dim=1).values
        maximum = torch.where(count > 0.0, maximum, torch.zeros_like(maximum))
        normalized_count = torch.log1p(count) / math.log1p(self.token_cap)
        logits = self.head(torch.cat((scalar, mean, maximum, normalized_count), dim=1))
        logits = logits[:, 0]
        return torch.sigmoid(logits), logits


def _view(dataset: PrimitiveProbeDataset, horizon: int):
    try:
        return next(row for row in dataset.horizons if row.horizon == horizon)
    except StopIteration as error:
        raise ValueError("primitive hazard horizon is absent") from error


def _model_kwargs(state: dict[str, object]) -> dict[str, int]:
    architecture = state["architecture"]
    return {
        "token_width": int(architecture["token_width"]),
        "token_hidden": int(architecture["token_hidden"]),
        "head_hidden_1": int(architecture["head_hidden_1"]),
        "head_hidden_2": int(architecture["head_hidden_2"]),
        "token_cap": int(architecture["token_cap"]),
    }


def _serialized_model(model: nn.Module) -> dict[str, object]:
    return {
        name: {
            "shape": list(value.shape),
            "values": value.detach().cpu().tolist(),
        }
        for name, value in model.state_dict().items()
    }


def _load_model(state: dict[str, object], model_name: str) -> PrimitiveSetHazard:
    model = PrimitiveSetHazard(**_model_kwargs(state))
    expected = model.state_dict()
    unresolved = state["models"][model_name]
    if set(unresolved) != set(expected):
        raise ValueError("primitive model parameter names changed")
    resolved = {}
    for name, template in expected.items():
        row = unresolved[name]
        value = torch.as_tensor(row["values"], dtype=torch.float32)
        if list(value.shape) != row.get("shape") or value.shape != template.shape:
            raise ValueError("primitive model parameter shape changed")
        resolved[name] = value
    model.load_state_dict(resolved, strict=True)
    model.eval()
    return model


def _variant_inputs(
    scalar: torch.Tensor,
    mask: torch.Tensor,
    model_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if model_name == "object_full":
        return scalar, mask
    if model_name == "scalar_only":
        return scalar, torch.zeros_like(mask)
    if model_name == "object_current_action_ablated":
        ablated = scalar.clone()
        ablated[:, ACTION_RELATIVE_START:] = 0.0
        return ablated, mask
    raise ValueError(f"unknown primitive hazard model: {model_name}")


def _predict_model(
    model: PrimitiveSetHazard,
    scalar: np.ndarray,
    tokens: np.ndarray,
    masks: np.ndarray,
    *,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("primitive prediction batch size must be positive")
    scalar_tensor = torch.from_numpy(scalar)
    token_tensor = torch.from_numpy(tokens)
    mask_tensor = torch.from_numpy(masks)
    probabilities = []
    logits = []
    with torch.inference_mode():
        for start in range(0, len(scalar), batch_size):
            stop = min(start + batch_size, len(scalar))
            batch_scalar, batch_mask = _variant_inputs(
                scalar_tensor[start:stop], mask_tensor[start:stop], model_name
            )
            probability, logit = model(
                batch_scalar,
                token_tensor[start:stop],
                batch_mask,
            )
            probabilities.append(probability.cpu().numpy())
            logits.append(logit.cpu().numpy())
    return (
        np.concatenate(probabilities).astype(np.float64),
        np.concatenate(logits).astype(np.float64),
    )


def primitive_hazard_predictions(
    state: dict[str, object],
    scalar: np.ndarray,
    tokens: np.ndarray,
    masks: np.ndarray,
    *,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Return sigmoid probabilities and disclose saturation and logit range."""
    if state.get("schema") != PRIMITIVE_HAZARD_FIT_SCHEMA:
        raise ValueError("primitive hazard fit schema mismatch")
    if state.get("torch_version") != torch.__version__:
        raise ValueError("primitive hazard PyTorch version mismatch")
    if (
        scalar.shape != (len(scalar), len(PROBE_FEATURE_NAMES))
        or tokens.shape != (
            len(scalar),
            int(state["architecture"]["token_cap"]),
            len(TOKEN_FEATURE_NAMES),
        )
        or masks.shape != tokens.shape[:2]
        or not np.all(np.isfinite(scalar))
        or not np.all(np.isfinite(tokens))
    ):
        raise ValueError("primitive hazard prediction tensors changed")
    model = _load_model(state, model_name)
    old_threads = torch.get_num_threads()
    torch.set_num_threads(int(state["fit"]["threads"]))
    try:
        probability, logits = _predict_model(
            model,
            scalar,
            tokens,
            masks,
            model_name=model_name,
            batch_size=batch_size,
        )
    finally:
        torch.set_num_threads(old_threads)
    if not np.all(np.isfinite(probability)) or np.any(
        (probability < 0.0) | (probability > 1.0)
    ):
        raise ValueError("primitive sigmoid produced an invalid probability")
    saturated = (probability <= 1e-7) | (probability >= 1.0 - 1e-7)
    return probability, {
        "minimum_probability": float(np.min(probability)),
        "maximum_probability": float(np.max(probability)),
        "minimum_logit": float(np.min(logits)),
        "maximum_logit": float(np.max(logits)),
        "saturated_rows": int(np.sum(saturated)),
        "saturated_fraction": float(np.mean(saturated)),
    }


def fit_primitive_hazard_models(
    dataset: PrimitiveProbeDataset,
    *,
    horizon: int,
    token_cap: int,
    token_hidden: int,
    head_hidden_1: int,
    head_hidden_2: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    seed: int,
    threads: int,
    expected_torch_version: str,
) -> dict[str, object]:
    """Fit three shared-initialization representation comparators."""
    if torch.__version__ != expected_torch_version:
        raise ValueError("preregistered PyTorch version is unavailable")
    if (
        token_cap <= 0
        or min(token_hidden, head_hidden_1, head_hidden_2) <= 0
        or epochs <= 0
        or batch_size <= 0
        or not 0.0 < learning_rate <= 1.0
        or weight_decay < 0.0
        or gradient_clip_norm <= 0.0
        or threads <= 0
    ):
        raise ValueError("primitive hazard fit settings are invalid")
    view = _view(dataset, horizon)
    if view.token_cap != token_cap:
        raise ValueError("primitive token cap differs from dataset")
    if int(np.sum(view.truncated_token_counts)) != 0:
        raise ValueError("preregistered primitive train rows may not truncate")
    positives = int(np.sum(view.hit_labels))
    if not positives or positives == view.rows:
        raise ValueError("primitive hazard target requires both classes")

    old_threads = torch.get_num_threads()
    old_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(threads)
    torch.use_deterministic_algorithms(True)
    architecture = {
        "token_width": len(TOKEN_FEATURE_NAMES),
        "token_hidden": token_hidden,
        "head_hidden_1": head_hidden_1,
        "head_hidden_2": head_hidden_2,
        "token_cap": token_cap,
        "pooling": "masked-mean-plus-maximum-plus-normalized-log-count",
        "activation": "relu",
        "output_link": "sigmoid",
    }
    torch.manual_seed(seed)
    initial = PrimitiveSetHazard(
        token_width=len(TOKEN_FEATURE_NAMES),
        token_hidden=token_hidden,
        head_hidden_1=head_hidden_1,
        head_hidden_2=head_hidden_2,
        token_cap=token_cap,
    )
    prevalence = positives / view.rows
    with torch.no_grad():
        initial.head[-1].bias.fill_(math.log(prevalence / (1.0 - prevalence)))
    initial_state = copy.deepcopy(initial.state_dict())
    parameter_count = sum(value.numel() for value in initial.parameters())

    scalar = torch.from_numpy(view.scalar_features)
    tokens = torch.from_numpy(view.primitive_tokens)
    masks = torch.from_numpy(view.primitive_masks)
    labels = torch.from_numpy(view.hit_labels.astype(np.float32))
    serialized = {}
    traces = {}
    try:
        for model_name in MODEL_NAMES:
            model = PrimitiveSetHazard(**{
                "token_width": len(TOKEN_FEATURE_NAMES),
                "token_hidden": token_hidden,
                "head_hidden_1": head_hidden_1,
                "head_hidden_2": head_hidden_2,
                "token_cap": token_cap,
            })
            model.load_state_dict(initial_state, strict=True)
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=weight_decay,
            )
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + 1)
            epoch_losses = []
            for _epoch in range(epochs):
                permutation = torch.randperm(view.rows, generator=generator)
                loss_sum = 0.0
                for start in range(0, view.rows, batch_size):
                    indices = permutation[start:start + batch_size]
                    batch_scalar, batch_mask = _variant_inputs(
                        scalar[indices], masks[indices], model_name
                    )
                    probability, _logit = model(
                        batch_scalar, tokens[indices], batch_mask
                    )
                    loss = torch.mean((probability - labels[indices]) ** 2)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                        raise ValueError("primitive hazard optimization became non-finite")
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(indices)
                epoch_losses.append(loss_sum / view.rows)
            serialized[model_name] = _serialized_model(model)
            traces[model_name] = {
                "epoch_brier": epoch_losses,
                "final_epoch_brier": epoch_losses[-1],
            }
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)

    state: dict[str, object] = {
        "schema": PRIMITIVE_HAZARD_FIT_SCHEMA,
        "model": MODEL_KIND,
        "feature_schema": PRIMITIVE_FEATURE_SCHEMA,
        "scalar_feature_names": list(PROBE_FEATURE_NAMES),
        "token_feature_names": list(TOKEN_FEATURE_NAMES),
        "architecture": architecture,
        "parameter_count_per_model": parameter_count,
        "shared_initialization": True,
        "shared_minibatch_order": True,
        "horizon_game_frames": horizon,
        "target": "physical-hit-within-fixed-horizon-under-behavior-continuation",
        "training_proper_score": "mean-unweighted-row-brier",
        "torch_version": torch.__version__,
        "fit": {
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": "adamw",
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "gradient_clip_norm": gradient_clip_norm,
            "seed": seed,
            "threads": threads,
            "dtype": "float32",
            "validation_early_stopping": False,
        },
        "models": serialized,
        "train": {
            "rows": view.rows,
            "positives": positives,
            "negatives": view.rows - positives,
            "prevalence": prevalence,
            "maximum_tokens": int(np.max(view.token_counts)),
            "truncated_tokens": int(np.sum(view.truncated_token_counts)),
            "optimization": traces,
        },
    }
    predictions = {}
    surfaces = {}
    for model_name in MODEL_NAMES:
        predictions[model_name], surfaces[model_name] = primitive_hazard_predictions(
            state,
            view.scalar_features,
            view.primitive_tokens,
            view.primitive_masks,
            model_name=model_name,
            batch_size=batch_size,
        )
    constant = np.full(view.rows, prevalence, dtype=np.float64)
    state["train"]["metrics"] = {
        **{
            name: _binary_metrics(predictions[name], view.hit_labels)
            for name in MODEL_NAMES
        },
        "constant_prevalence": _binary_metrics(constant, view.hit_labels),
    }
    state["train"]["probability_surfaces"] = surfaces
    return state


def benchmark_primitive_hazard(
    state: dict[str, object],
    dataset: PrimitiveProbeDataset,
    *,
    batch_rows: int,
    warmup_repetitions: int,
    measured_repetitions: int,
    threads: int,
) -> dict[str, object]:
    """Measure immutable scorer-only CPU latency on one fixed tensor batch."""
    if min(batch_rows, warmup_repetitions, measured_repetitions, threads) <= 0:
        raise ValueError("primitive benchmark settings are invalid")
    view = _view(dataset, int(state["horizon_game_frames"]))
    if view.rows < batch_rows:
        raise ValueError("primitive benchmark lacks rows")
    model = _load_model(state, "object_full")
    scalar = torch.from_numpy(view.scalar_features[:batch_rows])
    tokens = torch.from_numpy(view.primitive_tokens[:batch_rows])
    masks = torch.from_numpy(view.primitive_masks[:batch_rows])
    old_threads = torch.get_num_threads()
    torch.set_num_threads(threads)
    try:
        with torch.inference_mode():
            for _ in range(warmup_repetitions):
                model(scalar, tokens, masks)
            samples = []
            for _ in range(measured_repetitions):
                started = time.perf_counter_ns()
                model(scalar, tokens, masks)
                samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
    finally:
        torch.set_num_threads(old_threads)
    values = np.asarray(samples, dtype=np.float64)
    return {
        "scope": "scorer-only-fixed-batch",
        "batch_rows": batch_rows,
        "threads": threads,
        "warmup_repetitions": warmup_repetitions,
        "measured_repetitions": measured_repetitions,
        "median_ms": float(np.median(values)),
        "p99_ms": float(np.quantile(values, 0.99)),
        "maximum_ms": float(np.max(values)),
    }


def _masked_bootstrap(
    mask: np.ndarray,
    *,
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    episode_count: int,
    samples: int,
    seed: int,
) -> dict[str, object] | None:
    if not np.any(mask) or len(set(map(int, episode_indices[mask]))) != episode_count:
        return None
    return _episode_bootstrap_brier_delta(
        candidate[mask],
        baseline[mask],
        labels[mask],
        episode_indices[mask],
        episode_count=episode_count,
        samples=samples,
        seed=seed,
    )


def _episodes_favoring(
    candidate: np.ndarray,
    baseline: np.ndarray,
    labels: np.ndarray,
    episode_indices: np.ndarray,
    episode_ids: tuple[str, ...],
) -> tuple[int, list[dict[str, object]]]:
    target = labels.astype(np.float64)
    rows = []
    for index, episode_id in enumerate(episode_ids):
        members = episode_indices == index
        delta = (
            (candidate[members] - target[members]) ** 2
            - (baseline[members] - target[members]) ** 2
        )
        rows.append({
            "episode_id": episode_id,
            "rows": int(np.sum(members)),
            "candidate_minus_baseline_brier": float(np.mean(delta)),
        })
    return sum(row["candidate_minus_baseline_brier"] < 0.0 for row in rows), rows


def evaluate_primitive_hazard_models(
    state: dict[str, object],
    frozen_l2f_state: dict[str, object],
    dataset: PrimitiveProbeDataset,
    *,
    prediction_batch_size: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
    calibration_bins: int,
    minimum_overall_positives: int,
    minimum_overall_negatives: int,
    minimum_nonbaseline_positives: int,
    minimum_low_propensity_positives: int,
    minimum_prefirst_hit_positives: int,
    minimum_object_gain_episodes: int,
    minimum_overall_episodes_favoring_full: int,
    minimum_nonbaseline_episodes_favoring_full: int,
    minimum_low_propensity_episodes_favoring_full: int,
    minimum_prefirst_episodes_favoring_full: int,
    calibration_in_the_large_absolute_max: float,
    full_ece_over_action_ablated_max: float,
    maximum_saturated_fraction: float,
    maximum_parameter_count: int,
    maximum_batch18_p99_ms: float,
) -> dict[str, object]:
    """Evaluate frozen L2i once on reused complete L2d episodes."""
    horizon = int(state["horizon_game_frames"])
    view = _view(dataset, horizon)
    predictions = {}
    surfaces = {}
    for model_name in MODEL_NAMES:
        predictions[model_name], surfaces[model_name] = (
            primitive_hazard_predictions(
                state,
                view.scalar_features,
                view.primitive_tokens,
                view.primitive_masks,
                model_name=model_name,
                batch_size=prediction_batch_size,
            )
        )
    full = predictions["object_full"]
    scalar_only = predictions["scalar_only"]
    action_ablated = predictions["object_current_action_ablated"]
    frozen_l2f, frozen_surface = hazard_predictions(
        frozen_l2f_state,
        view.current_features,
        model_name="full_current_root_action",
    )
    labels = view.hit_labels
    all_rows = np.ones(view.rows, dtype=np.bool_)
    baseline_equal = np.asarray([
        published == baseline
        for published, baseline in zip(
            view.published_actions, view.baseline_actions, strict=True
        )
    ], dtype=np.bool_)
    nonbaseline = ~baseline_equal
    low_propensity = view.behavior_probabilities < 0.025
    prefirst = np.asarray(view.lifecycle_strata, dtype=object) == "pre-first-hit"
    episode_count = len(view.episode_ids)

    def action_stratum(mask: np.ndarray) -> dict[str, object]:
        return _stratum_result(
            mask,
            full=full,
            state_only=action_ablated,
            labels=labels,
            episode_indices=view.episode_indices,
            episode_ids=view.episode_ids,
            calibration_bins=calibration_bins,
        )

    strata = {
        "overall": action_stratum(all_rows),
        "published_equals_baseline": action_stratum(baseline_equal),
        "published_differs_from_baseline": action_stratum(nonbaseline),
        "behavior_propensity_below_0.025": action_stratum(low_propensity),
        "pre_first_hit": action_stratum(prefirst),
    }
    object_bootstrap = _episode_bootstrap_brier_delta(
        full,
        scalar_only,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed,
    )
    action_bootstrap = _episode_bootstrap_brier_delta(
        full,
        action_ablated,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 1,
    )
    frozen_bootstrap = _episode_bootstrap_brier_delta(
        full,
        frozen_l2f,
        labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=bootstrap_samples,
        seed=bootstrap_seed + 2,
    )
    boundary_masks = {
        "nonbaseline": nonbaseline,
        "low_propensity": low_propensity,
        "prefirst_hit": prefirst,
    }
    boundary_bootstraps = {
        name: _masked_bootstrap(
            mask,
            candidate=full,
            baseline=action_ablated,
            labels=labels,
            episode_indices=view.episode_indices,
            episode_count=episode_count,
            samples=bootstrap_samples,
            seed=bootstrap_seed + offset,
        )
        for offset, (name, mask) in enumerate(boundary_masks.items(), start=3)
    }
    object_favorable, object_per_episode = _episodes_favoring(
        full,
        scalar_only,
        labels,
        view.episode_indices,
        view.episode_ids,
    )
    calibration = {
        name: _calibration_summary(value, labels, bins=calibration_bins)
        for name, value in predictions.items()
    }
    positives = int(np.sum(labels))
    support = (
        positives >= minimum_overall_positives
        and view.rows - positives >= minimum_overall_negatives
        and int(np.min(np.bincount(
            view.episode_indices, minlength=episode_count
        ))) > 0
    )
    primitive_contract = int(np.sum(view.truncated_token_counts)) == 0
    object_gain = (
        float(object_bootstrap["upper_95"]) < 0.0
        and object_favorable >= minimum_object_gain_episodes
    )
    action_signal = (
        float(action_bootstrap["upper_95"]) < 0.0
        and int(strata["overall"]["episodes_favoring_full"])
        >= minimum_overall_episodes_favoring_full
    )

    def boundary_gate(
        result: dict[str, object],
        interval: dict[str, object] | None,
        *,
        minimum_positives: int,
        minimum_favorable: int,
    ) -> bool:
        return (
            int(result["positives"]) >= minimum_positives
            and int(result["episode_count"]) == episode_count
            and interval is not None
            and float(interval["upper_95"]) < 0.0
            and int(result["episodes_favoring_full"]) >= minimum_favorable
        )

    nonbaseline_signal = boundary_gate(
        strata["published_differs_from_baseline"],
        boundary_bootstraps["nonbaseline"],
        minimum_positives=minimum_nonbaseline_positives,
        minimum_favorable=minimum_nonbaseline_episodes_favoring_full,
    )
    low_propensity_signal = boundary_gate(
        strata["behavior_propensity_below_0.025"],
        boundary_bootstraps["low_propensity"],
        minimum_positives=minimum_low_propensity_positives,
        minimum_favorable=minimum_low_propensity_episodes_favoring_full,
    )
    lifecycle_signal = boundary_gate(
        strata["pre_first_hit"],
        boundary_bootstraps["prefirst_hit"],
        minimum_positives=minimum_prefirst_hit_positives,
        minimum_favorable=minimum_prefirst_episodes_favoring_full,
    )
    full_calibration = calibration["object_full"]
    ablated_calibration = calibration["object_current_action_ablated"]
    calibration_ready = (
        abs(float(full_calibration["calibration_in_the_large"]))
        <= calibration_in_the_large_absolute_max
        and float(full_calibration["expected_calibration_error"])
        - float(ablated_calibration["expected_calibration_error"])
        <= full_ece_over_action_ablated_max
    )
    surface_ready = (
        float(surfaces["object_full"]["saturated_fraction"])
        <= maximum_saturated_fraction
    )
    benchmark = state["train"]["inference_benchmark"]
    export_cost_ready = (
        int(state["parameter_count_per_model"]) <= maximum_parameter_count
        and int(benchmark["batch_rows"]) == 18
        and int(benchmark["threads"]) == 1
        and float(benchmark["p99_ms"]) <= maximum_batch18_p99_ms
    )
    selected = all((
        support,
        primitive_contract,
        export_cost_ready,
        surface_ready,
        object_gain,
        action_signal,
        nonbaseline_signal,
        low_propensity_signal,
        lifecycle_signal,
        calibration_ready,
    ))
    return {
        "schema": PRIMITIVE_HAZARD_EVALUATION_SCHEMA,
        "horizon_game_frames": horizon,
        "rows": {
            "total": view.rows,
            "positives": positives,
            "negatives": view.rows - positives,
        },
        "primitive_contract": {
            "token_cap": view.token_cap,
            "maximum_observed_tokens": int(np.max(view.token_counts)),
            "truncated_rows": int(np.sum(view.truncated_token_counts > 0)),
            "truncated_tokens": int(np.sum(view.truncated_token_counts)),
        },
        "metrics": {
            **{
                name: _binary_metrics(value, labels)
                for name, value in predictions.items()
            },
            "frozen_l2f_full_same_rows": _binary_metrics(frozen_l2f, labels),
        },
        "calibration": calibration,
        "probability_surfaces": {
            **surfaces,
            "frozen_l2f_full_same_rows": frozen_surface,
        },
        "whole_episode_bootstrap": {
            "object_full_minus_scalar_only_brier": {
                **object_bootstrap,
                "episodes_favoring_object": object_favorable,
                "per_episode": object_per_episode,
            },
            "object_full_minus_current_action_ablated_brier": action_bootstrap,
            "object_full_minus_frozen_l2f_brier": frozen_bootstrap,
            "nonbaseline_full_minus_action_ablated_brier": (
                boundary_bootstraps["nonbaseline"]
            ),
            "low_propensity_full_minus_action_ablated_brier": (
                boundary_bootstraps["low_propensity"]
            ),
            "prefirst_hit_full_minus_action_ablated_brier": (
                boundary_bootstraps["prefirst_hit"]
            ),
        },
        "strata": strata,
        "gates": {
            "overall_support_sufficient": support,
            "primitive_contract_passed": primitive_contract,
            "export_cost_passed": export_cost_ready,
            "probability_surface_passed": surface_ready,
            "object_set_gain": object_gain,
            "current_action_signal": action_signal,
            "nonbaseline_action_signal": nonbaseline_signal,
            "low_propensity_action_signal": low_propensity_signal,
            "prefirst_hit_lifecycle_signal": lifecycle_signal,
            "calibration_readiness_passed": calibration_ready,
            "selected_for_fresh_confirmation": selected,
        },
        "summary": {
            "decision": (
                "select-observed-primitive-set-h16-hazard-for-fresh-confirmation"
                if selected
                else "reject-observed-primitive-set-h16-hazard"
            ),
            "independent_confirmation": False,
            "counterfactual_successors": False,
            "causal_action_value_claimed": False,
            "object_set_tested": True,
            "object_set_selected": selected,
            "history_admitted": False,
            "value_learning_admitted": False,
            "online_policy_admitted": False,
        },
    }
