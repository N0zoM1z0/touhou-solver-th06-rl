"""BCE-with-logits correction for the frozen L2i primitive-set model."""

from __future__ import annotations

import copy
import math

import numpy as np
import torch
from torch import nn

from .factual_primitive_dataset import (
    PRIMITIVE_FEATURE_SCHEMA,
    TOKEN_FEATURE_NAMES,
    PrimitiveProbeDataset,
)
from .factual_primitive_hazard_model import (
    MODEL_NAMES,
    PRIMITIVE_HAZARD_FIT_SCHEMA,
    PrimitiveSetHazard,
    _episodes_favoring,
    _serialized_model,
    _variant_inputs,
    _view,
    benchmark_primitive_hazard,
    evaluate_primitive_hazard_models,
    primitive_hazard_predictions,
)
from .factual_probes import (
    PROBE_FEATURE_NAMES,
    _binary_metrics,
    _episode_bootstrap_brier_delta,
)


PRIMITIVE_LOG_HAZARD_FIT_SCHEMA = (
    "th06-rl-l2j-primitive-set-log-hazard-fit-v1"
)
PRIMITIVE_LOG_HAZARD_EVALUATION_SCHEMA = (
    "th06-rl-l2j-primitive-set-log-hazard-evaluation-v1"
)
MODEL_KIND = "shared-sigmoid-deepsets-bernoulli-log-regressor"
TRAINING_PROPER_SCORE = "mean-unweighted-bernoulli-log-score-with-logits"


def _as_l2i_state(state: dict[str, object]) -> dict[str, object]:
    if state.get("schema") != PRIMITIVE_LOG_HAZARD_FIT_SCHEMA:
        raise ValueError("primitive log-hazard fit schema mismatch")
    adapted = dict(state)
    adapted["schema"] = PRIMITIVE_HAZARD_FIT_SCHEMA
    return adapted


def primitive_log_hazard_predictions(
    state: dict[str, object],
    scalar: np.ndarray,
    tokens: np.ndarray,
    masks: np.ndarray,
    *,
    model_name: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[str, object]]:
    """Evaluate the loss-only model through the frozen L2i scorer."""
    return primitive_hazard_predictions(
        _as_l2i_state(state),
        scalar,
        tokens,
        masks,
        model_name=model_name,
        batch_size=batch_size,
    )


def fit_primitive_log_hazard_models(
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
    """Fit the three L2i comparators with unweighted BCE-with-logits only."""
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
        raise ValueError("primitive log-hazard fit settings are invalid")
    view = _view(dataset, horizon)
    if view.token_cap != token_cap:
        raise ValueError("primitive token cap differs from dataset")
    if int(np.sum(view.truncated_token_counts)) != 0:
        raise ValueError("preregistered primitive train rows may not truncate")
    positives = int(np.sum(view.hit_labels))
    if not positives or positives == view.rows:
        raise ValueError("primitive log-hazard target requires both classes")

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
    criterion = nn.BCEWithLogitsLoss(reduction="mean")
    serialized = {}
    traces = {}
    try:
        for model_name in MODEL_NAMES:
            model = PrimitiveSetHazard(
                token_width=len(TOKEN_FEATURE_NAMES),
                token_hidden=token_hidden,
                head_hidden_1=head_hidden_1,
                head_hidden_2=head_hidden_2,
                token_cap=token_cap,
            )
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
                    _probability, logits = model(
                        batch_scalar, tokens[indices], batch_mask
                    )
                    loss = criterion(logits, labels[indices])
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    gradient_norm = torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm
                    )
                    if not torch.isfinite(loss) or not torch.isfinite(gradient_norm):
                        raise ValueError(
                            "primitive log-hazard optimization became non-finite"
                        )
                    optimizer.step()
                    loss_sum += float(loss.detach()) * len(indices)
                epoch_losses.append(loss_sum / view.rows)
            serialized[model_name] = _serialized_model(model)
            traces[model_name] = {
                "epoch_negative_log_likelihood": epoch_losses,
                "final_epoch_negative_log_likelihood": epoch_losses[-1],
            }
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_deterministic)

    state: dict[str, object] = {
        "schema": PRIMITIVE_LOG_HAZARD_FIT_SCHEMA,
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
        "training_proper_score": TRAINING_PROPER_SCORE,
        "positive_class_weight": 1.0,
        "negative_class_weight": 1.0,
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
        predictions[model_name], surfaces[model_name] = (
            primitive_log_hazard_predictions(
                state,
                view.scalar_features,
                view.primitive_tokens,
                view.primitive_masks,
                model_name=model_name,
                batch_size=batch_size,
            )
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


def benchmark_primitive_log_hazard(
    state: dict[str, object],
    dataset: PrimitiveProbeDataset,
    **kwargs,
) -> dict[str, object]:
    return benchmark_primitive_hazard(_as_l2i_state(state), dataset, **kwargs)


def evaluate_primitive_log_hazard_models(
    state: dict[str, object],
    frozen_l2f_state: dict[str, object],
    frozen_l2i_state: dict[str, object],
    dataset: PrimitiveProbeDataset,
    *,
    minimum_loss_correction_episodes: int,
    **kwargs,
) -> dict[str, object]:
    """Apply the frozen L2i gates plus a paired loss-only improvement gate."""
    base = evaluate_primitive_hazard_models(
        _as_l2i_state(state), frozen_l2f_state, dataset, **kwargs
    )
    view = _view(dataset, int(state["horizon_game_frames"]))
    batch_size = int(kwargs["prediction_batch_size"])
    full, _surface = primitive_log_hazard_predictions(
        state,
        view.scalar_features,
        view.primitive_tokens,
        view.primitive_masks,
        model_name="object_full",
        batch_size=batch_size,
    )
    frozen_l2i, frozen_surface = primitive_hazard_predictions(
        frozen_l2i_state,
        view.scalar_features,
        view.primitive_tokens,
        view.primitive_masks,
        model_name="object_full",
        batch_size=batch_size,
    )
    episode_count = len(view.episode_ids)
    bootstrap = _episode_bootstrap_brier_delta(
        full,
        frozen_l2i,
        view.hit_labels,
        view.episode_indices,
        episode_count=episode_count,
        samples=int(kwargs["bootstrap_samples"]),
        seed=int(kwargs["bootstrap_seed"]) + 6,
    )
    favorable, per_episode = _episodes_favoring(
        full,
        frozen_l2i,
        view.hit_labels,
        view.episode_indices,
        view.episode_ids,
    )
    loss_correction = (
        float(bootstrap["upper_95"]) < 0.0
        and favorable >= minimum_loss_correction_episodes
    )
    prior_selection = bool(base["gates"]["selected_for_fresh_confirmation"])
    selected = prior_selection and loss_correction
    base["schema"] = PRIMITIVE_LOG_HAZARD_EVALUATION_SCHEMA
    base["metrics"]["frozen_l2i_object_full_same_rows"] = _binary_metrics(
        frozen_l2i, view.hit_labels
    )
    base["probability_surfaces"][
        "frozen_l2i_object_full_same_rows"
    ] = frozen_surface
    base["whole_episode_bootstrap"][
        "logscore_object_full_minus_frozen_l2i_object_full_brier"
    ] = {
        **bootstrap,
        "episodes_favoring_logscore": favorable,
        "per_episode": per_episode,
    }
    base["gates"]["l2i_base_selection_gates_passed"] = prior_selection
    base["gates"]["loss_only_improves_frozen_l2i"] = loss_correction
    base["gates"]["selected_for_fresh_confirmation"] = selected
    base["summary"].update({
        "decision": (
            "select-logscore-observed-primitive-h16-hazard-for-fresh-confirmation"
            if selected
            else "reject-logscore-observed-primitive-h16-hazard"
        ),
        "object_set_selected": selected,
        "logscore_probability_model_tested": True,
        "loss_only_improves_frozen_l2i": loss_correction,
    })
    return base
