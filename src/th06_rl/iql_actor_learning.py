"""Behavior-supported IQL actor extraction from factual implicit-Q members."""

from __future__ import annotations

from dataclasses import dataclass
import ctypes
import hashlib
import math
from pathlib import Path

from .advantage_learning import OptionStep
from .low_rank_learning import FeatureRoleLayout


NATIVE_ACTOR_ABSOLUTE_TOLERANCE = 1e-4
NATIVE_ACTOR_FLOAT32_RELATIVE_TOLERANCE = 4.0 * 1.1920928955078125e-7


@dataclass(frozen=True)
class ActorArrays:
    states: object
    actions: object
    masks: object
    factual: object
    behavior_probabilities: object
    episode_ids: tuple[str, ...]
    base_weights: object
    state_mean: object
    state_scale: object
    action_mean: object
    action_scale: object


@dataclass(frozen=True)
class IqlActorModel:
    layout: FeatureRoleLayout
    state_mean: object
    state_scale: object
    action_mean: object
    action_scale: object
    state_hidden_weight: object
    state_hidden_bias: object
    state_latent_weight: object
    state_latent_bias: object
    action_hidden_weight: object
    action_hidden_bias: object
    action_latent_weight: object
    action_latent_bias: object
    action_score_weight: object
    action_score_bias: float

    def predict(self, rows):
        import numpy as np

        matrix = np.asarray(rows, dtype=np.float32)
        states = matrix[:, self.layout.state_indices]
        actions = matrix[:, self.layout.action_indices]
        if not np.allclose(states, states[0], atol=1e-7):
            raise ValueError("actor candidates do not share one factual state")
        state = (states[0] - self.state_mean) / self.state_scale
        action = (actions - self.action_mean) / self.action_scale
        state_hidden = np.tanh(
            state @ self.state_hidden_weight + self.state_hidden_bias
        )
        state_latent = (
            state_hidden @ self.state_latent_weight + self.state_latent_bias
        )
        action_hidden = np.tanh(
            action @ self.action_hidden_weight + self.action_hidden_bias
        )
        action_latent = (
            action_hidden @ self.action_latent_weight + self.action_latent_bias
        )
        return (
            np.sum(action_latent * state_latent[None, :], axis=1)
            / math.sqrt(len(state_latent))
            + action_hidden @ self.action_score_weight
            + self.action_score_bias
        )


@dataclass(frozen=True)
class IqlActorMember:
    model: IqlActorModel
    bootstrap: dict[str, int]
    advantage_scale: float
    diagnostics: dict[str, float]


def categorical_kl_from_logits(probabilities, logits) -> float:
    """Finite KL(mu || softmax(logits)) without probability underflow."""
    import numpy as np

    probabilities = np.asarray(probabilities, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if (
        probabilities.ndim != 1
        or logits.shape != probabilities.shape
        or not np.isfinite(probabilities).all()
        or not np.isfinite(logits).all()
        or (probabilities <= 0.0).any()
        or not math.isclose(float(probabilities.sum()), 1.0, rel_tol=1e-9)
    ):
        raise ValueError("categorical KL inputs are invalid")
    shifted = logits - logits.max()
    log_probabilities = shifted - math.log(float(np.exp(shifted).sum()))
    return float(np.sum(
        probabilities * (np.log(probabilities) - log_probabilities)
    ))


def native_actor_prediction_tolerance_ratio(expected, actual) -> float:
    """Return the worst float32-aware native/portable error ratio.

    Dense float32 accumulation error scales with the magnitude of a logit, so
    a pure absolute threshold is not closed under otherwise irrelevant actor
    score scaling.  Action conformance remains a separate exact gate.
    """
    import numpy as np

    expected = np.asarray(expected, dtype=np.float64)
    actual = np.asarray(actual, dtype=np.float64)
    if (
        expected.shape != actual.shape
        or not expected.size
        or not np.isfinite(expected).all()
        or not np.isfinite(actual).all()
    ):
        raise ValueError("native actor predictions are invalid")
    scale = np.maximum(np.abs(expected), np.abs(actual))
    tolerance = (
        NATIVE_ACTOR_ABSOLUTE_TOLERANCE
        + NATIVE_ACTOR_FLOAT32_RELATIVE_TOLERANCE * scale
    )
    return float(np.max(np.abs(expected - actual) / tolerance))


def iql_actor_model_artifact(model: IqlActorModel) -> dict[str, object]:
    """Serialize one immutable portable actor without framework state."""
    import numpy as np

    def values(array):
        return np.asarray(array, dtype=np.float32).tolist()

    return {
        "schema": "autonomous-iql-actor-model-v1",
        "feature_names": list(model.layout.names),
        "state_indices": list(model.layout.state_indices),
        "action_indices": list(model.layout.action_indices),
        "state_mean": values(model.state_mean),
        "state_scale": values(model.state_scale),
        "action_mean": values(model.action_mean),
        "action_scale": values(model.action_scale),
        "state_hidden_weight": values(model.state_hidden_weight),
        "state_hidden_bias": values(model.state_hidden_bias),
        "state_latent_weight": values(model.state_latent_weight),
        "state_latent_bias": values(model.state_latent_bias),
        "action_hidden_weight": values(model.action_hidden_weight),
        "action_hidden_bias": values(model.action_hidden_bias),
        "action_latent_weight": values(model.action_latent_weight),
        "action_latent_bias": values(model.action_latent_bias),
        "action_score_weight": values(model.action_score_weight),
        "action_score_bias": float(model.action_score_bias),
    }


def iql_actor_model_from_artifact(artifact: dict[str, object]) -> IqlActorModel:
    import numpy as np

    if artifact.get("schema") != "autonomous-iql-actor-model-v1":
        raise ValueError("unsupported IQL actor artifact")
    names = tuple(map(str, artifact.get("feature_names", ())))
    state_indices = tuple(map(int, artifact.get("state_indices", ())))
    action_indices = tuple(map(int, artifact.get("action_indices", ())))
    if (
        not names or not state_indices or not action_indices
        or set(state_indices) & set(action_indices)
        or sorted((*state_indices, *action_indices)) != list(range(len(names)))
    ):
        raise ValueError("IQL actor feature layout is invalid")
    layout = FeatureRoleLayout(
        names=names,
        state_indices=state_indices,
        action_indices=action_indices,
    )

    def array(name: str, shape: tuple[int, ...]):
        value = np.asarray(artifact.get(name), dtype=np.float32)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"IQL actor {name} shape is invalid")
        return value

    state_count, action_count = len(state_indices), len(action_indices)
    state_hidden_bias = np.asarray(
        artifact.get("state_hidden_bias"), dtype=np.float32
    )
    state_latent_bias = np.asarray(
        artifact.get("state_latent_bias"), dtype=np.float32
    )
    if state_hidden_bias.ndim != 1 or state_latent_bias.ndim != 1:
        raise ValueError("IQL actor hidden dimensions are invalid")
    hidden, rank = len(state_hidden_bias), len(state_latent_bias)
    model = IqlActorModel(
        layout=layout,
        state_mean=array("state_mean", (state_count,)),
        state_scale=array("state_scale", (state_count,)),
        action_mean=array("action_mean", (action_count,)),
        action_scale=array("action_scale", (action_count,)),
        state_hidden_weight=array(
            "state_hidden_weight", (state_count, hidden)
        ),
        state_hidden_bias=array("state_hidden_bias", (hidden,)),
        state_latent_weight=array("state_latent_weight", (hidden, rank)),
        state_latent_bias=array("state_latent_bias", (rank,)),
        action_hidden_weight=array(
            "action_hidden_weight", (action_count, hidden)
        ),
        action_hidden_bias=array("action_hidden_bias", (hidden,)),
        action_latent_weight=array("action_latent_weight", (hidden, rank)),
        action_latent_bias=array("action_latent_bias", (rank,)),
        action_score_weight=array("action_score_weight", (hidden,)),
        action_score_bias=float(artifact.get("action_score_bias", math.nan)),
    )
    if (
        not math.isfinite(model.action_score_bias)
        or (model.state_scale <= 0.0).any()
        or (model.action_scale <= 0.0).any()
    ):
        raise ValueError("IQL actor normalization or bias is invalid")
    return model


class NativeIqlActorPopulation:
    """One-call native dense scorer for the complete seven-actor teacher."""

    def __init__(
        self,
        path: Path,
        *,
        expected_sha256: str,
        models: list[IqlActorModel],
    ) -> None:
        import numpy as np

        if not models:
            raise ValueError("native IQL actor population is empty")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            raise ValueError("native IQL actor scorer SHA-256 mismatch")
        reference = models[0]
        for model in models[1:]:
            if model.layout != reference.layout or any(
                not np.array_equal(getattr(model, name), getattr(reference, name))
                for name in (
                    "state_mean", "state_scale", "action_mean", "action_scale"
                )
            ):
                raise ValueError("native IQL actor population layout drifted")
        hidden = len(reference.state_hidden_bias)
        rank = len(reference.state_latent_bias)
        if hidden > 256 or rank > 128:
            raise ValueError("native IQL actor dimensions exceed fixed kernel")
        library = ctypes.CDLL(str(path))
        function = library.th06_rl_score_iql_actor_population_v1
        pointer = ctypes.POINTER(ctypes.c_float)
        function.argtypes = [
            pointer, pointer,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            ctypes.c_int32, ctypes.c_int32, ctypes.c_int32,
            *(pointer for _index in range(10)),
            pointer,
        ]
        function.restype = ctypes.c_int

        def packed(name: str):
            values = np.concatenate([
                np.asarray(getattr(model, name), dtype=np.float32).reshape(-1)
                for model in models
            ])
            return (ctypes.c_float * len(values))(*values)

        self.library = library
        self.function = function
        self.models = tuple(models)
        self.layout = reference.layout
        self.state_mean = np.asarray(reference.state_mean, dtype=np.float32)
        self.state_scale = np.asarray(reference.state_scale, dtype=np.float32)
        self.action_mean = np.asarray(reference.action_mean, dtype=np.float32)
        self.action_scale = np.asarray(reference.action_scale, dtype=np.float32)
        self.model_count = len(models)
        self.hidden = hidden
        self.rank = rank
        self.arrays = tuple(packed(name) for name in (
            "state_hidden_weight", "state_hidden_bias",
            "state_latent_weight", "state_latent_bias",
            "action_hidden_weight", "action_hidden_bias",
            "action_latent_weight", "action_latent_bias",
            "action_score_weight", "action_score_bias",
        ))

    def predict(self, rows) -> tuple[tuple[float, ...], ...]:
        import numpy as np

        matrix = np.asarray(rows, dtype=np.float32)
        states = matrix[:, self.layout.state_indices]
        if not np.allclose(states, states[0], atol=1e-7):
            raise ValueError("native actor candidates changed factual state")
        state = (states[0] - self.state_mean) / self.state_scale
        actions = (
            matrix[:, self.layout.action_indices] - self.action_mean
        ) / self.action_scale
        state_input = (ctypes.c_float * len(state))(*state)
        flat_actions = actions.reshape(-1)
        action_input = (ctypes.c_float * len(flat_actions))(*flat_actions)
        output = (ctypes.c_float * (self.model_count * len(matrix)))()
        status = self.function(
            state_input,
            action_input,
            len(matrix),
            len(state),
            actions.shape[1],
            self.model_count,
            self.hidden,
            self.rank,
            *self.arrays,
            output,
        )
        if status != 0:
            raise RuntimeError(f"native IQL actor scorer failed with status {status}")
        return tuple(
            tuple(float(output[model * len(matrix) + row]) for row in range(len(matrix)))
            for model in range(self.model_count)
        )


def actor_arrays(
    samples: list[OptionStep], layout: FeatureRoleLayout
) -> ActorArrays:
    import numpy as np
    from .implicit_learning import _episodes

    episodes = _episodes(samples)
    maximum = max(len(sample.legal_actions) for sample in samples)
    states = []
    actions = np.zeros(
        (len(samples), maximum, len(layout.action_indices)), dtype=np.float32
    )
    masks = np.zeros((len(samples), maximum), dtype=np.bool_)
    factual = np.zeros(len(samples), dtype=np.int64)
    behavior_probabilities = np.zeros(
        (len(samples), maximum), dtype=np.float32
    )
    episode_ids = []
    base_weights = []
    mean_options = sum(map(len, episodes.values())) / len(episodes)
    index = 0
    all_action_rows = []
    for episode, rows in episodes.items():
        episode_weight = mean_options / len(rows)
        for sample in rows:
            candidates = np.asarray(sample.candidate_vectors, dtype=np.float32)
            if candidates.shape[1] != len(layout.names):
                raise ValueError("actor feature schema and candidate width differ")
            state_candidates = candidates[:, layout.state_indices]
            if not np.allclose(state_candidates, state_candidates[0], atol=1e-7):
                raise ValueError("actor candidates changed state-only features")
            action_rows = candidates[:, layout.action_indices]
            count = len(action_rows)
            states.append(state_candidates[0])
            actions[index, :count] = action_rows
            masks[index, :count] = True
            factual[index] = sample.legal_actions.index(sample.action)
            probabilities = np.asarray(
                sample.behavior_probabilities, dtype=np.float64
            )
            if (
                len(probabilities) != count
                or np.any(~np.isfinite(probabilities))
                or np.any(probabilities <= 0.0)
                or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9)
            ):
                raise ValueError("IQL actor needs complete behavior propensities")
            behavior_probabilities[index, :count] = probabilities
            episode_ids.append(episode)
            base_weights.append(episode_weight)
            all_action_rows.append(action_rows)
            index += 1
    state_values = np.asarray(states, dtype=np.float32)
    action_values = np.concatenate(all_action_rows, axis=0)
    state_mean = state_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    state_scale = state_values.std(axis=0, dtype=np.float64).astype(np.float32)
    action_mean = action_values.mean(axis=0, dtype=np.float64).astype(np.float32)
    action_scale = action_values.std(axis=0, dtype=np.float64).astype(np.float32)
    state_scale[state_scale < 1e-6] = 1.0
    action_scale[action_scale < 1e-6] = 1.0
    return ActorArrays(
        states=state_values,
        actions=actions,
        masks=masks,
        factual=factual,
        behavior_probabilities=behavior_probabilities,
        episode_ids=tuple(episode_ids),
        base_weights=np.asarray(base_weights, dtype=np.float32),
        state_mean=state_mean,
        state_scale=state_scale,
        action_mean=action_mean,
        action_scale=action_scale,
    )


def factual_cost_advantage(
    samples: list[OptionStep], member
) -> object:
    """Return Q(s,A)-V(s) using factual propensity-centered action effect."""
    import numpy as np
    from .implicit_learning import _arrays, _centered_layout, _episodes

    episodes = _episodes(samples)
    _q, offline_states, row_episodes, _weights = _arrays(episodes)
    effect_rows, coefficients, starts, effect_episodes, _samples = (
        _centered_layout(episodes)
    )
    if row_episodes != effect_episodes:
        raise RuntimeError("actor Bellman and centered layouts differ")
    common = member.outcome_model.predict(offline_states)
    raw_effect = member.q_model.predict(effect_rows)
    centered = np.add.reduceat(coefficients * raw_effect, starts)
    factual_q = common + centered
    value = member.value_model.predict(offline_states)
    advantage = np.asarray(factual_q - value, dtype=np.float32)
    if not np.all(np.isfinite(advantage)):
        raise ValueError("IQL actor advantage is non-finite")
    return advantage


def _weighted_rms(values, weights) -> float:
    import numpy as np

    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError("actor bootstrap removed every episode")
    result = math.sqrt(float(np.average(
        np.square(values[positive], dtype=np.float64),
        weights=weights[positive],
    )))
    return max(result, 1e-3)


def action_centered_actor_losses(
    factual_losses, behavior_losses, advantage_weights
):
    """Unbiased known-behavior control variate for factual AWR losses.

    For any state and any fixed action loss ``L``, the factual action is drawn
    from the known behavior distribution ``mu``.  Therefore

    ``w(A)L(A) - (L(A) - E_mu[L])``

    has expectation ``E_mu[w(A)L(A)]`` without an inverse propensity.  Do not
    center the entire ``(w - 1)L`` term: ``w`` is not conditionally normalized
    at every state and doing so introduces bias.
    """
    return behavior_losses + (advantage_weights - 1.0) * factual_losses


def fit_iql_actor_member(
    samples: list[OptionStep],
    critic_member=None,
    *,
    layout: FeatureRoleLayout,
    prepared: ActorArrays,
    seed: int,
    threads: int,
    advantage=None,
    bootstrap: dict[str, int] | None = None,
    hidden: int = 64,
    rank: int = 24,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    log_weight_clip: float = 4.0,
) -> IqlActorMember:
    import numpy as np
    import torch

    if min(threads, hidden, rank, epochs, batch_size) < 1:
        raise ValueError("IQL actor dimensions and resources must be positive")
    if log_weight_clip <= 0.0:
        raise ValueError("IQL actor log-weight clip must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    if advantage is None:
        if critic_member is None:
            raise ValueError("actor needs a critic or frozen advantages")
        advantage = factual_cost_advantage(samples, critic_member)
    else:
        advantage = np.asarray(advantage, dtype=np.float32)
        if advantage.shape != (len(prepared.episode_ids),):
            raise ValueError("frozen actor advantages and options differ")
    if bootstrap is None:
        if critic_member is None:
            raise ValueError("actor needs an explicit episode bootstrap")
        bootstrap = critic_member.bootstrap
    bootstrap_weights = np.asarray([
        bootstrap.get(episode, 0)
        for episode in prepared.episode_ids
    ], dtype=np.float32)
    group_weights = prepared.base_weights * bootstrap_weights
    temperature = _weighted_rms(advantage, group_weights)
    log_weights = np.clip(
        -advantage / temperature, -log_weight_clip, log_weight_clip
    )
    raw_advantage_weights = np.exp(log_weights).astype(np.float32)
    positive = group_weights > 0.0
    normalization = float(np.average(
        raw_advantage_weights[positive], weights=group_weights[positive]
    ))
    normalized_advantage_weights = raw_advantage_weights / normalization

    states = (
        np.asarray(prepared.states) - prepared.state_mean
    ) / prepared.state_scale
    actions = (
        np.asarray(prepared.actions) - prepared.action_mean
    ) / prepared.action_scale
    state_tensor = torch.from_numpy(states.astype(np.float32, copy=False))
    action_tensor = torch.from_numpy(actions.astype(np.float32, copy=False))
    mask_tensor = torch.from_numpy(np.asarray(prepared.masks))
    factual_tensor = torch.from_numpy(np.asarray(prepared.factual))
    propensity_tensor = torch.from_numpy(
        np.asarray(prepared.behavior_probabilities)
    )
    base_weight_tensor = torch.from_numpy(group_weights)
    advantage_weight_tensor = torch.from_numpy(normalized_advantage_weights)

    class Actor(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state_hidden = torch.nn.Linear(states.shape[1], hidden)
            self.state_latent = torch.nn.Linear(hidden, rank)
            self.action_hidden = torch.nn.Linear(actions.shape[2], hidden)
            self.action_latent = torch.nn.Linear(hidden, rank)
            self.action_score = torch.nn.Linear(hidden, 1)

        def forward(self, state, action):
            state_hidden = torch.tanh(self.state_hidden(state))
            state_latent = self.state_latent(state_hidden)
            action_hidden = torch.tanh(self.action_hidden(action))
            action_latent = self.action_latent(action_hidden)
            return (
                (action_latent * state_latent[:, None, :]).sum(dim=2)
                / math.sqrt(rank)
                + self.action_score(action_hidden).squeeze(2)
            )

    model = Actor()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    indices = torch.nonzero(base_weight_tensor > 0.0, as_tuple=False).squeeze(1)
    generator = torch.Generator().manual_seed(seed + 1)
    final_loss = math.inf
    for _epoch in range(epochs):
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        weighted_loss = 0.0
        weight_sum = 0.0
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            logits = model(state_tensor[batch], action_tensor[batch])
            logits = logits.masked_fill(~mask_tensor[batch], float("-inf"))
            log_probabilities = torch.nn.functional.log_softmax(logits, dim=1)
            factual_losses = -log_probabilities.gather(
                1, factual_tensor[batch, None]
            ).squeeze(1)
            behavior_losses = -(
                propensity_tensor[batch]
                * torch.where(
                    mask_tensor[batch], log_probabilities, 0.0
                )
            ).sum(dim=1)
            # Exact known-behavior expectation plus a zero-mean factual-action
            # control variate.  This is unbiased for advantage-weighted
            # behavior cloning and never divides by propensity.
            centered_losses = action_centered_actor_losses(
                factual_losses,
                behavior_losses,
                advantage_weight_tensor[batch],
            )
            weights = base_weight_tensor[batch]
            loss = (weights * centered_losses).sum() / weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            weighted_loss += float(loss.detach()) * float(weights.sum())
            weight_sum += float(weights.sum())
        final_loss = weighted_loss / weight_sum

    def array(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32, copy=True)

    artifact = IqlActorModel(
        layout=layout,
        state_mean=prepared.state_mean,
        state_scale=prepared.state_scale,
        action_mean=prepared.action_mean,
        action_scale=prepared.action_scale,
        state_hidden_weight=array(model.state_hidden.weight).T,
        state_hidden_bias=array(model.state_hidden.bias),
        state_latent_weight=array(model.state_latent.weight).T,
        state_latent_bias=array(model.state_latent.bias),
        action_hidden_weight=array(model.action_hidden.weight).T,
        action_hidden_bias=array(model.action_hidden.bias),
        action_latent_weight=array(model.action_latent.weight).T,
        action_latent_bias=array(model.action_latent.bias),
        action_score_weight=array(model.action_score.weight).reshape(-1),
        action_score_bias=float(array(model.action_score.bias)[0]),
    )
    return IqlActorMember(
        model=artifact,
        bootstrap=dict(sorted(bootstrap.items())),
        advantage_scale=temperature,
        diagnostics={
            # A finite-sample control-variate estimate may be negative even
            # though the underlying cross entropy is nonnegative.
            "action_centered_risk_estimate": float(final_loss),
            "advantage_rms": temperature,
            "minimum_log_weight": float(log_weights.min()),
            "maximum_log_weight": float(log_weights.max()),
            "advantage_weight_mean_before_normalization": normalization,
            "advantage_weight_ess": float(
                np.square((group_weights * normalized_advantage_weights).sum())
                / np.square(
                    group_weights * normalized_advantage_weights,
                    dtype=np.float64,
                ).sum()
            ),
            "effective_groups": float(np.count_nonzero(group_weights)),
            "parameters": float(sum(value.numel() for value in model.parameters())),
        },
    )


def fit_iql_actor_population(
    samples: list[OptionStep],
    critic_population: list[object],
    *,
    layout: FeatureRoleLayout,
    seed: int = 460_813,
    threads: int = 1,
    hidden: int = 64,
    rank: int = 24,
    epochs: int = 10,
    batch_size: int = 512,
    learning_rate: float = 1e-3,
    log_weight_clip: float = 4.0,
) -> list[IqlActorMember]:
    if not critic_population:
        raise ValueError("IQL actor population needs critic members")
    prepared = actor_arrays(samples, layout)
    return [
        fit_iql_actor_member(
            samples,
            critic_member,
            layout=layout,
            prepared=prepared,
            seed=seed + member * 10_000,
            threads=threads,
            hidden=hidden,
            rank=rank,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            log_weight_clip=log_weight_clip,
        )
        for member, critic_member in enumerate(critic_population)
    ]


def cross_fitted_factual_advantages(
    samples: list[OptionStep],
    *,
    folds: int = 3,
    critic_iterations: int = 2,
    n_step_options: int = 8,
    q_trees: int = 8,
    value_trees: int = 8,
    seed: int = 860_813,
    threads: int = 1,
) -> tuple[object, dict[str, object]]:
    """Generate every AWR label with a critic excluding its entire episode."""
    import numpy as np
    from .advantage_learning import _folds
    from .implicit_learning import _episodes, fit_implicit_q_member

    episodes = _episodes(samples)
    groups = list(episodes)
    if not 2 <= folds <= len(groups):
        raise ValueError("actor advantage cross-fit needs complete-episode folds")
    partitions = _folds(groups, count=folds, seed=seed)
    by_option: dict[tuple[str, str], float] = {}
    reports = []
    for fold, heldout_groups in enumerate(partitions):
        heldout_set = set(heldout_groups)
        train = [
            sample for sample in samples
            if sample.episode_id not in heldout_set
        ]
        heldout = [
            sample for sample in samples
            if sample.episode_id in heldout_set
        ]
        train_groups = tuple(_episodes(train))
        critic = fit_implicit_q_member(
            train,
            iterations=critic_iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            seed=seed + fold * 10_000,
            threads=threads,
            bootstrap={episode: 1 for episode in train_groups},
        )
        values = factual_cost_advantage(heldout, critic)
        ordered = [
            sample for episode in _episodes(heldout).values()
            for sample in episode
        ]
        for sample, value in zip(ordered, values, strict=True):
            key = (sample.episode_id, sample.option_id)
            if key in by_option:
                raise RuntimeError("actor advantage was cross-fitted twice")
            by_option[key] = float(value)
        reports.append({
            "fold": fold,
            "fit_episodes": sorted(train_groups),
            "heldout_episodes": sorted(heldout_set),
            "options": len(heldout),
            "critic_iterations": critic.iterations,
        })
    ordered = [
        sample for episode in episodes.values() for sample in episode
    ]
    values = np.asarray([
        by_option[(sample.episode_id, sample.option_id)] for sample in ordered
    ], dtype=np.float32)
    if len(by_option) != len(ordered) or not np.all(np.isfinite(values)):
        raise RuntimeError("actor advantage cross-fit is incomplete")
    return values, {
        "folds": reports,
        "episode_groups": len(groups),
        "options": len(ordered),
        "all_labels_out_of_episode": all(
            not set(report["fit_episodes"]) & set(report["heldout_episodes"])
            for report in reports
        ),
    }


def fit_cross_fitted_iql_actor_population(
    samples: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    advantage_folds: int = 3,
    critic_iterations: int = 2,
    n_step_options: int = 8,
    q_trees: int = 8,
    value_trees: int = 8,
    seed: int = 760_813,
    threads: int = 1,
    hidden: int = 64,
    rank: int = 24,
    epochs: int = 8,
    batch_size: int = 1024,
    learning_rate: float = 1e-3,
    log_weight_clip: float = 4.0,
) -> tuple[list[IqlActorMember], dict[str, object]]:
    from .implicit_learning import _bootstrap_counts, _episodes

    prepared = actor_arrays(samples, layout)
    advantages, crossfit = cross_fitted_factual_advantages(
        samples,
        folds=advantage_folds,
        critic_iterations=critic_iterations,
        n_step_options=n_step_options,
        q_trees=q_trees,
        value_trees=value_trees,
        seed=seed + 100_000,
        threads=threads,
    )
    groups = tuple(_episodes(samples))
    bootstraps = [
        _bootstrap_counts(groups, seed=seed + member * 10_000)
        for member in range(7)
    ]
    actors = [
        fit_iql_actor_member(
            samples,
            layout=layout,
            prepared=prepared,
            seed=seed + member * 10_000 + 500_000,
            threads=threads,
            advantage=advantages,
            bootstrap=bootstraps[member],
            hidden=hidden,
            rank=rank,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            log_weight_clip=log_weight_clip,
        )
        for member in range(7)
    ]
    return actors, crossfit


def pessimistic_actor_action(
    population: list[IqlActorMember],
    sample: OptionStep,
    *,
    supported: list[bool] | None = None,
    member_indices: tuple[int, ...] | None = None,
) -> str:
    import numpy as np

    if not population:
        raise ValueError("IQL actor population is empty")
    selected = member_indices or tuple(range(len(population)))
    scores = np.asarray([
        population[member].model.predict(sample.candidate_vectors)
        for member in selected
    ], dtype=np.float64)
    return actor_population_choice(
        scores, sample, supported=supported
    )


def actor_population_choice(
    scores,
    sample: OptionStep,
    *,
    supported: list[bool] | None = None,
) -> str:
    """Select from precomputed member scores with population-range pessimism."""
    import numpy as np

    scores = np.asarray(scores, dtype=np.float64)
    if scores.ndim != 2 or scores.shape[1] != len(sample.legal_actions):
        raise ValueError("actor score matrix and safe set differ")
    baseline = sample.legal_actions.index(sample.baseline_action)
    differences = scores - scores[:, [baseline]]
    mask = supported or [True] * len(sample.legal_actions)
    candidates = []
    for index, action in enumerate(sample.legal_actions):
        if action == sample.baseline_action or not mask[index]:
            continue
        values = differences[:, index]
        lower_bound = float(2.0 * values.min() - values.max())
        if lower_bound > 0.0:
            candidates.append((lower_bound, action))
    return max(candidates, default=(0.0, sample.baseline_action))[1]


# Linux fork jobs are integer-indexed so large immutable augmented samples stay
# copy-on-write instead of being serialized once per complete-episode fold.
_FORKED_ACTOR_JOBS = None


def evaluate_iql_actor_fold(
    train: list[OptionStep],
    heldout: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    episode_cohorts: dict[str, str],
    fold: int,
    critic_iterations: int,
    n_step_options: int,
    q_trees: int,
    value_trees: int,
    seed: int,
    threads: int,
    actor_hidden: int,
    actor_rank: int,
    actor_epochs: int,
    actor_batch_size: int,
    actor_learning_rate: float,
    actor_log_weight_clip: float,
    actor_advantage_folds: int = 0,
    intervention_probability_cap: float = 0.10,
    intervention_density_ratio_cap: float = 2.0,
    fit_representation_on_train: bool = False,
    frozen_actors: list[IqlActorMember] | None = None,
    frozen_support: dict[str, object] | None = None,
    intervention_minimum_uniform_mass: float | None = None,
) -> dict[str, object]:
    from collections import Counter
    import numpy as np
    from .implicit_learning import (
        _support_artifacts,
        _supported,
        _arrays,
        _episodes,
        _n_step_targets,
        fit_implicit_q_member,
        fit_implicit_q_population,
    )

    if fit_representation_on_train:
        from .advantage_learning import (
            _augment_steps,
            fit_hazard_codebook,
            rich_feature_names,
        )
        from .low_rank_learning import named_feature_roles

        representation = fit_hazard_codebook(
            train, seed=seed + fold * 100_000 + 40_000
        )
        train = _augment_steps(train, representation)
        heldout = _augment_steps(heldout, representation)
        layout = named_feature_roles(rich_feature_names())
        representation_report = {
            "fit_episodes": sorted({sample.episode_id for sample in train}),
            "heldout_episodes": sorted({
                sample.episode_id for sample in heldout
            }),
            "heldout_excluded": not (
                {sample.episode_id for sample in train}
                & {sample.episode_id for sample in heldout}
            ),
        }
    else:
        representation_report = {
            "scope": "caller-prepared",
            "heldout_excluded": False,
        }

    if frozen_actors is not None:
        if len(frozen_actors) != 7:
            raise ValueError("frozen actor candidate must contain seven members")
        actors = list(frozen_actors)
        critics = []
        advantage_crossfit = {
            "scope": "frozen-development-candidate",
            "qualification_labels_used": False,
        }
    elif actor_advantage_folds:
        actors, advantage_crossfit = fit_cross_fitted_iql_actor_population(
            train,
            layout=layout,
            advantage_folds=actor_advantage_folds,
            critic_iterations=critic_iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            seed=seed + fold * 100_000 + 50_000,
            threads=threads,
            hidden=actor_hidden,
            rank=actor_rank,
            epochs=actor_epochs,
            batch_size=actor_batch_size,
            learning_rate=actor_learning_rate,
            log_weight_clip=actor_log_weight_clip,
        )
        critics = []
    else:
        critics = fit_implicit_q_population(
            train,
            members=7,
            iterations=critic_iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            seed=seed + fold * 100_000,
            total_threads=threads,
            parallel_members=False,
        )
        actors = fit_iql_actor_population(
            train,
            critics,
            layout=layout,
            seed=seed + fold * 100_000 + 50_000,
            threads=threads,
            hidden=actor_hidden,
            rank=actor_rank,
            epochs=actor_epochs,
            batch_size=actor_batch_size,
            learning_rate=actor_learning_rate,
            log_weight_clip=actor_log_weight_clip,
        )
        advantage_crossfit = None

    train_groups = tuple(_episodes(train))
    evaluation_critic = fit_implicit_q_member(
        train,
        iterations=critic_iterations,
        n_step_options=n_step_options,
        q_trees=q_trees,
        value_trees=value_trees,
        seed=seed + fold * 100_000 + 90_000,
        threads=threads,
        bootstrap={episode: 1 for episode in train_groups},
    )
    heldout_episodes = _episodes(heldout)
    heldout_targets = _n_step_targets(
        heldout_episodes,
        evaluation_critic.value_model,
        n_step_options=n_step_options,
    )
    _q_rows, heldout_states, row_episodes, _base_weights = _arrays(
        heldout_episodes
    )
    common_predictions = evaluation_critic.outcome_model.predict(
        heldout_states
    )
    ordered_heldout = [
        sample for rows in heldout_episodes.values() for sample in rows
    ]
    heldout_common = {
        (episode, sample.option_id): float(common)
        for episode, sample, common in zip(
            row_episodes, ordered_heldout, common_predictions, strict=True
        )
    }
    if frozen_support is None:
        support, support_report = _support_artifacts(
            train, seed=seed + fold * 100_000 + 80_000
        )
    else:
        support = frozen_support
        factual = []
        for sample in train:
            mask = _supported(sample, support)
            factual.append(mask[sample.legal_actions.index(sample.action)])
        support_report = {
            "scope": "frozen-development-candidate",
            "rows": len(factual),
            "factual_coverage": sum(factual) / len(factual),
            "threshold": float(support["threshold"]),
        }
    all_members = tuple(range(len(actors)))
    left_members = (0, 1, 2)
    right_members = (3, 4, 5, 6)
    episode_reports: dict[str, dict[str, object]] = {}
    action_counts: Counter[str] = Counter()
    mean_action_counts: Counter[str] = Counter()
    unsupported_candidates = 0
    for sample in heldout:
        report = episode_reports.setdefault(sample.episode_id, {
            "cohort": episode_cohorts[sample.episode_id],
            "options": 0,
            "full_proposals": 0,
            "full_proposal_loo_exact": 0,
            "loo_union": 0,
            "loo_exact": 0,
            "split_union": 0,
            "split_exact": 0,
            "individual_proposals": 0,
            "individual_union": 0,
            "individual_exact": 0,
            "mean_proposals": 0,
            "mean_loo_union": 0,
            "mean_loo_exact": 0,
            "policy_intervention_exposure": 0.0,
            "policy_model_effect": 0.0,
            "policy_dr_effect": 0.0,
            "policy_max_abs_correction": 0.0,
            "policy_loo_intervention_exposure": [0.0] * 7,
            "policy_loo_model_effect": [0.0] * 7,
            "policy_loo_dr_effect": [0.0] * 7,
            "policy_loo_max_abs_correction": [0.0] * 7,
            "behavior_kl_sum": 0.0,
        })
        report["options"] += 1
        mask = _supported(sample, support)
        unsupported_candidates += sum(
            action != sample.baseline_action and not mask[index]
            for index, action in enumerate(sample.legal_actions)
        )
        scores = np.asarray([
            actor.model.predict(sample.candidate_vectors) for actor in actors
        ], dtype=np.float64)
        individual = tuple(
            actor_population_choice(
                scores[[member]], sample, supported=mask
            )
            for member in all_members
        )
        individual_union = any(
            choice != sample.baseline_action for choice in individual
        )
        report["individual_proposals"] += sum(
            choice != sample.baseline_action for choice in individual
        )
        report["individual_union"] += int(individual_union)
        report["individual_exact"] += int(
            individual_union and len(set(individual)) == 1
        )
        mean_choice = actor_population_choice(
            scores.mean(axis=0, keepdims=True), sample, supported=mask
        )
        report["mean_proposals"] += int(
            mean_choice != sample.baseline_action
        )
        if mean_choice != sample.baseline_action:
            mean_action_counts[mean_choice] += 1
        mean_leave_one_out = tuple(
            actor_population_choice(
                scores[[
                    member for member in all_members if member != omitted
                ]].mean(axis=0, keepdims=True),
                sample,
                supported=mask,
            )
            for omitted in all_members
        )
        mean_loo_union = (
            mean_choice != sample.baseline_action
            or any(
                choice != sample.baseline_action
                for choice in mean_leave_one_out
            )
        )
        report["mean_loo_union"] += int(mean_loo_union)
        report["mean_loo_exact"] += int(
            mean_loo_union
            and all(choice == mean_choice for choice in mean_leave_one_out)
        )

        policy_choices = (mean_choice, *mean_leave_one_out)
        if any(choice != sample.baseline_action for choice in policy_choices):
            baseline = sample.legal_actions.index(sample.baseline_action)
            factual = sample.legal_actions.index(sample.action)
            propensity = np.asarray(
                sample.behavior_probabilities, dtype=np.float64
            )
            q_effect = np.asarray(
                evaluation_critic.q_model.predict(sample.candidate_vectors),
                dtype=np.float64,
            )
            centered_factual = (
                q_effect[factual] - float(propensity @ q_effect)
            )
            key = (sample.episode_id, sample.option_id)
            factual_q = heldout_common[key] + centered_factual
            residual = heldout_targets[key] - factual_q
            effects = []
            for choice in policy_choices:
                if choice == sample.baseline_action:
                    effects.append((0.0, 0.0, 0.0, 0.0))
                    continue
                candidate = sample.legal_actions.index(choice)
                if intervention_minimum_uniform_mass is None:
                    exposure = min(
                        intervention_probability_cap,
                        intervention_density_ratio_cap * propensity[candidate],
                        intervention_density_ratio_cap * propensity[baseline],
                    )
                else:
                    minimum = intervention_minimum_uniform_mass / len(
                        sample.legal_actions
                    )
                    if propensity.min() + 1e-12 < minimum:
                        raise ValueError(
                            "recorded behavior violates deployable propensity floor"
                        )
                    exposure = min(
                        intervention_probability_cap,
                        intervention_density_ratio_cap * minimum,
                    )
                model_effect = exposure * (
                    q_effect[candidate] - q_effect[baseline]
                )
                correction = exposure * (
                    int(factual == candidate) - int(factual == baseline)
                ) / propensity[factual]
                effects.append((
                    float(exposure),
                    float(model_effect),
                    float(model_effect + correction * residual),
                    abs(float(correction)),
                ))
            exposure, model_effect, dr_effect, abs_correction = effects[0]
            report["policy_intervention_exposure"] += exposure
            report["policy_model_effect"] += model_effect
            report["policy_dr_effect"] += dr_effect
            report["policy_max_abs_correction"] = max(
                report["policy_max_abs_correction"], abs_correction
            )
            for omitted, values in enumerate(effects[1:]):
                exposure, model_effect, dr_effect, abs_correction = values
                report["policy_loo_intervention_exposure"][omitted] += exposure
                report["policy_loo_model_effect"][omitted] += model_effect
                report["policy_loo_dr_effect"][omitted] += dr_effect
                report["policy_loo_max_abs_correction"][omitted] = max(
                    report["policy_loo_max_abs_correction"][omitted],
                    abs_correction,
                )
        full = actor_population_choice(scores, sample, supported=mask)
        leave_one_out = tuple(
            actor_population_choice(
                scores[[
                    member for member in all_members if member != omitted
                ]],
                sample,
                supported=mask,
            )
            for omitted in all_members
        )
        stable = all(choice == full for choice in leave_one_out)
        loo_either = (
            full != sample.baseline_action
            or any(
                choice != sample.baseline_action for choice in leave_one_out
            )
        )
        report["loo_union"] += int(loo_either)
        report["loo_exact"] += int(loo_either and stable)
        if full != sample.baseline_action:
            report["full_proposals"] += 1
            report["full_proposal_loo_exact"] += int(stable)
            action_counts[full] += 1
        left = actor_population_choice(
            scores[list(left_members)], sample, supported=mask
        )
        right = actor_population_choice(
            scores[list(right_members)], sample, supported=mask
        )
        split_either = (
            left != sample.baseline_action or right != sample.baseline_action
        )
        report["split_union"] += int(split_either)
        report["split_exact"] += int(split_either and left == right)

        probabilities = np.asarray(
            sample.behavior_probabilities, dtype=np.float64
        )
        member_kl = []
        for member_scores in scores:
            member_kl.append(categorical_kl_from_logits(
                probabilities, member_scores
            ))
        report["behavior_kl_sum"] += sum(member_kl) / len(member_kl)
    return {
        "fold": fold,
        "fit_episodes": sorted({sample.episode_id for sample in train}),
        "heldout_episodes": sorted(episode_reports),
        "support": support_report,
        "critic_bootstrap": [member.bootstrap for member in critics],
        "critic_iterations": [member.iterations for member in critics],
        "evaluation_critic_iterations": evaluation_critic.iterations,
        "actor_advantage_crossfit": advantage_crossfit,
        "representation": representation_report,
        "actor_diagnostics": [actor.diagnostics for actor in actors],
        "actor_advantage_scales": [actor.advantage_scale for actor in actors],
        "episodes": episode_reports,
        "unsupported_candidates": unsupported_candidates,
        "proposal_actions": dict(sorted(action_counts.items())),
        "mean_policy_actions": dict(sorted(mean_action_counts.items())),
    }


def _run_forked_actor_job(index: int):
    if _FORKED_ACTOR_JOBS is None:
        raise RuntimeError("forked IQL actor job context is absent")
    return evaluate_iql_actor_fold(**_FORKED_ACTOR_JOBS[index])


def summarize_iql_actor_episodes(
    episode_reports: dict[str, dict[str, object]],
    *,
    cohort_names: tuple[str, ...],
) -> dict[str, dict[str, object]]:
    """Aggregate policy effects only across complete factual episodes."""
    import numpy as np

    def cohort(name: str | None) -> dict[str, object]:
        rows = [
            episode_reports[episode]
            for episode in sorted(episode_reports)
            if name is None or episode_reports[episode]["cohort"] == name
        ]
        if not rows:
            raise ValueError("policy-effect cohort is empty")
        options = sum(row["options"] for row in rows)
        proposals = sum(row["full_proposals"] for row in rows)
        proposal_exact = sum(row["full_proposal_loo_exact"] for row in rows)
        loo_union = sum(row["loo_union"] for row in rows)
        loo_exact = sum(row["loo_exact"] for row in rows)
        split_union = sum(row["split_union"] for row in rows)
        split_exact = sum(row["split_exact"] for row in rows)
        individual_proposals = sum(
            row["individual_proposals"] for row in rows
        )
        individual_union = sum(row["individual_union"] for row in rows)
        individual_exact = sum(row["individual_exact"] for row in rows)
        mean_proposals = sum(row["mean_proposals"] for row in rows)
        mean_loo_union = sum(row["mean_loo_union"] for row in rows)
        mean_loo_exact = sum(row["mean_loo_exact"] for row in rows)
        policy_dr_effects = np.asarray([
            row["policy_dr_effect"] for row in rows
        ], dtype=np.float64)
        policy_model_effects = np.asarray([
            row["policy_model_effect"] for row in rows
        ], dtype=np.float64)
        policy_mean = float(policy_dr_effects.mean())
        policy_se = float(
            policy_dr_effects.std(ddof=1) / math.sqrt(len(rows))
        ) if len(rows) > 1 else math.inf
        cohort_seed = 960_813 + sum(map(ord, name or "overall"))
        generator = np.random.default_rng(cohort_seed)
        bootstrap_means = policy_dr_effects[generator.integers(
            0, len(policy_dr_effects), size=(4096, len(policy_dr_effects))
        )].mean(axis=1)
        loo_effects = np.asarray([
            [row["policy_loo_dr_effect"][member] for row in rows]
            for member in range(7)
        ], dtype=np.float64)
        loo_bootstrap_upper = []
        for member in range(7):
            member_generator = np.random.default_rng(
                cohort_seed + (member + 1) * 10_000
            )
            member_means = loo_effects[member][member_generator.integers(
                0, len(rows), size=(4096, len(rows))
            )].mean(axis=1)
            loo_bootstrap_upper.append(float(np.quantile(member_means, 0.95)))
        return {
            "episode_groups": len(rows),
            "options": options,
            "full_proposals": proposals,
            "full_proposal_rate": proposals / options,
            "full_proposal_loo_exact_rate": (
                proposal_exact / proposals if proposals else 0.0
            ),
            "loo_union_stability": (
                loo_exact / loo_union if loo_union else 0.0
            ),
            "split_conditional_agreement": (
                split_exact / split_union if split_union else 0.0
            ),
            "individual_member_proposal_rate": (
                individual_proposals / (7 * options)
            ),
            "individual_union_rate": individual_union / options,
            "individual_unanimous_conditional_rate": (
                individual_exact / individual_union
                if individual_union else 0.0
            ),
            "mean_population_proposal_rate": mean_proposals / options,
            "mean_population_loo_stability": (
                mean_loo_exact / mean_loo_union if mean_loo_union else 0.0
            ),
            "policy_intervention_exposure_rate": (
                sum(row["policy_intervention_exposure"] for row in rows)
                / options
            ),
            "policy_dr_hit_effect_mean": policy_mean,
            "policy_dr_hit_effect_standard_error": policy_se,
            "policy_dr_hit_effect_normal_upper_95": (
                policy_mean + 1.6448536269514722 * policy_se
            ),
            "policy_dr_hit_effect_normal_lower_95": (
                policy_mean - 1.6448536269514722 * policy_se
            ),
            "policy_dr_hit_effect_bootstrap_upper_95": float(
                np.quantile(bootstrap_means, 0.95)
            ),
            "policy_dr_hit_effect_bootstrap_lower_95": float(
                np.quantile(bootstrap_means, 0.05)
            ),
            "policy_dr_bootstrap_episode_groups": len(policy_dr_effects),
            "policy_dr_bootstrap_resamples": 4096,
            "policy_loo_dr_hit_effect_means": [
                float(values.mean()) for values in loo_effects
            ],
            "policy_loo_dr_hit_effect_bootstrap_upper_95": (
                loo_bootstrap_upper
            ),
            "policy_loo_worst_bootstrap_upper_95": max(
                loo_bootstrap_upper
            ),
            "policy_dr_beneficial_episode_rate": float(
                np.mean(policy_dr_effects < 0.0)
            ),
            "policy_model_hit_effect_mean": float(
                policy_model_effects.mean()
            ),
            "policy_max_abs_correction": max(
                row["policy_max_abs_correction"] for row in rows
            ),
            "mean_behavior_kl": (
                sum(row["behavior_kl_sum"] for row in rows) / options
            ),
        }

    return {
        "overall": cohort(None),
        **{name: cohort(name) for name in sorted(cohort_names)},
    }


def crossfit_iql_actor_report(
    samples: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    episode_cohorts: dict[str, str],
    folds: int = 5,
    critic_iterations: int = 2,
    n_step_options: int = 8,
    q_trees: int = 8,
    value_trees: int = 8,
    seed: int = 460_813,
    total_threads: int = 32,
    actor_hidden: int = 64,
    actor_rank: int = 24,
    actor_epochs: int = 8,
    actor_batch_size: int = 1024,
    actor_learning_rate: float = 1e-3,
    actor_log_weight_clip: float = 4.0,
    actor_advantage_folds: int = 0,
    intervention_probability_cap: float = 0.10,
    intervention_density_ratio_cap: float = 2.0,
    fit_representation_on_train: bool = False,
) -> dict[str, object]:
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing
    import numpy as np
    import sys
    from .advantage_learning import _folds
    from .implicit_learning import _episodes

    episodes = _episodes(samples)
    groups = list(episodes)
    if set(episode_cohorts) != set(groups):
        raise ValueError("every actor development episode needs one cohort")
    if not 1 <= folds <= min(len(groups) // 2, total_threads):
        raise ValueError("invalid actor complete-episode fold count")
    fold_threads = max(1, total_threads // folds)
    partitions = _folds(groups, count=folds, seed=seed)
    jobs = []
    for fold, heldout_groups in enumerate(partitions):
        heldout_set = set(heldout_groups)
        jobs.append({
            "train": [
                sample for sample in samples
                if sample.episode_id not in heldout_set
            ],
            "heldout": [
                sample for sample in samples
                if sample.episode_id in heldout_set
            ],
            "layout": layout,
            "episode_cohorts": episode_cohorts,
            "fold": fold,
            "critic_iterations": critic_iterations,
            "n_step_options": n_step_options,
            "q_trees": q_trees,
            "value_trees": value_trees,
            "seed": seed,
            "threads": fold_threads,
            "actor_hidden": actor_hidden,
            "actor_rank": actor_rank,
            "actor_epochs": actor_epochs,
            "actor_batch_size": actor_batch_size,
            "actor_learning_rate": actor_learning_rate,
            "actor_log_weight_clip": actor_log_weight_clip,
            "actor_advantage_folds": actor_advantage_folds,
            "intervention_probability_cap": intervention_probability_cap,
            "intervention_density_ratio_cap": intervention_density_ratio_cap,
            "fit_representation_on_train": fit_representation_on_train,
        })
    if folds == 1:
        reports = [evaluate_iql_actor_fold(**jobs[0])]
    else:
        if sys.platform != "linux" or "fork" not in multiprocessing.get_all_start_methods():
            raise RuntimeError("IQL actor cross-fit requires Linux fork")
        global _FORKED_ACTOR_JOBS
        _FORKED_ACTOR_JOBS = tuple(jobs)
        try:
            with ProcessPoolExecutor(
                max_workers=folds,
                mp_context=multiprocessing.get_context("fork"),
            ) as executor:
                reports = list(executor.map(
                    _run_forked_actor_job, range(folds)
                ))
        finally:
            _FORKED_ACTOR_JOBS = None
    episode_reports = {
        episode: report
        for fold_report in reports
        for episode, report in fold_report["episodes"].items()
    }

    return {
        "execution": {
            "total_thread_budget": total_threads,
            "fold_workers": folds,
            "threads_per_fold": fold_threads,
        },
        "folds": reports,
        "episodes": episode_reports,
        "cohorts": summarize_iql_actor_episodes(
            episode_reports,
            cohort_names=tuple(sorted(set(episode_cohorts.values()))),
        ),
        "unsupported_candidates": sum(
            report["unsupported_candidates"] for report in reports
        ),
        "proposal_actions": dict(sorted(sum(
            (Counter(report["proposal_actions"]) for report in reports),
            Counter(),
        ).items())),
        "mean_policy_actions": dict(sorted(sum(
            (Counter(report["mean_policy_actions"]) for report in reports),
            Counter(),
        ).items())),
    }


def run_iql_actor_causal_smoke(*, threads: int = 8) -> dict[str, object]:
    """Recover delayed action value and abstain in a propensity-known null."""
    from .implicit_learning import (
        delayed_effect_episodes,
        fit_implicit_q_population,
    )

    count, options, delay, horizon = 64, 72, 12, 4
    layout = FeatureRoleLayout(
        names=tuple([
            *(f"observation:pending-{index}" for index in range(delay)),
            "action:treatment",
        ]),
        state_indices=tuple(range(delay)),
        action_indices=(delay,),
    )
    critic_parameters = {
        "members": 7,
        "iterations": 4,
        "n_step_options": horizon,
        "q_trees": 40,
        "value_trees": 32,
        "total_threads": threads,
        "parallel_members": False,
    }
    actor_parameters = {
        "threads": threads,
        "hidden": 32,
        "rank": 12,
        "epochs": 8,
        "batch_size": 512,
        "learning_rate": 1e-3,
        "log_weight_clip": 4.0,
    }

    def fit(null_effect: bool):
        samples = delayed_effect_episodes(
            count=count,
            options=options,
            delay=delay,
            null_effect=null_effect,
        )
        critics = fit_implicit_q_population(samples, **critic_parameters)
        actors = fit_iql_actor_population(
            samples, critics, layout=layout, **actor_parameters
        )
        decisions = [
            pessimistic_actor_action(actors, sample) for sample in samples
        ]
        interior = [
            sample for sample in samples
            if 8 <= sample.sequence < options - delay
        ]
        effects = [
            sum(
                float(
                    actor.model.predict(sample.candidate_vectors)[1]
                    - actor.model.predict(sample.candidate_vectors)[0]
                )
                for sample in interior
            ) / len(interior)
            for actor in actors
        ]
        return samples, critics, actors, decisions, effects

    delayed, critics, actors, decisions, effects = fit(False)
    null, _null_critics, null_actors, null_decisions, null_effects = fit(True)
    gates = {
        "effect_delayed_beyond_backup": delay > horizon,
        "all_critic_members_beat_zero_effect": all(
            member.iterations[-1]["q_weighted_mse"]
            < member.iterations[-1]["zero_effect_weighted_mse"]
            for member in critics
        ),
        "all_actor_members_prefer_beneficial_action": all(
            effect > 0.0 for effect in effects
        ),
        "pessimistic_actor_exercises_beneficial_action": (
            decisions.count("left") > 0
        ),
        "null_pessimistic_actor_abstains": null_decisions.count("left") == 0,
        "complete_behavior_probability_used": all(
            sample.behavior_probabilities == (0.5, 0.5)
            for sample in (*delayed, *null)
        ),
    }
    return {
        "schema": "autonomous-generation-6-iql-actor-causal-smoke-v1",
        "evidence_eligible": False,
        "parameters": {
            "episode_groups": count,
            "options_per_episode": options,
            "effect_delay": delay,
            "critic": critic_parameters,
            "actor": actor_parameters,
            "actor_objective": (
                "known-behavior-expectation-plus-centered-factual-advantage-residual"
            ),
        },
        "actor_member_mean_logit_effects": effects,
        "beneficial_overrides": decisions.count("left"),
        "null_actor_member_mean_logit_effects": null_effects,
        "null_overrides": null_decisions.count("left"),
        "options": len(delayed),
        "actor_diagnostics": [actor.diagnostics for actor in actors],
        "null_actor_diagnostics": [
            actor.diagnostics for actor in null_actors
        ],
        "gates": gates,
        "passed": all(gates.values()),
    }


def run_cross_fitted_iql_actor_policy_smoke(
    *, threads: int = 8
) -> dict[str, object]:
    """Test nested episode cross-fitting and policy-level HIT uncertainty."""
    from .implicit_learning import delayed_effect_episodes

    count, options, delay, horizon = 64, 72, 12, 4
    layout = FeatureRoleLayout(
        names=tuple([
            *(f"observation:pending-{index}" for index in range(delay)),
            "action:treatment",
        ]),
        state_indices=tuple(range(delay)),
        action_indices=(delay,),
    )
    parameters = {
        "folds": 4,
        "critic_iterations": 4,
        "n_step_options": horizon,
        "q_trees": 40,
        "value_trees": 32,
        "total_threads": threads,
        "actor_hidden": 32,
        "actor_rank": 12,
        "actor_epochs": 8,
        "actor_batch_size": 512,
        "actor_advantage_folds": 3,
        "intervention_probability_cap": 0.10,
        "intervention_density_ratio_cap": 2.0,
    }

    def evaluate(null_effect: bool):
        samples = delayed_effect_episodes(
            count=count,
            options=options,
            delay=delay,
            null_effect=null_effect,
        )
        report = crossfit_iql_actor_report(
            samples,
            layout=layout,
            episode_cohorts={
                sample.episode_id: "synthetic" for sample in samples
            },
            **parameters,
        )
        return report

    delayed = evaluate(False)
    null = evaluate(True)
    effect = delayed["cohorts"]["overall"]
    null_effect = null["cohorts"]["overall"]
    gates = {
        "effect_delayed_beyond_backup": delay > horizon,
        "all_actor_labels_out_of_episode": all(
            fold["actor_advantage_crossfit"]["all_labels_out_of_episode"]
            for report in (delayed, null) for fold in report["folds"]
        ),
        "beneficial_policy_exercised": (
            effect["mean_population_proposal_rate"] > 0.0
            and effect["policy_intervention_exposure_rate"] > 0.0
        ),
        "beneficial_policy_upper_bound_below_zero": (
            effect["policy_dr_hit_effect_bootstrap_upper_95"] < 0.0
        ),
        "all_leave_one_out_policies_upper_bound_below_zero": (
            effect["policy_loo_worst_bootstrap_upper_95"] < 0.0
        ),
        "null_policy_interval_contains_zero": (
            null_effect["policy_dr_hit_effect_bootstrap_lower_95"] <= 0.0
            <= null_effect["policy_dr_hit_effect_bootstrap_upper_95"]
        ),
        "bounded_policy_correction": max(
            effect["policy_max_abs_correction"],
            null_effect["policy_max_abs_correction"],
        ) <= parameters["intervention_density_ratio_cap"] + 1e-9,
    }
    return {
        "schema": "autonomous-generation-6-cross-fitted-policy-smoke-v1",
        "evidence_eligible": False,
        "parameters": {
            "episode_groups": count,
            "options_per_episode": options,
            "effect_delay": delay,
            **parameters,
        },
        "delayed_effect_policy": effect,
        "null_effect_policy": null_effect,
        "gates": gates,
        "passed": all(gates.values()),
    }
