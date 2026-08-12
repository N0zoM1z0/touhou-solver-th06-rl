"""Behavior-supported IQL actor extraction from factual implicit-Q members."""

from __future__ import annotations

from dataclasses import dataclass
import math

from .advantage_learning import OptionStep
from .low_rank_learning import FeatureRoleLayout


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


def fit_iql_actor_member(
    samples: list[OptionStep],
    critic_member,
    *,
    layout: FeatureRoleLayout,
    prepared: ActorArrays,
    seed: int,
    threads: int,
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

    bootstrap_weights = np.asarray([
        critic_member.bootstrap.get(episode, 0)
        for episode in prepared.episode_ids
    ], dtype=np.float32)
    group_weights = prepared.base_weights * bootstrap_weights
    advantage = factual_cost_advantage(samples, critic_member)
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
            # Exact known-behavior expectation plus an action-centered factual
            # residual is an unbiased, no-inverse-propensity estimate of the
            # normalized advantage-weighted behavior-cloning objective.
            centered_losses = behavior_losses + (
                advantage_weight_tensor[batch] - 1.0
            ) * (factual_losses - behavior_losses)
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
        bootstrap=critic_member.bootstrap,
        advantage_scale=temperature,
        diagnostics={
            "weighted_cross_entropy": float(final_loss),
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
