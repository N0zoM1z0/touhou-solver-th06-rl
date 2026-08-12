"""Low-rank action-centered sequential offline-RL development primitives."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import math
import random
from typing import Iterable

from .advantage_learning import OptionStep


@dataclass(frozen=True)
class FeatureRoleLayout:
    names: tuple[str, ...]
    state_indices: tuple[int, ...]
    action_indices: tuple[int, ...]

    def __post_init__(self) -> None:
        indices = (*self.state_indices, *self.action_indices)
        if (
            not self.names
            or not self.state_indices
            or not self.action_indices
            or len(set(indices)) != len(indices)
            or set(indices) != set(range(len(self.names)))
        ):
            raise ValueError("feature roles must partition the complete vector")


def named_feature_roles(names: Iterable[str]) -> FeatureRoleLayout:
    """Derive game-neutral state/action roles from the learner schema."""
    values = tuple(names)
    action = tuple(
        index for index, name in enumerate(values)
        if name.startswith("action:")
        or name.startswith("delta_from_baseline:")
        or name in {"matches_baseline", "matches_current"}
    )
    action_set = set(action)
    state = tuple(index for index in range(len(values)) if index not in action_set)
    return FeatureRoleLayout(values, state, action)


@dataclass(frozen=True)
class LowRankEffectModel:
    layout: FeatureRoleLayout
    state_mean: object
    state_scale: object
    action_mean: object
    action_scale: object
    state_hidden_weight: object
    state_hidden_bias: object
    state_latent_weight: object
    state_latent_bias: object
    action_latent_weight: object
    action_bias_weight: object

    def _state_latent(self, states):
        import numpy as np

        normalized = (states - self.state_mean) / self.state_scale
        hidden = np.tanh(
            normalized @ self.state_hidden_weight + self.state_hidden_bias
        )
        return hidden @ self.state_latent_weight + self.state_latent_bias

    def _action_latent(self, actions):
        normalized = (actions - self.action_mean) / self.action_scale
        return normalized @ self.action_latent_weight, normalized

    def predict_grouped(self, states, actions, starts):
        import numpy as np

        state_latent = self._state_latent(states)
        action_latent, normalized = self._action_latent(actions)
        counts = np.diff(np.append(starts, len(actions)))
        repeated = np.repeat(state_latent, counts, axis=0)
        return (
            np.sum(repeated * action_latent, axis=1)
            / math.sqrt(state_latent.shape[1])
            + normalized @ self.action_bias_weight
        )

    def predict_centered(self, states, centered_actions):
        """Predict E-centered action effects; feature means cancel exactly."""
        import numpy as np

        state_latent = self._state_latent(states)
        normalized = centered_actions / self.action_scale
        action_latent = normalized @ self.action_latent_weight
        return (
            np.sum(state_latent * action_latent, axis=1)
            / math.sqrt(state_latent.shape[1])
            + normalized @ self.action_bias_weight
        )

    def predict(self, rows):
        import numpy as np

        matrix = np.asarray(rows, dtype=np.float32)
        states = matrix[:, self.layout.state_indices]
        actions = matrix[:, self.layout.action_indices]
        starts = np.arange(len(matrix), dtype=np.int64)
        return self.predict_grouped(states, actions, starts)


@dataclass(frozen=True)
class LowRankImplicitMember:
    outcome_model: object
    q_model: LowRankEffectModel
    value_model: object
    bootstrap: dict[str, int]
    iterations: tuple[dict[str, float], ...]


@dataclass(frozen=True)
class StructuredArrays:
    states: object
    centered_actions: object
    action_mean: object
    action_scale: object
    row_episodes: tuple[str, ...]
    base_weights: object


def _episodes(samples: Iterable[OptionStep]) -> dict[str, list[OptionStep]]:
    from .implicit_learning import _episodes as implicit_episodes

    return implicit_episodes(samples)


def _structured_arrays(
    episodes: dict[str, list[OptionStep]], layout: FeatureRoleLayout
) -> StructuredArrays:
    import numpy as np

    states = []
    centered_actions = []
    row_episodes = []
    base_weights = []
    action_sum = np.zeros(len(layout.action_indices), dtype=np.float64)
    action_square_sum = np.zeros(len(layout.action_indices), dtype=np.float64)
    action_count = 0
    mean_episode_options = sum(map(len, episodes.values())) / len(episodes)
    for episode, rows in episodes.items():
        episode_weight = mean_episode_options / len(rows)
        for sample in rows:
            candidates = np.asarray(sample.candidate_vectors, dtype=np.float32)
            if candidates.shape[1] != len(layout.names):
                raise ValueError("low-rank feature schema and candidate width differ")
            state_candidates = candidates[:, layout.state_indices]
            if not np.allclose(state_candidates, state_candidates[0], atol=1e-7):
                raise ValueError("candidate-invariant state features changed by action")
            actions = candidates[:, layout.action_indices]
            probabilities = np.asarray(
                sample.behavior_probabilities, dtype=np.float64
            )
            if (
                len(probabilities) != len(actions)
                or np.any(~np.isfinite(probabilities))
                or np.any(probabilities <= 0.0)
                or not math.isclose(float(probabilities.sum()), 1.0, abs_tol=1e-9)
            ):
                raise ValueError("low-rank critic needs complete propensities")
            factual = sample.legal_actions.index(sample.action)
            states.append(state_candidates[0])
            centered_actions.append(
                actions[factual] - probabilities @ actions
            )
            row_episodes.append(episode)
            base_weights.append(episode_weight)
            action_sum += actions.sum(axis=0, dtype=np.float64)
            action_square_sum += np.square(actions, dtype=np.float64).sum(axis=0)
            action_count += len(actions)
    action_mean = action_sum / action_count
    variance = np.maximum(action_square_sum / action_count - action_mean ** 2, 0.0)
    action_scale = np.sqrt(variance)
    action_scale[action_scale < 1e-6] = 1.0
    return StructuredArrays(
        states=np.asarray(states, dtype=np.float32),
        centered_actions=np.asarray(centered_actions, dtype=np.float32),
        action_mean=action_mean.astype(np.float32),
        action_scale=action_scale.astype(np.float32),
        row_episodes=tuple(row_episodes),
        base_weights=np.asarray(base_weights, dtype=np.float64),
    )


def _fit_effect_model(
    prepared: StructuredArrays,
    *,
    layout: FeatureRoleLayout,
    targets,
    row_weights,
    seed: int,
    threads: int,
    hidden: int,
    rank: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> tuple[LowRankEffectModel, dict[str, float]]:
    import numpy as np
    import torch

    if min(hidden, rank, epochs, batch_size, threads) < 1:
        raise ValueError("low-rank dimensions, epochs, batch, and threads must be positive")
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)

    states = np.asarray(prepared.states, dtype=np.float32)
    actions = np.asarray(prepared.centered_actions, dtype=np.float32)
    weights = np.asarray(row_weights, dtype=np.float32)
    labels = np.asarray(targets, dtype=np.float32)
    positive = weights > 0.0
    if not np.any(positive):
        raise ValueError("low-rank bootstrap removed every episode")
    state_mean = np.average(states, axis=0, weights=weights).astype(np.float32)
    state_variance = np.average(
        (states - state_mean) ** 2, axis=0, weights=weights
    )
    state_scale = np.sqrt(np.maximum(state_variance, 0.0)).astype(np.float32)
    state_scale[state_scale < 1e-6] = 1.0
    normalized_states = (states - state_mean) / state_scale
    normalized_actions = actions / prepared.action_scale

    state_tensor = torch.from_numpy(normalized_states)
    action_tensor = torch.from_numpy(normalized_actions)
    label_tensor = torch.from_numpy(labels)
    weight_tensor = torch.from_numpy(weights)

    class Critic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.state_hidden = torch.nn.Linear(states.shape[1], hidden)
            self.state_latent = torch.nn.Linear(hidden, rank)
            self.action_latent = torch.nn.Linear(actions.shape[1], rank, bias=False)
            self.action_bias = torch.nn.Linear(actions.shape[1], 1, bias=False)

        def forward(self, state, action):
            latent_state = self.state_latent(torch.tanh(self.state_hidden(state)))
            latent_action = self.action_latent(action)
            return (
                (latent_state * latent_action).sum(dim=1) / math.sqrt(rank)
                + self.action_bias(action).squeeze(1)
            )

    model = Critic()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4
    )
    generator = torch.Generator().manual_seed(seed + 1)
    indices = torch.nonzero(weight_tensor > 0.0, as_tuple=False).squeeze(1)
    final_loss = math.inf
    for _epoch in range(epochs):
        permutation = indices[torch.randperm(len(indices), generator=generator)]
        loss_sum = 0.0
        weight_sum = 0.0
        for start in range(0, len(permutation), batch_size):
            batch = permutation[start:start + batch_size]
            batch_weights = weight_tensor[batch]
            prediction = model(state_tensor[batch], action_tensor[batch])
            loss = (
                batch_weights * (prediction - label_tensor[batch]) ** 2
            ).sum() / batch_weights.sum()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            loss_sum += float(loss.detach()) * float(batch_weights.sum())
            weight_sum += float(batch_weights.sum())
        final_loss = loss_sum / weight_sum

    def array(tensor):
        return tensor.detach().cpu().numpy().astype(np.float32, copy=True)

    fitted = LowRankEffectModel(
        layout=layout,
        state_mean=state_mean,
        state_scale=state_scale,
        action_mean=prepared.action_mean,
        action_scale=prepared.action_scale,
        state_hidden_weight=array(model.state_hidden.weight).T,
        state_hidden_bias=array(model.state_hidden.bias),
        state_latent_weight=array(model.state_latent.weight).T,
        state_latent_bias=array(model.state_latent.bias),
        action_latent_weight=array(model.action_latent.weight).T,
        action_bias_weight=array(model.action_bias.weight).reshape(-1),
    )
    return fitted, {
        "training_weighted_mse": float(final_loss),
        "parameters": float(sum(value.numel() for value in model.parameters())),
        "maximum_centered_action": float(np.abs(actions).max()),
    }


def fit_low_rank_implicit_member(
    samples: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    iterations: int,
    n_step_options: int,
    q_trees: int,
    value_trees: int,
    expectile: float,
    seed: int,
    threads: int,
    bootstrap: dict[str, int],
    hidden: int,
    rank: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    prepared: StructuredArrays | None = None,
) -> LowRankImplicitMember:
    import numpy as np
    from .implicit_learning import (
        _arrays,
        _n_step_targets,
        _q_regressor,
        _value_regressor,
    )

    episodes = _episodes(samples)
    prepared = prepared or _structured_arrays(episodes, layout)
    _q_rows, offline_states, row_episodes, base_weights = _arrays(episodes)
    if row_episodes != prepared.row_episodes:
        raise RuntimeError("low-rank and Bellman row order differs")
    row_weights = base_weights * np.asarray([
        bootstrap.get(episode, 0) for episode in row_episodes
    ], dtype=np.float64)
    value_model = outcome_model = q_model = None
    reports = []
    for iteration in range(iterations):
        target_map = _n_step_targets(
            episodes, value_model, n_step_options=n_step_options
        )
        labels = np.asarray([
            target_map[(episode, sample.option_id)]
            for episode, rows in episodes.items() for sample in rows
        ], dtype=np.float32)
        outcome_model = _q_regressor(
            trees=q_trees, seed=seed + iteration * 3, threads=threads
        )
        outcome_model.fit(offline_states, labels, sample_weight=row_weights)
        common = outcome_model.predict(offline_states)
        residual = labels - common
        q_model, effect_report = _fit_effect_model(
            prepared,
            layout=layout,
            targets=residual,
            row_weights=row_weights,
            seed=seed + iteration * 3 + 1,
            threads=threads,
            hidden=hidden,
            rank=rank,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        )
        centered = q_model.predict_centered(
            prepared.states, prepared.centered_actions
        )
        q_predictions = common + centered
        value_model = _value_regressor(
            trees=value_trees,
            seed=seed + iteration * 3 + 2,
            threads=threads,
            expectile=expectile,
            row_weights=row_weights,
        )
        value_model.fit(offline_states, q_predictions)
        reports.append({
            "iteration": float(iteration + 1),
            "target_mean": float(np.average(labels, weights=row_weights)),
            "q_weighted_mse": float(np.average(
                (q_predictions - labels) ** 2, weights=row_weights
            )),
            "zero_effect_weighted_mse": float(np.average(
                residual ** 2, weights=row_weights
            )),
            **effect_report,
        })
    assert outcome_model is not None and q_model is not None and value_model is not None
    return LowRankImplicitMember(
        outcome_model=outcome_model,
        q_model=q_model,
        value_model=value_model,
        bootstrap=dict(sorted(bootstrap.items())),
        iterations=tuple(reports),
    )


def _bootstrap_counts(groups: tuple[str, ...], *, seed: int) -> dict[str, int]:
    generator = random.Random(seed)
    return dict(Counter(generator.choice(groups) for _ in groups))


def fit_low_rank_population(
    samples: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    members: int = 7,
    iterations: int = 2,
    n_step_options: int = 8,
    q_trees: int = 8,
    value_trees: int = 8,
    expectile: float = 0.10,
    seed: int = 360_813,
    threads: int = 1,
    hidden: int = 48,
    rank: int = 12,
    epochs: int = 8,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
) -> list[LowRankImplicitMember]:
    episodes = _episodes(samples)
    groups = tuple(episodes)
    prepared = _structured_arrays(episodes, layout)
    if members < 1:
        raise ValueError("low-rank population cannot be empty")
    return [
        fit_low_rank_implicit_member(
            samples,
            layout=layout,
            iterations=iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            expectile=expectile,
            seed=seed + member * 10_000,
            threads=threads,
            bootstrap=_bootstrap_counts(groups, seed=seed + member * 10_000),
            hidden=hidden,
            rank=rank,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            prepared=prepared,
        )
        for member in range(members)
    ]


def _prediction_layout(
    episodes: dict[str, list[OptionStep]], layout: FeatureRoleLayout
):
    import numpy as np

    states = []
    actions = []
    coefficients = []
    starts = []
    samples = []
    row_episodes = []
    for episode, rows in episodes.items():
        for sample in rows:
            candidates = np.asarray(sample.candidate_vectors, dtype=np.float32)
            state_candidates = candidates[:, layout.state_indices]
            if not np.allclose(state_candidates, state_candidates[0], atol=1e-7):
                raise ValueError("candidate-invariant state features changed by action")
            action_rows = candidates[:, layout.action_indices]
            probabilities = np.asarray(sample.behavior_probabilities, dtype=np.float64)
            factual = sample.legal_actions.index(sample.action)
            starts.append(len(actions))
            states.append(state_candidates[0])
            actions.extend(action_rows)
            coefficients.extend(
                float(index == factual) - probabilities[index]
                for index in range(len(action_rows))
            )
            samples.append(sample)
            row_episodes.append(episode)
    return (
        np.asarray(states, dtype=np.float32),
        np.asarray(actions, dtype=np.float32),
        np.asarray(coefficients, dtype=np.float64),
        np.asarray(starts, dtype=np.int64),
        tuple(samples),
        tuple(row_episodes),
    )


def _population_choice(predictions, sample, supported, member_indices):
    import numpy as np

    baseline = sample.legal_actions.index(sample.baseline_action)
    selected = np.asarray(member_indices, dtype=np.int64)
    advantages = predictions[selected, :] - predictions[selected, [baseline]][:, None]
    candidates = []
    for index, action in enumerate(sample.legal_actions):
        if action == sample.baseline_action or not supported[index]:
            continue
        values = advantages[:, index]
        bound = float(2.0 * values.max() - values.min())
        if bound < 0.0:
            candidates.append((bound, action))
    return min(candidates, default=(0.0, sample.baseline_action))[1]


def evaluate_low_rank_fold(
    train: list[OptionStep],
    heldout: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    fold: int,
    episode_cohorts: dict[str, str],
    members: int,
    iterations: int,
    n_step_options: int,
    q_trees: int,
    value_trees: int,
    seed: int,
    threads: int,
    hidden: int,
    rank: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
) -> dict[str, object]:
    import numpy as np
    from .advantage_learning import (
        _augment_steps,
        fit_hazard_codebook,
        rich_feature_names,
    )
    from .implicit_learning import _arrays, _n_step_targets, _support_artifacts, _supported

    representation = fit_hazard_codebook(train, seed=seed + fold * 100_000 + 30_000)
    train = _augment_steps(train, representation)
    heldout = _augment_steps(heldout, representation)
    rich_layout = named_feature_roles(rich_feature_names())
    if layout.names != rich_layout.names:
        raise ValueError("cross-fit rich feature layout drifted")
    population = fit_low_rank_population(
        train,
        layout=layout,
        members=members,
        iterations=iterations,
        n_step_options=n_step_options,
        q_trees=q_trees,
        value_trees=value_trees,
        seed=seed + fold * 100_000,
        threads=threads,
        hidden=hidden,
        rank=rank,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    support, support_report = _support_artifacts(
        train, seed=seed + fold * 100_000 + 50_000
    )
    heldout_episodes = _episodes(heldout)
    _factual, offline_states, array_episodes, _weights = _arrays(heldout_episodes)
    states, actions, coefficients, starts, samples, prediction_episodes = (
        _prediction_layout(heldout_episodes, layout)
    )
    if array_episodes != prediction_episodes:
        raise RuntimeError("low-rank held-out layouts differ")
    predictions = np.asarray([
        member.q_model.predict_grouped(states, actions, starts)
        for member in population
    ], dtype=np.float64)
    centered = np.asarray([
        np.add.reduceat(coefficients * values, starts)
        for values in predictions
    ])
    common = np.asarray([
        member.outcome_model.predict(offline_states) for member in population
    ], dtype=np.float64)
    targets = []
    for member in population:
        target_map = _n_step_targets(
            heldout_episodes, member.value_model, n_step_options=n_step_options
        )
        targets.append([
            target_map[(episode, sample.option_id)]
            for episode, rows in heldout_episodes.items() for sample in rows
        ])
    targets = np.asarray(targets, dtype=np.float64)
    zero_errors = np.mean((targets - common) ** 2, axis=0)
    q_errors = np.mean((targets - common - centered) ** 2, axis=0)

    stops = [*starts[1:], len(actions)]
    episode_reports: dict[str, dict[str, object]] = {}
    full_proposals = split_proposals = split_union = split_agreements = 0
    loo_union = loo_exact = full_proposal_loo_exact = 0
    unsupported_candidates = 0
    full_actions: Counter[str] = Counter()
    split_actions: Counter[str] = Counter()
    all_members = tuple(range(members))
    left = tuple(range(members // 2))
    right = tuple(range(members // 2, members))
    for index, (start, stop) in enumerate(zip(starts, stops, strict=True)):
        sample = samples[index]
        episode = sample.episode_id
        report = episode_reports.setdefault(episode, {
            "cohort": episode_cohorts[episode],
            "zero_squared_error": 0.0,
            "q_squared_error": 0.0,
            "options": 0,
            "full_proposals": 0,
        })
        report["zero_squared_error"] += float(zero_errors[index])
        report["q_squared_error"] += float(q_errors[index])
        report["options"] += 1
        mask = _supported(sample, support)
        unsupported_candidates += sum(
            action != sample.baseline_action and not mask[action_index]
            for action_index, action in enumerate(sample.legal_actions)
        )
        option_predictions = predictions[:, start:stop]
        full = _population_choice(
            option_predictions, sample, mask, all_members
        )
        loo = tuple(
            _population_choice(
                option_predictions,
                sample,
                mask,
                tuple(member for member in all_members if member != omitted),
            )
            for omitted in all_members
        )
        left_choice = _population_choice(option_predictions, sample, mask, left)
        right_choice = _population_choice(option_predictions, sample, mask, right)
        split_either = (
            left_choice != sample.baseline_action
            or right_choice != sample.baseline_action
        )
        split_union += int(split_either)
        split_agreements += int(split_either and left_choice == right_choice)
        if left_choice == right_choice and left_choice != sample.baseline_action:
            split_proposals += 1
            split_actions[left_choice] += 1
        loo_either = (
            full != sample.baseline_action
            or any(choice != sample.baseline_action for choice in loo)
        )
        stable = all(choice == full for choice in loo)
        loo_union += int(loo_either)
        loo_exact += int(loo_either and stable)
        full_proposal_loo_exact += int(
            full != sample.baseline_action and stable
        )
        if full != sample.baseline_action:
            full_proposals += 1
            report["full_proposals"] += 1
            full_actions[full] += 1
    for report in episode_reports.values():
        report["q_beats_zero"] = (
            report["q_squared_error"] < report["zero_squared_error"]
        )
    return {
        "fold": fold,
        "fit_episodes": sorted({sample.episode_id for sample in train}),
        "heldout_episodes": sorted(episode_reports),
        "support": support_report,
        "bootstrap": [member.bootstrap for member in population],
        "iterations": [member.iterations for member in population],
        "episodes": episode_reports,
        "options": len(samples),
        "full_proposals": full_proposals,
        "full_proposal_loo_exact": full_proposal_loo_exact,
        "loo_union": loo_union,
        "loo_exact": loo_exact,
        "split_proposals": split_proposals,
        "split_union": split_union,
        "split_agreements": split_agreements,
        "unsupported_candidates": unsupported_candidates,
        "full_actions": dict(sorted(full_actions.items())),
        "split_actions": dict(sorted(split_actions.items())),
    }


def crossfit_low_rank_report(
    samples: list[OptionStep],
    *,
    layout: FeatureRoleLayout,
    episode_cohorts: dict[str, str],
    folds: int = 3,
    members: int = 7,
    iterations: int = 2,
    n_step_options: int = 8,
    q_trees: int = 16,
    value_trees: int = 12,
    seed: int = 360_813,
    threads: int = 32,
    hidden: int = 48,
    rank: int = 12,
    epochs: int = 12,
    batch_size: int = 2048,
    learning_rate: float = 1e-3,
) -> dict[str, object]:
    from .advantage_learning import _folds

    episodes = _episodes(samples)
    groups = list(episodes)
    if set(episode_cohorts) != set(groups):
        raise ValueError("every development episode needs exactly one cohort")
    partitions = _folds(groups, count=folds, seed=seed)
    reports = []
    for fold, heldout_groups in enumerate(partitions):
        heldout_set = set(heldout_groups)
        reports.append(evaluate_low_rank_fold(
            [sample for sample in samples if sample.episode_id not in heldout_set],
            [sample for sample in samples if sample.episode_id in heldout_set],
            layout=layout,
            fold=fold,
            episode_cohorts=episode_cohorts,
            members=members,
            iterations=iterations,
            n_step_options=n_step_options,
            q_trees=q_trees,
            value_trees=value_trees,
            seed=seed,
            threads=threads,
            hidden=hidden,
            rank=rank,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
        ))
    episode_reports = {
        episode: report
        for fold_report in reports
        for episode, report in fold_report["episodes"].items()
    }

    def cohort(name: str | None) -> dict[str, object]:
        rows = [
            row for row in episode_reports.values()
            if name is None or row["cohort"] == name
        ]
        zero = sum(row["zero_squared_error"] for row in rows)
        q_loss = sum(row["q_squared_error"] for row in rows)
        return {
            "episode_groups": len(rows),
            "options": sum(row["options"] for row in rows),
            "zero_squared_error": zero,
            "q_squared_error": q_loss,
            "relative_q_loss": q_loss / zero if zero > 0.0 else math.inf,
            "episodes_beating_zero": sum(row["q_beats_zero"] for row in rows),
            "full_proposals": sum(row["full_proposals"] for row in rows),
        }

    options = sum(report["options"] for report in reports)
    full_proposals = sum(report["full_proposals"] for report in reports)
    full_proposal_loo_exact = sum(
        report["full_proposal_loo_exact"] for report in reports
    )
    loo_union = sum(report["loo_union"] for report in reports)
    loo_exact = sum(report["loo_exact"] for report in reports)
    split_union = sum(report["split_union"] for report in reports)
    split_agreements = sum(report["split_agreements"] for report in reports)
    return {
        "folds": reports,
        "episodes": episode_reports,
        "cohorts": {
            "overall": cohort(None),
            **{
                name: cohort(name)
                for name in sorted(set(episode_cohorts.values()))
            },
        },
        "options": options,
        "full_proposals": full_proposals,
        "full_proposal_rate": full_proposals / options,
        "full_proposal_loo_exact_rate": (
            full_proposal_loo_exact / full_proposals if full_proposals else 0.0
        ),
        "loo_union_stability": loo_exact / loo_union if loo_union else 0.0,
        "split_conditional_agreement": (
            split_agreements / split_union if split_union else 0.0
        ),
        "split_proposals": sum(report["split_proposals"] for report in reports),
        "unsupported_candidates": sum(
            report["unsupported_candidates"] for report in reports
        ),
        "full_actions": dict(sorted(sum(
            (Counter(report["full_actions"]) for report in reports), Counter()
        ).items())),
    }


def run_low_rank_causal_smoke(*, threads: int = 8) -> dict[str, object]:
    """Recover delayed HIT effects and abstain under a known null process."""
    import numpy as np
    from .implicit_learning import delayed_effect_episodes

    parameters = {
        "members": 7,
        "iterations": 4,
        "n_step_options": 4,
        "q_trees": 40,
        "value_trees": 32,
        "threads": threads,
        "hidden": 24,
        "rank": 6,
        "epochs": 24,
        "batch_size": 512,
        "learning_rate": 1e-3,
    }
    count, options, delay = 64, 72, 12
    layout = FeatureRoleLayout(
        names=tuple([
            *(f"observation:pending-{index}" for index in range(delay)),
            "action:treatment",
        ]),
        state_indices=tuple(range(delay)),
        action_indices=(delay,),
    )

    def fit(null_effect: bool):
        samples = delayed_effect_episodes(
            count=count, options=options, delay=delay, null_effect=null_effect
        )
        population = fit_low_rank_population(
            samples, layout=layout, **parameters
        )
        return samples, population

    delayed, population = fit(False)
    interior = [
        sample for sample in delayed
        if 8 <= sample.sequence < options - delay
    ]
    effects = [
        float(np.mean([
            member.q_model.predict(sample.candidate_vectors)[1]
            - member.q_model.predict(sample.candidate_vectors)[0]
            for sample in interior
        ]))
        for member in population
    ]
    q_beats_zero = [
        member.iterations[-1]["q_weighted_mse"]
        < member.iterations[-1]["zero_effect_weighted_mse"]
        for member in population
    ]
    null, null_population = fit(True)
    null_overrides = 0
    for sample in null:
        member_effects = []
        for member in null_population:
            prediction = member.q_model.predict(sample.candidate_vectors)
            member_effects.append(float(prediction[1] - prediction[0]))
        null_overrides += int(
            2.0 * max(member_effects) - min(member_effects) < 0.0
        )
    gates = {
        "all_members_recover_beneficial_direction": all(
            effect < 0.0 for effect in effects
        ),
        "all_members_beat_zero_effect": all(q_beats_zero),
        "null_population_abstains": null_overrides == 0,
        "delayed_effect_exceeds_single_backup": delay > parameters["n_step_options"],
    }
    return {
        "schema": "autonomous-generation-6-low-rank-causal-smoke-v1",
        "evidence_eligible": False,
        "parameters": {
            **parameters,
            "episode_groups": count,
            "options_per_episode": options,
            "effect_delay": delay,
        },
        "member_mean_effects": effects,
        "member_q_beats_zero": q_beats_zero,
        "null_overrides": null_overrides,
        "null_options": len(null),
        "gates": gates,
        "passed": all(gates.values()),
    }
