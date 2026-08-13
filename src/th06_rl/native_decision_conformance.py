"""Decision-level numerical certification for the native IQL actor.

The learner is trained and inspected with array libraries, while deployment
uses a fixed scalar C++ kernel.  Raw logits are not a policy identity: common
offsets cancel when the baseline-centred proposal is formed.  This module
therefore provides independent references for the historical float32 scorer
and the float64-intermediate serving successor:

* ``native_order_float32_scores`` mirrors the historical C++ reduction order;
* ``actor_forward_reference`` evaluates the mathematical network in float64
  and propagates a conservative float32 forward-error envelope;
* ``native_order_centered_portability_reference`` follows the declared scalar
  serving precision and bounds target-local ``tanh`` variation.

The envelope is intentionally based on absolute intermediate products and
operation counts.  It never grows or shrinks according to the final logit.
Wine/Linux exact-action differential tests remain mandatory because no
portable bound can specify every platform libm implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

from .iql_actor_learning import IqlActorModel


FLOAT32_UNIT_ROUNDOFF = 2.0 ** -24
# Both frozen targets use a float32 tanh result.  Eight unit roundoffs is a
# deliberately conservative allowance around the real tanh value; the frozen
# Linux and Win32 differentials independently check the actual libm targets.
TANH_ABSOLUTE_ALLOWANCE = 8.0 * FLOAT32_UNIT_ROUNDOFF


@dataclass(frozen=True)
class ActorForwardReference:
    scores: object
    error_bounds: object
    maximum_intermediate_absolute_sum: float


@dataclass(frozen=True)
class DecisionCertificate:
    choice: str
    selected_index: int
    selected_advantage: float
    decision_margin: float
    error_envelope: float
    margin_ratio: float
    certified: bool
    comparison: str


def _gamma(operations: int) -> float:
    if operations < 0:
        raise ValueError("floating-point operation count cannot be negative")
    product = operations * FLOAT32_UNIT_ROUNDOFF
    if product >= 1.0:
        raise ValueError("floating-point error bound is not finite")
    return product / (1.0 - product)


def _normalization_reference(values, mean, scale):
    import numpy as np

    values = np.asarray(values, dtype=np.float32).astype(np.float64)
    mean = np.asarray(mean, dtype=np.float32).astype(np.float64)
    scale = np.asarray(scale, dtype=np.float32).astype(np.float64)
    if values.shape[-1] != len(mean) or mean.shape != scale.shape:
        raise ValueError("actor normalization shape differs")
    exact = (values - mean) / scale
    eta = FLOAT32_UNIT_ROUNDOFF / (1.0 - FLOAT32_UNIT_ROUNDOFF)
    subtraction = eta * (np.abs(values) + np.abs(mean))
    division_input = subtraction / np.abs(scale)
    error = division_input + eta * (np.abs(exact) + division_input)
    return exact, error


def _affine_reference(inputs, input_error, weight, bias):
    import numpy as np

    inputs = np.asarray(inputs, dtype=np.float64)
    input_error = np.asarray(input_error, dtype=np.float64)
    weight = np.asarray(weight, dtype=np.float32).astype(np.float64)
    bias = np.asarray(bias, dtype=np.float32).astype(np.float64)
    if inputs.shape != input_error.shape or inputs.shape[-1] != weight.shape[0]:
        raise ValueError("actor affine input shape differs")
    exact = inputs @ weight + bias
    absolute_sum = np.abs(inputs) @ np.abs(weight) + np.abs(bias)
    gamma = _gamma(2 * weight.shape[0] + 1)
    propagated = input_error @ np.abs(weight)
    error = gamma * absolute_sum + (1.0 + gamma) * propagated
    return exact, error, absolute_sum


def _tanh_reference(values, errors):
    import numpy as np

    values = np.asarray(values, dtype=np.float64)
    errors = np.asarray(errors, dtype=np.float64)
    result = np.tanh(values)
    closest = np.maximum(np.abs(values) - errors, 0.0)
    derivative = 1.0 - np.square(np.tanh(closest))
    return result, derivative * errors + TANH_ABSOLUTE_ALLOWANCE


def actor_forward_reference(
    model: IqlActorModel, rows,
) -> ActorForwardReference:
    """Evaluate scores and a per-score float32 forward-error envelope.

    Input rows and model parameters are interpreted as their serialized
    float32 values.  Normalization, both affine towers, tanh, the low-rank dot
    product, and the final score addition are all included in the bound.
    """
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(model.layout.names):
        raise ValueError("actor reference input shape differs")
    states = matrix[:, model.layout.state_indices]
    actions = matrix[:, model.layout.action_indices]
    state, state_error = _normalization_reference(
        states, model.state_mean, model.state_scale
    )
    action, action_error = _normalization_reference(
        actions, model.action_mean, model.action_scale
    )

    state_pre, state_pre_error, state_hidden_sum = _affine_reference(
        state, state_error, model.state_hidden_weight, model.state_hidden_bias
    )
    state_hidden, state_hidden_error = _tanh_reference(
        state_pre, state_pre_error
    )
    state_latent, state_latent_error, state_latent_sum = _affine_reference(
        state_hidden, state_hidden_error,
        model.state_latent_weight, model.state_latent_bias,
    )
    action_pre, action_pre_error, action_hidden_sum = _affine_reference(
        action, action_error,
        model.action_hidden_weight, model.action_hidden_bias,
    )
    action_hidden, action_hidden_error = _tanh_reference(
        action_pre, action_pre_error
    )
    action_latent, action_latent_error, action_latent_sum = _affine_reference(
        action_hidden, action_hidden_error,
        model.action_latent_weight, model.action_latent_bias,
    )
    action_score, action_score_error, action_score_sum = _affine_reference(
        action_hidden, action_hidden_error,
        np.asarray(model.action_score_weight, dtype=np.float32)[:, None],
        np.asarray([model.action_score_bias], dtype=np.float32),
    )
    action_score = action_score[:, 0]
    action_score_error = action_score_error[:, 0]

    products = action_latent * state_latent
    absolute_dot_sum = np.sum(
        np.abs(action_latent) * np.abs(state_latent), axis=1
    )
    product_input_error = np.sum(
        np.abs(state_latent) * action_latent_error
        + np.abs(action_latent) * state_latent_error
        + action_latent_error * state_latent_error,
        axis=1,
    )
    dot_gamma = _gamma(2 * products.shape[1] + 1)
    dot = np.sum(products, axis=1)
    dot_error = (
        dot_gamma * absolute_dot_sum
        + (1.0 + dot_gamma) * product_input_error
    )
    rank_scale = float(np.float32(
        1.0 / math.sqrt(float(np.float32(products.shape[1])))
    ))
    scaled_dot = dot * rank_scale
    scale_gamma = _gamma(1)
    scaled_error = (
        (1.0 + scale_gamma) * abs(rank_scale) * dot_error
        + scale_gamma * np.abs(scaled_dot)
    )
    scores = action_score + scaled_dot
    final_gamma = _gamma(1)
    error = (
        action_score_error + scaled_error
        + final_gamma * (np.abs(action_score) + np.abs(scaled_dot))
    )
    maximum_sum = max(
        float(np.max(value)) for value in (
            state_hidden_sum, state_latent_sum, action_hidden_sum,
            action_latent_sum, action_score_sum, absolute_dot_sum[:, None],
        )
    )
    if not np.isfinite(scores).all() or not np.isfinite(error).all():
        raise ValueError("actor reference produced a non-finite result")
    return ActorForwardReference(
        scores=scores,
        error_bounds=error,
        maximum_intermediate_absolute_sum=maximum_sum,
    )


def actor_centered_forward_reference(
    model: IqlActorModel, rows, *, baseline_index: int,
) -> ActorForwardReference:
    """Reference the deployment kernel's directly centred advantages.

    The policy never consumes an absolute actor logit.  Computing the action
    tower differences before the final dot products removes the common bias
    and state-dependent offset instead of first rounding two large logits and
    subtracting them.  This function supplies the corresponding, materially
    tighter forward-error envelope.
    """
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float32)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(model.layout.names)
        or not 0 <= baseline_index < len(matrix)
    ):
        raise ValueError("centered actor reference input shape differs")
    states = matrix[:, model.layout.state_indices]
    if not np.array_equal(states, np.broadcast_to(states[0], states.shape)):
        raise ValueError("centered actor candidates changed factual state")
    actions = matrix[:, model.layout.action_indices]
    state, state_error = _normalization_reference(
        states[:1], model.state_mean, model.state_scale
    )
    action, action_error = _normalization_reference(
        actions, model.action_mean, model.action_scale
    )
    state_pre, state_pre_error, state_hidden_sum = _affine_reference(
        state, state_error, model.state_hidden_weight, model.state_hidden_bias
    )
    state_hidden, state_hidden_error = _tanh_reference(
        state_pre, state_pre_error
    )
    state_latent, state_latent_error, state_latent_sum = _affine_reference(
        state_hidden, state_hidden_error,
        model.state_latent_weight, model.state_latent_bias,
    )
    action_pre, action_pre_error, action_hidden_sum = _affine_reference(
        action, action_error,
        model.action_hidden_weight, model.action_hidden_bias,
    )
    action_hidden, action_hidden_error = _tanh_reference(
        action_pre, action_pre_error
    )
    action_latent, action_latent_error, action_latent_sum = _affine_reference(
        action_hidden, action_hidden_error,
        model.action_latent_weight, model.action_latent_bias,
    )

    baseline_hidden = action_hidden[baseline_index]
    baseline_hidden_error = action_hidden_error[baseline_index]
    hidden_delta = action_hidden - baseline_hidden
    hidden_delta_error = action_hidden_error + baseline_hidden_error
    subtraction_gamma = _gamma(1)
    hidden_delta_error += subtraction_gamma * (
        np.abs(action_hidden) + np.abs(baseline_hidden)
    )
    score_products = hidden_delta * np.asarray(
        model.action_score_weight, dtype=np.float32
    ).astype(np.float64)
    score_absolute_sum = np.sum(np.abs(score_products), axis=1)
    score_gamma = _gamma(2 * hidden_delta.shape[1] + 1)
    score = np.sum(score_products, axis=1)
    score_error = (
        score_gamma * score_absolute_sum
        + (1.0 + score_gamma) * (
            hidden_delta_error @ np.abs(np.asarray(
                model.action_score_weight, dtype=np.float32
            ).astype(np.float64))
        )
    )

    baseline_latent = action_latent[baseline_index]
    baseline_latent_error = action_latent_error[baseline_index]
    latent_delta = action_latent - baseline_latent
    latent_delta_error = action_latent_error + baseline_latent_error
    latent_delta_error += subtraction_gamma * (
        np.abs(action_latent) + np.abs(baseline_latent)
    )
    dot_products = latent_delta * state_latent[0]
    dot_absolute_sum = np.sum(np.abs(dot_products), axis=1)
    dot_input_error = np.sum(
        np.abs(state_latent[0]) * latent_delta_error
        + np.abs(latent_delta) * state_latent_error[0]
        + latent_delta_error * state_latent_error[0],
        axis=1,
    )
    dot_gamma = _gamma(2 * latent_delta.shape[1] + 1)
    dot = np.sum(dot_products, axis=1)
    dot_error = (
        dot_gamma * dot_absolute_sum
        + (1.0 + dot_gamma) * dot_input_error
    )
    rank_scale = float(np.float32(
        1.0 / math.sqrt(float(np.float32(latent_delta.shape[1])))
    ))
    scale_gamma = _gamma(1)
    scaled_dot = dot * rank_scale
    scaled_error = (
        (1.0 + scale_gamma) * abs(rank_scale) * dot_error
        + scale_gamma * np.abs(scaled_dot)
    )
    advantages = score + scaled_dot
    final_gamma = _gamma(1)
    error = (
        score_error + scaled_error
        + final_gamma * (np.abs(score) + np.abs(scaled_dot))
    )
    # The baseline is defined to be exactly zero in the centered kernel.
    advantages[baseline_index] = 0.0
    error[baseline_index] = 0.0
    maximum_sum = max(
        float(np.max(value)) for value in (
            state_hidden_sum, state_latent_sum, action_hidden_sum,
            action_latent_sum, score_absolute_sum[:, None],
            dot_absolute_sum[:, None],
        )
    )
    if not np.isfinite(advantages).all() or not np.isfinite(error).all():
        raise ValueError("centered actor reference produced a non-finite result")
    return ActorForwardReference(
        scores=advantages,
        error_bounds=error,
        maximum_intermediate_absolute_sum=maximum_sum,
    )


def actor_centered_float64_scores(
    model: IqlActorModel, rows, *, baseline_index: int,
):
    """Fast array reference for the frozen float32-parameter/f64 policy."""
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float32)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(model.layout.names)
        or not 0 <= baseline_index < len(matrix)
    ):
        raise ValueError("float64 centered actor input shape differs")
    states = matrix[:, model.layout.state_indices]
    if not np.array_equal(states, np.broadcast_to(states[0], states.shape)):
        raise ValueError("float64 centered actor candidates changed state")
    # Online normalization is intentionally retained as float32.  Only the
    # fitted actor tower is promoted, so representation/support semantics do
    # not change and serialized weights remain the sole learned parameters.
    state = np.asarray(
        (states[0] - model.state_mean) / model.state_scale,
        dtype=np.float32,
    ).astype(np.float64)
    action = np.asarray(
        (matrix[:, model.layout.action_indices] - model.action_mean)
        / model.action_scale,
        dtype=np.float32,
    ).astype(np.float64)

    def f64(value):
        return np.asarray(value, dtype=np.float32).astype(np.float64)

    state_hidden = np.tanh(
        state @ f64(model.state_hidden_weight)
        + f64(model.state_hidden_bias)
    )
    state_latent = (
        state_hidden @ f64(model.state_latent_weight)
        + f64(model.state_latent_bias)
    )
    action_hidden = np.tanh(
        action @ f64(model.action_hidden_weight)
        + f64(model.action_hidden_bias)
    )
    action_latent = (
        action_hidden @ f64(model.action_latent_weight)
        + f64(model.action_latent_bias)
    )
    hidden_delta = action_hidden - action_hidden[baseline_index]
    latent_delta = action_latent - action_latent[baseline_index]
    result = (
        hidden_delta @ f64(model.action_score_weight)
        + np.sum(latent_delta * state_latent[None, :], axis=1)
        / math.sqrt(len(state_latent))
    )
    result[baseline_index] = 0.0
    if not np.isfinite(result).all():
        raise ValueError("float64 centered actor produced a non-finite result")
    return result


def native_order_float32_scores(model: IqlActorModel, rows):
    """Mirror ``th06_rl_score_iql_actor_population_v1`` scalar order.

    This deliberately avoids BLAS.  Every multiplication and addition is
    rounded to float32 at the same loop boundary as the C++ kernel.  Tanh is
    rounded to float32 after evaluation; target-specific libm differences are
    handled by the forward envelope and direct Linux/Win32 differential.
    """
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[1] != len(model.layout.names):
        raise ValueError("actor scalar-reference input shape differs")

    def f32(value):
        return np.float32(value)

    def normalized(row, indices, mean, scale):
        result = []
        for source, center, width in zip(
            indices, mean, scale, strict=True
        ):
            result.append(f32(f32(row[source] - center) / width))
        return result

    state = normalized(
        matrix[0], model.layout.state_indices,
        model.state_mean, model.state_scale,
    )
    state_indices = list(model.layout.state_indices)
    if any(not np.array_equal(
        matrix[0, state_indices], row[state_indices]
    ) for row in matrix[1:]):
        raise ValueError("actor scalar-reference candidates changed state")

    hidden_count = len(model.state_hidden_bias)
    rank_count = len(model.state_latent_bias)
    state_hidden = []
    for hidden in range(hidden_count):
        value = f32(model.state_hidden_bias[hidden])
        for feature, input_value in enumerate(state):
            product = f32(input_value * model.state_hidden_weight[feature, hidden])
            value = f32(value + product)
        state_hidden.append(f32(math.tanh(float(value))))
    state_latent = []
    for rank in range(rank_count):
        value = f32(model.state_latent_bias[rank])
        for hidden, input_value in enumerate(state_hidden):
            product = f32(input_value * model.state_latent_weight[hidden, rank])
            value = f32(value + product)
        state_latent.append(value)

    rank_scale = f32(1.0 / math.sqrt(float(f32(rank_count))))
    result = []
    for row in matrix:
        action = normalized(
            row, model.layout.action_indices,
            model.action_mean, model.action_scale,
        )
        action_hidden = []
        score = f32(model.action_score_bias)
        for hidden in range(hidden_count):
            value = f32(model.action_hidden_bias[hidden])
            for feature, input_value in enumerate(action):
                product = f32(
                    input_value * model.action_hidden_weight[feature, hidden]
                )
                value = f32(value + product)
            value = f32(math.tanh(float(value)))
            action_hidden.append(value)
            score = f32(score + f32(value * model.action_score_weight[hidden]))
        action_latent = []
        for rank in range(rank_count):
            value = f32(model.action_latent_bias[rank])
            for hidden, input_value in enumerate(action_hidden):
                product = f32(
                    input_value * model.action_latent_weight[hidden, rank]
                )
                value = f32(value + product)
            action_latent.append(value)
        dot = f32(0.0)
        for left, right in zip(action_latent, state_latent, strict=True):
            dot = f32(dot + f32(left * right))
        result.append(f32(score + f32(dot * rank_scale)))
    return np.asarray(result, dtype=np.float32)


def native_order_float32_centered_advantages(
    model: IqlActorModel, rows, *, baseline_index: int,
):
    """Mirror the native directly centred deployment reduction order."""
    import numpy as np

    matrix = np.asarray(rows, dtype=np.float32)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(model.layout.names)
        or not 0 <= baseline_index < len(matrix)
    ):
        raise ValueError("centered scalar-reference input shape differs")

    def f32(value):
        return np.float32(value)

    def normalized(row, indices, mean, scale):
        return [
            f32(f32(row[source] - center) / width)
            for source, center, width in zip(indices, mean, scale, strict=True)
        ]

    state_indices = list(model.layout.state_indices)
    if any(not np.array_equal(
        matrix[0, state_indices], row[state_indices]
    ) for row in matrix[1:]):
        raise ValueError("centered scalar-reference candidates changed state")
    state = normalized(
        matrix[0], model.layout.state_indices,
        model.state_mean, model.state_scale,
    )
    hidden_count = len(model.state_hidden_bias)
    rank_count = len(model.state_latent_bias)
    state_hidden = []
    for hidden in range(hidden_count):
        value = f32(model.state_hidden_bias[hidden])
        for feature, input_value in enumerate(state):
            value = f32(value + f32(
                input_value * model.state_hidden_weight[feature, hidden]
            ))
        state_hidden.append(f32(math.tanh(float(value))))
    state_latent = []
    for rank in range(rank_count):
        value = f32(model.state_latent_bias[rank])
        for hidden, input_value in enumerate(state_hidden):
            value = f32(value + f32(
                input_value * model.state_latent_weight[hidden, rank]
            ))
        state_latent.append(value)

    def action_tower(row):
        action = normalized(
            row, model.layout.action_indices,
            model.action_mean, model.action_scale,
        )
        hidden_values = []
        for hidden in range(hidden_count):
            value = f32(model.action_hidden_bias[hidden])
            for feature, input_value in enumerate(action):
                value = f32(value + f32(
                    input_value * model.action_hidden_weight[feature, hidden]
                ))
            hidden_values.append(f32(math.tanh(float(value))))
        latent_values = []
        for rank in range(rank_count):
            value = f32(model.action_latent_bias[rank])
            for hidden, input_value in enumerate(hidden_values):
                value = f32(value + f32(
                    input_value * model.action_latent_weight[hidden, rank]
                ))
            latent_values.append(value)
        return hidden_values, latent_values

    baseline_hidden, baseline_latent = action_tower(matrix[baseline_index])
    rank_scale = f32(1.0 / math.sqrt(float(f32(rank_count))))
    result = []
    for index, row in enumerate(matrix):
        if index == baseline_index:
            result.append(f32(0.0))
            continue
        action_hidden, action_latent = action_tower(row)
        score = f32(0.0)
        for hidden, (value, baseline) in enumerate(zip(
            action_hidden, baseline_hidden, strict=True
        )):
            delta = f32(value - baseline)
            score = f32(score + f32(
                delta * model.action_score_weight[hidden]
            ))
        dot = f32(0.0)
        for value, baseline, state_value in zip(
            action_latent, baseline_latent, state_latent, strict=True
        ):
            delta = f32(value - baseline)
            dot = f32(dot + f32(delta * state_value))
        result.append(f32(score + f32(dot * rank_scale)))
    return np.asarray(result, dtype=np.float32)


def native_order_centered_portability_reference(
    model: IqlActorModel, rows, *, baseline_index: int,
    serving_precision: str = "float32",
) -> ActorForwardReference:
    """Bound target variation around the specified scalar float32 policy.

    IEEE operations have identical operands, serving precision, and prescribed
    ordering on the frozen targets; only ``tanh`` is allowed a target-local
    perturbation.
    Once that perturbation exists, each subsequent multiply/add includes one
    local-ULP rounding discontinuity.  This is much tighter—and more relevant
    to deployment—than bounding the distance from float32 to an unused
    infinite-precision network.
    """
    import numpy as np

    if serving_precision == "float32":
        serving_type = np.float32
        tanh_allowance = TANH_ABSOLUTE_ALLOWANCE
    elif serving_precision == "float64":
        serving_type = np.float64
        tanh_allowance = 8.0 * (2.0 ** -53)
    else:
        raise ValueError("unsupported centered actor serving precision")
    matrix = np.asarray(rows, dtype=np.float32)
    if (
        matrix.ndim != 2 or matrix.shape[1] != len(model.layout.names)
        or not 0 <= baseline_index < len(matrix)
    ):
        raise ValueError("portability reference input shape differs")

    def cast(value):
        return serving_type(value)

    maximum_sum = 0.0

    def rounding_jump(value, variation: float) -> float:
        if variation <= 0.0:
            return 0.0
        magnitude = serving_type(abs(float(value)) + variation)
        spacing = float(np.spacing(magnitude))
        if not math.isfinite(spacing):
            raise ValueError("portability reference overflowed")
        return max(spacing, float(np.nextafter(
            serving_type(0.0), serving_type(1.0)
        )))

    def add(left, left_error, right, right_error):
        variation = float(left_error) + float(right_error)
        value = cast(left + right)
        return value, variation + rounding_jump(value, variation)

    def subtract(left, left_error, right, right_error):
        variation = float(left_error) + float(right_error)
        value = cast(left - right)
        return value, variation + rounding_jump(value, variation)

    def multiply(left, left_error, right, right_error):
        variation = (
            abs(float(left)) * float(right_error)
            + abs(float(right)) * float(left_error)
            + float(left_error) * float(right_error)
        )
        value = cast(left * right)
        return value, variation + rounding_jump(value, variation)

    def normalized(row, indices, mean, scale):
        result = []
        for source, center, width in zip(indices, mean, scale, strict=True):
            # With equal serialized operands, IEEE subtraction/division are
            # part of the canonical policy rather than a cross-target error.
            normalized = np.float32(np.float32(row[source] - center) / width)
            result.append((cast(normalized), 0.0))
        return result

    state_indices = list(model.layout.state_indices)
    if any(not np.array_equal(
        matrix[0, state_indices], row[state_indices]
    ) for row in matrix[1:]):
        raise ValueError("portability reference candidates changed state")
    state = normalized(
        matrix[0], model.layout.state_indices,
        model.state_mean, model.state_scale,
    )
    hidden_count = len(model.state_hidden_bias)
    rank_count = len(model.state_latent_bias)
    state_hidden = []
    for hidden in range(hidden_count):
        value, error = cast(model.state_hidden_bias[hidden]), 0.0
        absolute_sum = abs(float(value))
        for feature, (input_value, input_error) in enumerate(state):
            weight = model.state_hidden_weight[feature, hidden]
            product, product_error = multiply(
                input_value, input_error, weight, 0.0
            )
            absolute_sum += abs(float(product))
            value, error = add(value, error, product, product_error)
        maximum_sum = max(maximum_sum, absolute_sum)
        closest = max(abs(float(value)) - error, 0.0)
        derivative = 1.0 - math.tanh(closest) ** 2
        state_hidden.append((
            cast(math.tanh(float(value))),
            derivative * error + tanh_allowance,
        ))
    state_latent = []
    for rank in range(rank_count):
        value, error = cast(model.state_latent_bias[rank]), 0.0
        absolute_sum = abs(float(value))
        for hidden, (input_value, input_error) in enumerate(state_hidden):
            weight = model.state_latent_weight[hidden, rank]
            product, product_error = multiply(
                input_value, input_error, weight, 0.0
            )
            absolute_sum += abs(float(product))
            value, error = add(value, error, product, product_error)
        maximum_sum = max(maximum_sum, absolute_sum)
        state_latent.append((value, error))

    def action_tower(row):
        nonlocal maximum_sum
        action = normalized(
            row, model.layout.action_indices,
            model.action_mean, model.action_scale,
        )
        hidden_values = []
        for hidden in range(hidden_count):
            value, error = cast(model.action_hidden_bias[hidden]), 0.0
            absolute_sum = abs(float(value))
            for feature, (input_value, input_error) in enumerate(action):
                weight = model.action_hidden_weight[feature, hidden]
                product, product_error = multiply(
                    input_value, input_error, weight, 0.0
                )
                absolute_sum += abs(float(product))
                value, error = add(value, error, product, product_error)
            maximum_sum = max(maximum_sum, absolute_sum)
            closest = max(abs(float(value)) - error, 0.0)
            derivative = 1.0 - math.tanh(closest) ** 2
            hidden_values.append((
                cast(math.tanh(float(value))),
                derivative * error + tanh_allowance,
            ))
        latent_values = []
        for rank in range(rank_count):
            value, error = cast(model.action_latent_bias[rank]), 0.0
            absolute_sum = abs(float(value))
            for hidden, (input_value, input_error) in enumerate(hidden_values):
                weight = model.action_latent_weight[hidden, rank]
                product, product_error = multiply(
                    input_value, input_error, weight, 0.0
                )
                absolute_sum += abs(float(product))
                value, error = add(value, error, product, product_error)
            maximum_sum = max(maximum_sum, absolute_sum)
            latent_values.append((value, error))
        return hidden_values, latent_values

    baseline_hidden, baseline_latent = action_tower(matrix[baseline_index])
    rank_scale = cast(1.0 / math.sqrt(float(serving_type(rank_count))))
    outputs = []
    output_errors = []
    for index, row in enumerate(matrix):
        if index == baseline_index:
            outputs.append(cast(0.0))
            output_errors.append(0.0)
            continue
        action_hidden, action_latent = action_tower(row)
        score, score_error = cast(0.0), 0.0
        absolute_score_sum = 0.0
        for hidden, (value, baseline) in enumerate(zip(
            action_hidden, baseline_hidden, strict=True
        )):
            delta, delta_error = subtract(*value, *baseline)
            product, product_error = multiply(
                delta, delta_error, model.action_score_weight[hidden], 0.0
            )
            absolute_score_sum += abs(float(product))
            score, score_error = add(
                score, score_error, product, product_error
            )
        dot, dot_error = cast(0.0), 0.0
        absolute_dot_sum = 0.0
        for value, baseline, state_value in zip(
            action_latent, baseline_latent, state_latent, strict=True
        ):
            delta, delta_error = subtract(*value, *baseline)
            product, product_error = multiply(
                delta, delta_error, *state_value
            )
            absolute_dot_sum += abs(float(product))
            dot, dot_error = add(dot, dot_error, product, product_error)
        maximum_sum = max(
            maximum_sum, absolute_score_sum, absolute_dot_sum
        )
        scaled, scaled_error = multiply(dot, dot_error, rank_scale, 0.0)
        output, output_error = add(
            score, score_error, scaled, scaled_error
        )
        outputs.append(output)
        output_errors.append(output_error)
    result = np.asarray(outputs, dtype=serving_type)
    errors = np.asarray(output_errors, dtype=np.float64)
    if not np.isfinite(result).all() or not np.isfinite(errors).all():
        raise ValueError("portability reference produced a non-finite result")
    return ActorForwardReference(
        scores=result,
        error_bounds=errors,
        maximum_intermediate_absolute_sum=maximum_sum,
    )


def certify_mean_population_decision(
    member_scores,
    member_error_bounds,
    legal_actions,
    baseline_action: str,
    supported,
) -> DecisionCertificate:
    """Certify the exact deployed mean-population baseline-centred choice."""
    import numpy as np

    scores = np.asarray(member_scores, dtype=np.float64)
    errors = np.asarray(member_error_bounds, dtype=np.float64)
    legal = tuple(map(str, legal_actions))
    mask = tuple(bool(value) for value in supported)
    if (
        scores.ndim != 2 or scores.shape != errors.shape
        or scores.shape[1] != len(legal) or len(mask) != len(legal)
        or baseline_action not in legal or not np.isfinite(scores).all()
        or not np.isfinite(errors).all() or (errors < 0.0).any()
    ):
        raise ValueError("actor decision-certificate inputs are invalid")
    baseline = legal.index(baseline_action)
    means = scores.mean(axis=0)
    mean_errors = errors.mean(axis=0)
    advantages = means - means[baseline]
    eligible = [
        index for index in range(len(legal))
        if index != baseline and mask[index]
    ]
    positive = [index for index in eligible if advantages[index] > 0.0]
    if positive:
        selected = max(positive, key=lambda index: (
            advantages[index], legal[index]
        ))
        comparisons = [(baseline, "positive-vs-baseline")]
        comparisons.extend(
            (index, "winner-vs-alternative")
            for index in eligible if index != selected
        )
        margins = []
        for other, label in comparisons:
            margin = means[selected] - means[other]
            envelope = mean_errors[selected] + mean_errors[other]
            margins.append((margin / envelope if envelope else math.inf,
                            margin, envelope, label))
        ratio, margin, envelope, comparison = min(margins)
    else:
        selected = baseline
        if eligible:
            margins = []
            for other in eligible:
                margin = means[baseline] - means[other]
                envelope = mean_errors[baseline] + mean_errors[other]
                margins.append((margin / envelope if envelope else math.inf,
                                margin, envelope, "baseline-vs-alternative"))
            ratio, margin, envelope, comparison = min(margins)
        else:
            ratio, margin, envelope, comparison = (
                math.inf, math.inf, 0.0, "no-supported-alternative"
            )
    return DecisionCertificate(
        choice=legal[selected],
        selected_index=selected,
        selected_advantage=float(advantages[selected]),
        decision_margin=float(margin),
        error_envelope=float(envelope),
        margin_ratio=float(ratio),
        certified=bool(margin > envelope),
        comparison=comparison,
    )
