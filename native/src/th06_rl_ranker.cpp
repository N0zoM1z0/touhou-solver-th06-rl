#include "th06_rl_ranker.h"

#include <cmath>
#include <cstdint>

extern "C" TH06_RL_RANKER_API int th06_rl_score_xgboost_v1(
    const Th06RlTreeNode* nodes,
    const std::int32_t node_count,
    const std::int32_t* tree_offsets,
    const std::int32_t tree_count,
    const float* features,
    const std::int32_t row_count,
    const std::int32_t feature_count,
    const float base_score,
    float* outputs) {
    if (nodes == nullptr || tree_offsets == nullptr || features == nullptr ||
        outputs == nullptr || node_count <= 0 || tree_count <= 0 ||
        row_count <= 0 || feature_count <= 0 || tree_offsets[0] != 0 ||
        tree_offsets[tree_count] != node_count) {
        return 1;
    }
    for (std::int32_t tree = 0; tree < tree_count; ++tree) {
        if (tree_offsets[tree] < 0 ||
            tree_offsets[tree] >= tree_offsets[tree + 1] ||
            tree_offsets[tree + 1] > node_count) {
            return 2;
        }
    }
    for (std::int32_t row = 0; row < row_count; ++row) {
        float score = base_score;
        const float* row_features = features + row * feature_count;
        for (std::int32_t tree = 0; tree < tree_count; ++tree) {
            const std::int32_t start = tree_offsets[tree];
            const std::int32_t size = tree_offsets[tree + 1] - start;
            std::int32_t node_index = 0;
            for (std::int32_t steps = 0; steps <= size; ++steps) {
                if (node_index < 0 || node_index >= size) {
                    return 3;
                }
                const Th06RlTreeNode& node = nodes[start + node_index];
                if (node.feature < 0) {
                    score += node.leaf;
                    break;
                }
                if (node.feature >= feature_count) {
                    return 4;
                }
                const float value = row_features[node.feature];
                node_index = std::isnan(value)
                    ? node.missing
                    : value < node.threshold ? node.left : node.right;
                if (steps == size) {
                    return 5;
                }
            }
        }
        outputs[row] = score;
    }
    return 0;
}

extern "C" TH06_RL_RANKER_API int th06_rl_score_xgboost_population_v1(
    const Th06RlTreeNode* nodes,
    const std::int32_t node_count,
    const std::int32_t* tree_offsets,
    const std::int32_t tree_count,
    const std::int32_t* model_tree_offsets,
    const std::int32_t model_count,
    const float* features,
    const std::int32_t row_count,
    const std::int32_t feature_count,
    const float* base_scores,
    float* outputs) {
    if (nodes == nullptr || tree_offsets == nullptr ||
        model_tree_offsets == nullptr || features == nullptr ||
        base_scores == nullptr || outputs == nullptr || node_count <= 0 ||
        tree_count <= 0 || model_count <= 0 || row_count <= 0 ||
        feature_count <= 0 || tree_offsets[0] != 0 ||
        tree_offsets[tree_count] != node_count ||
        model_tree_offsets[0] != 0 ||
        model_tree_offsets[model_count] != tree_count) {
        return 1;
    }
    for (std::int32_t tree = 0; tree < tree_count; ++tree) {
        if (tree_offsets[tree] < 0 ||
            tree_offsets[tree] >= tree_offsets[tree + 1] ||
            tree_offsets[tree + 1] > node_count) {
            return 2;
        }
    }
    for (std::int32_t model = 0; model < model_count; ++model) {
        if (model_tree_offsets[model] < 0 ||
            model_tree_offsets[model] >= model_tree_offsets[model + 1] ||
            model_tree_offsets[model + 1] > tree_count ||
            !std::isfinite(base_scores[model])) {
            return 3;
        }
    }
    for (std::int32_t model = 0; model < model_count; ++model) {
        const std::int32_t first_tree = model_tree_offsets[model];
        const std::int32_t last_tree = model_tree_offsets[model + 1];
        for (std::int32_t row = 0; row < row_count; ++row) {
            outputs[model * row_count + row] = base_scores[model];
        }
        // Keep one tree's compact node array hot while traversing every
        // candidate row. Per-row floating-point accumulation remains in the
        // original tree order, so this is bit-for-bit the same population.
        for (std::int32_t tree = first_tree; tree < last_tree; ++tree) {
            const std::int32_t start = tree_offsets[tree];
            const std::int32_t size = tree_offsets[tree + 1] - start;
            for (std::int32_t row = 0; row < row_count; ++row) {
                const float* row_features = features + row * feature_count;
                std::int32_t node_index = 0;
                for (std::int32_t steps = 0; steps <= size; ++steps) {
                    if (node_index < 0 || node_index >= size) return 4;
                    const Th06RlTreeNode& node = nodes[start + node_index];
                    if (node.feature < 0) {
                        outputs[model * row_count + row] += node.leaf;
                        break;
                    }
                    if (node.feature >= feature_count) return 5;
                    const float value = row_features[node.feature];
                    node_index = std::isnan(value)
                        ? node.missing
                        : value < node.threshold ? node.left : node.right;
                    if (steps == size) return 6;
                }
            }
        }
    }
    return 0;
}

extern "C" TH06_RL_RANKER_API int th06_rl_min_support_distance_v1(
    const float* features,
    const std::int32_t row_count,
    const std::int32_t feature_count,
    const float* feature_mean,
    const float* feature_scale,
    const float* standardized_prototypes,
    const std::int32_t prototype_count,
    const std::int32_t* action_offsets,
    const std::int32_t action_count,
    const std::int32_t* row_actions,
    float* outputs) {
    if (features == nullptr || feature_mean == nullptr ||
        feature_scale == nullptr || standardized_prototypes == nullptr ||
        action_offsets == nullptr || row_actions == nullptr ||
        outputs == nullptr || row_count <= 0 || feature_count <= 0 ||
        prototype_count <= 0 || action_count <= 0 || action_offsets[0] != 0 ||
        action_offsets[action_count] != prototype_count) {
        return 1;
    }
    for (std::int32_t feature = 0; feature < feature_count; ++feature) {
        if (!std::isfinite(feature_mean[feature]) ||
            !std::isfinite(feature_scale[feature]) ||
            feature_scale[feature] <= 0.0f) {
            return 2;
        }
    }
    for (std::int32_t action = 0; action < action_count; ++action) {
        if (action_offsets[action] < 0 ||
            action_offsets[action] >= action_offsets[action + 1] ||
            action_offsets[action + 1] > prototype_count) {
            return 3;
        }
    }
    for (std::int32_t row = 0; row < row_count; ++row) {
        const std::int32_t action = row_actions[row];
        if (action < 0 || action >= action_count) return 4;
        float best = INFINITY;
        for (std::int32_t prototype = action_offsets[action];
             prototype < action_offsets[action + 1]; ++prototype) {
            float sum = 0.0f;
            for (std::int32_t feature = 0; feature < feature_count; ++feature) {
                const float value = features[row * feature_count + feature];
                if (!std::isfinite(value)) return 5;
                const float normalized =
                    (value - feature_mean[feature]) / feature_scale[feature];
                const float delta = normalized - standardized_prototypes[
                    prototype * feature_count + feature];
                sum += delta * delta;
            }
            best = std::fmin(best, sum / static_cast<float>(feature_count));
        }
        outputs[row] = best;
    }
    return 0;
}

extern "C" TH06_RL_RANKER_API int th06_rl_score_iql_actor_population_v1(
    const float* states,
    const float* actions,
    const std::int32_t row_count,
    const std::int32_t state_count,
    const std::int32_t action_count,
    const std::int32_t model_count,
    const std::int32_t hidden_count,
    const std::int32_t rank_count,
    const float* state_hidden_weight,
    const float* state_hidden_bias,
    const float* state_latent_weight,
    const float* state_latent_bias,
    const float* action_hidden_weight,
    const float* action_hidden_bias,
    const float* action_latent_weight,
    const float* action_latent_bias,
    const float* action_score_weight,
    const float* action_score_bias,
    float* outputs) {
    if (states == nullptr || actions == nullptr ||
        state_hidden_weight == nullptr || state_hidden_bias == nullptr ||
        state_latent_weight == nullptr || state_latent_bias == nullptr ||
        action_hidden_weight == nullptr || action_hidden_bias == nullptr ||
        action_latent_weight == nullptr || action_latent_bias == nullptr ||
        action_score_weight == nullptr || action_score_bias == nullptr ||
        outputs == nullptr || row_count <= 0 || state_count <= 0 ||
        action_count <= 0 || model_count <= 0 || hidden_count <= 0 ||
        rank_count <= 0) {
        return 1;
    }
    const float rank_scale = 1.0f / std::sqrt(static_cast<float>(rank_count));
    for (std::int32_t model = 0; model < model_count; ++model) {
        const std::int64_t shw = static_cast<std::int64_t>(model) *
            state_count * hidden_count;
        const std::int64_t shb = static_cast<std::int64_t>(model) * hidden_count;
        const std::int64_t slw = static_cast<std::int64_t>(model) *
            hidden_count * rank_count;
        const std::int64_t slb = static_cast<std::int64_t>(model) * rank_count;
        const std::int64_t ahw = static_cast<std::int64_t>(model) *
            action_count * hidden_count;
        const std::int64_t ahb = static_cast<std::int64_t>(model) * hidden_count;
        const std::int64_t alw = static_cast<std::int64_t>(model) *
            hidden_count * rank_count;
        const std::int64_t alb = static_cast<std::int64_t>(model) * rank_count;
        float state_hidden[256];
        float state_latent[128];
        if (hidden_count > 256 || rank_count > 128) return 2;
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            state_hidden[hidden] = state_hidden_bias[shb + hidden];
        }
        // Weights are feature-major. Traverse each contiguous hidden row once
        // while retaining every output's original feature accumulation order.
        for (std::int32_t feature = 0; feature < state_count; ++feature) {
            const float input = states[feature];
            const float* weights = state_hidden_weight + shw +
                static_cast<std::int64_t>(feature) * hidden_count;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                state_hidden[hidden] += input * weights[hidden];
            }
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            state_hidden[hidden] = std::tanh(state_hidden[hidden]);
        }
        for (std::int32_t rank = 0; rank < rank_count; ++rank) {
            state_latent[rank] = state_latent_bias[slb + rank];
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            const float input = state_hidden[hidden];
            const float* weights = state_latent_weight + slw +
                static_cast<std::int64_t>(hidden) * rank_count;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                state_latent[rank] += input * weights[rank];
            }
        }
        for (std::int32_t row = 0; row < row_count; ++row) {
            float action_hidden[256];
            float score = action_score_bias[model];
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                action_hidden[hidden] = action_hidden_bias[ahb + hidden];
            }
            for (std::int32_t feature = 0; feature < action_count; ++feature) {
                const float input = actions[row * action_count + feature];
                const float* weights = action_hidden_weight + ahw +
                    static_cast<std::int64_t>(feature) * hidden_count;
                for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                    action_hidden[hidden] += input * weights[hidden];
                }
            }
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                action_hidden[hidden] = std::tanh(action_hidden[hidden]);
                score += action_hidden[hidden] * action_score_weight[
                    ahb + hidden];
            }
            float action_latent[128];
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                action_latent[rank] = action_latent_bias[alb + rank];
            }
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                const float input = action_hidden[hidden];
                const float* weights = action_latent_weight + alw +
                    static_cast<std::int64_t>(hidden) * rank_count;
                for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                    action_latent[rank] += input * weights[rank];
                }
            }
            float dot = 0.0f;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                dot += action_latent[rank] * state_latent[rank];
            }
            outputs[model * row_count + row] = score + dot * rank_scale;
        }
    }
    return 0;
}

extern "C" TH06_RL_RANKER_API int
th06_rl_score_centered_iql_actor_population_v1(
    const float* states,
    const float* actions,
    const std::int32_t row_count,
    const std::int32_t state_count,
    const std::int32_t action_count,
    const std::int32_t baseline_row,
    const std::int32_t model_count,
    const std::int32_t hidden_count,
    const std::int32_t rank_count,
    const float* state_hidden_weight,
    const float* state_hidden_bias,
    const float* state_latent_weight,
    const float* state_latent_bias,
    const float* action_hidden_weight,
    const float* action_hidden_bias,
    const float* action_latent_weight,
    const float* action_latent_bias,
    const float* action_score_weight,
    const float* action_score_bias,
    float* outputs) {
    if (states == nullptr || actions == nullptr ||
        state_hidden_weight == nullptr || state_hidden_bias == nullptr ||
        state_latent_weight == nullptr || state_latent_bias == nullptr ||
        action_hidden_weight == nullptr || action_hidden_bias == nullptr ||
        action_latent_weight == nullptr || action_latent_bias == nullptr ||
        action_score_weight == nullptr || action_score_bias == nullptr ||
        outputs == nullptr || row_count <= 0 || state_count <= 0 ||
        action_count <= 0 || baseline_row < 0 || baseline_row >= row_count ||
        model_count <= 0 || hidden_count <= 0 || hidden_count > 256 ||
        rank_count <= 0 || rank_count > 128) {
        return 1;
    }
    const float rank_scale = 1.0f / std::sqrt(static_cast<float>(rank_count));
    for (std::int32_t model = 0; model < model_count; ++model) {
        const std::int64_t shw = static_cast<std::int64_t>(model) *
            state_count * hidden_count;
        const std::int64_t shb = static_cast<std::int64_t>(model) * hidden_count;
        const std::int64_t slw = static_cast<std::int64_t>(model) *
            hidden_count * rank_count;
        const std::int64_t slb = static_cast<std::int64_t>(model) * rank_count;
        const std::int64_t ahw = static_cast<std::int64_t>(model) *
            action_count * hidden_count;
        const std::int64_t ahb = static_cast<std::int64_t>(model) * hidden_count;
        const std::int64_t alw = static_cast<std::int64_t>(model) *
            hidden_count * rank_count;
        const std::int64_t alb = static_cast<std::int64_t>(model) * rank_count;
        float state_hidden[256];
        float state_latent[128];
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            state_hidden[hidden] = state_hidden_bias[shb + hidden];
        }
        for (std::int32_t feature = 0; feature < state_count; ++feature) {
            const float input = states[feature];
            const float* weights = state_hidden_weight + shw +
                static_cast<std::int64_t>(feature) * hidden_count;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                state_hidden[hidden] += input * weights[hidden];
            }
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            state_hidden[hidden] = std::tanh(state_hidden[hidden]);
        }
        for (std::int32_t rank = 0; rank < rank_count; ++rank) {
            state_latent[rank] = state_latent_bias[slb + rank];
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            const float input = state_hidden[hidden];
            const float* weights = state_latent_weight + slw +
                static_cast<std::int64_t>(hidden) * rank_count;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                state_latent[rank] += input * weights[rank];
            }
        }

        float baseline_hidden[256];
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            baseline_hidden[hidden] = action_hidden_bias[ahb + hidden];
        }
        const float* baseline_action = actions + baseline_row * action_count;
        for (std::int32_t feature = 0; feature < action_count; ++feature) {
            const float input = baseline_action[feature];
            const float* weights = action_hidden_weight + ahw +
                static_cast<std::int64_t>(feature) * hidden_count;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                baseline_hidden[hidden] += input * weights[hidden];
            }
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            baseline_hidden[hidden] = std::tanh(baseline_hidden[hidden]);
        }
        float baseline_latent[128];
        for (std::int32_t rank = 0; rank < rank_count; ++rank) {
            baseline_latent[rank] = action_latent_bias[alb + rank];
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            const float input = baseline_hidden[hidden];
            const float* weights = action_latent_weight + alw +
                static_cast<std::int64_t>(hidden) * rank_count;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                baseline_latent[rank] += input * weights[rank];
            }
        }

        for (std::int32_t row = 0; row < row_count; ++row) {
            if (row == baseline_row) {
                outputs[model * row_count + row] = 0.0f;
                continue;
            }
            float action_hidden[256];
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                action_hidden[hidden] = action_hidden_bias[ahb + hidden];
            }
            for (std::int32_t feature = 0; feature < action_count; ++feature) {
                const float input = actions[row * action_count + feature];
                const float* weights = action_hidden_weight + ahw +
                    static_cast<std::int64_t>(feature) * hidden_count;
                for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                    action_hidden[hidden] += input * weights[hidden];
                }
            }
            float score = 0.0f;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                action_hidden[hidden] = std::tanh(action_hidden[hidden]);
                score += (action_hidden[hidden] - baseline_hidden[hidden]) *
                    action_score_weight[ahb + hidden];
            }
            float action_latent[128];
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                action_latent[rank] = action_latent_bias[alb + rank];
            }
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                const float input = action_hidden[hidden];
                const float* weights = action_latent_weight + alw +
                    static_cast<std::int64_t>(hidden) * rank_count;
                for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                    action_latent[rank] += input * weights[rank];
                }
            }
            float dot = 0.0f;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                dot += (action_latent[rank] - baseline_latent[rank]) *
                    state_latent[rank];
            }
            outputs[model * row_count + row] = score + dot * rank_scale;
        }
    }
    return 0;
}

extern "C" TH06_RL_RANKER_API int
th06_rl_score_centered_iql_actor_population_f64_v1(
    const float* states,
    const float* actions,
    const std::int32_t row_count,
    const std::int32_t state_count,
    const std::int32_t action_count,
    const std::int32_t baseline_row,
    const std::int32_t model_count,
    const std::int32_t hidden_count,
    const std::int32_t rank_count,
    const float* state_hidden_weight,
    const float* state_hidden_bias,
    const float* state_latent_weight,
    const float* state_latent_bias,
    const float* action_hidden_weight,
    const float* action_hidden_bias,
    const float* action_latent_weight,
    const float* action_latent_bias,
    const float* action_score_weight,
    const float* action_score_bias,
    double* outputs) {
    if (states == nullptr || actions == nullptr ||
        state_hidden_weight == nullptr || state_hidden_bias == nullptr ||
        state_latent_weight == nullptr || state_latent_bias == nullptr ||
        action_hidden_weight == nullptr || action_hidden_bias == nullptr ||
        action_latent_weight == nullptr || action_latent_bias == nullptr ||
        action_score_weight == nullptr || action_score_bias == nullptr ||
        outputs == nullptr || row_count <= 0 || state_count <= 0 ||
        action_count <= 0 || baseline_row < 0 || baseline_row >= row_count ||
        model_count <= 0 || hidden_count <= 0 || hidden_count > 256 ||
        rank_count <= 0 || rank_count > 128) {
        return 1;
    }
    const double rank_scale = 1.0 / std::sqrt(static_cast<double>(rank_count));
    for (std::int32_t model = 0; model < model_count; ++model) {
        const std::int64_t shw = static_cast<std::int64_t>(model) *
            state_count * hidden_count;
        const std::int64_t shb = static_cast<std::int64_t>(model) * hidden_count;
        const std::int64_t slw = static_cast<std::int64_t>(model) *
            hidden_count * rank_count;
        const std::int64_t slb = static_cast<std::int64_t>(model) * rank_count;
        const std::int64_t ahw = static_cast<std::int64_t>(model) *
            action_count * hidden_count;
        const std::int64_t ahb = static_cast<std::int64_t>(model) * hidden_count;
        const std::int64_t alw = static_cast<std::int64_t>(model) *
            hidden_count * rank_count;
        const std::int64_t alb = static_cast<std::int64_t>(model) * rank_count;
        double state_hidden[256];
        double state_latent[128];
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            state_hidden[hidden] = static_cast<double>(
                state_hidden_bias[shb + hidden]);
        }
        for (std::int32_t feature = 0; feature < state_count; ++feature) {
            const double input = static_cast<double>(states[feature]);
            const float* weights = state_hidden_weight + shw +
                static_cast<std::int64_t>(feature) * hidden_count;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                state_hidden[hidden] += input * static_cast<double>(weights[hidden]);
            }
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            state_hidden[hidden] = std::tanh(state_hidden[hidden]);
        }
        for (std::int32_t rank = 0; rank < rank_count; ++rank) {
            state_latent[rank] = static_cast<double>(
                state_latent_bias[slb + rank]);
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            const double input = state_hidden[hidden];
            const float* weights = state_latent_weight + slw +
                static_cast<std::int64_t>(hidden) * rank_count;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                state_latent[rank] += input * static_cast<double>(weights[rank]);
            }
        }

        double baseline_hidden[256];
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            baseline_hidden[hidden] = static_cast<double>(
                action_hidden_bias[ahb + hidden]);
        }
        const float* baseline_action = actions + baseline_row * action_count;
        for (std::int32_t feature = 0; feature < action_count; ++feature) {
            const double input = static_cast<double>(baseline_action[feature]);
            const float* weights = action_hidden_weight + ahw +
                static_cast<std::int64_t>(feature) * hidden_count;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                baseline_hidden[hidden] +=
                    input * static_cast<double>(weights[hidden]);
            }
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            baseline_hidden[hidden] = std::tanh(baseline_hidden[hidden]);
        }
        double baseline_latent[128];
        for (std::int32_t rank = 0; rank < rank_count; ++rank) {
            baseline_latent[rank] = static_cast<double>(
                action_latent_bias[alb + rank]);
        }
        for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
            const double input = baseline_hidden[hidden];
            const float* weights = action_latent_weight + alw +
                static_cast<std::int64_t>(hidden) * rank_count;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                baseline_latent[rank] +=
                    input * static_cast<double>(weights[rank]);
            }
        }

        for (std::int32_t row = 0; row < row_count; ++row) {
            if (row == baseline_row) {
                outputs[model * row_count + row] = 0.0;
                continue;
            }
            double action_hidden[256];
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                action_hidden[hidden] = static_cast<double>(
                    action_hidden_bias[ahb + hidden]);
            }
            for (std::int32_t feature = 0; feature < action_count; ++feature) {
                const double input = static_cast<double>(
                    actions[row * action_count + feature]);
                const float* weights = action_hidden_weight + ahw +
                    static_cast<std::int64_t>(feature) * hidden_count;
                for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                    action_hidden[hidden] +=
                        input * static_cast<double>(weights[hidden]);
                }
            }
            double score = 0.0;
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                action_hidden[hidden] = std::tanh(action_hidden[hidden]);
                score += (action_hidden[hidden] - baseline_hidden[hidden]) *
                    static_cast<double>(action_score_weight[ahb + hidden]);
            }
            double action_latent[128];
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                action_latent[rank] = static_cast<double>(
                    action_latent_bias[alb + rank]);
            }
            for (std::int32_t hidden = 0; hidden < hidden_count; ++hidden) {
                const double input = action_hidden[hidden];
                const float* weights = action_latent_weight + alw +
                    static_cast<std::int64_t>(hidden) * rank_count;
                for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                    action_latent[rank] +=
                        input * static_cast<double>(weights[rank]);
                }
            }
            double dot = 0.0;
            for (std::int32_t rank = 0; rank < rank_count; ++rank) {
                dot += (action_latent[rank] - baseline_latent[rank]) *
                    state_latent[rank];
            }
            outputs[model * row_count + row] = score + dot * rank_scale;
        }
    }
    return 0;
}

extern "C" TH06_RL_RANKER_API int th06_rl_encode_hazard_codebook_v1(
    const float* primitives,
    const std::int32_t primitive_count,
    const std::int32_t feature_count,
    const float* feature_mean,
    const float* feature_scale,
    const float* standardized_prototypes,
    const std::int32_t prototype_count,
    float* outputs,
    const std::int32_t output_count) {
    const std::int32_t expected = 2 * prototype_count + 2 * feature_count + 2;
    if ((primitive_count > 0 && primitives == nullptr) ||
        feature_mean == nullptr || feature_scale == nullptr ||
        standardized_prototypes == nullptr || outputs == nullptr ||
        primitive_count < 0 || primitive_count > 256 || feature_count <= 0 ||
        prototype_count <= 0 || output_count != expected) {
        return 1;
    }
    for (std::int32_t feature = 0; feature < feature_count; ++feature) {
        if (!std::isfinite(feature_mean[feature]) ||
            !std::isfinite(feature_scale[feature]) ||
            feature_scale[feature] <= 0.0f) {
            return 2;
        }
    }
    for (std::int32_t index = 0; index < output_count; ++index) {
        outputs[index] = 0.0f;
    }
    if (primitive_count == 0) {
        outputs[output_count - 1] = 1.0f;
        return 0;
    }
    const std::int32_t min_offset = prototype_count;
    const std::int32_t mean_offset = 2 * prototype_count;
    const std::int32_t max_offset = mean_offset + feature_count;
    for (std::int32_t prototype = 0; prototype < prototype_count; ++prototype) {
        outputs[min_offset + prototype] = INFINITY;
    }
    for (std::int32_t primitive = 0; primitive < primitive_count; ++primitive) {
        std::int32_t assignment = -1;
        float assignment_distance = INFINITY;
        for (std::int32_t prototype = 0; prototype < prototype_count; ++prototype) {
            float distance = 0.0f;
            for (std::int32_t feature = 0; feature < feature_count; ++feature) {
                const float value = primitives[primitive * feature_count + feature];
                if (!std::isfinite(value)) return 3;
                const float normalized =
                    (value - feature_mean[feature]) / feature_scale[feature];
                const float delta = normalized - standardized_prototypes[
                    prototype * feature_count + feature];
                distance += delta * delta;
            }
            distance /= static_cast<float>(feature_count);
            outputs[min_offset + prototype] = std::fmin(
                outputs[min_offset + prototype], distance);
            if (distance < assignment_distance) {
                assignment_distance = distance;
                assignment = prototype;
            }
        }
        if (assignment < 0) return 4;
        outputs[assignment] += 1.0f;
        for (std::int32_t feature = 0; feature < feature_count; ++feature) {
            const float normalized =
                (primitives[primitive * feature_count + feature] -
                 feature_mean[feature]) / feature_scale[feature];
            outputs[mean_offset + feature] += normalized;
            outputs[max_offset + feature] = std::fmax(
                outputs[max_offset + feature], std::fabs(normalized));
        }
    }
    const float count = static_cast<float>(primitive_count);
    for (std::int32_t prototype = 0; prototype < prototype_count; ++prototype) {
        outputs[prototype] /= count;
    }
    for (std::int32_t feature = 0; feature < feature_count; ++feature) {
        outputs[mean_offset + feature] /= count;
    }
    outputs[output_count - 2] = std::log1p(count);
    return 0;
}

extern "C" TH06_RL_RANKER_API int th06_rl_score_supported_iql_actor_v1(
    const float* features,
    const std::int32_t row_count,
    const std::int32_t feature_count,
    const std::int32_t* row_actions,
    const std::int32_t baseline_row,
    const float* support_mean,
    const float* support_scale,
    const float* support_prototypes,
    const std::int32_t support_prototype_count,
    const std::int32_t* support_action_offsets,
    const std::int32_t support_action_count,
    const std::int32_t* state_indices,
    const std::int32_t state_count,
    const std::int32_t* actor_action_indices,
    const std::int32_t actor_action_count,
    const float* state_mean,
    const float* state_scale,
    const float* action_mean,
    const float* action_scale,
    const std::int32_t model_count,
    const std::int32_t hidden_count,
    const std::int32_t rank_count,
    const float* state_hidden_weight,
    const float* state_hidden_bias,
    const float* state_latent_weight,
    const float* state_latent_bias,
    const float* action_hidden_weight,
    const float* action_hidden_bias,
    const float* action_latent_weight,
    const float* action_latent_bias,
    const float* action_score_weight,
    const float* action_score_bias,
    float* support_outputs,
    double* actor_outputs) {
    if (features == nullptr || row_actions == nullptr ||
        state_indices == nullptr || actor_action_indices == nullptr ||
        state_mean == nullptr || state_scale == nullptr ||
        action_mean == nullptr || action_scale == nullptr ||
        support_outputs == nullptr || actor_outputs == nullptr ||
        row_count <= 0 || row_count > 64 || feature_count <= 0 ||
        baseline_row < 0 || baseline_row >= row_count ||
        state_count <= 0 || state_count > 256 ||
        actor_action_count <= 0 || actor_action_count > 128) {
        return 10;
    }
    const int support_status = th06_rl_min_support_distance_v1(
        features, row_count, feature_count, support_mean, support_scale,
        support_prototypes, support_prototype_count, support_action_offsets,
        support_action_count, row_actions, support_outputs);
    if (support_status != 0) return 20 + support_status;

    float state[256];
    float actions[64 * 128];
    for (std::int32_t feature = 0; feature < state_count; ++feature) {
        const std::int32_t source = state_indices[feature];
        if (source < 0 || source >= feature_count ||
            !std::isfinite(state_mean[feature]) ||
            !std::isfinite(state_scale[feature]) ||
            state_scale[feature] <= 0.0f) {
            return 11;
        }
        state[feature] = (
            features[source] - state_mean[feature]
        ) / state_scale[feature];
    }
    for (std::int32_t feature = 0; feature < actor_action_count; ++feature) {
        if (actor_action_indices[feature] < 0 ||
            actor_action_indices[feature] >= feature_count ||
            !std::isfinite(action_mean[feature]) ||
            !std::isfinite(action_scale[feature]) ||
            action_scale[feature] <= 0.0f) {
            return 12;
        }
    }
    for (std::int32_t row = 0; row < row_count; ++row) {
        for (std::int32_t feature = 0;
             feature < actor_action_count; ++feature) {
            actions[row * actor_action_count + feature] = (
                features[row * feature_count + actor_action_indices[feature]] -
                action_mean[feature]
            ) / action_scale[feature];
        }
    }
    const int actor_status = th06_rl_score_centered_iql_actor_population_f64_v1(
        state, actions, row_count, state_count, actor_action_count,
        baseline_row, model_count, hidden_count, rank_count, state_hidden_weight,
        state_hidden_bias, state_latent_weight, state_latent_bias,
        action_hidden_weight, action_hidden_bias, action_latent_weight,
        action_latent_bias, action_score_weight, action_score_bias,
        actor_outputs);
    return actor_status == 0 ? 0 : 40 + actor_status;
}

extern "C" TH06_RL_RANKER_API int th06_rl_evaluate_iql_policy_v1(
    const float* observation,
    const std::int32_t observation_count,
    const float* action_features,
    const std::int32_t row_count,
    const std::int32_t action_feature_count,
    const std::int32_t baseline_row,
    const std::int32_t current_row,
    const float* hazard_primitives,
    const std::int32_t hazard_primitive_count,
    const std::int32_t hazard_feature_count,
    const float* hazard_mean,
    const float* hazard_scale,
    const float* hazard_prototypes,
    const std::int32_t hazard_prototype_count,
    const std::int32_t hazard_output_count,
    const float* history,
    const std::int32_t history_count,
    const std::int32_t* row_actions,
    const std::int32_t* row_supported,
    const std::int32_t* row_tie_break_ranks,
    const double support_threshold,
    const float* support_mean,
    const float* support_scale,
    const float* support_prototypes,
    const std::int32_t support_prototype_count,
    const std::int32_t* support_action_offsets,
    const std::int32_t support_action_count,
    const std::int32_t* state_indices,
    const std::int32_t state_count,
    const std::int32_t* actor_action_indices,
    const std::int32_t actor_action_count,
    const float* state_mean,
    const float* state_scale,
    const float* action_mean,
    const float* action_scale,
    const std::int32_t model_count,
    const std::int32_t hidden_count,
    const std::int32_t rank_count,
    const float* state_hidden_weight,
    const float* state_hidden_bias,
    const float* state_latent_weight,
    const float* state_latent_bias,
    const float* action_hidden_weight,
    const float* action_hidden_bias,
    const float* action_latent_weight,
    const float* action_latent_bias,
    const float* action_score_weight,
    const float* action_score_bias,
    std::int32_t* proposal_row,
    std::int32_t* supported_alternative_count) {
    const std::int32_t feature_count = observation_count +
        2 * action_feature_count + 2 + hazard_output_count + history_count;
    if (observation == nullptr || action_features == nullptr ||
        history == nullptr || row_actions == nullptr ||
        row_supported == nullptr || row_tie_break_ranks == nullptr ||
        proposal_row == nullptr || supported_alternative_count == nullptr ||
        !std::isfinite(support_threshold) || support_threshold < 0.0 ||
        row_count <= 0 || row_count > 64 || model_count <= 0 ||
        model_count > 16 ||
        observation_count <= 0 || action_feature_count <= 0 ||
        baseline_row < 0 || baseline_row >= row_count || current_row < -1 ||
        current_row >= row_count || hazard_output_count <= 0 ||
        history_count <= 0 || feature_count <= 0 || feature_count > 512) {
        return 50;
    }
    float hazard[256];
    if (hazard_output_count > 256) return 51;
    const int hazard_status = th06_rl_encode_hazard_codebook_v1(
        hazard_primitives, hazard_primitive_count, hazard_feature_count,
        hazard_mean, hazard_scale, hazard_prototypes, hazard_prototype_count,
        hazard, hazard_output_count);
    if (hazard_status != 0) return 60 + hazard_status;

    float rows[64 * 512];
    const float* baseline = action_features +
        baseline_row * action_feature_count;
    for (std::int32_t row = 0; row < row_count; ++row) {
        float* output = rows + row * feature_count;
        std::int32_t offset = 0;
        for (std::int32_t index = 0; index < observation_count; ++index) {
            output[offset++] = observation[index];
        }
        const float* selected = action_features + row * action_feature_count;
        for (std::int32_t index = 0; index < action_feature_count; ++index) {
            output[offset++] = selected[index];
        }
        for (std::int32_t index = 0; index < action_feature_count; ++index) {
            output[offset++] = selected[index] - baseline[index];
        }
        output[offset++] = row == baseline_row ? 1.0f : 0.0f;
        output[offset++] = row == current_row ? 1.0f : 0.0f;
        for (std::int32_t index = 0; index < hazard_output_count; ++index) {
            output[offset++] = hazard[index];
        }
        for (std::int32_t index = 0; index < history_count; ++index) {
            output[offset++] = history[index];
        }
        if (offset != feature_count) return 52;
    }
    float support_outputs[64];
    double actor_outputs[64 * 16];
    const int status = th06_rl_score_supported_iql_actor_v1(
        rows, row_count, feature_count, row_actions, baseline_row,
        support_mean, support_scale, support_prototypes,
        support_prototype_count, support_action_offsets,
        support_action_count, state_indices, state_count,
        actor_action_indices, actor_action_count, state_mean, state_scale,
        action_mean, action_scale, model_count, hidden_count, rank_count,
        state_hidden_weight, state_hidden_bias, state_latent_weight,
        state_latent_bias, action_hidden_weight, action_hidden_bias,
        action_latent_weight, action_latent_bias, action_score_weight,
        action_score_bias, support_outputs, actor_outputs);
    if (status != 0) return 80 + status;

    std::int32_t selected = baseline_row;
    std::int32_t selected_rank = row_tie_break_ranks[baseline_row];
    std::int32_t supported_count = 0;
    double best_advantage = 0.0;
    for (std::int32_t row = 0; row < row_count; ++row) {
        if (row == baseline_row || row_supported[row] == 0 ||
            static_cast<double>(support_outputs[row]) > support_threshold) {
            continue;
        }
        ++supported_count;
        double mean = 0.0;
        for (std::int32_t model = 0; model < model_count; ++model) {
            mean += actor_outputs[model * row_count + row];
        }
        mean /= static_cast<double>(model_count);
        const double advantage = mean;
        if (advantage > best_advantage ||
            (advantage == best_advantage && advantage > 0.0 &&
             row_tie_break_ranks[row] > selected_rank)) {
            best_advantage = advantage;
            selected = row;
            selected_rank = row_tie_break_ranks[row];
        }
    }
    *proposal_row = selected;
    *supported_alternative_count = supported_count;
    return 0;
}
