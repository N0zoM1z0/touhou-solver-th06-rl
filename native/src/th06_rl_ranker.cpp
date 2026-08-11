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
