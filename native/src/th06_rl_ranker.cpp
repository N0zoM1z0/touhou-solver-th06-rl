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
