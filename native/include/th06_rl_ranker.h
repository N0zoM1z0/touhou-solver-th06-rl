#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(TH06_RL_RANKER_BUILD)
#    define TH06_RL_RANKER_API __declspec(dllexport)
#  else
#    define TH06_RL_RANKER_API __declspec(dllimport)
#  endif
#else
#  define TH06_RL_RANKER_API __attribute__((visibility("default")))
#endif

extern "C" {

struct Th06RlTreeNode {
    std::int32_t feature;
    float threshold;
    std::int32_t left;
    std::int32_t right;
    std::int32_t missing;
    float leaf;
};

// Batch-score already encoded candidate rows. This library ranks no actions
// and has no game/collision access; the native gate remains authoritative.
TH06_RL_RANKER_API int th06_rl_score_xgboost_v1(
    const Th06RlTreeNode* nodes,
    std::int32_t node_count,
    const std::int32_t* tree_offsets,
    std::int32_t tree_count,
    const float* features,
    std::int32_t row_count,
    std::int32_t feature_count,
    float base_score,
    float* outputs);

}
