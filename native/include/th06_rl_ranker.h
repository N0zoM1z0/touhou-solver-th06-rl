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

// Score a complete immutable model population over one shared candidate
// matrix. Tree offsets address the concatenated node array; model-tree
// offsets partition that tree array. Outputs are model-major. This preserves
// every tree/member while avoiding repeated feature marshaling across the FFI.
TH06_RL_RANKER_API int th06_rl_score_xgboost_population_v1(
    const Th06RlTreeNode* nodes,
    std::int32_t node_count,
    const std::int32_t* tree_offsets,
    std::int32_t tree_count,
    const std::int32_t* model_tree_offsets,
    std::int32_t model_count,
    const float* features,
    std::int32_t row_count,
    std::int32_t feature_count,
    const float* base_scores,
    float* outputs);

// Compute one locally calibrated support distance for each already-encoded
// candidate row. Prototype offsets partition a common standardized prototype
// matrix by stable action index. This function has no game or safety access.
TH06_RL_RANKER_API int th06_rl_min_support_distance_v1(
    const float* features,
    std::int32_t row_count,
    std::int32_t feature_count,
    const float* feature_mean,
    const float* feature_scale,
    const float* standardized_prototypes,
    std::int32_t prototype_count,
    const std::int32_t* action_offsets,
    std::int32_t action_count,
    const std::int32_t* row_actions,
    float* outputs);

// Encode one bounded, permutation-invariant observed-hazard set. Output is
// prototype fractions, prototype minimum distances, normalized feature means,
// normalized max-absolute values, log1p(count), and an empty-set indicator.
TH06_RL_RANKER_API int th06_rl_encode_hazard_codebook_v1(
    const float* primitives,
    std::int32_t primitive_count,
    std::int32_t feature_count,
    const float* feature_mean,
    const float* feature_scale,
    const float* standardized_prototypes,
    std::int32_t prototype_count,
    float* outputs,
    std::int32_t output_count);

// Score a complete population of low-rank listwise actors. All arrays are
// model-major row-major float matrices. State/action inputs are already
// normalized by the immutable portable artifact. Outputs are model-major.
TH06_RL_RANKER_API int th06_rl_score_iql_actor_population_v1(
    const float* states,
    const float* actions,
    std::int32_t row_count,
    std::int32_t state_count,
    std::int32_t action_count,
    std::int32_t model_count,
    std::int32_t hidden_count,
    std::int32_t rank_count,
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
    float* outputs);

}
