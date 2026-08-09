#pragma once

#include <cstdint>

#if defined(_WIN32)
#  if defined(TH06_RL_NATIVE_BUILD)
#    define TH06_RL_API __declspec(dllexport)
#  else
#    define TH06_RL_API __declspec(dllimport)
#  endif
#else
#  define TH06_RL_API __attribute__((visibility("default")))
#endif

extern "C" {

struct Th06RlAabb {
    float left;
    float top;
    float right;
    float bottom;
};

// Matches Player::CalcLaserHitbox after rotating the player center into the
// laser's source-local coordinate system.
struct Th06RlLaserRect {
    float origin_x;
    float origin_y;
    float angle;
    float center_offset;
    float size_x;
    float size_y;
};

// Source state for the common final 0x080 player-aim turn. The Hard kernel
// advances these bullets separately for each action/delivery scenario so a
// trajectory aimed at one mutually exclusive candidate cannot reject another.
struct Th06RlPlayerAimedBullet {
    float x;
    float y;
    float vx;
    float vy;
    float half_width;
    float half_height;
    float speed;
    float angle;
    float turn_speed;
    float direction_rotation;
    float timer_float;
    std::int32_t timer;
    std::int32_t direction_interval;
    std::int32_t direction_num_times;
    std::int32_t direction_max_times;
};

struct Th06RlPlanResult {
    std::int32_t selected_action;
    std::int32_t effort_horizon;
    std::uint32_t surviving_first_action_mask;
    float min_clearance;
    float cumulative_risk;
    float terminal_x;
    float terminal_y;
    float terminal_boundary_deficit;
    std::int32_t endpoint_count;
    std::int32_t continuation_action_count;
};

// Action indices are stable: focused stay/up/down/left/right/diagonals 0..8,
// followed by the same nine unfocused actions at 9..17. Bomb is absent.
TH06_RL_API int th06_rl_certify_actions_v1(
    float player_x,
    float player_y,
    float player_half_width,
    float player_half_height,
    float normal_speed,
    float focus_speed,
    float normal_diagonal_speed,
    float focus_diagonal_speed,
    std::int32_t current_action,
    std::int32_t horizon,
    const std::int32_t* delivery_delays,
    std::int32_t delivery_delay_count,
    std::uint32_t candidate_mask,
    const std::uint32_t* aabb_frame_offsets,
    const Th06RlAabb* aabbs,
    const std::uint32_t* laser_frame_offsets,
    const Th06RlLaserRect* lasers,
    float collision_margin,
    std::uint32_t* safe_mask,
    float* action_min_clearance,
    float* action_final_xy);

// Candidate-coupled extension of the same Hard contract. Common AABB/laser
// hazards retain their v1 meaning; player-aim bullets are source-advanced
// against the candidate's exact bounded delivery paths.
TH06_RL_API int th06_rl_certify_actions_aimed_v1(
    float player_x,
    float player_y,
    float player_half_width,
    float player_half_height,
    float normal_speed,
    float focus_speed,
    float normal_diagonal_speed,
    float focus_diagonal_speed,
    std::int32_t current_action,
    std::int32_t horizon,
    const std::int32_t* delivery_delays,
    std::int32_t delivery_delay_count,
    std::uint32_t candidate_mask,
    const std::uint32_t* aabb_frame_offsets,
    const Th06RlAabb* aabbs,
    const std::uint32_t* laser_frame_offsets,
    const Th06RlLaserRect* lasers,
    const Th06RlPlayerAimedBullet* aimed_bullets,
    std::int32_t aimed_bullet_count,
    float collision_margin,
    std::uint32_t* safe_mask,
    float* action_min_clearance,
    float* action_final_xy);

// Feature-only straight-action profiles. This API never certifies an action;
// callers must intersect candidates with th06_rl_certify_actions_v1 first.
TH06_RL_API int th06_rl_profile_actions_v1(
    float player_x,
    float player_y,
    float player_half_width,
    float player_half_height,
    float normal_speed,
    float focus_speed,
    float normal_diagonal_speed,
    float focus_diagonal_speed,
    std::int32_t current_action,
    std::int32_t horizon,
    const std::int32_t* delivery_delays,
    std::int32_t delivery_delay_count,
    std::uint32_t candidate_mask,
    const std::uint32_t* aabb_frame_offsets,
    const Th06RlAabb* aabbs,
    const std::uint32_t* laser_frame_offsets,
    const Th06RlLaserRect* lasers,
    float collision_margin,
    const std::int32_t* checkpoints,
    std::int32_t checkpoint_count,
    float* action_checkpoint_min_clearance);

TH06_RL_API int th06_rl_local_plan_v1(
    float player_x,
    float player_y,
    float player_half_width,
    float player_half_height,
    float normal_speed,
    float focus_speed,
    float normal_diagonal_speed,
    float focus_diagonal_speed,
    std::int32_t current_action,
    std::int32_t horizon,
    std::int32_t control_delay,
    std::int32_t action_hold_frames,
    std::int32_t beam_width,
    float position_quantization,
    float comfort_clearance,
    float boundary_reserve,
    float risk_scale,
    float direction_switch_cost,
    float direction_reverse_cost,
    float focus_switch_cost,
    float collision_margin,
    std::uint32_t hard_first_action_mask,
    std::uint32_t continuation_action_mask,
    const float* hard_min_clearance,
    const std::uint32_t* aabb_frame_offsets,
    const Th06RlAabb* aabbs,
    const std::uint32_t* laser_frame_offsets,
    const Th06RlLaserRect* lasers,
    Th06RlPlanResult* result);

}
