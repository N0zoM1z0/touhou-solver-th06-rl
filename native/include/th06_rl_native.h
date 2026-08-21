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

}
