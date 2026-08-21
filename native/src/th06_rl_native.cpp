#include "th06_rl_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace {

constexpr std::int32_t kActionCount = 18;
constexpr float kLeft = 8.0f;
constexpr float kRight = 376.0f;
constexpr float kTop = 16.0f;
constexpr float kBottom = 432.0f;

constexpr std::uint16_t kFocus = 0x04;
constexpr std::uint16_t kUp = 0x10;
constexpr std::uint16_t kDown = 0x20;
constexpr std::uint16_t kLeftButton = 0x40;
constexpr std::uint16_t kRightButton = 0x80;

struct Action {
    std::int32_t dx;
    std::int32_t dy;
    bool focused;
};

constexpr std::array<Action, kActionCount> kActions{{
    {0, 0, true}, {0, -1, true}, {0, 1, true}, {-1, 0, true},
    {1, 0, true}, {-1, -1, true}, {1, -1, true}, {-1, 1, true},
    {1, 1, true},
    {0, 0, false}, {0, -1, false}, {0, 1, false}, {-1, 0, false},
    {1, 0, false}, {-1, -1, false}, {1, -1, false}, {-1, 1, false},
    {1, 1, false},
}};

struct Kinematics {
    float normal_speed;
    float focus_speed;
    float normal_diagonal_speed;
    float focus_diagonal_speed;
};

struct Position {
    float x;
    float y;
};

struct LaserBasis {
    float sine;
    float cosine;
};

struct HazardView {
    std::int32_t horizon;
    const std::uint32_t* aabb_offsets;
    const Th06RlAabb* aabbs;
    const std::uint32_t* laser_offsets;
    const Th06RlLaserRect* lasers;
    const LaserBasis* laser_bases;
};

struct HazardSample {
    std::int32_t collisions;
    float clearance;
};

struct HazardAccumulator {
    std::int32_t collisions{0};
    float overlap_clearance{std::numeric_limits<float>::infinity()};
    float minimum_distance_squared{std::numeric_limits<float>::infinity()};
    float minimum_gap_x{0.0f};
    float minimum_gap_y{0.0f};
};

bool valid_action(std::int32_t action) {
    return 0 <= action && action < kActionCount;
}

std::int32_t direction_index(std::int32_t dx, std::int32_t dy) {
    if (dx == 0 && dy == 0) return 0;
    if (dx == 0 && dy == -1) return 1;
    if (dx == 0 && dy == 1) return 2;
    if (dx == -1 && dy == 0) return 3;
    if (dx == 1 && dy == 0) return 4;
    if (dx == -1 && dy == -1) return 5;
    if (dx == 1 && dy == -1) return 6;
    if (dx == -1 && dy == 1) return 7;
    return 8;
}

std::uint16_t action_mask(std::int32_t index) {
    const auto action = kActions[static_cast<std::size_t>(index)];
    std::uint16_t mask = action.focused ? kFocus : 0;
    if (action.dx < 0) mask |= kLeftButton;
    if (action.dx > 0) mask |= kRightButton;
    if (action.dy < 0) mask |= kUp;
    if (action.dy > 0) mask |= kDown;
    return mask;
}

std::int32_t action_from_mask(std::uint16_t mask) {
    std::int32_t dx = 0;
    std::int32_t dy = 0;
    if ((mask & kUp) != 0) {
        dy = -1;
        if ((mask & kLeftButton) != 0) dx = -1;
        if ((mask & kRightButton) != 0) dx = 1;
    } else if ((mask & kDown) != 0) {
        dy = 1;
        if ((mask & kLeftButton) != 0) dx = -1;
        if ((mask & kRightButton) != 0) dx = 1;
    } else {
        if ((mask & kLeftButton) != 0) dx = -1;
        if ((mask & kRightButton) != 0) dx = 1;
    }
    return direction_index(dx, dy) + (((mask & kFocus) != 0) ? 0 : 9);
}

std::vector<std::int32_t> transition_prefix_actions(
    std::int32_t current,
    std::int32_t target) {
    const auto current_mask = action_mask(current);
    const auto target_mask = action_mask(target);
    auto prefix_mask = current_mask;
    std::vector<std::int32_t> result;
    // Keyboard::_sync sorts these key names for releases and then presses.
    constexpr std::array<std::uint16_t, 5> event_order{{
        kDown, kFocus, kLeftButton, kRightButton, kUp,
    }};
    for (const bool pressing : {false, true}) {
        for (const auto bit : event_order) {
            const bool in_current = (current_mask & bit) != 0;
            const bool in_target = (target_mask & bit) != 0;
            if ((!pressing && in_current && !in_target)
                || (pressing && !in_current && in_target)) {
                prefix_mask ^= bit;
                const auto prefix = action_from_mask(prefix_mask);
                if (prefix != current && prefix != target
                    && std::find(result.begin(), result.end(), prefix)
                        == result.end()) {
                    result.push_back(prefix);
                }
            }
        }
    }
    return result;
}

Position advance(
    Position position,
    std::int32_t action_index,
    const Kinematics& kinematics) {
    const auto action = kActions[static_cast<std::size_t>(action_index)];
    const bool diagonal = action.dx != 0 && action.dy != 0;
    float speed = 0.0f;
    if (action.focused) {
        speed = diagonal
            ? kinematics.focus_diagonal_speed
            : kinematics.focus_speed;
    } else {
        speed = diagonal
            ? kinematics.normal_diagonal_speed
            : kinematics.normal_speed;
    }
    position.x = std::clamp(
        position.x + static_cast<float>(action.dx) * speed,
        kLeft,
        kRight);
    position.y = std::clamp(
        position.y + static_cast<float>(action.dy) * speed,
        kTop,
        kBottom);
    return position;
}

void accumulate_aabb(
    float player_x,
    float player_y,
    float player_half_width,
    float player_half_height,
    const Th06RlAabb& hazard,
    float collision_margin,
    HazardAccumulator& result) {
    const float gap_x = std::max(
        hazard.left - (player_x + player_half_width),
        (player_x - player_half_width) - hazard.right);
    const float gap_y = std::max(
        hazard.top - (player_y + player_half_height),
        (player_y - player_half_height) - hazard.bottom);
    if (gap_x <= 0.0f && gap_y <= 0.0f) {
        const float clearance = std::max(gap_x, gap_y);
        result.overlap_clearance = std::min(
            result.overlap_clearance,
            clearance);
        result.collisions += static_cast<std::int32_t>(
            clearance <= collision_margin);
        return;
    }
    const float positive_x = std::max(gap_x, 0.0f);
    const float positive_y = std::max(gap_y, 0.0f);
    const float distance_squared = positive_x * positive_x
        + positive_y * positive_y;
    if (distance_squared < result.minimum_distance_squared) {
        result.minimum_distance_squared = distance_squared;
        result.minimum_gap_x = positive_x;
        result.minimum_gap_y = positive_y;
    }
    // max(x, y) is a cheap lower bound on hypot(x, y).  The exact
    // comparison is still performed for every possible collision.
    if (collision_margin >= 0.0f
        && std::max(positive_x, positive_y) <= collision_margin) {
        result.collisions += static_cast<std::int32_t>(
            std::hypot(positive_x, positive_y) <= collision_margin);
    }
}

void accumulate_laser(
    float player_x,
    float player_y,
    float player_half_width,
    float player_half_height,
    const Th06RlLaserRect& laser,
    const LaserBasis& basis,
    float collision_margin,
    HazardAccumulator& result) {
    const float dx = player_x - laser.origin_x;
    const float dy = player_y - laser.origin_y;
    const float local_x = basis.cosine * dx + basis.sine * dy;
    const float local_y = basis.cosine * dy - basis.sine * dx;
    const Th06RlAabb local{
        laser.center_offset - laser.size_x / 2.0f,
        -laser.size_y / 2.0f,
        laser.center_offset + laser.size_x / 2.0f,
        laser.size_y / 2.0f,
    };
    accumulate_aabb(
        local_x,
        local_y,
        player_half_width,
        player_half_height,
        local,
        collision_margin,
        result);
}

HazardSample sample_hazards(
    const HazardView& hazards,
    std::int32_t frame,
    Position position,
    float player_half_width,
    float player_half_height,
    float collision_margin) {
    HazardAccumulator result;
    const auto frame_index = static_cast<std::size_t>(frame - 1);
    const auto aabb_start = hazards.aabb_offsets[frame_index];
    const auto aabb_end = hazards.aabb_offsets[frame_index + 1];
    for (auto index = aabb_start; index < aabb_end; ++index) {
        accumulate_aabb(
            position.x,
            position.y,
            player_half_width,
            player_half_height,
            hazards.aabbs[index],
            collision_margin,
            result);
    }
    const auto laser_start = hazards.laser_offsets[frame_index];
    const auto laser_end = hazards.laser_offsets[frame_index + 1];
    for (auto index = laser_start; index < laser_end; ++index) {
        accumulate_laser(
            position.x,
            position.y,
            player_half_width,
            player_half_height,
            hazards.lasers[index],
            hazards.laser_bases[index],
            collision_margin,
            result);
    }
    const float clearance = std::isfinite(result.overlap_clearance)
        ? result.overlap_clearance
        : (
            std::isfinite(result.minimum_distance_squared)
                ? std::hypot(result.minimum_gap_x, result.minimum_gap_y)
                : std::numeric_limits<float>::infinity()
        );
    return HazardSample{result.collisions, clearance};
}

std::vector<LaserBasis> prepare_laser_bases(
    std::int32_t horizon,
    const std::uint32_t* laser_offsets,
    const Th06RlLaserRect* lasers) {
    const auto count = laser_offsets[static_cast<std::size_t>(horizon)];
    std::vector<LaserBasis> bases;
    bases.reserve(count);
    for (std::uint32_t index = 0; index < count; ++index) {
        bases.push_back(LaserBasis{
            std::sin(lasers[index].angle),
            std::cos(lasers[index].angle),
        });
    }
    return bases;
}

bool valid_common_inputs(
    float player_half_width,
    float player_half_height,
    const Kinematics& kinematics,
    std::int32_t current_action,
    std::int32_t horizon,
    const std::uint32_t* aabb_offsets,
    const std::uint32_t* laser_offsets) {
    return player_half_width >= 0.0f
        && player_half_height >= 0.0f
        && kinematics.normal_speed > 0.0f
        && kinematics.focus_speed > 0.0f
        && kinematics.normal_diagonal_speed > 0.0f
        && kinematics.focus_diagonal_speed > 0.0f
        && valid_action(current_action)
        && horizon > 0
        && aabb_offsets != nullptr
        && laser_offsets != nullptr;
}

int certify_actions_impl(
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
    float* action_final_xy) {
    const Kinematics kinematics{
        normal_speed,
        focus_speed,
        normal_diagonal_speed,
        focus_diagonal_speed,
    };
    if (!valid_common_inputs(
            player_half_width,
            player_half_height,
            kinematics,
            current_action,
            horizon,
            aabb_frame_offsets,
            laser_frame_offsets)
        || delivery_delays == nullptr
        || delivery_delay_count <= 0
        || safe_mask == nullptr
        || action_min_clearance == nullptr
        || action_final_xy == nullptr) {
        return 2;
    }
    const auto laser_bases = prepare_laser_bases(
        horizon,
        laser_frame_offsets,
        lasers);
    const HazardView hazards{
        horizon,
        aabb_frame_offsets,
        aabbs,
        laser_frame_offsets,
        lasers,
        laser_bases.data(),
    };
    *safe_mask = 0;
    for (std::int32_t action = 0; action < kActionCount; ++action) {
        action_min_clearance[action] = -std::numeric_limits<float>::infinity();
        action_final_xy[action * 2] = player_x;
        action_final_xy[action * 2 + 1] = player_y;
        if ((candidate_mask & (1u << action)) == 0) continue;
        bool valid = true;
        float minimum = std::numeric_limits<float>::infinity();
        for (std::int32_t delay_index = 0;
             delay_index < delivery_delay_count && valid;
             ++delay_index) {
            const auto delay = delivery_delays[delay_index];
            if (delay < 0 || delay > horizon) return 2;
            std::vector<std::int32_t> prefixes{-1};
            if (delay > 0) {
                const auto transition = transition_prefix_actions(
                    current_action,
                    action);
                prefixes.insert(
                    prefixes.end(), transition.begin(), transition.end());
            }
            for (const auto prefix : prefixes) {
                Position position{player_x, player_y};
                for (std::int32_t frame = 1; frame <= horizon; ++frame) {
                    std::int32_t applied = action;
                    if (prefix >= 0 && frame == delay) {
                        applied = prefix;
                    } else if (frame < delay || (prefix < 0 && frame <= delay)) {
                        applied = current_action;
                    }
                    position = advance(position, applied, kinematics);
                    const auto common = sample_hazards(
                        hazards,
                        frame,
                        position,
                        player_half_width,
                        player_half_height,
                        collision_margin);
                    minimum = std::min(minimum, common.clearance);
                    if (common.collisions > 0) {
                        valid = false;
                        break;
                    }
                }
                if (prefix < 0
                    && delay_index == delivery_delay_count - 1) {
                    action_final_xy[action * 2] = position.x;
                    action_final_xy[action * 2 + 1] = position.y;
                }
                if (!valid) break;
            }
        }
        action_min_clearance[action] = minimum;
        if (valid) *safe_mask |= 1u << action;
    }
    return 0;
}

}  // namespace

extern "C" int th06_rl_certify_actions_v1(
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
    float* action_final_xy) {
    return certify_actions_impl(
        player_x,
        player_y,
        player_half_width,
        player_half_height,
        normal_speed,
        focus_speed,
        normal_diagonal_speed,
        focus_diagonal_speed,
        current_action,
        horizon,
        delivery_delays,
        delivery_delay_count,
        candidate_mask,
        aabb_frame_offsets,
        aabbs,
        laser_frame_offsets,
        lasers,
        collision_margin,
        safe_mask,
        action_min_clearance,
        action_final_xy);
}
