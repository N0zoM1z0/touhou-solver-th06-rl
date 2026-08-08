#include "th06_rl_native.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <set>
#include <tuple>
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

bool opposed(std::int32_t first_index, std::int32_t second_index) {
    const auto first = kActions[static_cast<std::size_t>(first_index)];
    const auto second = kActions[static_cast<std::size_t>(second_index)];
    return (first.dx != 0 || first.dy != 0)
        && first.dx == -second.dx
        && first.dy == -second.dy;
}

float boundary_deficit(Position position, float reserve) {
    return std::max(reserve - (position.x - kLeft), 0.0f)
        + std::max(reserve - (kRight - position.x), 0.0f)
        + std::max(reserve - (position.y - kTop), 0.0f)
        + std::max(reserve - (kBottom - position.y), 0.0f);
}

struct Node {
    Position position;
    std::int32_t first_action;
    std::int32_t last_action;
    float min_clearance;
    float risk;
    std::int32_t direction_switches;
    std::int32_t reversals;
};

using NodeKey = std::tuple<
    float, float, float, std::int32_t, std::int32_t, float,
    std::int32_t, std::int32_t>;

NodeKey node_key(
    const Node& node,
    float comfort_clearance,
    float boundary_reserve) {
    return {
        std::max(comfort_clearance - node.min_clearance, 0.0f),
        boundary_deficit(node.position, boundary_reserve),
        node.risk,
        node.reversals,
        node.direction_switches,
        -node.min_clearance,
        node.first_action,
        node.last_action,
    };
}

struct ActionEvaluation {
    std::int32_t action;
    const Node* best;
    float min_clearance;
    float boundary;
    std::int32_t endpoints;
    std::int32_t continuation_actions;
};

using EvaluationKey = std::tuple<
    float, float, std::int32_t, std::int32_t, float,
    std::int32_t, std::int32_t, float, std::int32_t>;

EvaluationKey evaluation_key(
    const ActionEvaluation& value,
    std::int32_t current_action,
    float comfort_clearance) {
    return {
        std::max(comfort_clearance - value.min_clearance, 0.0f),
        value.boundary,
        -value.continuation_actions,
        -value.endpoints,
        value.best->risk,
        static_cast<std::int32_t>(opposed(value.action, current_action)),
        static_cast<std::int32_t>(value.action != current_action),
        -value.min_clearance,
        value.action,
    };
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
                    const auto sample = sample_hazards(
                        hazards,
                        frame,
                        position,
                        player_half_width,
                        player_half_height,
                        collision_margin);
                    minimum = std::min(minimum, sample.clearance);
                    if (sample.collisions > 0) {
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

extern "C" int th06_rl_profile_actions_v1(
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
    float* action_checkpoint_min_clearance) {
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
        || checkpoints == nullptr
        || checkpoint_count <= 0
        || action_checkpoint_min_clearance == nullptr) {
        return 2;
    }
    for (std::int32_t index = 0; index < checkpoint_count; ++index) {
        if (checkpoints[index] <= 0
            || checkpoints[index] > horizon
            || (index > 0 && checkpoints[index] <= checkpoints[index - 1])) {
            return 2;
        }
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
    for (std::int32_t action = 0; action < kActionCount; ++action) {
        for (std::int32_t checkpoint = 0;
             checkpoint < checkpoint_count;
             ++checkpoint) {
            action_checkpoint_min_clearance[action * checkpoint_count + checkpoint]
                = -std::numeric_limits<float>::infinity();
        }
        if ((candidate_mask & (1u << action)) == 0) continue;
        for (std::int32_t checkpoint = 0;
             checkpoint < checkpoint_count;
             ++checkpoint) {
            action_checkpoint_min_clearance[action * checkpoint_count + checkpoint]
                = std::numeric_limits<float>::infinity();
        }
        for (std::int32_t delay_index = 0;
             delay_index < delivery_delay_count;
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
                float running_minimum = std::numeric_limits<float>::infinity();
                std::int32_t checkpoint_index = 0;
                for (std::int32_t frame = 1; frame <= horizon; ++frame) {
                    std::int32_t applied = action;
                    if (prefix >= 0 && frame == delay) {
                        applied = prefix;
                    } else if (frame < delay || (prefix < 0 && frame <= delay)) {
                        applied = current_action;
                    }
                    position = advance(position, applied, kinematics);
                    const auto sample = sample_hazards(
                        hazards,
                        frame,
                        position,
                        player_half_width,
                        player_half_height,
                        collision_margin);
                    running_minimum = std::min(running_minimum, sample.clearance);
                    if (checkpoint_index < checkpoint_count
                        && frame == checkpoints[checkpoint_index]) {
                        auto& output = action_checkpoint_min_clearance[
                            action * checkpoint_count + checkpoint_index];
                        output = std::min(output, running_minimum);
                        ++checkpoint_index;
                    }
                }
            }
        }
    }
    return 0;
}

extern "C" int th06_rl_local_plan_v1(
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
    Th06RlPlanResult* result) {
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
        || control_delay < 0
        || control_delay >= horizon
        || action_hold_frames <= 0
        || beam_width <= 0
        || position_quantization <= 0.0f
        || comfort_clearance <= 0.0f
        || boundary_reserve <= 0.0f
        || risk_scale <= 0.0f
        || hard_first_action_mask == 0
        || continuation_action_mask == 0
        || hard_min_clearance == nullptr
        || result == nullptr) {
        return 2;
    }
    *result = Th06RlPlanResult{
        -1, 0, 0, -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(), player_x, player_y,
        std::numeric_limits<float>::infinity(), 0, 0,
    };
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
    std::vector<Node> beam;
    for (std::int32_t action = 0; action < kActionCount; ++action) {
        if ((hard_first_action_mask & (1u << action)) == 0) continue;
        beam.push_back(Node{
            {player_x, player_y},
            action,
            current_action,
            hard_min_clearance[action],
            0.0f,
            0,
            0,
        });
    }
    std::vector<Node> last_complete;
    std::int32_t effort_horizon = 0;
    using DedupKey = std::tuple<long, long, std::int32_t, std::int32_t>;
    for (std::int32_t frame = 1; frame <= horizon; ++frame) {
        std::map<DedupKey, Node> deduplicated;
        for (const auto& node : beam) {
            std::vector<std::int32_t> choices;
            if (frame <= control_delay) {
                choices.push_back(current_action);
            } else if (frame == control_delay + 1) {
                choices.push_back(node.first_action);
            } else if ((frame - control_delay - 1) % action_hold_frames == 0) {
                for (std::int32_t action = 0; action < kActionCount; ++action) {
                    if ((continuation_action_mask & (1u << action)) != 0) {
                        choices.push_back(action);
                    }
                }
            } else {
                choices.push_back(node.last_action);
            }
            for (const auto action : choices) {
                const auto position = advance(node.position, action, kinematics);
                const auto sample = sample_hazards(
                    hazards,
                    frame,
                    position,
                    player_half_width,
                    player_half_height,
                    collision_margin);
                if (sample.collisions > 0) continue;
                const bool changed = action != node.last_action;
                const bool reversed = opposed(action, node.last_action);
                float transition_risk = std::isfinite(sample.clearance)
                    ? std::exp(-std::max(sample.clearance, 0.0f) / risk_scale)
                    : 0.0f;
                if (changed) transition_risk += direction_switch_cost;
                if (reversed) transition_risk += direction_reverse_cost;
                if (kActions[static_cast<std::size_t>(action)].focused
                    != kActions[static_cast<std::size_t>(node.last_action)].focused) {
                    transition_risk += focus_switch_cost;
                }
                Node candidate{
                    position,
                    node.first_action,
                    action,
                    std::min(node.min_clearance, sample.clearance),
                    node.risk + transition_risk,
                    node.direction_switches + static_cast<std::int32_t>(changed),
                    node.reversals + static_cast<std::int32_t>(reversed),
                };
                const DedupKey key{
                    std::lround(position.x / position_quantization),
                    std::lround(position.y / position_quantization),
                    candidate.first_action,
                    candidate.last_action,
                };
                const auto retained = deduplicated.find(key);
                if (retained == deduplicated.end()
                    || node_key(candidate, comfort_clearance, boundary_reserve)
                        < node_key(retained->second, comfort_clearance, boundary_reserve)) {
                    deduplicated.insert_or_assign(key, candidate);
                }
            }
        }
        if (deduplicated.empty()) break;
        std::vector<Node> expanded;
        expanded.reserve(deduplicated.size());
        for (const auto& [key, node] : deduplicated) {
            static_cast<void>(key);
            expanded.push_back(node);
        }
        std::sort(expanded.begin(), expanded.end(), [&](const Node& left, const Node& right) {
            return node_key(left, comfort_clearance, boundary_reserve)
                < node_key(right, comfort_clearance, boundary_reserve);
        });
        if (expanded.size() > static_cast<std::size_t>(beam_width)) {
            expanded.resize(static_cast<std::size_t>(beam_width));
        }
        beam = std::move(expanded);
        last_complete = beam;
        effort_horizon = frame;
    }
    if (last_complete.empty()) return 1;

    std::vector<ActionEvaluation> evaluations;
    for (std::int32_t action = 0; action < kActionCount; ++action) {
        if ((hard_first_action_mask & (1u << action)) == 0) continue;
        const Node* best = nullptr;
        std::set<std::pair<long, long>> endpoints;
        std::set<std::int32_t> continuations;
        for (const auto& node : last_complete) {
            if (node.first_action != action) continue;
            endpoints.emplace(
                std::lround(node.position.x / position_quantization),
                std::lround(node.position.y / position_quantization));
            continuations.insert(node.last_action);
            if (best == nullptr
                || node_key(node, comfort_clearance, boundary_reserve)
                    < node_key(*best, comfort_clearance, boundary_reserve)) {
                best = &node;
            }
        }
        if (best == nullptr) continue;
        evaluations.push_back(ActionEvaluation{
            action,
            best,
            std::min(hard_min_clearance[action], best->min_clearance),
            boundary_deficit(best->position, boundary_reserve),
            static_cast<std::int32_t>(endpoints.size()),
            static_cast<std::int32_t>(continuations.size()),
        });
        result->surviving_first_action_mask |= 1u << action;
    }
    if (evaluations.empty()) return 1;
    const auto selected = std::min_element(
        evaluations.begin(),
        evaluations.end(),
        [&](const ActionEvaluation& left, const ActionEvaluation& right) {
            return evaluation_key(left, current_action, comfort_clearance)
                < evaluation_key(right, current_action, comfort_clearance);
        });
    result->selected_action = selected->action;
    result->effort_horizon = effort_horizon;
    result->min_clearance = selected->min_clearance;
    result->cumulative_risk = selected->best->risk;
    result->terminal_x = selected->best->position.x;
    result->terminal_y = selected->best->position.y;
    result->terminal_boundary_deficit = selected->boundary;
    result->endpoint_count = selected->endpoints;
    result->continuation_action_count = selected->continuation_actions;
    return 0;
}
