"""Read-only feature schema for retained portable tree-scorer artifacts.

This module preserves the byte-level model format consumed by historical
generation policies. It does not define a learner, labels, rewards, or a data
selection path.
"""

FEATURE_SCHEMA = "th06-rl-offline-feature-v1"
CATEGORICAL_FEATURES = (
    "source_context",
    "action",
    "baseline_action",
    "current_action",
    "legal_mask",
    "hard_mask",
    "context_quality",
    "transition_schema",
)
NUMERIC_FEATURES = (
    "player_x",
    "player_y",
    "edge_reserve",
    "power",
    "bullet_count",
    "laser_count",
    "hard_action_count",
    "legal_action_count",
    "phase_elapsed_frames",
    "action_dx",
    "action_dy",
    "action_focused",
    "action_stationary",
    "action_diagonal",
    "baseline_dx",
    "baseline_dy",
    "baseline_focused",
    "matches_baseline",
    "matches_current",
    "phase_number_0",
    "phase_number_1",
    "phase_number_2",
    "phase_number_3",
    "phase_number_4",
    "phase_number_5",
)
FEATURE_NAMES = (*CATEGORICAL_FEATURES, *NUMERIC_FEATURES)
