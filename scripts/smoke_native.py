#!/usr/bin/env python3
"""Fast native ABI smoke: a late incoming box must cause a lateral dodge."""

from th06_rl.core import Kinematics, LocalPlannerConfig, movement_actions
from th06_rl.native import Aabb, NativeKernel, PackedHazards


def main() -> int:
    actions = movement_actions()
    by_name = {action.name: action for action in actions}
    horizon = 12
    hazards = PackedHazards(
        aabb_frames=tuple(
            (
                Aabb(
                    188.0,
                    340.0 + 3.0 * frame,
                    196.0,
                    348.0 + 3.0 * frame,
                ),
            )
            for frame in range(1, horizon + 1)
        ),
        laser_frames=((),) * horizon,
    )
    kernel = NativeKernel()
    kinematics = Kinematics(4.0, 2.0, 2.828427, 1.414214)
    hard = kernel.certify_actions(
        x=192.0,
        y=380.0,
        half_width=1.25,
        half_height=1.25,
        kinematics=kinematics,
        current_action=by_name["stay"],
        hazards=PackedHazards(
            hazards.aabb_frames[:4], hazards.laser_frames[:4]
        ),
        candidates=(by_name["stay"], by_name["left"], by_name["right"]),
    )
    # Reuse the Hard-4 certificate as immutable first-action authority while
    # the soft beam sees the longer forecast.
    plan = kernel.plan(
        x=192.0,
        y=380.0,
        half_width=1.25,
        half_height=1.25,
        kinematics=kinematics,
        current_action=by_name["stay"],
        hazards=hazards,
        hard=hard,
        continuation_actions=(
            by_name["stay"], by_name["left"], by_name["right"]
        ),
        config=LocalPlannerConfig(horizon=horizon),
    )
    if plan is None or plan.action.name not in {"left", "right"}:
        raise RuntimeError(f"native planner failed dodge smoke: {plan}")
    print(
        f"native={kernel.path} hard={len(hard)} action={plan.action.name} "
        f"h={plan.effort_horizon} clearance={plan.min_clearance:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

