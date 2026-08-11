# Stage 6 sub10 native-safe scan result (2026-08-10)

## Decision

The scan creates zero candidates and closes the r1 anchor. `up_right` was the
unique discovery winner, but it was not locally admissible at the predeclared
r2 confirmation checkpoint. The native gate rejected the confirmation before
any alternative branch was accepted. No r3/r8 fallback confirmation is
allowed.

## Discovery

The exhaustive protocol was committed as `52c97f0` before execution. At r1
sequence 2904 / retail frame 3296, all 14 recorded retail-native-safe actions
were branched under restored frozen-UCB continuation. The factual
`down_right` branch again matched the exact recorded suffix and frame-3303
Hard-empty terminal.

Thirteen actions reached authority failure after seven ticks. `up_right`
reached the same kind of authority failure after 45 ticks with zero physical
death, making it the single robust winner without relying on a reserve
tie-break. The COW document SHA-256 is
`19cafa9556586a6461e36d28b7b49e47cb121befdb77d4d209536a9cc17016fc`.
The discovery audit SHA-256 is
`51426db5447a6becba6bc09ff801f23259dd8320d9d232756e702cfced592f75`.

## Confirmation rejection

At the fixed r2 sequence 3193 / retail frame 3566 checkpoint, factual
`down_right` remained reproducible, but `up_right` was absent from the native
local admissible set. `label_retail_policy_cow.py` failed closed with
`first action up_right is not locally admissible` and emitted no confirmation
document. This is the predeclared rejection path, not an infrastructure
failure and not permission to choose the discovery runner-up.

## Consequence

A fixed-direction residual does not generalize across the two independent
sub10 checkpoints. Do not activate `up_right`, choose another r1 action, move
the r1 frame, or substitute r3/r8 for confirmation. A future experiment may
only use a separately predeclared action-relative selector that ranks whatever
actions the current native gate admits, trained and held out by physical
episode rather than memorizing a fixed direction or checkpoint identity.

No complete Stage was run because no candidate reached Wine shadow. Complete
natural original-retail Wine Stage 6 HIT count remains the promotion metric;
headless survival ticks are only a rejection/selection filter.
