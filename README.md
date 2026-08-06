# touhou-solver-th06-rl

A source-grounded TH06 agent built around a non-learning reactive dodge
baseline. The local planner searches short movement continuations, ranks
survival and maneuver reserve before soft objectives, and can only choose from
actions certified by an external authority layer. A fresh certification is
required immediately before input publication.

Learning is intentionally downstream of that baseline. Online UCB and future
imitation/RL policies may rank survival-equivalent actions, but cannot enlarge the safe
set, change collision physics, lower margins, request Bomb, or bypass
fail-close behavior.

The active collection target is Hard / Reimu-A / Stage 4. Difficulty,
character, shot type, stage, and automatically derived source context remain
separate corpus/model scopes.

`run_hard_stage4_learning.bat` repeatedly starts an exact Practice Stage 4
trial, records one complete gzip-sharded run, checkpoints the bounded online
model, cleans up the exact game PID, and starts the next trial. It continues
after an ordinary HIT or a certified local control dead-end, but stops on Bomb
or an infrastructure/source-authority failure. Create
`artifacts\pause-hard-stage4` to pause between trials.

Each corpus frame retains the complete coherent source snapshot, Hard set,
local survivor set, policy probability and planner diagnostics. Stable
timeline/ECL tables are content-addressed once per run. Transitions retain raw
outcome terms instead of only a reward, so later GPU training can rebuild
features, temporal windows and reward functions without recollecting play.

Non-boss play is partitioned by the next authoritative timeline event. Boss
and midboss play is partitioned by boss ID, ECL subroutine, life/timer
callbacks, and spell/nonspell state. These labels condition learning only;
they never select a handwritten movement route.
