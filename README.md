# touhou-solver-th06-rl

A source-grounded TH06 agent built around a non-learning reactive dodge
baseline. The local planner searches short movement continuations, ranks
survival and maneuver reserve before soft objectives, and can only choose from
actions certified by an external authority layer. A fresh certification is
required immediately before input publication.

Learning is intentionally downstream of that baseline. Future imitation and
RL policies may rank survival-equivalent actions, but cannot enlarge the safe
set, change collision physics, lower margins, request Bomb, or bypass
fail-close behavior.

The initial physical target is Normal / Reimu-A / Stage 1. Difficulty,
character, shot type, stage, and automatically derived source context remain
separate corpus/model scopes.

