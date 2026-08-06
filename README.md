# touhou-solver-th06-rl

A source-grounded TH06 learning agent built around a small native dodge gate.
The online loop projects already-observed physical hazards, certifies a bounded
movement set, lets a lightweight learned policy rank that set, and freshly
certifies the selected action immediately before publication. It performs no
timeline/ECL expansion or combinatorial beam search in the resident hot path.

Online UCB and future imitation/RL policies own long-horizon/global-local
tradeoffs but cannot enlarge the native safe set, change collision physics,
lower margins, request Bomb, or bypass fail-close behavior. Cold start uses
only a generic clearance/boundary-reserve fallback, never phase rules.

The active collection target is Hard / Reimu-A / Stage 4. Difficulty,
character, shot type, stage, and automatically derived source context remain
separate corpus/model scopes.

`run_hard_stage4_learning.bat` repeatedly starts an exact Practice Stage 4
trial, records one complete gzip-sharded stage trajectory, checkpoints the
bounded online model, cleans up the exact game PID, and starts the next full
stage. The verified life patch prevents Game Over without hiding physical HIT:
each HIT is recorded as negative feedback, input is released through death and
spawn, and play resumes in the same stage. A certified local control dead-end
also becomes feedback rather than an external restart. Bomb or an
input-backend failure still stops the loop. Transient capture, source-context,
policy-checkpoint, trace, and corpus failures fail closed without destroying
the physical Stage episode. Create
`artifacts\pause-hard-stage4` to pause between complete stages.

Each corpus frame retains the complete coherent source snapshot, Hard set,
local survivor set, policy probability and planner diagnostics. Stable
timeline/ECL tables are content-addressed once per run. Transitions retain raw
outcome terms instead of only a reward, so later GPU training can rebuild
features, temporal windows and reward functions without recollecting play.

Non-boss play is partitioned by the next authoritative timeline event. Boss
and midboss play is partitioned by boss ID, ECL subroutine, life/timer
callbacks, and spell/nonspell state. These labels condition learning only;
they never select a handwritten movement route.

Inspect physical learning progress without touching the game process:

```bash
python scripts/evaluate_learning.py --recent 20
```

The report separates complete-stage trends from interrupted diagnostics and
includes per-source-phase HIT/dead-end/action coverage, policy support and
observed outcomes, capture/solve latency, stale retries, and compressed corpus
density. These are descriptive physical metrics; they are not presented as
counterfactual or causal off-policy proof.
