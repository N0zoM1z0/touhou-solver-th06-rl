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

The restartable online learner is hierarchical. Its original coarse
phase/position/threat/reserve UCB remains the hot-start backoff, while a fine
layer separates 30-frame phase-relative clock bins, current and baseline
actions, and the exact Hard/lookahead masks already computed by the native
gate. This avoids averaging visibly different physical frontiers without
reading raw bullet geometry or running a model in the resident hot path.

The active learning target is Lunatic / Reimu-A / Stages 4, 5, and 6. Hard /
Reimu-A / Stage 4 is retained only as prior baseline evidence. Difficulty,
character, shot type, stage, and automatically derived source context remain
separate corpus/model scopes.

`run_lunatic_stage456_learning.bat` rotates exact Practice Stages 4, 5, and 6,
records one complete gzip-sharded stage trajectory at a time, checkpoints each
stage's independent bounded online model, cleans up the exact game PID, and
starts the next full stage. The verified life patch prevents Game Over without
hiding physical HIT:
each HIT is recorded as negative feedback, input is released through death and
spawn, and play resumes in the same stage. A certified local control dead-end
also becomes feedback rather than an external restart. Bomb or an
input-backend failure still stops the loop. Transient capture, source-context,
policy-checkpoint, trace, and corpus failures fail closed without destroying
the physical Stage episode. Create
`artifacts\pause-lunatic-stage456` to pause between complete stages. The
single-stage `run_lunatic_stage4_learning.bat` remains useful for focused
experiments. `run_lunatic_stage5_learning.bat` and
`run_lunatic_stage6_learning.bat` are the focused Stage 5 and Stage 6 training
entry points and retain the same complete-episode/storage/audit contracts. The
`run_hard_stage4_learning.bat` entry point and its independent policy state are
retained for baseline reproduction, not for the active data flywheel.

Before launching each new Stage, both loops conservatively account the
`artifacts` tree and reserve the recorder's full 512 MiB per-run allowance.
Corpus usage comes from each atomic manifest plus a 2 MiB per-run metadata/open
shard reserve; the few live traces and policy files are measured directly, so
the Windows guard never performs a slow per-shard UNC scan. Collection stops
before game launch if the reservation could cross the 45 GiB local budget; no
corpus is deleted automatically. Complete, audited runs can later be mirrored
into a Hugging Face dataset before any user-approved local pruning.

Each control frame retains a coherent collision-authority root: player state,
all live bullet motion/collision fields, lasers, lethal enemy bodies, RNG,
resources, source context, exact Hard/local sets, behavior probability, and
decision diagnostics. Expensive player-attack, item/effect, sprite, callback,
and complete ECL state is retained in exhaustive anchors admitted only during
quiet/passive windows. Stable timeline/ECL tables are content-addressed.
Transitions retain raw outcome and latency terms instead of only a reward, so
later GPU training can rebuild features, temporal windows and reward functions
without teaching the policy to imitate missed control frames. A transition
with a capture gap or over-budget root remains evidence but is excluded from
the default online learner stratum.

The trajectory unit is one complete physical Practice Stage. Transition
schema v4 records that episode identity separately from source-context changes
and failure events: a life-patched HIT is negative evidence inside the same
Stage, not an episode boundary. Offline phase-local learners may split on the
source-context boundary, while a Stage-level learner can retain the full
trajectory without inferring either convention from a generic `done` bit.

Non-boss play is partitioned by the next authoritative timeline event. Boss
and midboss play is partitioned by boss ID, ECL subroutine, life/timer
callbacks, and spell/nonspell state. These labels condition learning only;
they never select a handwritten movement route.

Inspect physical learning progress without touching the game process:

```bash
python scripts/evaluate_learning.py --difficulty lunatic --stage 4 --recent 20
```

Run the same command separately for stages 5 and 6; the evaluator refuses to
mix their trends. The report separates complete-stage trends from interrupted diagnostics and
includes per-source-phase HIT/dead-end/action coverage, policy support and
observed outcomes, capture/solve latency, stale retries, and compressed corpus
density. These are descriptive physical metrics; they are not presented as
counterfactual or causal off-policy proof.

Measure whether the online state compression aliases distinct physical
frontiers in one completed run with:

```bash
PYTHONPATH=src python scripts/analyze_policy_aliasing.py \
  artifacts/corpus/RUN_ID
```

The report contrasts the preserved coarse UCB grouping with a counterfactual
hierarchical partition. It is an alias diagnostic, not an off-policy estimate.

Every batch invokes the post-Stage infra audit before starting the next game.
It writes `infra-audit.json` inside the ignored run directory and classifies
HITs as latency gaps, empty Hard sets, newly born hazards, possible geometry /
Hard-certificate counterexamples, missing publication, or unresolved local
traces. It never edits policy parameters or route behavior.

After a complete Lunatic Stage, verify the dense-frame control budget with:

```bash
python scripts/evaluate_latency.py artifacts/live/lunatic_reimu_a_stage4.jsonl
```

The initial qualification requires the 500+ live-bullet bin to keep control
P95 within one 60 Hz frame (`16.67 ms`) and stale retries at or below 2%.
