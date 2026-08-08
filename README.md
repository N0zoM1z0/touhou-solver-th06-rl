# touhou-solver-th06-rl

A source-grounded TH06 learning agent built around a small native dodge gate.
The online loop projects already-observed physical hazards, certifies a bounded
movement set, lets a lightweight learned policy rank that set, and freshly
certifies the selected action immediately before publication. It performs no
timeline/ECL expansion or combinatorial beam search in the resident hot path.

Its accelerated Linux simulation and lockstep environment backend lives in
[`N0zoM1z0/th06-headless`](https://github.com/N0zoM1z0/th06-headless). That
source-only fork is for corpus generation, replay, and learning acceleration;
the shipped Windows game remains this solver's final physical validation gate.
The currently paired local runtime revision is `4b8c90b` on the headless
fork's `th06-rl-headless-spike` branch; solver clients and benchmark protocol
changes are committed together with that dependency recorded in the evidence
note.

Online UCB and future imitation/RL policies own long-horizon/global-local
tradeoffs but cannot enlarge the native safe set, change collision physics,
lower margins, request Bomb, or bypass fail-close behavior. Cold start uses
only a generic clearance/boundary-reserve fallback, never phase rules.

The restartable online learner uses three bounded backoff levels. Its original
coarse phase/position/threat/reserve UCB remains the broad hot start. A middle
layer adds a 30-frame phase-relative clock plus current and baseline actions,
and a fine layer additionally separates the exact Hard/lookahead masks already
computed by the native gate. Unseen exact frontiers therefore reuse a more
specific phase/control prior before falling back to the heavily aliased coarse
state. No level reads raw bullet geometry or runs a model in the resident hot
path.

The active learning target is Lunatic / Reimu-A / Stages 1 through 6. Hard /
Reimu-A / Stage 4 is retained only as prior baseline evidence. Difficulty,
character, shot type, stage, and automatically derived source context remain
separate corpus/model scopes. The 2026-08-08 offline snapshot predates this
expansion and contains only Stages 4 through 6.

For source-simulator counterfactual evaluation,
`scripts/benchmark_headless_branches.py` replays one generic action prefix once,
forks all 18 Bomb-free actions from the resulting physical state, and compares
every terminal observation against a cold full-prefix replay. It runs at nice
15 / low I/O priority and writes a revisioned JSON report under ignored
`artifacts/`; this is an offline teacher/benchmark path, never resident search.

Generate a compact factual trajectory with the paired runtime, then audit it
independently:

```bash
PYTHONPATH=src python scripts/generate_headless_corpus.py \
  --seed 7 --difficulty 3 --character 0 --shot-type 0 --stage 6 \
  --max-ticks 1200 --epsilon 0.02 --teacher-horizon 12
PYTHONPATH=src python scripts/audit_headless_corpus.py artifacts/headless-corpus \
  --output artifacts/benchmarks/headless-corpus-audit.json
```

The offline teacher may run bounded local search, but its selected action is
rechecked against the four-frame native gate immediately before the synchronous
step. Every compact row stores exact behavior probability, the native legal
set, candidate clearance/reserve features, automatic source context, and both
observation hashes. Full source observations are gzip anchors every 120 ticks
and at terminal/failure boundaries. Neither timeline identity nor RNG selects
a handwritten movement route, and RNG is excluded from deployable state
features.

`run_lunatic_stage123456_learning.bat` rotates exact Practice Stages 1 through 6,
records one complete sharded stage trajectory at a time, checkpoints each
stage's independent bounded online model, cleans up the exact game PID, and
starts the next full stage. The verified life patch prevents Game Over without
hiding physical HIT:
each HIT is recorded as negative feedback, input is released through death and
spawn, and play resumes in the same stage. A certified local control dead-end
also becomes feedback rather than an external restart. Bomb or an
input-backend failure still stops the loop. Transient capture, source-context,
policy-reload, trace, and corpus failures fail closed without destroying
the physical Stage episode. Create
`artifacts\pause-lunatic-stage456` to pause between complete stages. The
Stages 4 through 6 can also be rotated with the older
`run_lunatic_stage456_learning.bat`. The single-stage
`run_lunatic_stage4_learning.bat` remains useful for focused experiments.
`run_lunatic_stage5_learning.bat` and
`run_lunatic_stage6_learning.bat` are the focused Stage 5 and Stage 6 training
entry points and retain the same complete-episode/storage/audit contracts. The
`run_hard_stage4_learning.bat` entry point and its independent policy state are
retained for baseline reproduction, not for the active data flywheel.

Policy durability is a Stage-boundary operation: the full restart checkpoint
is atomically written only after the complete Practice Stage, then committed
by the Stage transaction. The quiet-window maintenance path polls only the
small policy source signature. It never serializes the large learner while the
game can recover from a HIT, so persistence I/O cannot consume respawn control
frames.

Before launching each new Stage, both loops conservatively account the
`artifacts` tree and reserve the recorder's full 512 MiB per-run allowance.
Corpus usage comes from each atomic manifest plus a 2 MiB per-run metadata/open
shard reserve; the few live traces and policy files are measured directly, so
the Windows guard never performs a slow per-shard UNC scan. Collection stops
before game launch if the reservation could cross the 45 GiB local budget; no
corpus is deleted automatically. Complete, audited runs can later be mirrored
into a Hugging Face dataset before any user-approved local pruning.

During physical play, the batch writes gzip-0 shards only to the fast local
`D:\th06-rl-corpus-spool`; it does not deflate or transact manifests over WSL
UNC. After the exact game PID has stopped, the batch recompresses those shards
to gzip-3, verifies their content hashes while copying, atomically archives the
run under `artifacts/corpus`, and only then permits the next Stage. A failed
finalization returns status 78 and retains the local spool for retry.

Mirror every closed run and the last Stage-committed policy states with:

```bash
python scripts/sync_hf_corpus.py
```

The sync is incremental and excludes the currently open run. If a Stage is in
progress, its transaction backup is uploaded instead of the live, uncommitted
checkpoint. The public default dataset is
[`Joh1rreq/touhou-solver-th06-rl-corpus`](https://huggingface.co/datasets/Joh1rreq/touhou-solver-th06-rl-corpus).
`--dry-run` performs local structural validation; add `--verify-content` when
doing the slower full shard SHA-256
audit before local pruning. Successful upload alone never deletes local data.

Each control frame retains a coherent collision-authority root: player state,
all live bullet motion/collision fields, lasers, lethal enemy bodies, RNG,
resources, source context, exact Hard/local sets, behavior probability, and
decision diagnostics. Frame schema v4 stores each occupied bullet once as its
packed source tail plus a compact pointer-to-visual-size table; the redundant
resident Python bullet rows are reconstructed offline. This is lossless for
collision and boundary-reflection semantics while keeping the asynchronous
writer comfortably ahead of the 60 Hz producer. Expensive player-attack,
item/effect, sprite, callback,
and complete ECL state is retained in exhaustive anchors admitted only during
quiet/passive windows. Stable timeline/ECL tables are content-addressed.
Transitions retain raw outcome and latency terms instead of only a reward, so
later GPU training can rebuild features, temporal windows and reward functions
without teaching the policy to imitate missed control frames. A transition
with a capture gap or over-budget root remains evidence but is excluded from
the default online learner stratum.

The trajectory unit is one complete physical Practice Stage. Transition
schema v5 records that episode identity separately from source-context changes
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

Pass `--prior-run-dir artifacts/corpus/PRIOR_RUN_ID` once per loaded prior run
to measure how much of the current run actually reuses their union.
The report contrasts coarse, middle, and exact-fine partitions. It is an alias
and support diagnostic, not an off-policy estimate.
Fine legal-action opportunities and behavior choices are already losslessly
present in the corpus; the restart checkpoint retains only observed fine
counters so its RAM and disk cost scales with feedback actually consumed
rather than every action that happened to be legal or merely selected.

Run the immutable corpus audit and CPU-only offline policy zoo with:

```bash
PYTHONPATH=src python scripts/audit_offline_corpus.py CORPUS_DIR \
  --revision HUGGING_FACE_COMMIT --output artifacts/offline/corpus-audit.json
PYTHONPATH=src python scripts/train_offline_cpu.py CORPUS_DIR \
  --revision HUGGING_FACE_COMMIT --scope 3/0/0/6 --view exact-v5 \
  --threads 12 --output artifacts/offline/stage6-exact-policy-zoo
```

The trainer fits CatBoost, LightGBM, CPU XGBoost, and Extra Trees sequentially,
uses chronological complete-Stage splits, and evaluates only native-safe action
sets. Offline ranking never authorizes policy promotion. The source-grounded
geometry/planning benchmark, recorded physical-frame coherence check, effective
corpus ratios, exact fixed revision, and conservative VPS commands are recorded
in [`notes/OFFLINE_TRAINING_BENCHMARK_20260808.md`](notes/OFFLINE_TRAINING_BENCHMARK_20260808.md).

Online reward version `survival-reserve-hit-trace-v2` delivers a confirmed
physical HIT independently of the one-step publication outcome. It assigns a
discounted penalty to at most the prior 120 learning-eligible actions in the
same difficulty/character/shot/stage/source-phase scope, then clears that
episode trace. This is deliberately bounded: ordinary-frame work remains O(1)
and the O(120) scan runs only on HIT. Checkpoints from the earlier reward
version are rejected rather than silently mixed into these statistics.

Faithful older corpus can hot-start that same online learner without fitting a
different model or inventing outcomes for actions that were not taken:

```bash
PYTHONPATH=src python scripts/replay_online_ucb.py \
  --difficulty 3 --stage 6 \
  --code-commit COMMIT --native-kernel-sha256 SHA256 \
  --output artifacts/policy/lunatic_reimu_a_stage6.json \
  --report artifacts/replay/lunatic_reimu_a_stage6.json
```

The replay accepts only complete physical Stages in the exact requested
scope, registers the logged proposal, consumes only the recorded eligible
outcome, and delivers recorded HITs through the same bounded credit path as
live play. Optional code/native filters keep pre-fix infrastructure out of a
deployment hot start. Transition v5 stores the small exact UCB context beside
each outcome, so future replay does not need to decode the large raw hazard
root; older v4 runs remain usable through their lossless frame shards.

Every batch invokes the post-Stage infra audit before starting the next game.
It writes `infra-audit.json` inside the ignored run directory and classifies
HITs as latency gaps, empty Hard sets, newly born hazards, possible geometry /
Hard-certificate counterexamples, missing publication, or unresolved local
traces. It never edits policy parameters or route behavior.

Audit every bullet EX-motion combination retained by complete corpus runs,
including all occupied raw slots rather than only the online-reachable subset:

```bash
PYTHONPATH=src python scripts/audit_motion_flags.py artifacts/corpus
```

The report separates source-exact fired motion, source-bounded unknown
spawn-animation completion, candidate-dependent player retargeting, and
unknown fail-closed modes. This is an offline coverage census and adds no work
to the resident controller.

After a complete Lunatic Stage, verify the dense-frame control budget with:

```bash
python scripts/evaluate_latency.py artifacts/live/lunatic_reimu_a_stage4.jsonl
```

The initial qualification requires the 500+ live-bullet bin to keep control
P95 within one 60 Hz frame (`16.67 ms`) and stale retries at or below 2%.
