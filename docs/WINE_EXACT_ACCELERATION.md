# Exact Wine training throughput

## Decision

Do not accelerate the clock or frame multiplier of evidence-producing TH06
processes. The corpus controller suspends Wine at a coherent source root while
it captures, certifies, records, and publishes one action. This preserves the
retail per-update order and `frame_multiplier == 1.0`, but deliberately makes
wall-clock playback slower than 60 Hz. Scaling Wine timers or removing waits
would change the control process and is not admitted.

Training throughput instead scales horizontally: multiple isolated copies of
the unchanged original executable run concurrently under the same coherent
suspension contract. A final online candidate must additionally pass a
single-instance, non-suspending, real-time 60 Hz end-to-end gate; paused corpus
evidence cannot satisfy that gate.

## Required worker isolation

Every collection worker owns a distinct:

- retail game directory, because configuration and score files are mutable;
- Wine prefix and wineserver;
- X display;
- controller/runner artifact directory;
- corpus spool and finalized corpus root;
- predeclared RNG mode and policy-state assignment.

Worker game directories are copied only from a never-executed template
extracted from the attested RAR. Copying the canonical gameplay directory is
forbidden because TH06 mutates its score and configuration files. The template
marker binds the archive, executable, and full file inventory; any drift fails
closed.

No process, file, prefix, display, or run-discovery root is shared. A completed
worker is accepted only after the existing retail and transition audits pass,
its corpus run is immutable, and exact cleanup proves no leftover process.
Validated run directories are then merged by run identity and manifest hash;
partial or duplicate runs fail closed.

## Compatibility gate

Before parallel data becomes eligible, run the same frozen behavior contract
once sequentially and simultaneously on every isolated worker declared by the
pool. The differential uses the same fixed retail RNG and immutable policy
state. Every worker's physical HIT count and normalized factual digest must
match the serial reference exactly; changing startup or capture timing is a
semantic failure, not training diversity. The gate also compares executable/
native hashes, schemas, complete-Stage lifecycle, zero drops/failures/Bomb,
action-distribution invariants, frame/decision sequence, and exact cleanup.

Only after every concurrent worker reproduces the serial reference exactly may
ordinary natural-RNG collection use that pool. Changing pool width or ownership
requires a newly recorded resource contract and compatibility gate. Any
mismatch disables parallel collection and starts an infra investigation; it
never relaxes equality, alters the learner, or selects different gameplay data.

## Shared-host scheduling

On a host that is already running important CPU or I/O work, constrain the
entire pool command to an explicit CPU set and inherit the lowest ordinary CFS
priority plus idle I/O priority.  Generate the machine-local pool under the
same affinity that will run its gate:

```bash
TH06_CPU_SET=32-47
ionice -c 3 nice -n 19 taskset -c "$TH06_CPU_SET" \
  .venv/bin/python scripts/prepare_wine_workers.py \
  --worker-root reference/wine-workers-v2-shared-host \
  --workers 2 --cpus-per-worker 8
```

Launch the gate and any later collector through the same
`ionice`/`nice`/`taskset` prefix.  The generated worker CPU partitions remain
disjoint, and normal-priority background work wins scheduler contention.  Do
not use `SCHED_IDLE` for evidence production: a recorded negative Stage 4
serial run completed without semantic mismatch but produced an observation-gap
rate of 3.62%, above the unchanged 0.5% gate.  Host courtesy does not justify
admitting discontinuous data; change only the host scheduling mode and rerun
the exact gate.

## Implemented commands

Prepare a resource-bounded pool. The portable default remains two workers;
choose a larger width explicitly when the current CPU affinity, memory, and
storage preflight can satisfy every worker (for example eight workers):

```bash
.venv/bin/python scripts/prepare_wine_workers.py --workers 8
```

After freezing a collection policy, run the non-evidence fixed-seed gate. Do
not overlap it with a canonical baseline, canary, promotion run, or heavy fit:

```bash
.venv/bin/python scripts/gate_parallel_wine.py \
  --policy-plugin src/th06_rl/policies/uniform_shield_exploration.py \
  --policy-state config/uniform_shield_exploration.json \
  --diagnostic-rng-seed 0x1234 \
  --artifact-root artifacts/parallel-gate \
  --corpus-root corpora/parallel-gate \
  --output artifacts/parallel-gate/gate.json
```

Only the exact bound policy/commit/pool may then collect a predeclared schedule
of complete natural-RNG routes. Changing the policy requires a new gate:

```bash
.venv/bin/python scripts/collect_route_parallel.py \
  --gate artifacts/parallel-gate/gate.json \
  --policy-plugin src/th06_rl/policies/uniform_shield_exploration.py \
  --policy-state config/uniform_shield_exploration.json \
  --episodes 12 \
  --artifact-root artifacts/parallel-collection \
  --corpus-root corpora/parallel-collection \
  --output artifacts/parallel-collection/admission.json
```

The collector writes the immutable schedule before gameplay and is resumable
only at already clean scheduled episodes. It never redraws or skips an episode
because of HIT count. The admission ledger appears only when the full schedule
has passed lifecycle, HIT, player-successor, shield-replay, latency, shard-hash,
and cleanup checks.

## Offline throughput

Offline work is semantically exact and can exploit the host aggressively:

- hash-verified corpus parsing, replaceable feature views, and immutable fit
  outputs;
- encode each factual object history once and score every shield-admissible
  candidate action from that shared representation;
- fit whole-episode population members concurrently while bounding total CPU
  threads;
- retain all population members and benchmark the exact native online scorer;
- resume only from hash-bound atomic checkpoints.

Acceleration never changes reward, successor state, RNG eligibility, action
authority, model population, or promotion criteria. Final measurement uses the
original Wine process without debugger suspension at real-time speed over
complete natural-RNG Stages.
