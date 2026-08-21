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
option/propensity invariants, frame/decision sequence, and exact cleanup.

Only after every concurrent worker reproduces the serial reference exactly may
ordinary natural-RNG collection use that pool. Changing pool width or ownership
requires a newly recorded resource contract and compatibility gate. Any
mismatch disables parallel collection and starts an infra investigation; it
never relaxes equality, alters the learner, or selects different gameplay data.

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
  --policy-plugin "$TH06_POLICY_PLUGIN" \
  --policy-state "$TH06_POLICY_STATE" \
  --diagnostic-rng-seed 0x1234 \
  --artifact-root artifacts/parallel-gate \
  --corpus-root corpus/parallel-gate \
  --output artifacts/parallel-gate/gate.json
```

Only the exact bound policy/commit/pool may then collect a predeclared schedule
of complete natural-RNG routes. Changing the policy requires a new gate:

```bash
.venv/bin/python scripts/collect_route_parallel.py \
  --gate artifacts/parallel-gate/gate.json \
  --policy-plugin "$TH06_POLICY_PLUGIN" \
  --policy-state "$TH06_POLICY_STATE" \
  --episodes 12 \
  --artifact-root artifacts/parallel-collection \
  --corpus-root corpus/parallel-collection \
  --output artifacts/parallel-collection/admission.json
```

The collector writes the immutable schedule before gameplay and is resumable
only at already clean scheduled episodes. It never redraws or skips an episode
because of HIT count. The admission ledger appears only when the full schedule
has passed lifecycle, HIT, player-successor, shield-replay, latency, shard-hash,
and cleanup checks.

## Offline throughput

Offline work is semantically exact and can exploit the host aggressively:

- hash-cache corpus parsing, hazard codebook, augmented factual options, and
  immutable fit outputs;
- encode the state-level hazard set once per option and reuse it for all safe
  candidate actions;
- fit whole-episode population members concurrently while bounding total CPU
  threads;
- retain all population members and benchmark the exact native online scorer;
- resume only from hash-bound atomic checkpoints.

Acceleration never changes reward, successor state, RNG eligibility, action
authority, model population, or promotion criteria. Final measurement uses the
original Wine process without debugger suspension at real-time speed over
complete natural-RNG Stages.
