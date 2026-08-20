# Exact Wine training throughput

## Decision

Do not accelerate the clock of evidence-producing TH06 processes. The retail
timing loop is paced at 60 Hz, while capture, native-safe solving, and input
delivery consume real host time around that loop. Scaling Wine timers or
removing the wait can change how many coherent observations and input updates
fit between physical frames even if the game executable itself is unchanged.
That semantic risk would train on a different control process.

Training throughput instead scales horizontally: multiple isolated copies of
the unchanged original executable run concurrently, each at normal timing.
Canary and final evidence remain single-instance and sequential.

## Required worker isolation

Every collection worker owns a distinct:

- retail game directory, because configuration and score files are mutable;
- Wine prefix and wineserver;
- X display;
- controller/runner artifact directory;
- corpus spool and finalized corpus root;
- fixed-RNG and policy seed assignment.

No process, file, prefix, display, or run-discovery root is shared. A completed
worker is accepted only after the existing retail and transition audits pass,
its corpus run is immutable, and exact cleanup proves no leftover process.
Validated run directories are then merged by run identity and manifest hash;
partial or duplicate runs fail closed.

## Compatibility gate

Before parallel data becomes eligible, run the same frozen behavior contract
once sequentially and simultaneously on two isolated workers. The differential
uses the same fixed retail RNG and immutable policy state. Physical HIT count
and a normalized factual digest must match exactly; changing startup or capture
timing is a semantic failure, not training diversity. The gate also compares
executable/native hashes, schemas, complete-Stage lifecycle, zero drops/
failures/Bomb, option/propensity invariants, frame/decision sequence, and exact
cleanup.

Only after both concurrent workers reproduce the serial reference exactly may
ordinary natural-RNG collection use two workers. Expansion beyond two requires
a separately recorded resource and compatibility gate. Any mismatch disables
parallel collection and starts an infra investigation; it never relaxes
equality, alters the learner, or selects different gameplay data.

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
authority, model population, or promotion criteria. Final measurement remains
the original Wine process at normal speed over complete natural-RNG Stages.

The implemented factual-option cache lives under ignored
`artifacts/cache/audited-option-episodes/`. Its key contains the complete
manifest SHA-256 and loader-source contract SHA-256; its metadata also binds the
absolute run identity and payload SHA-256. It never accepts a partial pair or
mismatched digest. Cache creation calls the ordinary full loader first, so the
first use still verifies every transition shard and factual/HIT invariant.
