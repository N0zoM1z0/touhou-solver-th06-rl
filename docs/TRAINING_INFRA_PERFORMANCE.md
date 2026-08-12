# Training infrastructure performance log

This append-only log separates throughput engineering from learner efficacy.
Every optimization must preserve factual original-Wine transitions, physical
HIT cost, native-safe authority, behavior propensities, complete population,
and promotion gates. A faster run is never evidence that a policy is better.

## Measurement contract

For every material change record the workload, repository implementation,
wall time, CPU/thread behavior, peak RSS when observed, cache state, and exact
correctness gate. Failed or neutral attempts remain in this log. Canary and
natural complete-Stage evaluation always use one original retail Wine process
at normal timing; only training collection may scale across isolated normal-
speed workers after its differential gate.

## 2026-08-12: full online population native batching

Workload: seven 128-tree members, 18 safe candidates, 256 hazard primitives,
1,200 decisions. Calling the native scorer once per member measured 6.93 ms
p95. Passing one shared feature matrix to the complete native population and
traversing member/tree/row reduced p95 first to 3.16 ms. Reordering to keep a
compact tree hot across rows, while preserving each row's tree-addition order,
measured 2.19 ms p95 under competing host load and zero 60 Hz misses.

Correctness: every portable/native member prediction is checked on immutable
conformance vectors; the entire seven-member teacher remains deployed. No
distillation or winner selection was used.

## 2026-08-12: state-level hazard encoding reuse

Profile: 877,569 raw transition rows became 102,409 factual option boundaries.
The rich feature builder encoded an identical hazard set separately for every
candidate action, up to 18 times per boundary. It now encodes once per option
and appends the identical encoding and factual history to every candidate.

Correctness: the augmentation test asserts exact feature tuples for factual and
all candidate vectors and asserts one encoder call per option. This removes
duplicated work without changing a bit of learner input.

## 2026-08-12: audited factual-option cache

Baseline: the first 29-episode Generation-5 frozen-Wine smoke spent about 17
minutes on one Python thread before XGBoost began. RSS grew to approximately
13.9 GiB. This stage verifies transition shard hashes, decompresses JSONL,
validates schemas/propensities/factual execution/HIT conservation, and assembles
102,409 options. Repeating it at every fit is unnecessary when neither corpus
nor loader changed.

The ignored cache key binds cache schema, complete corpus manifest SHA-256, and
the full loader-source contract SHA-256. Metadata additionally binds absolute
run identity, payload SHA-256, and option count. Creation calls the original
audited loader; partial or mismatched entries fail closed. Tests cover miss,
hit, source-contract invalidation, and payload tampering.

An eight-thread first-build attempt was stopped after profile showed only 1.12
cores used and zero completed entries: Python JSON parsing remained GIL-bound.
The replacement runs each independent complete episode audit in a separate
process, writes only a small atomic cache entry, then loads all verified entries
in the parent. Large OptionStep objects are never copied through process IPC.
First-build and warm-hit timings use the same repeated 29-episode smoke.

The process-parallel repeated smoke completed all 29 first-build cache entries
and parent loading in 119.17 seconds, versus approximately 17 minutes for the
single-process baseline: at least an 8.6x wall-time improvement for the audited
load stage. The complete run then spent 50.72 seconds on representation and
augmentation and 447.38 seconds on five-fold low-tree cross-fitting, for 617.26
seconds total. All 29 entries were misses as expected. A later warm-cache run
measured steady-state load without changing the smoke workload.

The production-sized seven-member repeat hit all 29 cache entries and loaded
the complete audited corpus in 10.90 seconds. Relative to the approximately
17-minute uncached baseline this is about 94x faster; relative to the 119.17-
second parallel first build it is about 10.9x faster. Representation remained
50.82 seconds and the larger seven-member five-fold fit took 639.79 seconds,
for 701.51 seconds total. Cache metadata, payload hashes, row order, option
counts, and downstream diagnostics all passed unchanged.

This separates two costs clearly: repeated corpus audit/parse is now a small
warm-cache cost, while cross-fitted population fitting is the dominant steady-
state workload. Native work should target a measured fit kernel or data-layout
bottleneck, not replace the now-amortized audit path speculatively.

## 2026-08-12: population fit parallelism

Whole-episode bootstrap members are causally independent after the dataset and
seed schedule are frozen. Generation 5 fits them concurrently under one total
CPU budget and preserves member index, seed, bootstrap counts, tree count, and
output order.

The initial 29-episode smoke nevertheless averaged only about two cores during
the model section while creating over 130 threads. The Python custom-objective
callback for the centered residual is the current suspected limiter. The next
candidate optimization is an equivalent expanded native/weighted regression
or C++ objective path. It may become default only after synthetic and recorded
predictions, loss, selected actions, and exported native models match the
reference objective within a declared numerical tolerance.

## Native implementation priority

The repository should use native C/C++ for fixed, hot numerical kernels and
Python for orchestration and audit readability. Current priority is:

1. measure process-parallel parse/cache build and warm-cache fit;
2. remove the centered-objective Python callback bottleneck with an equivalent
   native formulation;
3. profile hazard aggregation and cross-fit policy metrics after the first two
   changes;
4. only if first-build parsing remains material, implement a native streaming
   transition parser/option assembler behind a Python-reference differential.

A native parser must reproduce OptionStep order, every float/int/string field,
propensity vector, interval/return HIT totals, exclusions, and final content
hash on recorded v9/v10 corpora. Faster but non-identical data is forbidden.
