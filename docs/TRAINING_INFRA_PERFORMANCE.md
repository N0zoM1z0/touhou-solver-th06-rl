# Training infrastructure performance log

This append-only log separates throughput engineering from learner efficacy.
Every optimization must preserve factual original-Wine transitions, physical
HIT cost, native-safe authority, behavior propensities, complete population,
and promotion gates. A faster run is never evidence that a policy is better.

The repository-wide learner CPU budget is capped at 32 logical cores because
this host also runs unrelated CPU-heavy work. Parallel fold and population
workers divide that single budget; it is never interpreted per worker. New
training entrypoints must reject a request above the cap. Canonical Wine
collection and learner fitting are not launched concurrently.

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

## 2026-08-12: process-parallel cross-fitting under a 32-core cap

The first differential ran the unchanged two-iteration, eight-Q-tree,
eight-V-tree, seven-member, five-fold frozen 29-episode workload. Five
copy-on-write Linux processes executed independent folds; each received six
threads from one global 32-thread budget. Members initially ran concurrently
inside each fold with one XGBoost thread each. Cross-fit wall time fell from
639.79 to 206.17 seconds, a 3.10x speedup. Including an expected cache rebuild
and unchanged representation, total internal wall time was 375.79 seconds.
`/usr/bin/time` measured 726% average CPU and 24.96 GB maximum RSS.

Profiling showed that concurrent members inside one process still contended on
the Python custom-objective GIL and transiently created about 1,217 OpenMP
threads per fold despite low actual CPU use. The accepted scheduler therefore
runs one member at a time inside each fold and gives that native XGBoost fit all
six fold threads. `threadpoolctl` bounds OpenMP to six and BLAS to one during
the fit. Five folds can use at most 30 cores. On the same workload cross-fit
fell again to 191.87 seconds, a 3.33x speedup over the serial baseline. Total
time with the final expected cache rebuild was 363.06 seconds; average CPU was
1,444%, observed peak was approximately 29 cores, and maximum RSS was 14.88 GB.

Correctness: deterministic synthetic parallel and serial reports match in
full. On the recorded Wine corpus, both 32-core schedulers produced exactly
5,432 final proposals with identical action counts and identical per-fold final
proposal counts. Aggregate relative Q loss differed by 8.6e-8. One of 37,528
single-panel-only boundary decisions changed under native floating-point
reduction order, but it remained an abstention and did not change a published
action. The original 48-thread smoke is not the reproducibility reference
after the explicit host-sharing cap changed; every subsequent evidence fit
uses the committed 32-thread scheduler.

The cache rebuild exposed over-broad invalidation: its contract included the
entire training CLI, so a resource-default edit invalidated factual rows even
though the loader was unchanged. The loader and its accepted schema logic now
live in a dedicated module. Cache identity binds that module plus its factual
parsing dependencies, while CLI orchestration, worker counts, and model code
are excluded. This preserves fail-closed invalidation for data-semantic changes
without paying a five-gigabyte re-audit for unrelated scheduling edits.

Native-code decision: the measured tree-building kernels already execute in
XGBoost C++. The demonstrated bottleneck was orchestration around a Python
custom objective, and process isolation yields a 3.33x gain without changing
the objective or portable/native artifact format. A new handwritten C++ GBDT
is therefore not justified at this checkpoint; native work remains conditional
on a new profile after cross-fitting and cache costs are amortized.

## 2026-08-12: isolated normal-speed Wine collection workers

Collection throughput scales horizontally without changing game time. Four
preassigned workers each own a copied original game directory, fresh Wine
prefix, X display, artifact subtree, and corpus subtree. A worker never runs two
episodes concurrently. The canonical canary and final natural Stage-6 A/B use
worker zero sequentially; learner fitting and Wine collection never overlap.

Before evidence collection, one fixed Stage-4 input runs once serially and then
simultaneously on workers zero and one. The gate requires identical physical
HIT count and an identical normalized factual-option digest. That digest covers
sequence, frame, action, incumbent, complete propensity vector, all candidate
features, interval HIT, duration, return-to-go, termination, hazards, and
history; only per-run episode/option identifiers are removed. This differential
is explicitly non-evidence. Failure disables parallel collection rather than
relaxing equality or changing training data.

The seed schedule, 32-thread learner cap, worker assignment, displays,
collection boundaries, canary order, normal-speed final order, original game
inventory hash, Wine binary, native libraries, and all execution/learner source
files are bound by a committed SHA-256 contract before the differential starts.
This makes concurrency a replayable infrastructure choice, not an outcome-
dependent data-distribution adjustment.

The frozen differential failed. The serial worker-zero reference produced 28
HITs and normalized factual digest `69c7a198...d55`; concurrent worker zero
produced 30 HITs and `6454c07f...bb95`; concurrent worker one produced 26 HITs
and `751d6cb3...57e0`. All three original-Wine Stages completed cleanly, their
isolated path gate passed, but neither physical outcome nor factual semantics
matched. The first observed gameplay frames were 89, 88, and 91 respectively,
and option-boundary counts were 9,011, 9,204, and 8,660. Thus a fixed retail RNG
does not fix the complete controller/capture closed loop; startup and scheduling
timing can alter later factual interaction.

Decision: parallel Wine collection is disabled. The frozen episode-to-worker,
RNG, policy seed, fit, canary, and final schedules remain unchanged, but each
assigned worker runs one at a time in schedule order. The differential corpus
is non-evidence and is never admitted to training. This is the predeclared
failure behavior, not a relaxed equivalence metric. Offline parse/cache and
cross-fit parallelism remain enabled under the 32-core cap because they operate
on already recorded immutable facts and passed deterministic differentials.

The first serial evidence wave then exposed a separate host-display collision.
Episodes zero and one completed and were preserved with 23 and 36 HITs. Episode
two stopped before gameplay because pre-existing socket `X99` already existed;
its report had zero trace rows and no physical outcome. Sockets `X99` through
`X104` dated from an unrelated July session and had no discoverable owning
process, but they were not deleted. A hash-bound infra migration retains worker
IDs and assigns workers two/three fresh directories, prefixes, and displays
`:105`/`:106`. RNG, policy seed, episode order, corpus, learner, and all gates
remain unchanged; resume reuses the two completed factual episodes.

The first resume exposed a control-plane-only bug before Wine started: serial-
fallback validation indexed a formerly one-row migration list instead of
selecting its stable ID. The repair performs ID lookup and is bound to the exact
failure and prior contract hash. It changes no resource assignment or outcome
contract and preserves the same two evidence episodes.

The next fixed episode completed the original-Wine Stage with 22 physical HITs
and 23,428 lossless transitions, but one near-terminal input pickup timed out.
The controller failed closed, released input, and completed the Stage; the
strict learner audit correctly rejected the entire run because it contains an
`authority_lost` transition. Validation is not weakened and the run is not
learner-visible. Instead, the Wine primitive now archives a rejected attempt
and reruns the identical frozen row, with a fixed maximum of three total
attempts. Game RNG, policy seed/state, worker, stage, scorer, reward, and data
admission are unchanged. Repeated failure still stops as an infra failure.

This retry is required for unattended execution: a recoverable one-frame host
pickup fault must not require a human to relaunch the orchestrator, but neither
may it silently enter offline RL or cause outcome-conditioned resampling. Every
failed report and corpus remains on disk and its triggering hashes are recorded
in the migration manifest.

One control-plane provenance detail was repaired before the retry started: the
earlier X99 display failure was originally referenced through the reusable
active episode path. Once a later attempt occupied that path, startup correctly
rejected the mismatch. The display migration now binds the already preserved
`episode-002.incomplete-001/report.json` archive with the same original hash.
This prevents normal retries from aliasing immutable failure evidence.

The run also sharpened the Wine concurrency diagnosis. Evidence runs begin
controller observation at frames 85--91 despite fixed RNG and the same menu
route. That makes stage-entry/controller handoff the leading generic source of
the failed parallel differential. A synchronization change may restore
parallelism only after a new serial-versus-concurrent differential proves exact
HIT and normalized factual-option equality; until then collection remains
normal-speed serial and uses about one CPU core.

## Native implementation priority

## 2026-08-12: logical thread budgets do not enforce host sharing

The Stage-4 boundary-10 smoke exposed a resource-contract defect. Although
five fold processes each received six learner threads from the declared
32-thread budget, aggregate sampled CPU briefly reached approximately 51
cores. Independent OpenMP runtimes can create and schedule extra teams, so
library `n_jobs` and `threadpoolctl` settings are not a sufficient host-level
limit.

The live process group was immediately restricted to CPUs 0--31; the
boundary-15 smoke and subsequent Wine child inherited that affinity and stayed
inside the requested host-sharing set. Generation 6 makes this an executable
contract rather than an operator action: the launcher selects at most the first
32 CPUs from its inherited affinity, applies `sched_setaffinity` before any
worker is created, and records the effective set. Library thread counts remain
an inner performance control, while OS affinity is the authoritative hard cap.

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
