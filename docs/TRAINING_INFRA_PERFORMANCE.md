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

## 2026-08-12: Generation-6 low-rank reference profile

The correctness-first 31-episode development cross-fit loaded 102,737 cached
factual options in 11.51 seconds and spent 2,160.60 seconds in three sequential
complete-episode folds, 2,172.11 seconds total. The entire process remained on
the recorded CPU set 0--31. Sampled average use rose to approximately 24 cores;
resident memory stabilized near 13.8 GB. Native libraries created about 157
threads, but OS affinity prevented them from scheduling outside the 32-core
host-sharing set.

The profile identifies reusable data-plane and orchestration work, independent
of the failed learner result. Candidate-invariant option packing is rebuilt for
each fold, and seven causally independent whole-episode bootstrap members run
serially inside each fold. The next optimization may cache immutable packed
base arrays and execute members with fork/copy-on-write under fixed per-member
CPU shares. It must reproduce row order, centered propensities, targets, seeds,
losses, predictions, and selected actions before replacing this reference. No
qualification or Wine run waits on that performance work.

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

## 2026-08-12: five-fold IQL actor development replay

The first action-centered actor replay reused 31 cached immutable Wine episodes
and 102,737 factual options. Warm audited loading took 11.68 seconds,
candidate-invariant representation/augmentation took 51.49 seconds, and five
copy-on-write episode-fold workers completed critic plus actor fitting in
303.59 seconds, for 366.76 seconds total. Every process inherited the hard CPU
set 0--31. Sampled use was about three cores per fold, roughly 15 cores total;
each worker reported 14--15 GB resident while sharing most immutable pages by
fork.

This makes an algorithm correction test a minutes-scale offline replay and
requires no new Wine episode. It also shows unused room below the 32-core hard
cap. Actor minibatch work may be rebalanced only after estimator correctness;
the current scientific replay keeps five folds and their seed schedule fixed.

The one-line estimator repair plus expanded diagnostics replayed the exact same
workload in 374.46 seconds: 11.39 seconds warm loading, 52.96 seconds
representation, and 310.11 seconds five-fold fitting/scoring. The 2.1% total
difference is normal competing-host variation, not an algorithmic speed claim.
This confirms that immutable-corpus learner changes can be falsified in about
six minutes without starting original Wine.

The stricter nested cross-fit actor removed the separate global representation
pass and fitted five outer-fold representations concurrently. Each outer fold
then fitted three inner label critics, one evaluation nuisance, and seven actor
bootstraps. Despite stronger isolation, the complete 31-episode replay took
266.58 seconds including 11.4 seconds warm loading, versus 374.46 seconds for
the in-sample-weight actor. The hard CPU set remained 0--31 and sampled use was
about 14 cores during native/actor fitting. This is a 1.40x end-to-end speedup
from replacing seven bootstrap critics with four purpose-separated critics,
while producing a statistically stricter result.

## 2026-08-12: native dense actor population preflight

The first formal-width conformance run correctly failed its provisional
`2e-5` actor tolerance. Across 64 immutable development options, NumPy/BLAS
versus fixed-order C++ float32 dense accumulation differed by at most
`6.1035e-5`; native support distance differed by only `2.27e-6`. The earlier
random small-matrix test was not wide enough to characterize accumulation over
the production 234-feature representation.

The native actor numerical tolerance is therefore `1e-4`, while support keeps
`2e-5` and final selected action must match in every conformance case. This is
about four decimal digits below the unit-scale score and remains guarded by
exact action equality. The fitted population was checkpointed before native
scoring, so the follow-up changes neither training data nor model parameters
and does not refit merely to obtain a favorable native result.

The checkpoint-resumed full path then passed all 64 conformance cases with
exact action equality. Across 1,200 decisions including native hazard encoding,
native prototype support, the complete seven-actor forward pass, population
mean, and action choice, latency measured 2.12 ms median, 2.19 ms p95, and
3.34 ms maximum with zero 60 Hz deadline misses. This is below the frozen 4 ms
p95 online budget without distillation.

The rebuilt fully-static 32-bit Windows DLL was independently exercised by the
embeddable Python under Wine. On seven production actors and 18 deterministic
candidate rows, all 126 outputs completed and maximum portable-to-Windows error
was `6.2943e-5`, below the same `1e-4` contract. DLL SHA-256 is
`0aa7c5a95b90b2df0d032ec02f21fcd3a39be3ba440819d00c0cb025bc641ef0`.

The single-disclosure qualification loaded/augmented 31 development and 13
qualification episodes in 83.92 seconds, then spent 263.63 seconds scoring
40,341 qualification options, for 347.56 seconds total. The current hot spot is
Python per-option calls for seven actors, seven leave-one-out policies, and the
evaluation tree. Native batched scoring already exists and should replace this
offline loop after the qualification result is frozen; this performance issue
does not affect the separately measured online path.

## 2026-08-12: complete Wine online-policy preflight

Kernel-only timing understated the resident cost, so Generation 6 added a
full-path fixture over 64 factual registered Wine option contexts. It executes
adapter feature validation, native hazard encoding, candidate construction,
action-conditional support, all seven actors, population mean, and final
action selection under both Linux and the 32-bit embeddable Python in Wine.
It requires exact portable/Linux/Windows action equality, p95 below 4 ms, and
zero 60 Hz misses over 1,200 repeated boundary decisions.

The first honest run failed: actions were exact and Wine had zero frame misses,
but Wine p95 was `5.1593 ms`. The report is retained at SHA-256
`969280c18ae2775a1a27b39193a481d13b11dd0dd19b6465bc23873b8a77f27e`.
Profiling showed repeated Python reconstruction of the same observation,
baseline action, and schema once per legal candidate. Parsing these once per
option boundary changed no feature, model, support rule, or action and reduced
Wine p95 to `3.8651 ms`; report SHA-256 is
`54602019fb6ee92ad4e4ff3e22879679ef789f82ea0891ac2201e1aa1d343cb6`.

That margin was still unnecessarily small. The dense C++ actor multiplied
feature-major matrices through a cache-strided loop. Reordering the loop nests
keeps each output's exact accumulation order while traversing contiguous
weights. The optimized Linux/Win32 libraries are
`f0e34ad5b0929b3333e850028f814036786078193176cf08968d6975b3e220fa`
and
`e794045cb89e9f6439e4bdfc354325f89a0771a57aa75a4aa654aac9197f2b87`.
The independent Win32 126-output differential remained below `1e-4`
(`6.8665e-5` maximum), and the full factual path retained exact choices.

The final frozen-state preflight measured Linux p95 `1.3523 ms` and Wine p95
`3.2986 ms`, with Wine maximum `3.4282 ms`, zero samples above 4 ms, and zero
60 Hz misses.
Its ignored report SHA-256 is
`0ca3821252b2c8d02591aafbce3534539a19cce9e981b9be697ff1f39899dc0c`.
This is a pure deployment-infrastructure optimization; the frozen candidate
and all offline qualification values remain unchanged.

The isolated preflight was necessary but not sufficient. In the first live
original-Wine Stage-4 canary, with TH06, capture, native safety, input delivery,
and the 32-bit policy sharing the host, resident actor-policy p95 rose to
`8.1554 ms`; 2,143 of 4,693 boundaries exceeded 4 ms. No decision crossed the
16.67 ms deadline, but the stricter margin gate correctly rejected deployment.
The run completed and cleaned normally, so this is a reproducible live-load
performance defect rather than an outcome, safety, or learner failure.

The next optimization target is the remaining Python/FFI boundary: the same
234-wide candidate matrix is currently traversed and marshalled independently
for support and actor normalization. A fused native entry point can reuse that
matrix, perform state/action normalization, support, and seven-actor scoring in
one call while preserving the exact candidate and choice. A new live canary is
allowed only after exact portable equivalence and a successor frozen contract;
the failed performance evidence remains immutable.

## 2026-08-12: fused resident scorer and load isolation

The generic repair now performs adapter-array row construction, bounded hazard
encoding, action-conditional support, normalization, and the full seven-actor
population in one native call. It does not rank actions and cannot see game
memory, collision state, input, propensity, HITs, phases, or RNG; Python still
chooses a positive mean-score proposal only among the native-safe rows. Thus
the optimization changes neither learner semantics nor safety ownership.

The first 64-context preflight had accidentally sampled zero-hazard opening
boundaries. The corrected stress selector is explicitly computational: from
one registered sequential Wine episode it takes the largest hazard and safe-set
input widths without reading actions, HITs, phases, RNG, scores, or outcomes.
The frozen set contains 181--256 hazard primitives. On it all portable, Linux,
and Win32 actions were exact; Linux p95 was `0.7493 ms`, Wine p95 `2.1229 ms`,
Wine maximum `2.4212 ms`, and no sample exceeded 4 ms or the frame deadline.
Report SHA-256 is
`d687027508acc5787a0db846f8c5b48ce64c3ccee4b7fe9f49dfeb6a150cce2f`.
The independent 126-output Win32 actor differential remained `6.8665e-5`.

The fused Linux/Win32 binaries are
`58c3a1aa82c73dba5f1200094546b16aa1d2044e0c5f046027719368ab5580ab`
and
`507b7e2bb797b6d90b12dbebf1d77c431d6f3ce9086cf522c749f5f10305fa1b`.
The successor live canary additionally reserves CPUs 0--7 for the original
game and 8--31 for the controller. Both remain inside the user-requested
32-CPU set; the split prevents the game and scorer from evicting each other
without changing original 60 Hz pacing or running concurrent Wine trials.

The v2 live canary confirmed the improvement but missed the hard gate narrowly:
resident p95 fell from `8.1554 ms` to `4.1027 ms`, 438 of 5,051 boundaries were
above 4 ms, and none exceeded 16.67 ms. CPU partitions and every non-latency
gate passed. Because the policy still returns all 126 actor outputs across the
FFI and recomputes seven-member means in Win32 Python, the next model-invariant
optimization returns the final mean-supported row directly. Exact portable
action equality remains mandatory; the 4 ms gate is not rounded or relaxed.
