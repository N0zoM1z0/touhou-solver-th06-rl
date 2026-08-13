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

The next exact path computes the seven-member double-precision mean from the
unchanged float actor outputs and applies the same positive-advantage/support
choice in native code. Lexical tie ranks are passed explicitly to reproduce
Python tuple `max`; the factual supported mask and native-safe rows are still
provided by the immutable policy/controller contracts. Python `array` buffers
also replace per-element ctypes argument expansion without sharing mutable
state between decisions.

On the same 181--256-hazard fixture, exact action equality held and isolated
Wine p95 fell again to `1.7113 ms` (maximum `1.8918 ms`, zero >4 ms); Linux p95
was `0.5227 ms`. Report SHA-256 is
`b8ffb838bbf72adde6e68c20719bff4225dd5b28f581f80de911e8ffd21c3c47`.
The third successor canary keeps the same 32-CPU partition and changes only
experiment identity, policy RNG seed, display, and the preflighted DLL hash.

The v3 live run completed without an infrastructure or deadline event, but its
resident p95 was `4.0678 ms`; 405 of 4,087 boundary scores exceeded 4 ms. All
other frozen gates passed. This is a strict rejection, not a rounding case,
and it leaves the learner and data untouched. Comparing isolated and live
timings (`1.7113 ms` versus `4.0678 ms`) shows that the remaining work is
generic tail-latency robustness under co-resident game/capture load. The next
profile target is Python boundary-time allocation and copying of observation,
action, hazard, and index arrays. Any repair must preserve exact portable /
Linux / Win32 choices on factual maximum-width contexts before a new Wine
identity can be frozen.

The component profile then identified a platform-specific deployment cost:
the 32-bit MinGW target used legacy x87 scalar arithmetic even though the
fitted network is float32. The reproducible build now selects the baseline
SSE2 scalar ABI for the Win32 ranker only (`-msse2 -mfpmath=sse`). It does not
change the actor graph, activation, weights, support, mean, or choice code; the
original 32-bit retail game and native safety kernel remain unchanged.

The canonical DLL is SHA-256
`471013f1daa40c57829722a61f1729726a803bcccd4d0a898738ecd096f8c01a`.
Its independent 126-output differential was `7.4387e-5`, below the unchanged
`1e-4` tolerance. On the same 64 maximum-width factual contexts, all portable,
Linux, and Win32 proposals remained exact. Isolated Wine p95 fell from
`1.7113 ms` to `1.2393 ms`, maximum was `1.3571 ms`, and no sample crossed 4
ms or the frame deadline. Kernel report SHA-256 is
`73e24bd2a9840d483c982bd0edb4c0c420948f8daee29bcf27d9fedc9912a13e`;
full-path report SHA-256 is
`45267dd8dc0c434702482ec71503c58b8a38d97500fd1b7fa67f65c806fa0071`.
This creates enough measured margin to freeze a new wiring identity; it does
not retroactively pass v3 or authorize Stage-6 efficacy by itself.

The frozen v4 live canary confirmed the margin under the complete co-resident
game/capture workload. Resident p95 was `2.8751 ms`, down from v3's
`4.0678 ms`; one of 3,885 boundary scores exceeded 4 ms and none exceeded the
frame deadline. Every non-performance gate and cleanup also passed. The SSE2
change therefore resolves the Generation-6 serving blocker without
distillation or a learner/data change. It does not accelerate original game
time and does not turn the canary's three HITs into efficacy evidence.

The succeeding six-run complete-Stage-6 panel confirmed that result over a
substantially longer live workload. The unchanged full seven-member scorer
processed 25,963 option boundaries and 182,048 controller decisions. Per-run
p95 ranged from `2.9619` to `3.0019 ms`; only 12 individual boundary scores
exceeded 4 ms and none missed the 16.67 ms deadline. All runs used the frozen
0--7 game / 8--31 controller allocation and cleaned their private prefix.
Thus the optimized native path is not merely a short Stage-4 canary result: it
is stable across six normal-speed complete Stage-6 executions without
distillation or changing the learner.

## 2026-08-13: all-registry Generation-6 round preflight

The autonomous successor now selects all 44 registry entries with
`sequential_offline_rl` rather than copying a generation-owned partition.
Existing audited-option cache identities remain unchanged, so 143,078 factual
options loaded from cache without rebuilding the 7.6 GiB history. Five
episode-grouped cross-fit folds completed in 369.70 seconds inside CPUs 0--31.
This is the practical fast loop for learner repairs: no Wine process starts and
no new gameplay data is required.

The replay exposed a report-only numerical defect: direct softmax probabilities
underflowed on large finite logits and made behavior KL infinite at JSON write.
A stable log-softmax calculation now keeps the same diagnostic finite without
changing fitting, policy choice, or DR estimation. A regression test covers a
1,000-logit gap.

Production fit then exposed scale dependence in the old pure `1e-4` portable /
native actor-score tolerance. The refit produced score magnitudes up to about
1,448; its worst absolute float32 accumulation difference was
`0.000244140625`, while the relative error at those large scores was below
`4e-7`, all 64 selected actions were exact, and the smallest action margin was
`0.06715`. The gate now uses `1e-4 + 4 * float32_epsilon * abs(score)` and still
requires exact action equality. Checkpoint-resumed smoke passed at tolerance
ratio 0.532, support error `1.27e-6`, p95 `1.532 ms`, and zero deadline misses.

The independently pinned PyTorch 2.8 CPU dependency was absent from the local
virtual environment even though it was already declared in
`requirements-cpu-train.txt`; installing that exact pinned build restored the
synthetic smoke. This is environment provisioning, not an algorithm or corpus
change.

A raw synthetic Win32 actor-kernel differential subsequently exceeded the new
score bound. It is not waived or treated as gameplay evidence. The formal
round adds a stronger test: the complete fused support/hazard/actor policy must
produce exact portable, Linux, and Wine/Win32 proposals on 64 immutable
computational-width factual contexts, with p95 below 4 ms and no deadline miss,
before an active state can be exported. The state used for this test is
shadow-only, so native validation cannot accidentally become a canary.

## 2026-08-13: Generation-6 CFS deadline-tail reproducer and repair

Autonomous round 1 stopped after its eleventh collection Stage because the
resident actor recorded two calls above the `16.67 ms` frame deadline. The
episode still had p95 `2.9630 ms`, and both misses appeared inside one short
window where capture and the complete controller solve also produced
simultaneous 18--53 ms tails. An isolated 20,000-call factual-width Win32 replay
had p95 `1.2508 ms`, maximum `1.4047 ms`, and no miss. This excluded a
deterministic model-width or hazard-input cost explosion.

The new controlled stress audit runs the same 64 maximum-width factual
contexts and canonical SSE2 DLL for 10,000 calls while 32 ordinary CFS workers
contend on CPUs 0--31. Equal-priority execution reproduced 29 deadline misses
and a `24.6883 ms` maximum despite p95 `1.3199 ms`. Exact nice `-10` under the
same load retained p95 `1.3137 ms`, reduced maximum to `9.4206 ms`, and had
zero misses. All portable/Linux/Win32 actions stayed exact. The ignored formal
report SHA-256 is
`8526220a0fc1d467bee4b9c24d4e6fa8b786560093a03ca91fabcb63ee5c591f`.

Wine children now use bounded `SCHED_OTHER` priority, never real-time
scheduling. A small root wrapper validates the explicit inherited CPU set and
nice range, applies them, drops completely to the invoking non-root user, and
writes a run-local attestation. The strict complete-run validator checks the
effective UID/GID, CPU list, scheduler, and nice value. The controller also
stops sorting the actor's 4,096-sample latency window and reconstructing all
action diagnostics on every factual frame: immutable identity stays per-frame,
while full metrics are emitted every 60 frames and in one exact final record.

These are model- and game-neutral scheduling/telemetry repairs. The 32-CPU cap,
normal Wine pacing, native safety, action choice, propensity, factual data,
reward, and zero-deadline gate do not change. The failed round remains invalid;
a new frozen contract is required. Full reasoning and reproduction steps are
in `GENERATION6_LATENCY_TAIL_AUDIT.md`.

## 2026-08-13: bounded-priority child identity

The first repaired-round startup exposed that sudo's monitor PID is not the
exec child's PID. The wrapper attested Wine PID `3909394`, while Popen returned
monitor PID `3909392`; attaching the startup GDB script to the monitor could
never reach TH06. No controller or new corpus was created. The generic runner
now resolves the live attested PID before GDB attach and records both
identities. Two no-corpus smokes then proved that `dbus-launch --autolaunch`
cannot exit while the run's private Xvfb is alive: increasing a pre-X-shutdown
wait from five to fifteen seconds had no effect, and the helper exited
immediately after Xvfb stopped. Cleanup now stops the owned Xvfb before a
bounded five-second helper grace. This avoids immediate retry races without
killing an unrelated or shared process.

## 2026-08-13: repaired-round collection and refit profile

The repaired round's two new original-Wine Stages covered 56,957 game frames
and completed in about 18 minutes wall time, with scorer p95 values 3.1006 and
3.1105 ms and zero deadline misses. Cleanup and priority attestations passed
on both. The scheduler repair therefore removes the observed blocker without
accelerating game time or consuming more than the 0--31 CPU allocation.

The following all-registry cross-fit loaded 56 episodes / 167,250 options and
completed in 685.42 seconds. Five folds ran as five processes and used roughly
12--15 cores in aggregate. The subsequent seven-member production fit is the
new dominant offline cost: after 40 minutes it was still actively computing,
used the full 32-thread allowance, and held about 29 GiB RSS. CPU time kept
increasing and there was no crash or wait deadlock. This frozen run is allowed
to complete unchanged so it yields a comparable artifact and exact duration;
the next general performance audit should profile production actor fitting,
especially repeated full-dataset augmentation/training and thread/memory
scaling. It must preserve model math and output before any optimization is
accepted.

## 2026-08-13: decision-level float32 serving

Round 3's raw-logit gate compared NumPy/OpenBLAS scores with a scalar native
kernel before subtracting their common baseline. On the frozen checkpoint the
raw maximum error reached `0.0009765625`, although a read-only panel retained
64/64 exact actions. A real-arithmetic forward bound was also needlessly loose:
it charged the decision for cancellation error in common bias and state terms
that the policy never consumes.

The successor keeps the seven trained members but makes the native hot kernel
accumulate baseline-centred hidden/latent differences directly. Both native
targets compile this kernel with `-ffp-contract=off`; no model distillation or
weight modification is involved. A target-portability envelope starts from a
declared eight-unit-roundoff `tanhf` allowance and propagates only target
variation through the exact scalar operation sequence using local ULPs and
intermediate absolute sums. Four previously unresolved factual cases improved
from real-arithmetic margin/envelope ratios `0.31`--`0.52` to portability ratios
`6.49`--`11.04`; Linux target differences were `1.34e-5`--`2.44e-4` and stayed
inside the corresponding bounds. This smoke is definition validation, not the
formal full-corpus or Win32 result.

The full audit reuses the fit checkpoint and option cache, so numerical-serving
iterations avoid both original-Wine runtime and the 53-minute production fit.
It must still traverse every registered factual option and then run the frozen
wide Win32 panel before online authorization.

The first serial profile processed 3,719 options in 25.3 seconds, implying
roughly 19 minutes for 56 episodes before the scalar panel. The general audit
now forks at the immutable episode boundary: 16 single-threaded workers share
the fitted arrays copy-on-write, return deterministic per-dimension panel
heaps, and the parent reduces them with content-hash tie breaks. The scalar
panel uses the same bounded worker pool. Each child initializes its own
BLAS/OpenMP thread pool limit at one, so bounded parallelism does not depend on
shell environment variables and cannot silently become process-by-thread
oversubscription after a learner change. A regression test proves panel
reduction is invariant to completion order and locks the default at 16 workers
under the repository-wide 32-CPU cap.

On the full 56-episode / 167,250-option / 2,415,808-candidate workload, the
parallel float64 serving smoke completed both the Linux differential and 320
scalar envelope cases in 106.5 seconds. The main corpus pass finished in about
83 seconds. The frozen successor allows a 180-second wall-clock ceiling, so
ordinary host contention has headroom while a return to the roughly 19-minute
serial path still fails automatically. The report splits Linux-corpus and
scalar-panel time. Future learner/export variants must use this runner and retain its
reported worker count, exact option identity, deterministic panel selection,
and serial-equivalent result; reverting to an unbounded pool or the old
single-process loop is an infrastructure regression.

The Win32 differential must reconstruct the exact 320 identities selected by
that audit from immutable raw transition shards. A first single-process
development attempt spent multiple minutes saturating one CPU in gzip/JSON
decoding and was stopped before producing evidence. Context extraction now
partitions by immutable episode across 16 single-threaded fork workers under
the same 32-CPU affinity contract. The preflight reports this stage separately
and rejects context loading above 120 seconds; later algorithm/export changes
may not silently restore the serial corpus scan or select an easier fixture.
