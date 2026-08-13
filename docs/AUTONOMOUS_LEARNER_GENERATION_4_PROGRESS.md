# Autonomous learner generation 4 progress

This append-only record begins after Generation 3 was explicitly stopped and
before any Generation-4 Wine outcome. It cannot amend the design contract
during an evidence run.

## 2026-08-12: generation declared

Generation 3 was stopped after 13 clean complete historical episodes and one
fit-ineligible round. Its next in-progress episode was terminated, has no
retail report, and is excluded. Exact process-group shutdown left no game,
controller, Wine runner, or Xvfb process.

Generation 4 is declared in
`AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md`. It replaces complete-Stage raw
AIPW labels with factual semi-Markov n-step Bellman targets and a generalized
action-centered R objective; adds recorded ESS/uncertainty-aware propensities;
calibrates the cross-fitted policy rather than the maximum counterfactual row;
and requires the full seven-member population to meet the native latency gate.
No new-generation Wine outcome has been created.

## 2026-08-12: sequential causal learner smoke passed

The first complete implementation slice constructs factual option intervals,
eight-decision undiscounted Bellman targets, five-fold frozen value/outcome
nuisances, and a seven-member generalized action-centered R critic. An initial
smoke failure exposed an implementation-scale bug: equal episode weights had
been normalized so far below the tree learner's fixed minimum child weight
that the nuisances remained at their base prediction. Correcting the weight
scale while preserving equal whole-episode influence made the full pipeline
learnable; no gameplay parameter or evidence threshold was involved.

The deterministic production-sized causal fixture then passed on 160 complete
episode groups. Its true candidate effect was -1 HIT while common state risk
changed independently. Results:

- all seven members predicted negative advantage at both risk levels;
- aggregate prediction: -1.0149 HIT;
- cross-fitted critic R loss: 12,675.41 versus zero-effect 14,415.47;
- aggregate state-risk leakage: 0.3535 HIT;
- maximum individual-member state-risk leakage: 0.7179 HIT;
- every centered objective coefficient was bounded by 0.75; no inverse
  propensity appeared in the objective.

Unit contracts separately prove recursive interval-HIT conservation at the
eight-decision/terminal boundary and direct recovery of -1 action effect from
the centered objective. This is synthetic algorithm evidence, not Wine
gameplay evidence or authorization.

## 2026-08-12: propensity and full-population implementation

The Generation-4 behavior policy now implements the declared 0.50 incumbent,
0.25 uniform, and 0.25 information mixture only over the current native-safe
set. Before a critic exists, information weight is inverse square-root
accumulated propensity ESS. When a fit-authorized shadow critic is embedded,
the same quantity is multiplied by bounded seven-member disagreement. The
uniform component supplies the declared minimum probability independently of
the incumbent, action name, or gameplay location.

Transition v10 records the complete boundary probability vector, normalized
information weights, and pre-assignment action ESS. Tentative boundary
statistics roll back if Wine did not execute the assignment. Rejected actions
remain non-factual. Historical transition-v9 corpora retain their exact known
mixture reconstruction and remain readable by the sequential loader.

The online policy loads all seven 128-tree critic members through the existing
bounded native batch scorer. For each safe candidate it subtracts that member's
incumbent score and permits an override only when every member is negative.
One optimistic member forces abstention; there is no winner or mean-only
distillation. Host/Win32 hash compatibility and per-member conformance remain
mandatory.

The production fitting entrypoint consumes both declared historical v9 and new
v10 factual Wine episodes, applies the fixed five-fold/eight-step/160-tree
nuisance and 128-tree critic contract, and writes an immutable shadow state.
All repository tests pass after the transition, exploration, rollback,
population-selection, and loader changes. No Generation-4 Wine outcome has
been launched.

## 2026-08-12: frozen Wine offline smoke and preflight contract

All 13 complete Generation-3 episodes were frozen as one indivisible
historical input by run identity, manifest SHA-256, and physical HIT count.
The 16 new collection seeds, policy seeds, nine paired-canary seeds, and short
non-evidence smoke seed were generated from the declared Generation-4 seed and
committed before any Generation-4 Wine outcome. Selecting episodes after a fit
or outcome is impossible under this contract.

A deliberately low-cost 24-tree version of the entire sequential pipeline was
run on the 13 frozen factual Wine episodes. It loaded 48,001 factual option
boundaries and conserved their complete-Stage HIT totals. Cross-fitted R loss
fell from 346,684.17 for the zero-effect critic to 337,162.73 for the learned
critic, a relative loss of 0.9725, with improvement in all 13 episode groups.
This is the first factual Wine indication that the new orthogonal objective
contains learnable treatment signal. It is not efficacy evidence: 13 groups
fail the predeclared minimum of 20 and the smoke used fewer trees than the
production contract.

Policy calibration support was additionally corrected to fit prototypes and
the 99th-percentile threshold from each fold's training episodes only. The
held-out episodes now influence neither the critic nor its abstention rule.
The preflight hard gate binds this implementation and the seed/corpus
contracts, repeats the production causal recovery, benchmarks all seven full
128-tree members across 1,200 maximum-sized native decisions, and audits a
short retail-Wine transition-v10 run for complete propensities, ESS,
information weights, native-safe execution, and option lifecycle. The short
Wine run is explicitly ineligible as training or evaluation evidence.

## 2026-08-12: unattended state machine completed

The Generation-4 runner now owns the complete fixed process: validate the
frozen historical corpus, run preflight, collect new retail-Wine v10 episodes,
fit at 8/12/16 new-episode boundaries, replay the last three new episodes as a
baseline-only native shadow, hash-authorize a candidate, run the three-pair
fixed-RNG canary, and—only after it passes—run the alternating 12-natural-Stage
per-arm evaluation. It checkpoints every completed Wine episode and can resume
without repeating or replacing evidence.

If a fit-authorized native shadow exists after an unsuccessful canary, later
collection may use its seven-member disagreement only to allocate the fixed
0.25 information mass. The 0.50 incumbent and 0.25 uniform components remain
unchanged, the native-safe set remains authoritative, and the shadow critic
cannot publish an action. After 16 new complete Stages the runner emits the
predeclared ineffective verdict if no candidate has earned canary
authorization; there is no manual fallback or winner selection.

## 2026-08-12: full population native path accelerated

The first maximum-load native smoke correctly rejected the unmodified online
path: seven 128-tree members over 18 safe actions and 256 hazard primitives
measured 6.93 ms p95, above the fixed 4 ms gate, although it had zero 60 Hz
deadline misses. The model was not reduced. Inspection showed that the same
feature matrix was flattened and copied through `ctypes` seven times, once per
member.

The isolated native scorer now accepts the complete immutable population and
one shared candidate matrix in a single batch call. It still traverses every
tree of every member and returns model-major predictions; import checks every
member against its portable conformance values. Under the identical
1,200-decision maximum-load smoke, p95 fell to 3.16 ms, maximum latency to 3.26
ms, and deadline misses remained zero. Thus Generation 4 retains all seven
128-tree members without distillation or winner selection.

A second run under competing host CPU load exposed p95 jitter to 4.45 ms even
though mean latency remained 2.99 ms. The gate was again kept unchanged. The
native traversal was reordered from member/row/tree to member/tree/row, keeping
each compact tree hot while evaluating all action rows and preserving each
row's floating-point tree-addition order. Under the same loaded host this
reduced p95 to 2.19 ms and maximum latency to 2.36 ms. Passed causal/native
offline preflight artifacts are now contract- and scorer-hash cached so an
unrelated Wine startup failure can be retried without refitting or weakening a
gate.

The first retail-Wine v10 smoke then exposed an audit distinction rather than
a safety failure. Twelve input-lease/observation-gap continuation rows carried
no boundary baseline, but every executed intent equalled their singleton
native-safe action. The audit had incorrectly required a baseline on
continuations. It also compared 998 fully recorded tentative assignments
against only 270 factually executed boundaries; 728 tentative boundaries had
been explicitly rejected and their ESS updates rolled back. The repaired
audit requires baseline membership only at treatment boundaries, requires
executed intent membership on every option row, verifies propensity/
information/ESS vectors for all 998 assignments, and admits only the 270
executed boundaries as factual treatments. The captured non-evidence corpus
then passed every wiring and safety gate without changing policy behavior.

## 2026-08-12: collection episode 0 and generic startup repair

The committed preflight passed, after which the unattended runner completed
the first new fixed-RNG retail-Wine Stage. Episode 0 contains 27,436 online
decisions, completed Lunatic Stage 6 with HIT continuation, recorded 40
physical HITs, produced one complete transition-v10 corpus, and left no prefix
processes. It was atomically checkpointed before the runner attempted the next
episode.

Episode 1 produced no controller rows or corpus because startup normalization
stopped before control. GDB had attached while Wine was legitimately handling
`SIGUSR1`; its default signal policy returned from `continue` at the Linux
signal trampoline (`0xf7ff4549`) instead of waiting for the unchanged TH06
timing-loop breakpoint (`0x0042097e`). The generic GDB startup script now
passes `SIGUSR1` through without stopping and still requires the exact timing-
loop address, instruction bytes, and menu state before writing timing state.
No game, learner, action, RNG, reward, or outcome rule changed.

The migration from the exact old preflight contract hash to the repaired hash
is explicitly declared in the Generation-4 infra migration manifest. Resume
is permitted only from the recorded `infra_failure`, only when the sole config
change is that preflight hash, and records that one complete v10 episode was
preserved with no outcome or schedule-field change. The empty episode-1
startup artifact is archived and retried; it is not evidence.

## 2026-08-12: evidence budget completed

The unattended runner completed all 16 declared new original-retail Wine
Stages and all three fit boundaries. Every new run was a clean Lunatic Stage 6
completion in HIT-continuation mode with complete factual v10 data. The
physical HIT counts were 40, 34, 41, 38, 42, 26, 31, 38, 47, 36, 35, 34, 46,
42, 38, and 30.

No fit was eligible. Round 1 worsened aggregate cross-fitted R loss and improved
only 4/21 episodes. Round 2 again worsened aggregate loss despite improving
13/25 episodes. Round 3 improved aggregate loss by only 0.0116% and improved
14/29 episodes, failing strict majority. Proposal rates were respectively
17.0%, 61.0%, and 52.1%, which is not a stable decision surface. The runner
therefore emitted the frozen ineffective decision after the 16-new-Stage
budget; it did not launch a canary or natural evaluation.

The finite-sample support calculation separately missed its declared 99%
coverage by using an interpolated percentile. This is a demonstrated generic
infra bug, but it is not causal to the verdict because the independent
episode-majority gate also failed. The result and next-generation reasoning are
recorded in `AUTONOMOUS_LEARNER_GENERATION_4_RESULT.md`.
