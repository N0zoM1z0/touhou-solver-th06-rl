# Wine-first learning reset, 2026-08-10

## Decision

The next learning generation is Wine-first and proceeds from Stage 6 through
Stage 1.  Original-retail Wine is the promotion domain.  Linux headless remains
an accelerator for geometry tests, counterfactual proposal generation, and
same-action-stream platform differentials; headless scores never authorize a
policy promotion.

This decision freezes the broad offline-training generation.  Do not resume a
large model zoo, global policy replacement, ensemble sweep, or hyperparameter
sweep before the Wine-first gates below have been satisfied.  The frozen UCB
plus generic native reactive fallback remains the incumbent.  Every learned
candidate is a residual above that incumbent and may rank only the native-safe
set.  Fresh issue certification, fail-close behavior, and Bomb prohibition are
unchanged.

The target is not perfect offline ranking.  The target is a small number of
repeatable interventions that improve original-retail Wine survival without
regressing delivery or native authority.

## Evidence that forced the reset

### The offline-to-Wine path works, but the tested ranker did not help

The portable XGBoost adapter, isolated Win32 scorer, immutable policy state,
native action gate, and original-retail runner all worked end to end.  In the
first natural Stage 6 A/B, however, frozen UCB recorded 8 and 9 HITs while the
offline support32/margin1 selector recorded 12 and 9.  Retail RNG is not yet
controlled, so this small panel is not a powered comparison, but it supplies
no evidence of benefit and the model was not promoted.  See
`docs/WINE_OFFLINE_RANKER_AB_2026-08-10.md`.

The model's apparent training size was misleading.  Its exact-v5 split had
56,533 training rows but only two complete training runs and one validation
run.  Adjacent frames are correlated observations, not independent routes.
The run or seed, not the frame, is the statistical group.

### Strong headless metrics did not predict headless closed-loop survival

The conservative Stage 6 COW value artifact used 392 informative groups and
6,480 branch outcomes.  On seeds 163/164 it reached 80% completed-or-best
top-1 while retaining the factual behavior on about 95.5% of ordinary holdout
groups.  Yet an earlier direct route-wide value candidate stopped after only
3,097/15,365 decisions on seeds 157/158, versus 18,810/20,040 for the
incumbent.  These were native-legal, Bomb-free policy-selection regressions.

This failure happened within Linux headless on fresh seeds.  It proves that
Linux-to-Wine platform drift cannot be the only explanation.  The dominant
demonstrated problem is closed-loop covariate shift: a global learner can fit
sparse corrections while perturbing many upstream decisions.

### Factual Wine failure prediction did not generalize

Twenty-three immutable Wine first-failure prefixes were replayed against the
exact frozen incumbent.  The factual action audit covered 95,768 policy calls
with zero incumbent mismatch, zero policy mismatch, and zero shadow action
contract violation.  That makes the rows factual incumbent evidence; it does
not make later failure a causal label for an alternative action.

The context-reactive v2 r11 guard initially looked precise on its training
folds.  On two later long disjoint Wine shadow prefixes it produced 30 and 51
candidates but no labeled true positive: the first had 26 false positives and
4 unlabeled rows, and the second had 50 false positives and 1 unlabeled row.

The subsequent exact residual audit retained only rows where the offline
ranker proposed a supported native-safe action different from both incumbent
and generic baseline.  After removing 39 non-training replayable frames, the
bound audit retained 983 rows across 21 nonempty run groups.  Only 51 rows were
within factual terminal windows.  The r13 group-held-out result was rejected:

- OOF average precision: 0.1322;
- OOF ROC AUC: 0.6116;
- best nonempty point under the activation limit: 2 TP / 4 FP, one protected
  run, one-sided 95% precision lower bound 0.1173;
- best point protecting two runs: 3 TP / 12 FP, lower bound 0.0829.

No deployable r13 model was produced.  Threshold tuning cannot turn
"incumbent will fail later" into evidence that a counterfactual action will
help.

### Headless and Wine have different execution contracts

Headless synchronous `STEP` has delivery delay exactly zero.  Original-retail
Wine includes coherent process capture, input publication, stale retries, and
fresh issue revalidation.  The current headless ranker schema has 73 features,
including source clocks, Boss-relative state, hazard sectors, and multi-horizon
clearance profiles.  The current portable Wine ranker schema has 33 coarser
features.  A headless model is not deployable merely because similarly named
state exists online.

The first source-platform differential later measured exact discrete delivery
but accumulating floating-point geometry drift beginning at tick 441.  The
subsequent retail-anchored replay work recovered the pre-Stage retail RNG and
replayed two complete frozen-UCB prefixes through both source builds.  Across
2,736 and 2,808 common retail snapshots, RNG, game state, hazard counts, and
player geometry matched at `1e-6`; the only categorical discrepancy was a
known dialogue/control input gap.  This narrows platform uncertainty but does
not establish general retail equivalence.  See
`WINE_FIRST_STAGE6_TARGETED_COW_2026-08-10.md`.

## What remains valuable from the offline generation

Keep the infrastructure:

- source-grounded native geometry and the native-safe first-action set;
- coherent capture, fresh issue certification, fail-close, and exact input
  cleanup;
- immutable policy states, hashes, manifests, and scope separation;
- lossless first-failure prefixes and benchmark-only continuation isolation;
- exact COW outcome generation and grouped seed/run evaluation;
- negative artifacts and rejected-model reports;
- compact native model scoring after a schema has been reproduced exactly.

Stop using the following experiment shapes as promotion evidence:

- frame-random splits or row counts presented as independent sample counts;
- high imitation, COW top-1, AP, or AUC without unseen closed-loop evidence;
- one global ranker replacing the incumbent across the route;
- one seed or one counterexample treated as a general correction;
- terminal-window prediction treated as causal alternative-action value;
- broad route-wide COW generation before Wine identifies a repeated need;
- many near-duplicate hyperparameter candidates selected on one holdout;
- HIT-continuation rows entering any training set;
- headless-only features silently approximated in Wine;
- a single favorable Wine route presented as promotion evidence.

## Wine-first data funnel

### Gold: immutable Wine first-failure episodes

The independent episode is the unit of evidence.  Use frozen UCB, exploration
zero, an immutable copied state, natural Practice, lossless corpus collection,
and the default stop on first HIT, native authority failure, or Bomb request.
Do not patch lives.  Record:

- no-HIT completion versus first terminal;
- game frame and wall time to first terminal;
- terminal kind and automatic source context;
- coherent-capture failures, stale retries, release rows, and solve latency;
- native-safe set, incumbent action, generic baseline, and bounded
  per-action geometry already computed by the gate;
- immutable hashes and exact cleanup results.

Use benchmark continuation only as a separate full-Stage HIT count.  It is
always `training_eligible=false`.

Current telemetry shows that Wine-first collection is affordable.  Across 47
recorded Stage 6 first-failure trials from several policy generations, total
trial wall time was about 1.62 hours; the median was 95.8 seconds and median
terminal frame was 3,400.  Natural full-Stage trials took roughly 7--8
minutes.  Therefore first-failure episodes provide the high-volume physical
stratum; full Stage is reserved for milestones and promotion.

Do not turbo the game or run canonical trials concurrently.  The controller
requires a normal frame multiplier, and CPU contention changes capture and
delivery behavior.  Parallelize replay, clustering, model scoring, and
headless COW instead.

### Silver: replay every candidate on each Wine prefix

One Wine corpus should screen the entire small candidate population offline.
Replay must reconstruct the exact frozen incumbent and assert recorded-action
equality.  It may report where each candidate would intervene, whether its
action was in the recorded native-safe set, normal-region activation, and
terminal-window overlap.  It never publishes the candidate action and cannot
prove causal benefit.

This is the primary scaling mechanism: collect one physical trajectory and
reuse it for every current and future schema-compatible candidate.

### Bronze: targeted multi-seed headless COW

Headless generates counterfactual action outcomes only after Wine identifies a
generic failure region repeated across independent physical episodes.  Match
regions using Wine-reproducible physical features, then sample several
independent headless seeds and branch every native-safe first action.  Require
agreement across seeds and preserve seed-grouped holdouts.

The current COW mechanism still does not copy an original-retail memory
snapshot.  A validated deterministic replay contract now provides a narrower
alternative: recover the pre-Stage RNG, reproduce actual published action and
Shoot delivery, compare the reconstructed state with the Wine snapshot at
`1e-6`, and require an identical native hard-action set before branching.  The
result is a Wine-anchored source COW root, not original-retail execution.  It
can reject a candidate but cannot promote one.

## Failure-region and candidate rules

Mine the existing audited Wine prefixes before collecting more data.  Group by
run, then identify repeated regions from bounded generic physical features:

- player position and boundary reserve;
- current, incumbent, and generic baseline actions;
- native hard/legal masks and action count;
- already-computed per-action clearance and final boundary reserve;
- bullet and laser counts;
- automatic source context as a partition/support key, never a handwritten
  movement branch.

Do not use captured frame, RNG seed, counterexample identity, Boss name, or a
hand-authored phase state as control input.  Collect a new Wine episode only
when it can increase independent support for an unresolved region or provide a
disjoint shadow/canary evaluation.

The first residual generation should be support-driven rather than another
global risk classifier:

1. insufficient independent Wine failure-region support -> incumbent;
2. insufficient multi-seed headless COW support -> incumbent;
3. small independent COW models disagree on the action -> incumbent;
4. proposed action absent from the current native-safe set -> incumbent;
5. fresh issue revalidation fails -> fail closed;
6. otherwise publish at most one native-safe residual action.

Measure intervention events, not only activated rows.  Repeated adjacent
frames can represent one decision region.  The first canary should target only
one to three intervention events in a natural Stage, with generic bounded
hysteresis/cooldown if needed.  This is an initial conservatism target, not a
claim that the number is already optimal.

## Small population, not one offline winner

Retain the population idea but change its role.  Keep:

- one frozen UCB incumbent;
- at most two to four small residual candidates representing materially
  different data partitions or learning hypotheses;
- the immutable historical archive of all rejected models and evidence.

Do not create a population from dozens of near-identical hyperparameters or
random seeds.  A candidate must differ by a declared hypothesis, data view, or
complete run/seed partition.  Use a fixed small committee only to measure
disagreement.  Unanimous action agreement and independent empirical support
grant shadow eligibility, not active authority.

Replay the whole small population on every Wine prefix.  Test only one active
canary at a time so survival changes remain attributable.  Prune or retain
candidates only on fixed disjoint Wine panels; never repeatedly select against
the same holdout.

## Ordered implementation, Stage 6 through Stage 1

For each Stage, beginning with Stage 6:

1. implement deterministic action-stream recording/replay and compare the
   same seed/action stream between native Linux source and MinGW source under
   Wine; report the first physical snapshot/HIT divergence;
2. audit existing frozen-UCB Wine prefixes with episode-grouped failure-region
   clustering;
3. separate repeated generic regions from single-run RNG events;
4. generate targeted multi-seed headless COW only for repeated unresolved
   regions;
5. build at most three small residual candidates using only exactly
   reproducible Wine features and complete run/seed group splits;
6. replay every candidate against all eligible existing and new Wine prefixes;
7. run one selected extremely-low-intervention candidate in new disjoint Wine
   shadow episodes;
8. after clean shadow evidence, alternate frozen-UCB and one active-canary
   first-failure trials; stop the canary immediately on a native contract
   violation or clear survival regression;
9. only after first-failure survival improves, run natural full Stage trials
   and seek repeatable no-HIT/no-Bomb completion.

Advance from Stage 6 to Stage 5 and then through Stage 1 only with separate
data, models, evaluation panels, and promotion records.  A Stage-specific
candidate never silently shares data or state with another Stage.

## Promotion evidence

Offline and headless evidence can reject a candidate but cannot promote it.
Wine shadow can reject a candidate but cannot prove its counterfactual action
helped.  Promotion requires active original-retail Wine evidence with:

- immutable frozen incumbent and candidate identities;
- alternating/disjoint trials rather than one unrelated route;
- native-safe and fresh-issue action contracts intact;
- zero Bomb request and exact input/process cleanup;
- no regression in capture/stale/release/latency behavior attributable to the
  candidate;
- improved no-HIT completion rate or time/frame to first terminal;
- later, repeated natural full-Stage NMNB completion.

Wine remains closer to shipped execution than reconstructed Linux.  Real
Windows is still the final equivalence gate for a product claim, but the
Wine-first loop is the authoritative development and promotion environment for
the Stage 6-to-1 program.

## Immediate next branch scope

The new branch starts with no large fit.  Its first deliverables are:

1. a bounded deterministic action-stream schema and recorder;
2. native Linux source replay and MinGW source-under-Wine replay using the same
   seed and action stream;
3. an exact first-divergence report with hashes and physical state fields;
4. an episode-grouped audit of the already validated Stage 6 Wine prefixes.

Only after these four deliverables are verified may targeted COW generation or
residual training begin.

The source-platform differential and action recorder are now implemented and
measured in `SOURCE_PLATFORM_DIFFERENTIAL_STAGE6_2026-08-10.md`.  The result
shows exact discrete action/RNG delivery but accumulating subpixel physical
drift beginning at tick 441; the next active deliverable is the episode-grouped
audit of existing Stage 6 Wine prefixes.  No model training has resumed.

The episode-grouped audit is now recorded in
`WINE_FAILURE_REGION_AUDIT_STAGE6_2026-08-10.md`.  It reduces 1,290 correlated
positive frames to 23 independent authority-failure episodes, separates 19
episodes in repeated contexts from four singletons, and queues only three
bounded families for targeted multi-seed COW.  No residual has been trained.

The first targeted result is now recorded in
`WINE_FIRST_STAGE6_TARGETED_COW_2026-08-10.md`.  Generic headless COW did not
confirm the proposed sub10 alternative.  Deterministic replay then reproduced
two complete Wine prefixes and verified exact checkpoint native hard sets;
three independent Wine-anchored 600-frame branches produced two incumbent
wins and one tie.  The sub10 residual has zero candidates.  The next active
gate was the same bounded replay/COW audit for sub31, not model fitting.

That late-family audit is now complete in
`WINE_FIRST_STAGE6_LATE_FAILURE_AUDIT_2026-08-10.md`.  One exact sub31
checkpoint strongly favored the incumbent `left` over `up_fast`.  The second
independent sub31 prefix and both repeated sub18 prefixes could not reach their
Wine checkpoints in reconstructed source because the old corpus omitted the
actual Ctrl/Shoot delivery edges inside dialogue capture gaps.  Linux source
and MinGW source-under-Wine agreed on the resulting premature source HIT, so
more Linux compute cannot recover that missing original-retail evidence.

All three queued Stage 6 regions therefore have zero residual candidates.
Frame schema v5 now retains a bounded, Bomb-free dialogue-delivery sample and
attaches it to the next coherent battle frame without creating a transition or
learning row.  The next active gate is a small new frozen-UCB original-retail
Wine first-failure panel collected with that schema.  First use the new panel
to verify exact retail/source delivery; only repeated, exact COW support may
create a residual.  No candidate, shadow policy, active canary, or new fit is
currently authorized.

The frame-v5 Wine panel is now complete and audited in
`WINE_FIRST_STAGE6_FRAMEV5_PANEL_2026-08-10.md`.  Across three independent
episodes, corrected tick-aligned replay matched every retained dialogue input
in both source domains.  Run one still exposed a pre-dialogue MinGW
source-versus-retail RNG divergence, while runs two and three matched retail
discrete state through their full prefixes.

The panel also found a COW preflight bug: it compared the synchronous source
STEP safe set `(0,)` against the retail Wine Hard set, whose publication
coverage is `(0,1,2,3)`.  COW-v2 now reconstructs the retail set with the
retail delivery contract while retaining synchronous delivery only inside the
source branch.  Two exact independent anchors then disagreed on the strongest
`down_right` to `down_fast` hypothesis under the robust outcome buckets; the
second repeated pair already favored its incumbent at an exact anchor.  The
completed panel therefore has zero residual candidates.  It cannot enter
shadow or active canary, and it must not be repeatedly threshold-mined.  The
next work is a separately predeclared small frozen-UCB Wine panel with no fit
between episodes, looking only for a new repeated generic opportunity.

That second panel is now complete in
`WINE_FIRST_STAGE6_PANEL2_RESULT_2026-08-10.md`. Three immutable original-retail
Wine prefixes reached the same late sub31 context with zero physical HIT before
native fail-close. The only newly repeated action pair favored frozen UCB at
its first exact checkpoint, making unanimous alternative support impossible;
no residual was created and no further COW compute was spent on that pair.

The panel also narrows the simulator boundary. Linux source and MinGW
source-under-Wine retained equal action/RNG streams, and every sampled retail
dialogue input matched, but both source domains temporarily reported zero
bullets where original retail reported 289--308. Therefore equal RNG and
delivery are not sufficient proxies for equal hazard state.

The required same-frame audit is now complete in
`WINE_FIRST_STAGE6_ENEMY_MOTION_AUDIT_2026-08-10.md`. It found a concrete
offline-infrastructure error: the old portable observation omitted enemy
interpolation state, so the solver conservatively but incorrectly extrapolated
the current enemy velocity as constant. The corrected additive observation and
bounded projector restore exact native-safe-set agreement at all 19,120
retained r4--r6 snapshots in both source domains. Hazard-count and numeric
platform drift still remain, so this agreement authorizes only locally matched
reject-only COW. It does not authorize a candidate or make source replay a
promotion domain. Re-running the one predeclared COW pair with the repaired
state still favored frozen UCB and retained zero candidates.

The next experiment must be separately predeclared and small. A bounded
exploratory scan may branch every retail-native-safe first action at the
already-fixed r5 discovery anchor, then use the already-fixed r6 anchor only to
confirm a unique non-incumbent winner. That scan is hypothesis generation, not
retroactive panel-2 evidence; at most one survivor may enter a new disjoint
original-retail Wine shadow episode. No broad fit, sweep, or panel-2 threshold
mining is authorized.

The decisive Stage 6 score remains complete natural original-retail Wine HIT
count versus the immutable frozen incumbent, including whether the stage is
cleared with zero HIT. First-failure survival and headless COW only decide
which candidates deserve that expensive alternating full-Stage A/B; they do
not replace it. Full-Stage HIT continuation remains benchmark-only and may
not flow back into fitting.

The bounded first-action scan is now closed in
`WINE_FIRST_STAGE6_FIRST_ACTION_SCAN_RESULT_2026-08-10.md`. At the fixed r5
anchor, frozen UCB's `down_right` was the unique robust winner across all 14
retail-native-safe alternatives. The conditional r6 confirmation was therefore
not run, and the population remains empty.

Do not respond by scanning adjacent Panel-2 frames. The current branch tool
substitutes one action and then follows `NativeOfflineTeacher`; it does not
continue the immutable frozen UCB policy. The next infrastructure audit should
determine whether frozen-UCB decision state can be restored at a retail replay
checkpoint and advanced under source COW. Only if exact factual continuation
is demonstrated should a separately predeclared policy-faithful action
comparison run. Otherwise future intervention evidence must come from new
disjoint original-retail Wine shadow episodes.

The feasibility audit now passes in
`WINE_FIRST_STAGE6_POLICY_CONTINUATION_AUDIT_2026-08-10.md`. Additive current
boss/callback/spell metadata reconstructs the exact resident context without
future ECL interpretation. At the fixed r5 checkpoint, source replay restored
6,175 prior immutable-UCB calls, matched all eight subsequent lookup keys and
actions, and reached the same Hard-empty stop at the same frame. A small
policy-faithful one-step counterfactual may now be predeclared at that anchor;
this does not relax the disjoint Wine shadow or complete-Stage HIT-count gates.

The policy-faithful discovery is closed in
`WINE_FIRST_STAGE6_POLICY_COW_RESULT_2026-08-10.md`. Its factual branch matched
the eight-action Wine suffix and terminal frame. Both `stay` and `stay_fast`
then survived 600 ticks, but their robust outcomes tied, so the unique-winner
gate produced zero candidates and the conditional r6 confirmation was not run.
The two focus variants may only seed a newly predeclared neutral-action family
test on disjoint Wine episodes; they may not be merged or selected from Panel 2.

That disjoint test is frozen in
`WINE_FIRST_STAGE6_NEUTRAL_PANEL_PROTOCOL_2026-08-10.md`: collect exactly two
new first-failure Wine episodes without intervening adaptation, select the
closest eligible generic sub31 anchor in each by a fixed rule, and test
`down_right`, `stay`, and `stay_fast` under restored frozen-UCB continuation.
Only the same unique neutral winner in both episodes can become one headless
hypothesis. It still requires a new Wine shadow and active canary before the
decisive complete natural Stage HIT-count A/B.

The r7/r8 neutral panel is closed in
`WINE_FIRST_STAGE6_NEUTRAL_PANEL_RESULT_2026-08-10.md`. r7 first failed in
sub18 and r8 in sub10, so the fixed sub31 selector found zero anchors and COW
was not run. A current-kernel r1--r8 audit reproduced 36,106 frozen-UCB calls
without mismatch and found sub10 support four, sub31 support three, and sub18
support one, but no new repeated action pair. Future work must not mine an
alternate r7/r8 row; the only next bounded counterfactual is a separately
predeclared policy-faithful continuation at an already fixed old anchor.

That counterfactual is fixed in
`WINE_FIRST_STAGE6_SUB10_POLICY_COW_PROTOCOL_2026-08-10.md`. It revisits no
selection decision: r1 sequence 2904 is discovery, r2 sequence 3193 is a
conditional confirmation, and only `down_right` versus `down_fast` is tested.
The new question is whether a one-action substitution helps when restored
frozen UCB owns every later choice, rather than when the offline teacher owns
the continuation. Any unanimous result remains a headless hypothesis only.

The fixed sub10 pair is rejected in
`WINE_FIRST_STAGE6_SUB10_POLICY_COW_RESULT_2026-08-10.md`: `down_right` and
`down_fast` both reproduced a seven-tick Hard-empty outcome and tied under the
robust rank. The conditional r2 run was therefore forbidden. At most one final
predeclared all-native-safe scan may inspect the unchanged r1 anchor; do not
select another pair iteratively from this result.

That final scan is frozen in
`WINE_FIRST_STAGE6_SUB10_SAFE_SCAN_PROTOCOL_2026-08-10.md`. It branches the 14
already recorded retail-native-safe first actions under restored UCB and uses
the unchanged unique robust-winner rule. Only one winner may proceed to the
fixed r2 confirmation; a discovery tie or incumbent win closes r1 permanently.

The exhaustive result is closed in
`WINE_FIRST_STAGE6_SUB10_SAFE_SCAN_RESULT_2026-08-10.md`. `up_right` uniquely
extended r1 from seven to 45 source ticks, but the native gate excluded it at
the fixed r2 checkpoint, so confirmation failed closed and the candidate count
is zero. This rules out a fixed action residual; any later learner must rank
the current native-safe set action-relatively and use episode-grouped holdout.

The corresponding label-support panel is frozen in
`WINE_FIRST_STAGE6_SUB10_ACTION_RELATIVE_PANEL_PROTOCOL_2026-08-10.md`. It uses
exactly one lag-seven checkpoint from each of four current-kernel sub10 Wine
episodes and branches the complete recorded native-safe set under restored
UCB. Three unique non-incumbent episode winners are required merely to permit
a separately predeclared fit of at most three small residual candidates.

The first action-relative panel is closed as insufficient in
`WINE_FIRST_STAGE6_SUB10_ACTION_RELATIVE_PANEL_RESULT_2026-08-10.md`. r2 wrote
a sealed document, while r3 failed before branching because restore counted an
`input-lease` carried proposal as a fictitious policy call. The implementation
now skips every non-`ok` decision and a factual r3 audit restores 3,327 calls
with zero mismatch, but the old panel remains closed and permits no fit.

The clean corrected rerun is separately predeclared in
`WINE_FIRST_STAGE6_SUB10_ACTION_RELATIVE_PANEL_V2_PROTOCOL_2026-08-10.md`.
It repeats all four fixed documents from new output paths under the repaired
restore code and preserves the original three-of-four unique non-incumbent
support threshold. The sealed v1 r2 file is not reused.

v2 is closed in
`WINE_FIRST_STAGE6_SUB10_ACTION_RELATIVE_PANEL_V2_RESULT_2026-08-10.md`. Its
corrected r1/r2 documents are valid, but r3's 14-action Hard set collapses to
the single factual `down_left` local action. The protocol had crossed the
learner authority boundary and was rejected before alternatives. COW now
preflights requests against the recorded local set; v2 permits no fit.

The authority-correct completion is separately frozen in
`WINE_FIRST_STAGE6_SUB10_ACTION_RELATIVE_PANEL_V3_PROTOCOL_2026-08-10.md`. It
reuses the exact corrected r1/r2 documents, treats r3's single factual local
action as a negative support unit, scans only r8's complete recorded local
set, and preserves the original three-of-four fit threshold.

v3 passes exactly in
`WINE_FIRST_STAGE6_SUB10_ACTION_RELATIVE_PANEL_V3_RESULT_2026-08-10.md`.
r1 `up_right`, r2 `down`, and r8 `up_left_fast` are unique non-incumbent robust
winners; r3 has only factual `down_left` locally admissible. The differing
directions rule out a fixed action rule but permit a separate fit protocol for
at most three generic action-relative candidates. No candidate is active yet.

The first possible implementation, an exact fine-context lookup of the three
positive Wine anchors, was rejected in
`WINE_FIRST_STAGE6_EXACT_LOOKUP_REJECTION_2026-08-10.md`: it activated only
three times in 36,106 r1-r8 policy calls, each time on its own training anchor,
with zero independent-episode reuse.  The next reject-only accelerator is the
fixed eight-seed, policy-faithful source panel in
`WINE_FIRST_STAGE6_SUB10_HEADLESS_SUPPORT_PROTOCOL_2026-08-10.md`.  Headless
support cannot promote a policy; complete natural retail-Wine Stage 6 HIT
count remains the final metric.

That source panel is closed in
`WINE_FIRST_STAGE6_SUB10_HEADLESS_SUPPORT_RESULT_2026-08-10.md`.  All eight
seeds supplied valid policy-faithful COW checkpoints, but unique
non-incumbent support was 0/4 development and 1/4 confirmation episodes.
The gate failed and candidate count remains zero.  The next run is therefore
the evaluation-only current-kernel complete-Stage incumbent calibration
predeclared in
`WINE_FIRST_STAGE6_CURRENT_INCUMBENT_FULL_STAGE_PROTOCOL_2026-08-10.md`.
It measures HIT count directly but its post-HIT continuation cannot train a
policy or replace a later fresh alternating A/B control.

The current-kernel calibration is complete in
`WINE_FIRST_STAGE6_CURRENT_INCUMBENT_FULL_STAGE_RESULT_2026-08-10.md`: frozen
UCB completed Stage 6 with 10 physical HITs, zero Bombs, and all ten HITs
preceded by a native Hard-empty interval.  This is the current point baseline,
not a powered comparison; the historical 8/9-HIT controls used a different
native DLL identity.  No candidate reached the full-Stage gate, so the next
data acquisition returns to a small first-failure Wine panel rather than
training on the evaluation-only continuation trace.

The next bounded physical acquisition is fixed in
`WINE_FIRST_STAGE6_FRESH_PANEL_R9_R10_PROTOCOL_2026-08-10.md`: two immutable
frame-v5 first-failure episodes, followed by an r1-r10 factual replay and
episode-grouped region audit.  It cannot reopen closed pairs or use the
10-HIT continuation trace for training.  A new COW protocol is allowed only
for an untested generic region with support from at least two physical
episodes, including at least one new episode.
