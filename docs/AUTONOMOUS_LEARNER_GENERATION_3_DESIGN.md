# Autonomous learner generation 3 design

## Decision and boundary

Generation 2 remains frozen and ineffective. Generation 3 is a new,
game-neutral algorithm generation motivated by aggregate evidence, not by a
TH06 location, spell, frame window, RNG seed, or hand-selected action. No new
Wine generation-3 outcome may be inspected before this contract and its
implementation tests are committed.

The retained architecture is:

`Wine safe options -> grouped offline learning -> immutable residual scorer -> Wine multi-seed canary -> natural complete-Stage evaluation`

Original retail Wine remains the only source of successor states, rewards,
intervention outcomes, and promotion evidence. Native geometry remains the
only authority over legal actions. Learning may rank or abstain; it may not
add an action to the safe set, alter the collision margin, request Bomb, or
invent a counterfactual successor.

Generation 2 established that its model learned state danger but not a stable
action advantage. On its round-2 held-out episodes, future 60-frame HIT-risk
AUC was 0.701, while the mean within-state legal-action prediction range was
0.0163 and mean committee standard deviation was 0.0273. Only 28 of 23,992
states passed the robust override bound. Fixed-RNG canary override rate was
0.196%, but natural-RNG final override rate was 0.0063%, a 30.8-fold observed
coverage collapse. These aggregate facts define the new algorithm problem;
they do not authorize a failure-specific exception.

## 1. Matched learner correctness and validation

Every exported model must be fitted to the last reported Bellman target. The
fit loop may not update a target after the final exported fit. Validation
predictions and comparators must use exactly the same physical horizon and
terminal/continuation semantics as the exported value.

The primary offline diagnostics are grouped and cross-fitted:

- matched-horizon factual target calibration;
- temporal-difference residual by complete held-out episode;
- action-advantage calibration and conformal coverage;
- effective sample size for every action and option duration;
- proposal coverage and abstention decomposition on untouched episodes;
- native scorer equivalence and latency.

Offline diagnostics may reject a candidate but never promote it. A finite
number with mismatched semantics is not a valid gate. Generation 3 also fixes
the generation-2 implementation defects where the last updated target was not
refitted and the long-horizon Q prediction was compared with a 60-frame cost.

## 2. Safe temporally extended option exploration

The randomized treatment unit is an option, not an independent one-frame
impulse. At an option boundary the behavior policy mixes the frozen incumbent
with a uniform draw over the complete native-safe action set. The selected
movement intent may persist for at most eight physical frames.

Safety is not leased for eight frames. On every physical frame the controller
recaptures Wine, reconstructs the native-safe set, and freshly certifies the
intended action. An option terminates immediately on source-unsafe intent,
Hard-empty, incoherent capture, physical HIT, stage transition, or its fixed
eight-frame horizon. Fail-closed release and the incumbent remain available at
all times.

The corpus records a stable option ID, boundary flag, selected intent,
boundary propensity, elapsed frames, termination reason, and the factual
per-frame actions actually published. Continuation frames have conditional
propensity one given the recorded boundary treatment; they are not counted as
new randomized assignments. The exploration mixture remains 0.10. Option
duration, mixture, and termination rules are generation constants and may not
be changed after observing gameplay outcomes.

## 3. Cross-fitted doubly robust residual advantage learning

Generation 3 predicts relative treatment effect rather than asking one model
to recover a small action difference on top of large common state risk.

Whole episodes form deterministic cross-fitting folds. For each fold, nuisance
models are fitted without that fold to estimate factual continuation value and
action-conditioned option outcome. The recorded behavior propensity supplies
the correction term for a multi-action doubly robust pseudo-outcome. The
learned deployment quantity is

`A(s, option) = Q(s, option) - Q(s, incumbent option)`.

Bellman updates operate on option boundaries as a semi-Markov process and use
physical elapsed frames. HIT remains the sole gameplay cost. No Hard-empty,
clearance, survival bonus, stage progress, source phase, or teacher score is
added to reward. Native geometry and self-supervised representation targets
may be inputs or auxiliary losses, never reward terms.

The learner may use a large offline teacher, distributional outcomes, and
cross-fitted nuisance ensembles. Only a compact immutable residual scorer and
its calibrated uncertainty artifact enter Wine.

## 4. Game-neutral hazard and history representation

The TH06 adapter exposes normalized observed hazard primitives through the
shared environment interface: player-relative position, velocity, collision
extent, finite lifetime/age when observed, and generic geometry kind. It does
not expose ECL, boss/spell identity, frame windows, RNG, or source phase.

A permutation-invariant set encoder consumes the observed hazards. A short
four-observation history consumes only factual Wine observations, published
actions, and elapsed frames. Candidate native trajectory profiles remain
available. This lets the offline teacher distinguish spatially different
hazard fields that share the same minimum-clearance summary while keeping the
interface reusable for TH08.

The teacher is distilled into a bounded native batch scorer. The online call
must reuse the state embedding across all candidates, score only the current
native-safe set, allocate bounded memory, and satisfy the predeclared latency
gate of at most 4 ms p95 per decision with zero controller-deadline misses.
Python or a training framework may not run in the resident scoring hot path on
Windows.

## 5. Calibrated uncertainty and pessimistic selection

Raw episode-bootstrap range is not treated as a calibrated confidence
interval. Generation 3 uses whole-episode cross-fitting residuals to calibrate
action-advantage intervals at 90% nominal coverage. Calibration data is never
used to fit the corresponding fold's prediction.

At runtime every locally supported candidate receives a pessimistic upper
bound on HIT-cost advantage. Selection examines all supported candidates and
chooses the smallest pessimistic bound; it does not choose by mean first and
then test only that action. An override is allowed only when the chosen
candidate's upper advantage bound is strictly below zero. Otherwise the policy
abstains to the incumbent.

Support and uncertainty thresholds come from the declared grouped calibration
procedure. Humans may not lower them after seeing a missed opportunity. The
fit requires at least nine independent training episodes before active canary;
frame count alone cannot satisfy this requirement.

## 6. Multi-seed and natural-RNG evidence

Generation-3 collection uses a deterministic seed schedule committed before
outcomes. The initial fit boundary is 12 complete stages with three whole
episodes held out. If no candidate passes, four new complete stages are added
per round, up to a fixed maximum of 24 collection stages. Fit boundaries are
therefore 12, 16, 20, and 24 stages. Existing generation-2 collection may be
used for representation pretraining and contract development, but its canary
and final runs are never learner or generation-3 evaluation data.

Every active candidate receives three sequential paired fixed-RNG complete-
Stage canaries on distinct precommitted seeds. Authorization requires:

- six clean complete stages;
- candidate exercise in at least two pairs;
- strictly fewer aggregate candidate physical HITs;
- candidate no worse in at least two of three pairs;
- no safety, authority, accounting, scorer, or latency regression.

Only an authorized candidate enters final evaluation. Final evaluation runs
12 natural-RNG complete stages per arm in the fixed alternating order
baseline/candidate. It uses normal Wine timing, continue-on-HIT, zero Bomb, no
training corpus, and immutable policies. All 24 stages run; there is no
outcome-dependent early stop. Efficacy requires strictly fewer aggregate
candidate HITs, a candidate exercised in at least six of its stages, and clean
infrastructure. Per-stage results, rate ratio, uncertainty interval, and
override coverage are reported even though the predeclared aggregate rule is
authoritative.

## Automation and stopping rule

One crash-resumable runner owns collection, grouped cross-fitting, shadow,
canary, final evaluation, cleanup, and atomic state. It may stop only for a
reproducible infrastructure failure, a successful final evaluation, or the
24-stage collection budget ending without authorization. Humans inspect after
that terminal state.

Poor play, low proposal coverage, or a rejected candidate does not authorize
manual data redistribution. A future change to option duration, observation,
reward, learner, uncertainty calibration, or evidence gates is a new algorithm
generation declared before new outcomes.
