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

The declared fit uses three deterministic whole-episode cross-fit folds, three
nuisance members per fold, and seven final population members. Members use
whole-episode bootstrap resampling and distinct committed learner seeds. These
counts are algorithm constants, not outcome-dependent search dimensions.
The first-generation teacher uses learner seed 260812, 96 trees per nuisance
member, and 128 trees per population member; changing these after outcomes is
a new learner generation.

Bellman returns operate on option boundaries as an undiscounted semi-Markov
process and retain physical elapsed frames. Undiscounted return is deliberate:
it is the recursive sum of remaining physical HITs and therefore matches the
final complete-Stage hit-count objective without a hand-chosen time preference.
HIT remains the sole gameplay cost. No Hard-empty,
clearance, survival bonus, stage progress, source phase, or teacher score is
added to reward. Native geometry and self-supervised representation targets
may be inputs or auxiliary losses, never reward terms.

The learner does not collapse to a single empirically best fit. It retains a
population whose members differ only through predeclared whole-episode
resampling, cross-fit folds, and learner seeds. Every member sees the same
algorithm, reward, feature contract, and option rules; there is no human-made
specialist or failure-region member. Population disagreement is epistemic
evidence, not a reason to pick the most optimistic member.

The learner may use a large offline teacher, distributional outcomes, and
cross-fitted nuisance ensembles. The deployment artifact preserves the
population's advantage distribution, either as bounded native batch members
or a conformance-tested distillation of its required statistics. It may not
distill only the population mean and discard uncertainty.

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

The bounded adapter projection retains at most the 256 player-nearest observed
primitives under a generic distance ordering. The offline encoder learns 24
prototypes from a deterministic reservoir of at most 65,536 training
primitives, then emits permutation-invariant prototype occupancy/min-distance
and primitive moment pools. These are generation constants fixed before Wine
outcomes; no prototype represents a TH06 phase, source address, or spell.

The teacher is distilled into a bounded native batch scorer. The online call
must reuse the state embedding across all candidates, score only the current
native-safe set, allocate bounded memory, and satisfy the predeclared latency
gate of at most 4 ms p95 per decision with zero controller-deadline misses.
Python or a training framework may not run in the resident scoring hot path on
Windows.

Distillation preserves all seven population members separately. Each 128-tree
teacher member is distilled to a 48-tree depth-four native student; population
mean-only distillation is forbidden. On untouched episodes, per-member
distillation absolute error must be at most 0.05 HIT at p95 and 0.25 HIT at the
maximum. A failure rejects the fit and does not authorize larger runtime
models or relaxed bounds inside this generation.

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

Wine canary outcomes never select an individual population member, seed, or
fold. The only evaluated candidate is the immutable aggregate policy produced
by the predeclared calibration rule. This prevents best-of-population online
selection and repeated-test leakage while retaining useful policy diversity.

Support and uncertainty thresholds come from the declared grouped calibration
procedure. Humans may not lower them after seeing a missed opportunity. The
fit requires at least nine independent training episodes before active canary;
frame count alone cannot satisfy this requirement.

## Pre-collection smoke gates

Long Stage collection cannot be the first end-to-end learner test. Before any
generation-3 evidence episode, automation runs a deterministic causal fixture
whose true option advantage is -1 HIT while state risk changes independently.
All seven population members must recover a negative action advantage, the
population mean must be within 0.5 HIT of the known effect, state-risk leakage
must be below 0.15 HIT, the residual fit must beat the zero-advantage
comparator, and the population may not collapse to identical predictions.

Automation then runs one short, explicitly non-evidence Wine pipeline smoke.
It requires at least 32 valid option boundaries, at least one randomized
non-incumbent boundary, valid known propensities, conditional-probability-one
continuations, a witnessed horizon termination, fresh native-safe membership
for every published intent, clean infrastructure, and valid corpus accounting.
The short trajectory cannot train, calibrate, authorize, or evaluate a policy.
After a scorer exists, the same smoke additionally requires native/teacher
conformance, at most 4 ms scorer p95, and zero controller deadline misses.

The runner makes these checks a hard dependency: a deterministic causal
learner smoke runs first, followed by a 45-second fixed-RNG retail-Wine option
wiring smoke. Only a passing `autonomous-generation-3-preflight-v1` artifact
allows the first complete evidence Stage to launch. The Wine smoke corpus is
marked `evidence_eligible: false` and is excluded from every fit and outcome
comparison.

## 6. Multi-seed and natural-RNG evidence

Generation-3 collection uses a deterministic seed schedule committed before
outcomes. The initial fit boundary is 12 complete stages with three whole
episodes held out. If no candidate passes, four new complete stages are added
per round, up to a fixed maximum of 24 collection stages. Fit boundaries are
therefore 12, 16, 20, and 24 stages. Existing generation-2 collection may be
used for representation pretraining and contract development, but its canary
and final runs are never learner or generation-3 evaluation data.

The immutable schedule is
`config/autonomous_generation3_seeds.json` (generation seed 260812). It fixes
all 24 collection RNG/policy-seed pairs and all four rounds of three canary RNG
pairs before any Generation-3 Wine outcome. The declared experiment is retail
TH06 1.02h, Lunatic Reimu-A Practice Stage 6.

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
