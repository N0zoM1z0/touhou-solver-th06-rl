# Wine-only autonomous learning

## Product decision

Original retail gameplay under Wine is the sole environment that creates
transitions, rewards, or evaluation outcomes. The reconstructed Linux/headless
runtime is retired. The active loop is:

`Wine exploration -> episode-grouped offline learning -> immutable candidate -> Wine canary`

The complete system must run repeated learning rounds without a person watching
play or adjusting individual failure cases. This document is authoritative over
historical artifacts and Git history.

## Separation of responsibilities

The environment adapter owns game-specific facts: coherent memory capture,
action delivery, HIT/lifecycle accounting, and conversion into a versioned
observation/action interface. Native geometry owns the complete publishable
Bomb-free safe action set and the fresh issue check. Unknown or incoherent
hazards fail closed.

The learning system owns only generic quantities: observation vectors,
native-safe action sets, randomized action propensities, factual transitions,
episode groups, rewards, uncertainty, and candidate selection. It may rank or
abstain inside the safe set; it cannot enlarge that set. Online state is
immutable and fitting happens offline.

A TH08 port should supply another adapter and scope configuration while reusing
the exploration, transition schema, grouped fitting, shadow/canary gates,
orchestration, and evaluation aggregation unchanged.

Collected facts are not owned by a learner generation. The immutable Wine data
plane, replaceable algorithm, and fitted model artifact are separate contracts.
Learners query the capability registry for the minimum semantics they need and
reuse every compatible episode; they may not recollect data merely because the
model family changed. See `IMMUTABLE_WINE_DATA_PLANE.md`.

## No human gameplay tuning

After a run, poor survival or a high HIT count is training data, not permission
to edit the policy. In particular, do not add or tune any of the following from
observed failure cases:

- stage, boss, spell, phase, frame-window, RNG-seed, or run-specific routes;
- bullet-count, screen-region, action-count, clearance, or boundary thresholds
  that decide where learning is allowed;
- hand-selected alternative actions or failure-region data quotas;
- reward bonuses/penalties chosen to repair a particular pattern;
- activation rules keyed to a counterexample or named gameplay situation.

Exploration distribution, feature schema, reward definition, model family,
validation split, promotion gates, and evidence budget are versioned before a
round begins. They remain fixed for the declared run. Any algorithm change
starts a new generation and is justified by general held-out evidence, never a
single gameplay location.

Human changes are allowed only for demonstrated infrastructure defects:
incorrect memory semantics, incoherent snapshots, capture/action timing,
native geometry or collision safety, factual transition/label alignment,
process isolation, artifact integrity, or HIT accounting. Each repair must have
a reproducer, a contract test, and an audit note. If those contracts pass, the
response to weak play is another autonomous data/fit round.

## Unattended round state machine

One resumable orchestrator owns the state below and writes every transition
atomically:

1. **Collect** sequential original-Wine episodes with a frozen behavior policy
   and algorithmic propensity-recorded exploration inside the native-safe set.
2. **Validate** provenance, complete transition windows, propensities, zero
   Bomb, capture/delivery contracts, and exact process cleanup. Contract failure
   stops as `infra_failure`; policy failure does not.
3. **Fit** from all accepted experience using episode-grouped train/validation
   splits. Adjacent frames from an episode never cross groups.
4. **Shadow** the immutable candidate on held-out Wine observations. Unsupported
   or uncertain actions abstain to the incumbent. Shadow is screening evidence,
   not causal promotion evidence.
5. **Canary** on disjoint Wine episodes only when predeclared support and
   held-out gates pass. Safety/authority regression rejects automatically.
6. **Evaluate** promising candidates with alternating, complete normal-speed
   Wine Stages in HIT-continuation mode.
7. **Decide**: promote only on the declared HIT aggregate; otherwise collect the
   next round until the evidence budget is exhausted. Never change gameplay
   rules between these states.

The process may be interrupted and resumed without replaying completed work.
Training collection may use concurrent original-Wine workers only after a
normal-speed differential compatibility gate. Each worker must have an
isolated game directory, Wine prefix, display, artifact directory, and corpus
root; each remains paced by the original 60 Hz executable and passes the same
per-run audit before merge. Canary and final evaluation remain single-instance,
alternating, normal-speed jobs. Offline fitting may parallelize.

There is currently no authorized learner-generation runner or gameplay
candidate. `scripts/run_generation6_autonomous_round.py` and the other
generation runners are retained as frozen audit implementations, not as the
current command. Generation 6 is conclusively rejected; its actor objective
must not be refit. The next learner starts as a separately frozen Generation-7
contract on the existing registry and must pass learner-only qualification
before any outcome-facing Wine run. See `HAND_OFF.md` for that boundary.

Every active candidate is hash-chained:

`fit state -> held-out shadow audit -> bounded canary state -> canary audit -> full-evaluation state`

The online state loader rejects a missing or stale link. Each generation
predeclares its canary exposure budget and causal promotion rule; only a clean
canary audit can authorize a complete-Stage evaluation.

## Exploration and factual learning

The bootstrap exploration rule is game-neutral: when the native-safe set has
more than one action, a seeded policy-local random stream applies a fixed,
predeclared probability of selecting uniformly from that set; otherwise it
uses the incumbent. The recorded probability is the exact probability of the
published action under the mixture. No location or hazard threshold selects an
opportunity. Later active collection may use model uncertainty only if its
formula and budget were declared before the generation began.

Every accepted sample records the observation, safe set, incumbent and chosen
actions, behavior probability, next factual observation, termination/HIT, and
episode/scope/provenance IDs. An action Wine did not execute has no successor
label. Fixed RNG can reduce training variance but does not create a
counterfactual pair; different physical roots remain different episodes.

The sole gameplay cost is the factual physical HIT count, with `gamma = 1` and
terminal value zero. Survival time, progress, phase, graze, clearance, and
continued availability of safe actions are diagnostics or observations, never
reward shaping. A generation freezes its learner, propensity treatment,
grouped validation, and uncertainty contract before fitting. Model complexity
is unrestricted offline provided the exact deployable policy remains bounded,
immutable, and fast enough online.

For option-boundary complete-Stage returns, a rejected tentative option is not
a treatment or a new decision boundary. Physical HITs that appear while
learning is paused for the normal death/invulnerability lifecycle remain part
of the preceding factual interval and all earlier returns. Prefix HITs before
the first factual boundary are reported separately. Transition HITs, factual
interval HITs, prefix HITs, and manifest HITs must conserve exactly; accounting
may neither discard a HIT nor assign one to an action Wine did not execute.

## Runtime and promotion invariants

- Capture is coherent and publication is preceded by a fresh issue check.
- Bomb bit `0x02` is forbidden.
- The learned component ranks only native-safe actions and can abstain.
- No online `observe`, weight update, or checkpoint mutation occurs.
- No movement depends on game RNG, frame, run identity, or handwritten phase.
- Fixed RNG, acceleration, snapshots, and first-failure runs are training tools
  only; they cannot promote a candidate.
- Every Wine worker is resource-isolated and exact cleanup is verified after
  every episode; canary and final-evaluation Wine jobs are sequential.

Final promotion uses original retail Wine at normal timing, without fixed RNG,
from a natural complete Practice Stage start through termination, with HIT
continuation enabled. Policies are immutable, Bomb remains zero, and incumbent
and candidate trials alternate. The authoritative measure is each run's total
physical HIT count followed by the predeclared aggregate. Offline loss,
first-failure survival, or shadow agreement may reject a candidate but cannot
promote it.

## Generation stopping rule

Before collection begins, record the maximum round/episode budget, grouped
validation gate, canary count, and complete-Stage A/B count. Stop for human
inspection only when:

- a reproducible infrastructure contract fails;
- the evidence budget ends without a supported candidate (`ineffective`); or
- a candidate passes canary and lowers the declared complete-Stage physical HIT
  aggregate without safety/latency regression (`effective`).

No intermediate offline metric is a verdict.

## Current evidence

Generation 1 completed two unattended Wine collection/fit rounds and was
rejected by its unchanged shadow gate. See
`AUTONOMOUS_GENERATION_1_RESULT.md`. Its negative result does not alter the
autonomous-learning boundary above. Generation 2 replaced the learner and
completed its factual complete-Stage contract. Its second round passed a
fixed-RNG canary, but normal-speed natural-RNG evaluation ended at baseline 17
HITs versus candidate 18, so it is frozen as ineffective. See
`AUTONOMOUS_LEARNER_GENERATION_2_DESIGN.md` and
`AUTONOMOUS_LEARNER_GENERATION_2_RESULT.md`. A later attempt is a new declared
algorithm generation, not a failure-region adjustment. Generation 3 is frozen
as superseded without canary authorization after its complete-return AIPW
estimator produced structurally high variance; see
`AUTONOMOUS_LEARNER_GENERATION_3_RESULT.md`. Generation 4 is declared before
new outcomes in `AUTONOMOUS_LEARNER_GENERATION_4_DESIGN.md`. It retains the
same Wine/native/HIT boundaries while replacing the estimator with sequential
semi-Markov offline RL, generalized action centering, autonomous propensity-
aware exploration, policy-level cross-fitting, and a full native population.
It completed 16 new Wine Stages but produced no stable held-out advantage and
never earned canary authorization, so it is frozen as ineffective; see
`AUTONOMOUS_LEARNER_GENERATION_4_RESULT.md`. Generation 5 used factual in-sample
implicit Bellman iteration, lower cost expectile value fitting, and supported
pessimistic population selection. Its Stage-4 boundary-15 smoke learned
held-out Bellman structure but failed to identify a stable action, so
collection stopped without a candidate. See
`AUTONOMOUS_LEARNER_GENERATION_5_RESULT.md`. Generation 6 separated critic and
policy extraction, accumulated a 56-episode sequential registry view, and
preserved important native-serving and performance repairs. Its larger clean
Wine confirmation nevertheless reached 42 candidate HITs versus 34 incumbent
HITs. Offline diagnosis then proved its action-centered empirical actor
objective unbounded below, so Generation 6 is frozen as ineffective and must
not be refit. See `GENERATION6_DECISION_GAMEPLAY_RESULT.md` and `HAND_OFF.md`.

Generation 7 must first qualify a bounded proper learner on frozen factual Wine
corpora split by complete episode. These offline checks exist to reject broken
algorithms quickly and cannot reinterpret historical gameplay as authorization
or promotion evidence. Only after the learner architecture and its gates are
frozen may it collect new Wine evidence.
