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
Learners consume every episode admitted under the current source-complete
contract; they may not recollect data merely because the model family changed.
The old schema-only registry is removed and its pre-authority corpora are not
eligible. See `IMMUTABLE_WINE_DATA_PLANE.md`.

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

1. **Collect** original-Wine episodes with a frozen behavior policy and
   algorithmic propensity-recorded exploration inside the native-safe set.
   Collection is sequential until the exact pool-wide gate passes.
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
6. **Evaluate** promising candidates with alternating, complete real-time Wine
   Stages in HIT-continuation mode, without debugger suspension.
7. **Decide**: promote only on the declared HIT aggregate; otherwise collect the
   next round until the evidence budget is exhausted. Never change gameplay
   rules between these states.

The process may be interrupted and resumed without replaying completed work.
Training collection may use concurrent original-Wine workers only after a
fixed-seed differential compatibility gate. Each worker must have an
isolated game directory, Wine prefix, display, artifact directory, and corpus
root; each uses the same coherent process-suspension contract and passes the
same per-run audit before merge. This proves per-update semantics, not 60 Hz
wall-clock control. Canary and final evaluation remain single-instance,
alternating, non-suspending real-time jobs. Offline fitting may parallelize.

There is currently no authorized learner-generation runner or gameplay
candidate on `main`. Failed Generation-1--6 executable paths and their active
registry were removed; they may not be restored as a fallback. The next learner
starts from newly admitted source-complete episodes and must pass learner-only
qualification before any candidate-facing Wine run. See
`LEARNER_AUDIT_AND_GENERATION7_DECISION.md` and `HAND_OFF.md`.

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
command intent, behavior probability, next factual observation, next-root
sampled input, witnessed physical movement action when identifiable,
termination/HIT, and episode/scope/provenance IDs. SendInput publication is
never substituted for a sampled game input. The causal option treatment is the
command intent issued through the same certified pickup mechanism used online;
an old/prefix input during that bounded pickup is a factual treatment outcome,
not a counterfactual successor. Fixed RNG can reduce training variance but does
not create a counterfactual pair; different physical roots remain different
episodes.

The sole gameplay cost is the factual physical HIT count, with `gamma = 1` and
terminal value zero. Survival time, progress, phase, graze, clearance, and
continued availability of safe actions are diagnostics or observations, never
reward shaping. A generation freezes its learner, propensity treatment,
grouped validation, and uncertainty contract before fitting. Model complexity
is unrestricted offline provided the exact deployable policy remains bounded,
immutable, and fast enough online.

For option-boundary complete-route returns, a rejected tentative option is not
a treatment or a new decision boundary. Physical HITs that appear while
learning is paused for the normal death/spawning lifecycle remain part of the
preceding factual interval and all earlier returns. Prefix HITs before
the first factual boundary are reported separately. Transition HITs, factual
interval HITs, prefix HITs, and manifest HITs must conserve exactly; accounting
may neither discard a HIT nor assign one to an unpublished command. Options
whose decision root is already invulnerable remain in the corpus and return
accounting but are excluded from the NMNB actor/critic fit: that post-HIT state
is unreachable under the target no-miss policy. Online and during exploration,
that state exposes only the source-safe reactive baseline with propensity one.
Held-out PDIS carries its incoming prefix weight across this forced interval;
it rejects any other ineligible reason or non-singleton propensity.

## Runtime and promotion invariants

- Capture is coherent and publication is preceded by a fresh issue check.
- Bomb bit `0x02` is forbidden.
- The learned component ranks only native-safe actions and can abstain.
- No online `observe`, weight update, or checkpoint mutation occurs.
- No movement depends on game RNG, frame, run identity, or handwritten phase.
- Fixed RNG and isolated parallel workers may accelerate training collection;
  snapshots and first-failure prefixes are diagnostic tools only. Neither can
  promote a candidate, and first-failure prefixes are not training episodes.
- Every Wine worker is resource-isolated and exact cleanup is verified after
  every episode; canary and final-evaluation Wine jobs are sequential.

Final promotion uses original retail Wine at real-time timing, without fixed RNG
or debugger suspension, from a natural complete Practice Stage start through
termination, with HIT
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

Generations 1--6 did not establish a repeatable complete-Stage physical-HIT
improvement. Their terminal evidence, mathematical failure analysis, corpus
statistics, and valid retained-infrastructure boundary are consolidated in
`LEARNER_AUDIT_AND_GENERATION7_DECISION.md`; executable historical algorithms
are intentionally absent.

The next learner must first qualify a bounded proper objective on newly
admitted factual Wine corpora split by complete episode. These offline checks exist to reject broken
algorithms quickly and cannot reinterpret historical gameplay as authorization
or promotion evidence. Only after the learner architecture and its gates are
frozen may it collect new Wine evidence.
