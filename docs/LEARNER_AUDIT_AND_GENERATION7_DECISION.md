# Learner failure audit and Generation-7 decision

Date: 2026-08-13

Status: authoritative analysis and research direction; not gameplay evidence

## Scope

This record consolidates three inputs:

1. the repository's Generation-1--6 Wine results and implementations;
2. an external GPT audit supplied as `/tmp/th06_rl.txt`;
3. an independent code, contract, corpus, and literature audit performed before
   the Generation-7 prune.

The temporary GPT file is not a durable dependency. Its claims are summarized
and adjudicated here. No corpus, fitted artifact, or Wine outcome was modified
during the audit.

## Terminal fact

No historical learner established repeatable improvement under normal-speed,
natural-RNG, complete-Stage original-Wine evaluation. The last candidate
reached 42 physical HITs versus the incumbent's 34 after four complete Stage-6
blocks. Infrastructure, native safety, decision serving, and latency checks
were clean, so this was a learner failure rather than authority or delivery
failure.

The final actor optimized, per factual row,

```text
E_mu[L] + (w(A) - 1) L(A)
```

The expression is unbiased as a control-variate estimate for a fixed model,
but it is not a proper empirical actor objective. A factual coefficient may be
negative, allowing optimization to lower the loss without bound by driving the
factual action probability toward zero. Production diagnostics matched this
failure: mean actor risk moved `+0.188 -> -9.579 -> -56.638` while Stage-6
behavior KL moved `2.83 -> 6.52 -> 46.13`. The implementation, its tests, and
all executable Generation-1--6 paths were therefore removed.

## Findings accepted from the GPT audit

- The last actor objective was mathematically invalid for ERM.
- Historical critics primarily learned common state risk; stable
  action-relative ranking remained weak.
- State sufficiency must be tested rather than assumed, especially under
  partial observability.
- Physical legality, statistical support, and learned forecast uncertainty are
  different masks and must not share authority.
- A bounded proper actor, exact deployment-policy evaluation, episode-level
  cross-fitting, and action-effect falsification must precede more Wine play.
- Richer causal history and permutation-invariant hazard encoders are valid
  ablations, but they are hypotheses rather than prerequisites.

## Independent findings that strengthen or correct the audit

### Future-length leakage was real and stronger than documented

The historical offline state added four episode-position values. Three used
the final option count: normalized index, normalized remaining count, and
`log1p(remaining)`. Final option count is post-treatment and unavailable at
deployment. It was also highly outcome-revealing: Pearson correlation between
episode option count and full-Stage HIT was `-0.887` on Stage 4, `-0.891` on
Stage 5, and `-0.502` on Stage 6. Only raw causal information such as the
current option index may be retained, and only with explicit availability
metadata.

### The corpus has randomized signal; the inferential unit is the bottleneck

The registry-selected sequential corpus contains:

| Quantity | Audited count |
| --- | ---: |
| complete episode groups | 56 |
| factual options | 167,250 |
| manifest physical HITs | 2,044 |
| HIT-positive factual option intervals | 2,043 |
| factual nonbaseline assignments | 52,448 (31.36%) |

A diagnostic binary IPW smoke comparing factual nonbaseline assignments with
baseline assignments found negative next-HIT differences from one through 32
options (for example `-0.00336` at horizon 1 and `-0.14515` at horizon 32,
using episode-clustered standard errors). This is not a candidate-policy value
estimate and mixes behavior policies, but it rejects the claim that the corpus
contains no action signal. The unresolved question is whether it supports a
stable contextual ordering among all native-safe actions.

Rows help learn representation and proximal conditional effects. Independent
episodes determine policy-value uncertainty and generalization. Both scales
must be reported; neither 167,250 rows nor 56 groups alone describes support.

### There is no single pooled behavior policy

The sequential data mixes at least these factual collection laws:

- a 0.10 incumbent/uniform mixture;
- a 0.50 incumbent + 0.25 uniform + 0.25 information mixture;
- a 0.50 actor + 0.25 uniform + 0.25 inverse-ESS mixture.

It also mixes 19 Stage-4, 4 Stage-5, and 33 Stage-6 episodes. Estimating one
unqualified `Q^mu` from the pool is therefore ill-defined. Generation 7 must
either define a shared reference policy or condition nuisance functions on the
logged source/cohort and relevant causal scope. Source/stage may be nuisance or
evaluation strata; it may not select a handwritten route.

### Existing action geometry is already action-conditioned

The adapter already records action direction/focus, endpoint, viability,
clearance, and rank at horizons 1, 2, 3, 4, 6, 8, 10, and 12. The representation
may still miss topology or longer causal history, but "no action-conditioned
geometry" is inaccurate. Baseline-relative deltas and state sufficiency should
be tested before committing to a large Deep Sets/GRU model.

The online hazard list's 256-item cap was reached in only about `0.02%--0.23%`
of audited cohort observations. Truncation is therefore unlikely to be the
first-order bottleneck, although set encoding remains a useful ablation.

### Scientific boundary corrections

- RNG seed and source/ECL context may be used only for privileged diagnostics,
  never for movement or the deployable policy.
- Offline geometry may score every current native-safe action against already
  observed hazards, but it may not invent future births, dynamics, or successor
  states for actions Wine did not execute.
- A learned forecast may rank or cause abstention; it cannot become collision
  authority.
- FQE and sequential doubly robust estimation are cross-checks, not oracles.
- Fixed paired seeds are diagnostic only. Final evidence uses natural unread
  RNG with alternating incumbent/candidate blocks.
- Cost-versus-reward sign is not itself a repair; consistency and property
  tests are what matter.
- Concurrent Wine collection is disabled. The fixed-seed differential changed
  HIT counts (`28` serial versus `30` and `26` concurrently), frame/boundary
  counts, and digests. Offline fitting and replay may parallelize.

## Generation-7 architecture decision

Generation 7 proceeds in three falsifiable layers.

### G7-A: causal and deployment contract repair

1. Every feature has availability metadata: source, earliest availability,
   online/offline scope, and whether it is permitted for deployment.
2. A linter rejects post-treatment/future-length, RNG, source-control, and
   non-deployable features from actor inputs.
3. Cost is factual physical HIT only, `gamma = 1`, terminal value zero. Property
   tests cover sign convention, terminal handling, HIT conservation, and
   episode boundaries.
4. The actor objective is a proper weighted maximum-likelihood/KL objective
   with finite lower bound and nonnegative weights. An extreme-logit smoke must
   show that suppressing a factual action cannot improve its loss without
   bound.
5. Define a common deployable reference distribution, initially

   ```text
   pi_ref(a|s) = (1-epsilon) 1[a=a_inc]
               + epsilon Uniform(A_safe(s)).
   ```

6. The actor emits the complete residual stochastic distribution. Fitting,
   direct/DR/FQE evaluation, shadow, and native deployment consume the same
   policy object. There is no deterministic proposal followed by a separate
   thinning sampler.
7. Physical-safe, statistically-supported, and forecast-risk masks remain
   distinct. Only the first has action-publication authority.

Proper nonnegative advantage-weighted maximum likelihood follows the core
supervised extraction idea in [Advantage-Weighted Regression](https://arxiv.org/abs/1910.00177).

### G7-B: prove action-effect identifiability first

The primary comparison is predeclared:

1. baseline-relative orthogonal/direct advantage learning using known factual
   propensities;
2. one-step constrained improvement around the shared reference policy;
3. repaired canonical IQL critic plus proper AWR policy extraction as a
   challenger.

[Offline RL Without OPE](https://arxiv.org/abs/2106.08909) motivates constrained
one-step improvement empirically; it is not a project-specific guarantee.
[IQL](https://arxiv.org/abs/2110.06169) avoids training-time queries of unseen
actions, but still depends on state sufficiency, behavior mixture, support, and
policy extraction.

Required gates:

- factual action-permutation null;
- reward-suffix-permutation null;
- synthetic delayed causal effect with known sign;
- episode-bootstrap sign and policy stability;
- action-specific direction stability across source/stage strata;
- compact state versus progressively richer causal state;
- exact deployed-policy direct, sequential-DR, and FQE cross-checks;
- exact action distribution, fallback, mask, and teacher/student conformance.

[Sequential doubly robust estimation](https://arxiv.org/abs/1511.03722) can
reduce error under suitable nuisance/support conditions, but it cannot rescue
simultaneously wrong propensities, support, and models.

### G7-C: collect only for a demonstrated support gap

New collection is forbidden until G7-A/B show an aggregate effect, an adequate
causal state, and a specific unsupported action/context region under an
otherwise proper learner. A collection contract must then predeclare:

- a generic eligibility rule;
- incumbent versus one to three algorithmic challengers;
- exactly one option deviation followed by incumbent continuation;
- balanced, recorded, complete propensities;
- independent natural episodes plus separately marked fixed-RNG diagnostics;
- no spell, frame, HIT-location, route, or counterexample selection.

The autonomous data flow is:

```text
immutable D0 -> fit pi1 -> constrained Wine D1 -> append facts
             -> refit D0 union D1 -> pi2
```

Evaluation episodes used to decide a candidate are not silently recycled into
that same candidate's training set.

## Priority decision

The leading hypothesis is not simply "too little data" or "the network cannot
understand bullets." The first-order problems are the causal feature contract,
the undefined mixed-behavior target, and policy extraction. Generation 7 must
separate and falsify those three before richer set/history models, new Wine
collection, or renewed concurrent-Wine work.
