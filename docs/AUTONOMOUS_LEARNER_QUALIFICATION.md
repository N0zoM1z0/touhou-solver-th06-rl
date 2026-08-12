# Offline learner qualification

## Purpose

Learner architecture errors must be discovered without paying for another long
original-Wine collection wave. Existing factual Wine episodes therefore form a
non-authorizing qualification funnel:

`synthetic causal contracts -> frozen Wine development -> one untouched Wine qualification -> native latency -> new Wine learning round`

This changes iteration cost, not the environment or the efficacy standard.
Offline replay never invents a successor for an unexecuted action. Historical
episodes can reject a learner but cannot authorize a candidate, promote a
policy, or substitute for normal-speed natural-RNG complete-Stage Wine HIT
evaluation.

## Frozen corpus split

`config/autonomous_generation6_qualification.json` binds two already immutable
Wine report hashes and every admitted run-manifest hash reachable through them.
Within each source, the split is chronological and was selected without HIT,
action, RNG, frame, or failure-location inspection:

| Source | Stage | Development | Qualification |
| --- | ---: | ---: | ---: |
| Generation-3/4 historical corpus | 6 | first 21 | last 8 |
| Generation-5 curriculum corpus | 4 | first 10 | last 5 |
| Total complete episodes | | 31 | 13 |

Adjacent options and frames never cross a partition. Stage cohorts are
reported separately as well as together, so a large Stage-6 corpus cannot hide
failure on Stage 4 or vice versa. Rejected or incomplete Wine attempts are not
reachable from the source reports and cannot enter either partition.

The development partition is reusable while comparing generic learner
architectures. The qualification partition is single-disclosure for Generation
6: its factual samples and learner metrics are not loaded until the candidate
source, hyperparameters, random seeds, numerical gates, and output identity are
hash-frozen. Loading the partition contract to validate paths and manifest
hashes does not disclose samples or learner outcomes.

After its one Generation-6 use, these 13 episodes become development history.
They cannot be called an untouched holdout again. A later generation must
declare a new qualification partition or collect new original-Wine episodes;
statistical independence cannot be restored by renaming a split.

## Three-layer rejection funnel

### Layer 1: deterministic causal contracts

A learner must recover delayed beneficial and harmful action effects in a
semi-Markov process whose only cost is terminal/intervening HIT, preserve exact
HIT conservation with terminal value zero, and abstain in a randomized
null-effect process. These fixtures also check complete propensity centering,
episode isolation, and absence of counterfactual successor labels.

### Layer 2: frozen Wine development

All metrics are cross-fitted by complete episode. At minimum a generation must
predeclare and report:

- factual Bellman target loss against the state-only comparator, overall and by
  Stage cohort;
- policy proposal coverage and native-support abstention;
- stability of the actual full pessimistic population under whole-episode
  bootstrap or leave-one-member-out perturbation;
- policy-level cost-effect uncertainty, rather than the maximum uncertainty of
  every candidate pseudo-label;
- sensitivity to independently frozen training seeds;
- offline fit wall time and effective CPU affinity.

Development failure permits generic model or calibration redesign. It never
permits selecting actions, episodes, stage locations, RNGs, shaped rewards, or
data quotas. Every attempted architecture and result is appended to its
generation progress record so a later iteration cannot quietly forget a
negative result.

### Layer 3: one qualification disclosure and native latency

Only one development-passing candidate is frozen and evaluated on the 13
qualification episodes. It must pass the numerical gates declared in that
generation's design both overall and separately on Stage 4 and Stage 6. The
complete immutable online population must also pass native equivalence, p95
below 4 ms, and zero 60 Hz deadline misses. Failure starts a successor design;
the gates are not relaxed after seeing qualification output.

Passing means only that another Wine learning round is worth its cost. New
propensity-recorded Wine episodes still own causal canary and final efficacy
evidence.

## Generation-6 single-disclosure freeze

`config/autonomous_generation6_candidate_freeze.json` binds the exact
candidate/fit hashes, all learner/evaluator source hashes, source commit,
partition, deterministic seeds, development causal/null and canonical Wine
summaries, native Linux/Windows binaries, latency evidence, output identity,
and numerical gates before any of the 13 qualification trajectories are
loaded.

Overall, Stage 4, and Stage 6 must each exercise a nonzero policy intervention
without exceeding 10% proposal or exposure, keep correction magnitude at most
two, have at least half of complete episode effects below zero, retain a
negative independent model term, and place both the complete policy and the
worst of seven leave-one-actor-out policies below zero at the episode-bootstrap
95% upper bound. All conditions are conjunctive. There is no framewise action
agreement gate: population uncertainty is calibrated on the complete policy's
physical-HIT effect. Failure discloses the set permanently and rejects the
candidate; no observed gate may be relaxed.

## Generation-6 result and consumed status

Freeze commit `522aff7` was pushed before the first sample load. All frozen
gates passed on the 13 qualification episodes. Overall estimated effect was
`-5.2970 HIT/stage` with 95% upper bound `-4.1620`; the worst leave-one-actor
upper bound was `-3.4153`. Stage 4 measured `-6.6269`, upper `-4.9269`, worst
LOO `-4.2145`; Stage 6 measured `-4.4659`, upper `-3.0617`, worst LOO
`-2.3678`. Every one of 13 episode effects was negative, all independent model
terms were negative, and correction magnitude stayed at two. Report SHA-256 is
`1da0212281902daf18c124d3e246a244ae19d4a92fa3177efd34711c460b3e34`.

These episodes are now permanently disclosed development history. The pass
permits canary design only; it does not authorize a policy or become Wine
efficacy evidence.

The first online handoff audit then found that the evaluated local target used
each historical option's recorded behavior propensity to set intervention
probability. That target is statistically well-defined, but a new episode
cannot reconstruct the exact history-dependent v10 behavior propensity. The
qualified numbers remain preserved, but deployment is blocked until an
implementable propensity-lower-bound target is evaluated. Because the set is
already disclosed, that follow-up is explicitly development, never a second
qualification.

The implementable audit passed without changing the actor. It defines
`rho = min(0.10, 2 * 0.10 / native_safe_set_size)`, using the common minimum
uniform mass guaranteed by both admitted v9 and v10 behavior contracts. Overall
effect was `-1.6025`, bootstrap upper `-1.1998`, and worst-LOO upper `-1.0373`.
Stage 4 was `-2.0218` / `-1.5161` / `-1.3036`; Stage 6 was `-1.3405` /
`-0.8285` / `-0.7076`. Twelve of 13 episodes were beneficial, every model term
remained negative, and the largest correction fell to `0.4335`. The
post-disclosure development report SHA-256 is
`f683abf05b0fc1165181c1b922882e8950eae48cb67060e54792e3bc6a86ba8f`.
This target is the only variant eligible for canary design.

## Host-sharing contract

Every offline qualification or fit launcher applies Linux process affinity
before parsing, importing native learner runtimes, or forking workers. It
chooses at most the first 32 CPUs from the process's inherited affinity and
records the effective set. Children inherit the restriction. Library thread
settings may be lower for performance, but they are not the resource authority.
Learner work and Wine gameplay never overlap.
