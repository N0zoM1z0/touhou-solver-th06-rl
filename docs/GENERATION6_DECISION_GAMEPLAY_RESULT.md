# Generation 6 decision-successor gameplay result

## Verdict

The float64 decision-serving successor is a clean infrastructure success but
does not provide an effective learning signal. Four complete alternating
Stage-6 blocks were sufficient to reject the frozen conjunctive criterion:
the candidate was no worse in only one block, while the contract required at
least four of six. With only two blocks remaining, the mathematical maximum
was three of six. The run therefore stopped at the first complete-block
boundary where a positive verdict became impossible.

This is an exact rejection of this fitted learner under its frozen rule, not a
claim that four blocks prove a statistically harmful policy. It also does not
invalidate the reusable Wine corpus, the sequential factual transition
contract, native safety, or the float64 serving repair.

## Frozen evidence and exact stopping argument

The gameplay contract SHA-256 is
`93b0060c46160a47a2dc9785a40578131fff2a56ee530cfd74a030d5d0bd35e6`.
It ran from commit `e2a9d7b`, at original Wine pacing, with natural unread RNG,
complete Practice Stage 6, continue-on-HIT, zero Bomb, immutable policy state,
and the exact original-retail executable. Both Stage-4 wiring canaries passed:

| Canary | HIT | Proposals | Interventions | p95 | Deadline misses |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 7 | 607 | 8 | 3.0597 ms | 0 |
| 2 | 9 | 602 | 8 | 3.0815 ms | 0 |

The completed Stage-6 blocks were:

| Block | First arm | Incumbent HIT | Candidate HIT | Incumbent minus candidate | Candidate interventions |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | incumbent | 11 | 11 | 0 | 9 |
| 1 | candidate | 10 | 11 | -1 | 9 |
| 2 | incumbent | 6 | 10 | -4 | 4 |
| 3 | candidate | 7 | 10 | -3 | 7 |
| **Partial total** | | **34** | **42** | **-8** | **29** |

All eight complete evaluation reports passed every frozen gate. Candidate
Stages contained 18,784 option boundaries, 2,259 proposals, and 29 actual
interventions. The 29/2,259 publication ratio of 1.284% agrees with the
declared propensity-bounded sampler; the candidate did not silently remain in
shadow mode. Across both canaries and all eight completed Stage-6 runs, the
largest resident scorer p95 was `3.2269 ms` and the total deadline-miss count
was zero. Incumbent interventions were exactly zero. Native-safe authority,
complete-Stage accounting, immutable hashes, bounded process priority, and
Wine-prefix cleanup all passed.

After block 3:

`observed no-worse + remaining blocks = 1 + 2 = 3 < required 4`

No possible outcomes in blocks 4 and 5 could satisfy the conjunction. Trial 9
had only just started when it was stopped; it has no report and is absent from
the atomic ledger. No partial trajectory is evidence or learner input. The
historical ledger deliberately remains `status: evaluation` with a null
machine decision because the original runner required all twelve rows before
calling `_paired_verdict`. Its eight-row SHA-256 is
`c76e98f8e3fa2bba9eb8c6a41d1f2262b8334de2d96a7117e80f4db8bd847abe`.
The conclusive rejection is the monotone logical consequence above; the
historical ledger is not rewritten after stopping.

## Offline-to-Wine contradiction

The 56-episode cross-fit predicted a Stage-6 mean effect of `-4.9632 HIT`, with
a bootstrap upper 95% endpoint of `-3.7991` and 90.9% beneficial episodes. The
model-only component was `-1.8516 HIT`. Fresh Wine instead observed a partial
mean contrast of `+2.0 candidate HIT/Stage`, because incumbent minus candidate
was `-8` over four blocks. Natural-RNG variance is substantial, but the frozen
positive rule is already impossible and the offline estimate is materially
miscalibrated for this fitted policy.

The production actor proposed on 26.92% of all registered factual options in
the full native replay. Cross-fitted Stage-6 mean-population proposal rate was
21.25% and expected intervention exposure was 1.026% of boundaries. Fresh
Wine candidate runs proposed on 12.03% of boundaries and actually intervened
on 0.154%. The sampler is behaving as implemented; the replay and deployed
state distributions/exposures are not calibrated closely enough for the
reported `-4.9632` value to be treated as a deployment forecast.

## Confirmed learner defect: an unbounded empirical actor objective

Generation 6 fits the actor with the per-row control-variate estimate

`L_hat = E_mu[L] + (w(A) - 1) L(A)`.

For a fixed model this is an unbiased estimate of `E_mu[w(A)L(A)]`. Unbiased
does not make it a safe empirical optimization objective. With two actions,
`mu=(0.5, 0.5)`, factual `A=0`, and `w(A)=0.1`, it is

`L_hat = -0.4 L(0) + 0.5 L(1)`.

Sending the fitted probability of action 0 to zero makes `L(0)` diverge while
`L(1)` approaches zero, so the empirical objective tends to negative infinity.
The neural actor can exploit this on nearly unique physical states. Critic
labels being cross-fitted by episode does not remove the negative coefficient
from actor ERM. A direct probe of the production helper gives empirical losses
`-1.837`, `-5.526`, `-11.052`, and `-27.631` as the factual probability falls
from `1e-2` through `1e-6`, `1e-12`, and `1e-30`; the corresponding proper
weighted cross-entropies remain positive and increase.

The immutable fits show the predicted failure mode as more corpus is reused:

| Actor fit | Episodes/options | Mean fitted centered risk | Member range |
| --- | ---: | ---: | ---: |
| Initial qualified fit | 31 / 102,737 | +0.188 | -1.596 to +1.349 |
| All-registry preflight fit | 44 / 143,078 | -9.579 | -20.347 to -2.741 |
| Round-3 fit used here | 56 / 167,250 | -56.638 | -75.066 to -28.145 |

A true weighted cross-entropy is nonnegative. The increasingly negative
training estimate is accompanied by Stage-6 behavior KL increasing from
`2.83` to `6.52` to `46.13`, mean-population proposal rate increasing from
5.73% to 12.80% to 21.25%, and offline DR optimism increasing from `-1.3405`
to `-3.6075` to `-4.9632 HIT`. This coherent progression is evidence of
objective exploitation, not evidence that adding corpus made gameplay better.

The deployed mean policy also ignores most population disagreement. On the
56-episode Stage-6 cross-fit, dropping one member preserved mean-population
proposal decisions on only 30.14% of the proposal union, and the independent
3-member/4-member split agreed on only 1.42% of contexts where either proposed.
The fully pessimistic seven-member policy proposed on only 28 of 114,580
Stage-6 options. Replacing that abstaining population by an unstable mean made
deployment possible, but it did not establish a trustworthy learned effect.

## Successor boundary

Generation 6 must not be refit again with
`action_centered_actor_losses` as the optimized actor objective. The next
learner may reuse every immutable Wine transition, the fitted critic work, the
game-neutral feature interface, and all native serving infrastructure, but it
must be frozen as a new learner generation and satisfy all of these before new
outcome-facing Wine play:

1. use a bounded proper actor/value objective; an unbiased but unbounded
   finite-sample control variate is forbidden;
2. add an extreme-logit smoke proving the optimized loss cannot improve by
   assigning vanishing probability to a low-weight factual action;
3. cross-fit the complete policy construction and report held-out proper loss,
   proposal stability, and decision-level population uncertainty;
4. deploy the uncertainty rule that was audited, rather than replacing a
   nearly abstaining population with its uncalibrated mean;
5. evaluate the exact deployed stochastic target, and require offline
   proposal/exposure predictions to agree with a fresh incumbent-occupancy
   Wine shadow panel under a predeclared calibration rule;
6. keep physical HIT as the only reward and retain native-safe authority,
   zero Bomb, natural-RNG complete-Stage evaluation, and the prohibition on
   hand-selected spell/frame/HIT/RNG data repair.

No new scenario-specific corpus is justified by this result. The first next
step is learner-only replay on the existing registry. New Wine collection is
appropriate only after a bounded replacement passes synthetic, grouped
offline, decision uncertainty, native equivalence, and live shadow-exposure
smokes.

## Reusable performance consequence

Future evidence runners should predeclare rejection-only logical fail-fast at
complete paired-block boundaries. For every monotone count gate, compute the
best possible value after all remaining blocks; reject when even that bound
cannot pass. Never accept early: a positive result still needs every
predeclared report, aggregate, integrity, latency, and safety gate. This exact
rule saved four unnecessary full Stage-6 runs here without inspecting any
spell, frame, failure location, or RNG outcome to change the policy.

The float64 native serving optimization remains required. A learner change
must retain the 16-worker deterministic full-corpus differential, one-thread
math libraries per worker, the repository-wide 32-CPU cap, exact Linux/Win32
action and support decisions, p95 below 4 ms, and zero deadline misses. The
negative gameplay result is not permission to regress those independently
validated infrastructure properties.
