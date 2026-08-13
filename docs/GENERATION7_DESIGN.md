# Generation-7 learner-only design

Status: causal contract corrected to proposal-level ITT; G7-B stopped at
matched OPE calibration; no Wine outcome-facing authorization

Contract: `config/generation7_offline_contract.json`

## Research question

Can the existing randomized factual Wine corpus identify a stable
baseline-relative action effect when causal availability, mixed behavior
policies, and policy extraction are specified correctly?

Generation 7 does not begin with a larger representation. It first separates
three possible failures: causal state, behavior-target definition, and proper
policy extraction.

## G7-A contract

- Every actor/nuisance/diagnostic feature has explicit availability and allowed
  use metadata. Unknown features fail closed.
- Actor features must exist at the decision and be online-deployable. Episode
  final length/remaining count, future HIT suffix, RNG, source context, cohort,
  and stage nuisance fields are rejected from the actor.
- Raw causal option index may be represented only as `option_index_log`; no
  normalization by final episode length is allowed.
- Factual physical HIT is the only cost, with gamma 1 and terminal value zero.
- The treatment is the randomized proposal assignment. Fresh native
  revalidation/fallback is a post-assignment part of the deployed transition;
  it must be retained and must never be used to filter learner rows.
- AWR weights are finite and nonnegative. Weighted negative log likelihood plus
  nonnegative reference KL is bounded below by zero and must pass the
  extreme-logit smoke.
- One `ResidualStochasticPolicy` produces the complete proposal distribution
  for fitting, OPE, shadow, and eventual deployment. It is composed with the
  same immutable native publication/fallback kernel everywhere. There is no
  second learned or stochastic thinning sampler.
- Native collision safety, statistical support, and forecast risk are separate
  vectors. Learning cannot alter native authority.

The shared reference distribution is a 0.05 incumbent/uniform mixture on the
current native-safe set. Supported non-risky alternatives receive a bounded
KL tilt; other actions retain reference probability rather than receiving an
invented counterfactual safety label.

## G7-B comparison and gates

The predeclared variants are baseline-relative orthogonal/direct advantage,
one-step constrained improvement, and canonical IQL with proper AWR. All
nuisance and policy construction is cross-fitted by complete physical episode.
Behavior source/cohort and stage may enter nuisance strata but not the actor.

The run must emit every gate named in the contract. Direct, sequential-DR, and
FQE estimates are cross-checks; disagreement rejects authorization. One-step
and repeated-policy estimands must be named and compared only to matched
estimators. Sequential DR must report unclipped cumulative-weight ESS and
maximum weight; failed overlap rejects its use. Null tests
must use factual rows only. Synthetic tests may generate abstract transitions
but cannot claim anything about TH06 dynamics.

Before any outcome model is fit, the reference/behavior proposal importance
ratio must be calibrated to mean one overall and within source/stage strata,
using complete episodes for uncertainty. This outcome-free gate prevents
post-assignment compliance conditioning from masquerading as action effect.
Because assignment changes native compliance and time to the next proposal,
per-proposal HIT is not by itself a fixed-exposure gameplay value. Any positive
short-horizon claim must additionally use a predeclared fixed physical-time
outcome or a validated semi-Markov value that accounts for holding time.

## Stop boundary

No new Wine collection, canary, or evaluation is authorized by this design.
If the existing corpus fails identifiability gates, the result must say whether
the failure is state sufficiency, episode-level uncertainty, action-specific
support, or estimator disagreement. Only a demonstrated specific support gap
can begin a separately frozen G7-C collection design.

## Recorded disposition

The implemented results and stop decision are in `GENERATION7_PROGRESS.md`.
The first G7-B report was invalid because it conditioned on native publication
compliance. Correct ITT data removes the apparent long-horizon effect. An
exploratory horizon-1 orthogonal score remains, but matched IPS/FQE/DR
calibration fails and the policy is almost identical to the reference. IQL,
deployment, Wine evaluation, and G7-C are therefore not authorized.
