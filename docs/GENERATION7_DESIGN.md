# Generation-7 learner-only design

Status: frozen offline contract; G7-A complete; G7-B failed/stopped; no Wine
outcome-facing authorization

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
- AWR weights are finite and nonnegative. Weighted negative log likelihood plus
  nonnegative reference KL is bounded below by zero and must pass the
  extreme-logit smoke.
- One `ResidualStochasticPolicy` produces the complete action distribution for
  fitting, OPE, shadow, and eventual deployment. There is no proposal/thinning
  split.
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

## Stop boundary

No new Wine collection, canary, or evaluation is authorized by this design.
If the existing corpus fails identifiability gates, the result must say whether
the failure is state sufficiency, episode-level uncertainty, action-specific
support, or estimator disagreement. Only a demonstrated specific support gap
can begin a separately frozen G7-C collection design.

## Recorded disposition

The implemented results and stop decision are in `GENERATION7_PROGRESS.md`.
G7-B found structured action signal but failed one-step estimator calibration
and sequential-support gates. IQL, deployment, Wine evaluation, and G7-C are
therefore not authorized by this design.
