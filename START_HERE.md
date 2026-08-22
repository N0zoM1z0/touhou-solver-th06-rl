# Start here

Read in this order:

1. `AGENTS.md` — non-negotiable project rules;
2. `paper/main.tex` — research questions, method, and experiment ledger;
3. `docs/ONLINE_OFFLINE_SAFETY_CONTRACT.md` — exact responsibility boundary;
4. `docs/IMMUTABLE_WINE_DATA_PLANE.md` — reusable factual episode contract;
5. `docs/PORTABLE_WINE_RUNTIME.md` — machine-independent Wine setup;
6. `docs/WINE_RETAIL_VALIDATION.md` — original-game runner details.
7. `docs/WORK_LOG.md` — ignored operational experiment narration.

E0 is complete at commit
`af9900524520b72934a4c55e2f44118f88094633`: Practice Stages 4--6 and one
natural-RNG six-stage Lunatic route all completed and passed their factual
audits. The full route retained 108 HITs and zero Bomb.

E2/L0 is complete at commit
`2818861f4079bebfbd8443638ed0cb34236bd5e0`: the causal decision-epoch
builder, current-observation feature parity, masked linear BC fixture, and
immutable exporter passed the repository test suite. Read-only replay of the
canonical E0 Practice Stage 4 episode conserved all 18 HITs across 24,385
decision epochs; the complete E0 route conserved all 108 HITs across 126,468
decision epochs. Excluded rows and HITs remain explicit.

Current checkpoint:

1. keep the audited E0 recorder/shield and corpus schema frozen;
2. retain the negative fixed-seed serial/parallel differential from commit
   `76782a4f37e0d12a9a2384561b53e68ceaf998ae`: every run passed its own audit,
   but exact factual/HIT equality failed, so no parallel pool is admitted;
3. retain the completed negative E3/L1 result from commit
   `559572a8f3699c06eb41080e6061e579a1156c33`: all 12 serial Stage 4 episodes
   passed on their first attempt, but the current-observation BC passed the
   held-out NLL criterion and failed the frozen calibration criterion;
4. retain the read-only L1 diagnosis: train and validation are both strongly
   under-confident, the frozen train gradient remains 19.1% of its initial
   norm, portable roots reproduce the reactive baseline exactly, and a real
   floating ECE-bin defect is far too small to change the negative decision;
5. retain the completed inconclusive offline L1b convergence follow-up from
   commit `7285ee76fe36eda1470844b3635eaddd64292d23`. It reused the exact L1
   corpus and split and improved every held-out score, but exhausted 2,000
   updates at gradient ratio 0.0196 rather than the frozen 0.01 threshold;
6. retain the completed negative offline L1c timebox extension from commit
   `171d92b95c55331ab23bec73226b6b81de6f64cf`. It changed only the maximum
   update count, reached the same train-only 0.01 gradient-ratio criterion at
   update 4,631, passed held-out NLL, and failed the unchanged calibration gate;
7. retain the completed read-only L1c residual diagnosis from commit
   `a4e8232fbae5e3c4df9ecb8d7a787c5138d54e32`. A train-only global scale
   failed the calibration limit, current features reconstructed the reactive
   collector exactly, and the frozen linear model reproduced only 66--67% of
   its choices; the selected ablation is one small current-observation MLP;
8. retain the completed negative offline L1d experiment from commit
   `c96845952572f41f4c263706fe76548b6927df7e`. The fixed 114--32--18 ReLU
   MLP converged at update 3,121 and decisively improved held-out NLL over
   converged L1c, but validation ECE 0.076007 failed the unchanged 0.028732
   bound. Its decision is `stop-l1d-small-current-observation-mlp`; no Wine
   canary or value learning is admitted. Do not post-hoc add temperature,
   history, auxiliary targets, IQL, object encoders, or Wine to this result.
9. retain the completed offline-only target-contract diagnosis from commit
   `b12efebd995832fd68669af163255419cd517dc7`. Every recorded propensity
   exactly matched the declared mixture. Sampled-label noise, 500 more hard
   updates, and use of the full propensity target did not meet their frozen
   explanatory thresholds. Current features reconstruct the collector exactly,
   but L1d reproduced only 74.9% of its validation actions and nearly failed
   the rare focused/lexical tie stages. The selected next experiment is one
   structured current-observation scorer; no history, Wine, or value learning
   is admitted.
10. retain the completed inconclusive L1e fit from commit
    `ec24e8ad05a6e25513ad0468e22bb53617ebc338`. The seven-parameter shared
    action score exhausted 10,000 updates at gradient ratio 0.011684 rather
    than 0.01. It was also worse than L1d: validation KL 0.986686, reactive
    agreement 65.5%, and zero focused/lexical final-tie agreement. Its formal
    decision is `inconclusive-l1e-shared-action-optimization-not-converged`;
    no Wine canary is admitted. Do not extend or repair it post hoc.
11. retain the completed positive offline L2 factual-probe pilot run at commit
    `2c2af520b041804cf3a4036adc4d8fd0c3fe4175`. It reused the immutable L1
    8/4 whole-episode split and ran no Wine. On 94,162 held-out roots,
    published and first executed actions agreed exactly and the executed-action
    delta-x/delta-y MSE was 0.00732 times the global comparator. HIT had
    insufficient support at 1 and 4 frames but passed the complete-episode
    Brier gate at 16 and 64 frames; observed-shield contraction passed at all
    four horizons. The decision is
    `proceed-current-observation-factual-signal`. This is predictive
    representation evidence only: it admits neither history nor a deployable
    policy, value learner, Wine canary, or gameplay claim.

Four preregistered research fits have completed, but none passed its
promotion gate or is an authorized learned candidate. No learned policy has
been run online. A complete route with many HITs is still a successful
infrastructure baseline when the factual and control contracts pass.

The research goal is complete-route Lunatic NMNB probability. The first value
optimization target is expected undiscounted physical HIT count over complete
episodes; it is a dense surrogate and is not claimed equivalent to the goal
metric. The completed current-observation BC improved held-out action NLL over
action frequency but failed calibration. The bounded diagnosis attributes the
first unresolved boundary to optimization convergence rather than missing
capture or validation drift. L1b then improved validation NLL from 2.014976 to
1.720755 and ECE from 0.124251 to 0.055148, but its gradient ratio 0.0196 did
not reach the frozen 0.01 convergence requirement. The result is inconclusive;
L1c then reached that train-only criterion at update 4,631 and improved
validation NLL to 1.691664, but its ECE was 0.059713 against the unchanged
0.028732 limit. Insufficient optimization time is no longer a sufficient
explanation. The linear current-observation candidate is rejected under the
joint gate; short history and offline value learning remain blocked pending a
single-variable representation experiment. The read-only residual diagnosis
then rejected a global scale as sufficient on train, proved that current
features exactly determine the reactive collector, and selected one small
current-observation MLP. No scaled validation distribution was inspected.
L1d then showed that the fixed nonlinear representation materially improves
held-out NLL, accuracy, and Brier over converged L1c, but its train and
validation ECE remained high and the joint supervised gate failed. This exact
MLP is rejected as an online candidate; the result neither demonstrates a
need for history nor admits a Wine canary or value learning. The subsequent
target-contract diagnosis verified every captured behavior distribution
exactly. Against that exact distribution L1d still had validation KL 0.563604
and calibration error 0.077039. Five hundred more hard-target updates reduced
train KL by only 0.009951, while the paired full-propensity target gained only
0.001338 nat over the hard target. The frozen selection rule therefore assigns
the failure to the flat model's inductive bias and selects a shared
action-conditioned current-observation scorer. Future probability gates should
use the captured full distribution and a proper score directly; this
prospective correction does not revise any negative L1--L1d decision.
The frozen L1e ablation was the smallest test of this attribution: one shared
linear scorer sees only seven action-relative facts already derivable from the
114-vector. It has no hidden layer, history, auxiliary head, value target, or
new data. It did not converge and missed every representation gate, performing
worse than L1d on train and validation. Its negative stationary, focused, and
lexical weights expose the deeper mismatch: a global additive score cannot
apply lower lexicographic priorities only inside their rare higher-field tie
strata. The behavior distribution and reactive winner are already exactly
known from capture and portable current features. They should remain data-plane
controls rather than forcing a sequence of larger BC models. The next bounded
question is factual action/execution and HIT-risk learnability, not another
collector-imitation architecture.
The completed factual-probe pilot is the smallest executable form of that
question. Its immediate movement label comes from the first contiguous raw
Wine transition and its witnessed `executed_action`, not from an aggregated
decision interval. Its risk labels use fixed game-frame windows and factual
`life_lost` outcomes only. The successful current-root probe shows that a
portable transparent vector contains held-out factual risk signal. It does not
show that current observation is sufficient for optimal control or that the
action features, rather than state features, cause the improvement. The next
bounded work should decompose calibration, lifecycle/support, and incremental
action signal before selecting one confirmatory risk model. History, an object
encoder, IQL, and Wine evaluation remain unadmitted.
Survival critics, IQL/CQL
sweeps, object/recurrent encoders, ensembles, active collection, learned MPC,
and Wine branching are not starting points.

The first training inventory was collected serially with
unchanged retail clock/update semantics, coherent capture suspension, the
frozen declared 20% uniform observed-shield mixture, and complete-episode
admission. Its episode count, seeds, stopping rule, and whole-episode splits
were frozen before collection or fitting. This initial inventory was not
uncertainty-guided or adaptive collection.

Before reconsidering parallel or exact-prefix branch collection, investigate
and pass the generic coherent-root/action-delivery determinism boundary. Do not
relax exact equality or edit gameplay logic in response to a failure location.
