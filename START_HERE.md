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
12. retain the completed positive-but-weak L2b attribution run at commit
    `3e01cfd4c8cc0a255f9ff0209da32c21ae291d14`. Removing exactly the nine
    action-relative features showed incremental full-model HIT Brier signal at
    16 frames with interval `[-5.95e-5, -4.68e-7]`, but not at 64 frames, whose
    interval crossed zero. The full model was less calibrated than state-only
    at both HIT horizons. Action-relative shield-contraction signal was robust
    at all four horizons and in every validation episode. The decision is
    `proceed-action-relative-current-root-signal`, but these were reused
    validation episodes: this is attribution, not independent confirmation,
    causal action value, or permission for Wine, history, or IQL.
13. retain the completed L2c support/lifecycle diagnosis recorded at commit
    `c17839cbaed0643e567193ee8dc2e630931b70cf`. It reproduced the source
    scores exactly, localized the old h16 increment to baseline-equal rows,
    retained a favorable pre-first-HIT point direction, and froze a fresh
    confirmation. It fit no model and admitted no history or value learner.
14. retain the completed L2d fresh confirmation recorded at commit
    `c2d285a47c56e5dae0278fc837585394481f070b`. All eight new serial Stage 4
    episodes passed on attempt one,
    retained 91 physical HITs and zero Bomb, and independently reproduced the
    h16 full-versus-state Brier increment in all eight episode directions. The
    known-nonbaseline point gate passed weakly, the pre-first-HIT gate passed,
    and calibration readiness failed. Its decision is
    `confirm-predictive-signal-calibration-not-ready`.
15. retain the completed negative offline L2e calibration experiment run at
    commit `9004e387249f1a1b3a2c2c92e3da05333a470733`. A train-only
    two-parameter Platt surface converged, reproduced the L2d source scores
    exactly, and passed the frozen calibration limits, but worsened Brier
    against both uncalibrated full and state-only. It also reversed the
    nonbaseline and pre-first-HIT point deltas. Its decision is
    `reject-train-only-platt-calibration`; no fresh confirmation, policy,
    history, or value learning is admitted.
16. retain the completed negative-but-informative offline L2f experiment run at
    commit `7289949466c4ce38b3f1b43d853fa4bd72e3d02d`. The direct 64-tree,
    depth-3 action-conditioned h16 model improved reused-L2d Brier from
    `0.00551139` to `0.00480785` and beat its same-architecture state-only
    ablation at `0.00500959`; both complete-episode intervals were wholly
    favorable and calibration readiness passed. The frozen joint gate still
    rejected it: the comparator had an invalidly clipping raw surface, the
    nonbaseline interval narrowly crossed zero, and pre-first-HIT directions
    were heterogeneous. No fresh confirmation or Wine run occurred.
17. retain the completed negative offline L2g experiment run at commit
    `adb355875a0c008fc057881bad2c7f99e08e4302`. Exact bounded
    uniform-shield target weighting passed its recorder contract but did not
    improve target-weighted Brier over frozen L2f and did not stabilize the
    overall, low-propensity, nonbaseline, or pre-first-HIT action increments.
    The full raw surface also clipped 13.53% of evaluation rows. Its decision is
    `reject-weighted-h16-hazard`; action-measure mismatch is rejected as the
    primary L2f root cause. No fresh confirmation or Wine run occurred.
18. retain the completed negative offline L2h experiment run at commit
    `bbf6e81a3203fb00443060a215949a37fdb13352`. Its exact unit-frame 16-root
    portable history improved reused-L2d point Brier to `0.00479407` from
    same-row current-only `0.00483026`, and current action fields beat their
    history-preserving ablation in all eight episode directions. The temporal
    interval crossed zero with only five favorable episodes, however, while
    low-propensity, nonbaseline, and pre-first-HIT action intervals also crossed
    zero and the full raw surface clipped 2.43% of rows. Its decision is
    `reject-fixed-history-h16-hazard`; no fresh confirmation or Wine run
    occurred.
19. retain the completed negative L2i experiment run at commit
    `b8344f60e3d0d654e1bc23b2108cf4ea7231cd9a`. Its cap-48 primitive
    contract, zero truncation, 8,929-parameter export bound, 0.388 ms scorer
    p99, and overall action ablation passed. Object-full evaluation Brier
    `0.00473906` was worse than scalar-only `0.00468783`; the object interval
    crossed zero with five favorable episodes. Low-propensity, nonbaseline,
    and pre-first-HIT action deltas reversed with only two favorable episodes.
    The full sigmoid surface saturated 65.05% of rows and underpredicted event
    rate by `0.003094`. Its decision is
    `reject-observed-primitive-set-h16-hazard`; no fresh confirmation, Wine,
    policy, or value fit occurred.
20. retain the frozen, not-yet-run L2j loss-only experiment in
    `experiments/l2j-logscore-primitive-hazard-v1.json`. It retains the exact
    L2i data, tokens, architecture, initialization, minibatch order, optimizer,
    epochs, comparators, and evaluation gates, changing only direct sigmoid
    Brier to unweighted BCE-with-logits. In addition to every L2i gate, its new
    paired gate requires Brier below frozen L2i object-full with a complete-
    episode bootstrap upper endpoint below zero and at least six favorable
    episodes. It must reproduce frozen L2f and L2i scores exactly. This tests
    the observed rare-positive vanishing-gradient/saturation mechanism; it is
    not scalar calibration, class weighting, a loss sweep, or evidence that
    the object set works.

Multiple preregistered research fits have completed, but none passed its
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
That decomposition and fresh confirmation are now complete. Fresh episodes
confirmed the h16 incremental predictive signal, including the frozen weak
nonbaseline point and pre-first-HIT boundaries, but failed calibration
readiness. The first train-only probability-surface test then showed that a
global monotone Platt mapping can repair calibration-in-the-large and ECE while
simultaneously destroying the Brier advantage. Scalar calibration of the
frozen ridge score is therefore rejected. The first aligned direct probability
model established that the nonlinear hazard family is learnable: it strongly
improved Brier, ranking, calibration, and the overall action-relative ablation.
Its formal gate nevertheless failed on the low-propensity/nonbaseline and
pre-first-HIT boundaries. Propensity stratification localized the missing
action increment to the bulk of randomized actions below behavior probability
0.025, while baseline-heavy rows improved in all eight episodes. The bounded
propensity-corrected test then passed its exact weight and calibration contracts
but failed its own target-score, exploration-action, lifecycle, and candidate-
surface gates. It worsened uniform-target Brier versus frozen L2f and retained
the baseline-equal signal pattern. Logged action measure is therefore not the
primary root cause. The subsequent exact 16-frame scalar history used temporal
facts and improved its point score, but temporal gain was episode-heterogeneous
and its action increment again vanished on exploratory and pre-first-HIT rows.
That scalar-history candidate is rejected. The subsequent bounded current-root
instantiated-object set improved ranking and the aggregate action ablation, but
failed object gain, exploration/lifecycle action, calibration, and saturation
gates. Direct sigmoid Brier drove 65.05% of evaluation rows below probability
`1e-7` and underpredicted the event rate by `0.003094`. The next single-variable
boundary is unweighted BCE-with-logits on the otherwise identical L2i design,
because it retains a corrective gradient for rare positives while targeting
the same conditional event probability. Value learning and Wine remain blocked
until a hazard candidate passes selection and wholly fresh confirmation.
Survival critics, IQL/CQL sweeps, recurrent encoders, ensembles, active
collection, learned MPC, and Wine branching are not starting points.

The first training inventory was collected serially with
unchanged retail clock/update semantics, coherent capture suspension, the
frozen declared 20% uniform observed-shield mixture, and complete-episode
admission. Its episode count, seeds, stopping rule, and whole-episode splits
were frozen before collection or fitting. This initial inventory was not
uncertainty-guided or adaptive collection.

Before reconsidering parallel or exact-prefix branch collection, investigate
and pass the generic coherent-root/action-delivery determinism boundary. Do not
relax exact equality or edit gameplay logic in response to a failure location.
