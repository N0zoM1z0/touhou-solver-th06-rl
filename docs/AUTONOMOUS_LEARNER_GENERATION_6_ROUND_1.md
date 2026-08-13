# Generation 6 autonomous round 1

## Decision and status

This is the first unattended collect-refit-evaluate round after the positive
Generation-6 Stage-6 pilot. Its purpose is to test whether a reusable offline
RL learner improves with additional original-Wine facts, not to repair an
observed gameplay location. The machine contract is frozen before collection;
after freezing, outcome-facing code, schedules, seeds, gates, and mixtures do
not change.

The machine contract is `config/autonomous_generation6_round1.json`, SHA-256
`d733d919726393b60d243b1be2501cc0a57888b1c8e588ed34d257a0aa081a52`.
It binds implementation checkpoint `c1db773`, the base registry, complete
source-game inventory, original executable, Wine binary, controller/scorer
binaries, current Generation-6 actor, all learner/runner inputs, every policy
seed, and the exact evidence schedule. The two policy files differ from their
pre-authorization hashes only by adding this contract digest to their explicit
allowlists.

The round state machine is:

`12 Wine Stages -> append registry -> all-corpus cross-fit/refit -> full native smoke -> 2 Wine canaries -> 12 Stage-6 A/B Stages`

All Wine executions are original Japanese retail 1.02h at normal speed.
Training collection uses natural unread game RNG, complete Practice Stages,
patched-life HIT continuation, and zero Bomb. Final evaluation uses the same
natural-RNG complete-Stage HIT-continuation semantics without writing corpus.
No headless transition, reward, or outcome enters any step.

## Frozen collection behavior

Collection contains four complete Stages each from Stage 4, Stage 5, and Stage
6 in a balanced predeclared serial schedule. The Stage choice is coverage of
the generic adapter at three trajectory lengths; it is not conditioned on an
observed HIT, spell, phase, frame, or RNG. A single isolated Wine worker is
used because the earlier original-speed concurrency differential failed. CPUs
0--7 are reserved for the game and 8--31 for capture/controller/scoring, so the
round never occupies more than the user-authorized 32 CPUs.

At each eight-frame factual option boundary, the behavior policy sees only the
controller's native-safe action set and generic adapter features. Its exact
probability distribution is:

`0.50 * frozen G6 actor proposal + 0.25 * uniform safe action + 0.25 * inverse-ESS information allocation`

Inverse ESS uses only earlier published assignments and their recorded
propensities. It cannot inspect Stage, source context, boss, spell, frame, RNG,
HIT, failure location, or later outcome. Every safe action receives at least
`0.25 / safe_set_size`; the complete probability vector, selected probability,
information weights, and running ESS are recorded. The policy cannot enlarge
the native-safe set and Bomb is absent from the action vocabulary.

Infrastructure retries repeat the same scheduled Stage and immutable behavior
state only after a strict capture/process/provenance failure. A gameplay HIT is
not a retry reason. Each accepted episode must conserve physical HITs across
option intervals and the manifest, have a complete factual successor chain,
record every propensity, contain no infrastructure event, and clean its Wine
prefix. The atomic move-to-registry ledger is crash recoverable: a restart
revalidates the already moved run instead of drawing another natural-RNG game.

## Data, learner, and fit boundary

The twelve accepted runs become one new immutable registry source. No existing
source is copied, removed, relabelled, or filtered. Both cross-fit and
production fit query every training source that declares
`sequential_offline_rl`; consequently the new fit consumes all 44 currently
registered sequential episodes plus all 12 new episodes.

The reward remains physical HIT only. The learner uses factual semi-Markov
transitions, `gamma = 1`, terminal value zero, cross-fitted n-step Q/value
targets, an action-centered IQL actor with shared action representation, and a
seven-member whole-episode bootstrap population. It receives no handcrafted
survival, phase, spell, frame, or route reward. Model family, seeds, backup
horizon, support, intervention probability, and uncertainty calculation remain
fixed for the whole round.

Synthetic causal/null smoke runs first. Cross-fit is grouped by complete Wine
episode and reports Stage 4, 5, and 6 cohorts separately. The deployable policy
must have a negative overall episode-bootstrap HIT-effect upper bound, a
negative worst leave-one-member-out upper bound, negative model effect and at
least half beneficial episodes in every Stage cohort, non-degenerate population
proposals, bounded policy intervention exposure at or below 10%, and density
correction at or below two. These are rejection gates, never promotion
evidence.

Production fit preserves the complete seven-member population. Portable and
Linux native choices must be exact on 64 factual contexts, prototype support
must remain within tolerance, and full native p95 must be below 4 ms with zero
60 Hz deadline miss. The complete fused policy path is then exercised on the
same computational-width factual selection under both Linux and the actual
32-bit SSE2 DLL through Wine; both must exactly match portable proposals and
meet the same latency/deadline gates. A preflight state is shadow-only and
cannot authorize intervention. Only the conjunctive audit can create an active
canary state.

## Wine evidence and stopping rule

If offline or native smoke fails, the round stops as `offline-rejected` and no
candidate gameplay is launched. If it passes, two natural-RNG complete Stage-4
active canaries must be clean and exercise at least one intervention. The final
panel then runs six alternating Stage-6 blocks, one incumbent/shadow Stage and
one active candidate Stage per block, for twelve complete Stages total.

An `effective-learning-signal` requires all twelve audits to pass, candidate
intervention in at least four of six candidate Stages, fewer aggregate
candidate HITs, and candidate no worse in at least four of six blocks. The
small confirmation remains `promotion_eligible: false` regardless of sign. A
negative result triggers learner/infra analysis or the next autonomous round;
it never authorizes manual collection around a spell, frame, RNG, action, HIT,
or failure location.

## 2026-08-13 pre-freeze replay

The implementation was exercised on the unchanged 44-episode registry before
any new Wine collection. The five-fold replay reused 143,078 cached factual
options and completed in 369.70 seconds on CPUs 0--31. Overall cross-fitted DR
HIT effect was `-4.4457`; its 95% episode-bootstrap upper bound was `-3.6571`,
and the worst leave-one-member-out upper bound was `-3.5038`. Stage 4 and Stage
6 independently retained negative bounds. Stage 5 is absent from the old
registry, so it is deliberately required from the balanced new collection
rather than manufactured by a manual data edit.

The first production smoke exposed two general diagnostics defects and failed
closed. Extreme logits made the behavior KL serializer underflow to infinity;
the calculation now uses stable log-softmax. A pure `1e-4` actor-score absolute
gate then rejected a model whose logits reached about 1,448: Linux native error
was exactly `0.000244140625`, all 64 actions were exact, and the smallest native
decision margin was `0.06715`. The repaired gate is `1e-4` absolute plus four
float32 epsilons relative, while exact action equality remains mandatory.

Checkpoint-resumed production smoke then passed: error consumed 0.532 of that
bound, support error was `1.27e-6`, 64/64 actions were exact, p95 was `1.532 ms`,
and all 1,200 scores met the frame deadline. A raw synthetic Win32 kernel check
still exceeded the score bound on this larger-scale actor. Because raw logits
are not the deployed decision, it was not waived; the frozen round instead
requires full fused-policy exact proposal equality on factual Wine contexts
before canary. These repairs change neither corpus, reward, fitted checkpoint,
nor gameplay distribution.

Historical audited-option caches remained valid: changing the collection
mixture did not change the stable complete-propensity interface identifier, and
the distinct state schema, contract hash, metrics, and registry capability bind
the new behavior. Thus learner iteration reuses the 7.6 GiB cache rather than
rebuilding old facts. The CPU training environment also now contains the
already pinned CPU build of PyTorch 2.8 required by the IQL actor; dependency
availability is checked before Wine time is spent.
