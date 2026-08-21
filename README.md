# touhou-solver-th06-rl

A Wine-only learning agent for original Japanese TH06 1.02h.

Original-retail Wine is the sole gameplay environment. The online contract is
to pause the exact process, capture one coherent physical state, build a
source-complete Hard-horizon collision envelope, let a native kernel certify
the Bomb-free safe first-action set, rank only that set with an immutable
lightweight policy, and publish through an exact-PID background input bridge
before resuming that same source epoch. Unknown source coverage fails closed.

The learning loop is deliberately asymmetric:

`Wine exploration -> grouped offline learning -> immutable candidate -> Wine canary`

Wine supplies every factual outcome. Offline jobs reuse Wine trajectories to
construct environment-neutral transitions, train grouped action-value
residuals, and shadow-score candidates. The resident policy performs no
learning and defaults to the frozen incumbent outside independently supported
regions. Policy quality is improved by unattended data rounds, never by
hand-tuning a failure location.

Dense roots keep collision authority separate from learning data: exact raw
hazard-producer records support safety audits, while same-epoch player attacks,
items, score/graze, rank/RNG, and NMNB resource counters support offline
training. Capped learner features are neither source evidence nor a substitute
for those facts.

Corpus, learner/framework, and fitted result have independent identities and
lifecycles. A corpus is immutable reusable Wine evidence, not property of the
algorithm that first used it. Only newly audited source-complete episodes are
eligible for the next inventory; the old transition-v6/v9/v10 registry was
removed because it predates the current paused-root safety authority. The
normative contract is
[docs/IMMUTABLE_WINE_DATA_PLANE.md](docs/IMMUTABLE_WINE_DATA_PLANE.md).

The reconstructed Linux/headless simulator and the pre-generation online-UCB
path have been removed from the tracked tree. Their old implementations remain
recoverable from Git history, but are not available learning backends.

For a new machine, provision and verify the self-contained original-retail
runtime with
[docs/PORTABLE_WINE_RUNTIME.md](docs/PORTABLE_WINE_RUNTIME.md). No historical
solver checkout is needed. Then start with [HAND_OFF.md](HAND_OFF.md) and
[START_HERE.md](START_HERE.md).
Rebuild the ignored, hash-pinned retail stage-script cache with
[docs/ECL_REFERENCE_CACHE.md](docs/ECL_REFERENCE_CACHE.md).
Provision the isolated offline learner environment with
`scripts/bootstrap_offline_rl.sh`; on Linux it deliberately uses CPU-only
XGBoost and the official CPU-only Torch wheel instead of CUDA runtimes.
Complete-route exploration is scheduled by `scripts/collect_route_parallel.py`
with a distinct immutable policy seed per episode.  Turn its collection
ledger into an algorithm-independent index with `scripts/build_g7_dataset.py`,
then fit only that admitted index with `scripts/train_g7.py`.  A training run
remains explicitly forbidden from Wine canary use until the separate online
integration, shadow, and latency gates bind the exact portable artifact.
The authoritative method and
evaluation contract is
[docs/WINE_ONLY_AUTONOMOUS_LEARNING.md](docs/WINE_ONLY_AUTONOMOUS_LEARNING.md).
The original-retail runner contract is
[docs/WINE_RETAIL_VALIDATION.md](docs/WINE_RETAIL_VALIDATION.md).

Final policy evidence is the physical HIT count in alternating, real-time,
complete original-retail Wine Practice Stages with HIT continuation. Fixed RNG,
first-failure prefixes, shadow replay, and offline metrics may reject a
candidate, but may not promote one. The current corpus controller suspends the
process at coherent decision roots: it preserves retail frame multiplier and
per-update order, but is not proof of 60 Hz wall-clock deployment. Collection
throughput comes from isolated parallel workers and offline computation; a
separate non-suspending end-to-end latency gate is required before promotion.

Parallel collection is fail-closed and policy-bound: prepare workers only from
the attested archive template with `scripts/prepare_wine_workers.py`, then run
the fixed-seed serial/pool-wide differential with
`scripts/gate_parallel_wine.py`. `scripts/collect_route_parallel.py` accepts only
that exact commit/pool/native/policy gate and publishes an admission ledger only
after every predeclared complete natural-RNG route passes the full source audit.
`collect_wine_parallel.py` is a Practice-stage diagnostic collector and cannot
produce Generation-7 training admission.

There is currently no authorized gameplay candidate. The tracked Wine smoke
policy is infrastructure-only and cannot create promotion evidence. Generation
1--6 code and its active corpus registry are pruned; first establish a complete
source-audited baseline and eligible inventory, then freeze the new offline
learner contract before candidate-facing Wine evaluation. See
[docs/REPOSITORY_PRUNE.md](docs/REPOSITORY_PRUNE.md) for the retired paths and
the retention boundary.
