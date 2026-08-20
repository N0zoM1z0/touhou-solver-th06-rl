# Repository prune boundary

## Decision

The active tree is an infrastructure/data baseline plus the current learner
generation. It is not an archive of executable failed algorithms. On
2026-08-13 the repository removed the complete Generation-1--6 learner graph:

- learner and policy implementations;
- fit, shadow, canary, evaluation, and autonomous-round runners;
- learner-specific authorization and qualification code;
- generation-specific configs and tests;
- scattered design/progress/result documents.

The durable negative evidence and decisions were condensed into
`LEARNER_AUDIT_AND_GENERATION7_DECISION.md`. Detailed source remains recoverable
from Git history, but it is quarantine history and must not be imported by
active code.

## What the prune deliberately kept

- current source-complete corpus recording and audit infrastructure;
- factual recorder, transition integrity validation, and source-dataset loaders;
- original-retail capture/control and physical HIT accounting;
- native geometry, collision safety, causal/numeric audits, and fresh issue checks;
- game-neutral observation/action features, bounded hazard primitives, and
  causal history projection;
- immutable policy API/loader and generic safe exploration primitives;
- generic native XGBoost population, support-distance, and hazard-codebook
  scoring functions, whose mechanics are independent of the failed actor loss;
- resource isolation, storage/host checks, latency tooling, and Wine validation;
- infrastructure performance history, including failures.

The old derived option cache was removed because its pickle type and cache key
were coupled to a deleted learner module. It is reproducible derived data, not
corpus. A successor cache must use a learner-neutral row schema and a new cache
version; ignored old cache files are left untouched.

On 2026-08-20 the observed-only `replay_corpus.py` and
`audit_wine_hard_empty.py` diagnostics were removed with their document. The
latter called an observed projection “source-exact” and recomputed the now
forbidden zero-margin fallback. Current Hard-empty classification comes only
from recorded source-complete Hard-4 evidence and `scripts/audit_run.py`.

## Enforced absence

`tests/test_repository_prune.py` rejects retired backend modules, the legacy
corpus registry, and known Generation-1--6 learner/runner/config/document
paths. Ignored historical artifacts remain quarantined and are not an active
data source.

## Recovery rule

If a future generation needs a capability that existed in deleted code, first
state the capability as a current data, safety, or deployment contract and
implement the smallest generation-neutral version. Do not restore a historical
module merely because it already contains a nearby implementation.
