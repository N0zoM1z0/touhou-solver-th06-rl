# Exact-state offline feasibility benchmark

This benchmark separates three failure axes before another learner is trained:

1. **geometry / authority / search ceiling** -- no tested continuation finds a
   no-HIT, no-Bomb, authority-preserving branch for any native-safe first
   action;
2. **policy selection** -- at least one tested continuation constructs a
   feasible branch, but the factual policy's first action has no witness;
3. **representation / learner separation** -- a leave-one-seed-out probe
   compares the current compact candidate view with a bounded richer feature
   vector derived from the exact physical snapshot.

The benchmark is offline-only. It does not change the resident controller,
interpret future ECL births, add an action, weaken the native gate, or publish
Bomb. Search and richer source-physical features are diagnostics and may not
become collision authority.

## Epistemic boundary

The generator exhausts the declared Cartesian product of native-safe first
actions and continuation policies. A successful branch is a constructive
feasibility witness. Failure to find one is `oracle-no-witness`, **not** proof
that no action sequence exists. Increasing the continuation population or
branch horizon can turn a no-witness checkpoint into a witness; reports retain
the exact population, horizon, runtime hash, corpus hashes, and implementation
commit for that reason.

Each continuation dynamically rebuilds the native safe set and repeats the
fresh issue check on every tick. A branch is feasible only when it reaches the
requested tick limit or a source-reported successful Stage/chain exit with
zero physical deaths, zero Bomb use, and no authority failure.

## Continuation population

The default population contains:

- the constant-time generic clearance/boundary fallback;
- native local planners at horizons 4, 12, 30, and 60;
- any explicitly supplied distilled rankers, each identified by SHA-256.

This removes the old single-teacher continuation assumption while retaining a
bounded and auditable experiment. Ranker artifacts must match the exact scope
and clean headless source build.

## Generate and audit

Select only reconstructable rows before the first physical HIT or benchmark
authority release. Existing event selection in `batch_label_headless_cow.py`
can be used to discover those sequences. For one run:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/label_headless_feasibility_oracle.py RUN \
  --checkpoint-sequence SEQUENCE \
  --branch-frames 1200 \
  --planner-horizon 4 --planner-horizon 12 \
  --planner-horizon 30 --planner-horizon 60 \
  --model artifacts/models/STAGE_INCUMBENT/teacher-ranker.joblib \
  --output artifacts/feasibility/STAGE-SEED.json

PYTHONPATH=.:src .venv/bin/python \
  scripts/audit_headless_feasibility_oracle.py \
  artifacts/feasibility/STAGE-SEED.json \
  --output artifacts/benchmarks/STAGE-feasibility-audit.json
```

Formal generation refuses a dirty repository by default. `--allow-dirty-code`
exists only for local smoke tests and is recorded as dirty provenance.

The independent audit recomputes branch feasibility, action/continuation
coverage, best-action summaries, policy verdicts, Bomb deltas, bounds, and
source identity. It rejects a missing branch or a summary that does not match
the physical outcomes.

## Representation probe

Every exact checkpoint records two non-authoritative learning views:

- the same compact state and candidate records consumed by the resident
  ranker;
- a bounded nearest-hazard vector derived from the exact bullet, laser, enemy,
  and player snapshot.

For checkpoints where some but not all first actions have feasibility
witnesses, the auditor trains an Extra Trees diagnostic with leave-one-seed-out
splits. It reports the feasible-action top-1 rate for compact and exact-derived
views, plus the factual/local-teacher rate.

Interpret the deltas conservatively:

- high `oracle-no-witness` rate points first to native feasibility, observation
  authority, insufficient continuation coverage, or insufficient horizon;
- higher exact-derived than compact held-out rate is evidence of compact
  representation loss;
- higher compact-probe than factual-policy rate is evidence that the compact
  view contains usable signal which the current learner did not extract;
- too few seeds, too few discriminative checkpoints, or missing class support
  produces an explicit insufficient-evidence status rather than a verdict.

Only complete-seed, exact-scope comparisons may be combined. The final NMNB
gate remains a natural full-Stage headless clear followed by differential and
physical Windows validation.
