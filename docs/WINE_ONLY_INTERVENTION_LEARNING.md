# Wine-only intervention learning

## Decision

Original Japanese TH06 1.02h under Wine is the only environment that may
advance game state for learning or evaluation. The retired Linux/headless
runtime contributes no trajectories, labels, candidate actions, or scores.

The active loop is:

`Wine intervention -> episode-grouped offline fit/replay -> small residual -> Wine canary`

This document is authoritative over historical experiment artifacts and Git
history.

## Invariants

- Native geometry constructs the complete publishable first-action set.
- A learned component ranks only that set and may always abstain to the frozen
  incumbent.
- Capture is coherent and publication is preceded by a fresh issue check.
- Bomb bit `0x02` is forbidden.
- The online policy is immutable: no `observe`, weight update, or checkpoint
  mutation occurs while the game is running.
- Movement is never keyed to frame, RNG, run identity, boss name, or a
  handwritten phase route.

## Evidence strata

### Training: normal or accelerated original-retail Wine

A training episode uses an immutable incumbent plus a bounded intervention
scheduler. At an eligible generic physical frontier it may randomize between
the incumbent and one or more native-safe alternatives. It records the exact
choice probability and then returns control to the incumbent. The default
budget is one intervention event per episode.

Fixed RNG, action-prefix replay, accelerated frame delivery, process snapshots,
and isolated parallel Wine workers are permitted only when the resulting
physical state is checked against normal Wine. A branch root must bind the
retail executable, native kernel, policy, RNG, action stream, delivery stream,
and coherent before-state hash. An unmatched root is discarded.

### Offline: grouped learning and shadow replay

Every intervention retains:

- the native-safe action set and incumbent action;
- generic action-relative clearance, future safe width, position, velocity,
  and boundary reserve features;
- selected action and exact propensity;
- survival horizon, minimum later safe width and reserve, HIT, and authority
  terminal outcomes;
- complete physical episode and fixed-RNG pair identifiers.

Training/validation splits keep all rows from one episode or branch family in
one group. Adjacent frames are never independent samples. Models are small
action-relative residuals. They default to the incumbent outside trained
support and abstain when a fixed committee disagrees.

Offline replay may screen arbitrarily many candidates against recorded Wine
states, but it is non-causal for actions Wine did not execute. Factual future
failure prediction alone cannot authorize an alternative action.

### Selection: Wine shadow and active canary

Shadow mode computes residual decisions while publishing the incumbent. It
checks scope, support, native membership, intervention count, latency, and
fresh-issue compatibility. Shadow never supplies causal benefit evidence.

An active first-failure canary may publish one residual only after disjoint
Wine intervention support and clean shadow replay. Canary runs stop on the
first HIT, authority failure, Bomb request, or contract violation.

### Final evaluation: complete normal-speed Wine Stage

Promotion uses alternating incumbent and candidate trials under all of these
conditions:

- original retail executable with verified identity;
- normal frame timing, no accelerated clock, no fixed RNG;
- natural full Practice Stage start and termination;
- HIT continuation enabled so the authoritative metric is total physical HITs;
- immutable policy states and zero Bomb;
- sequential trials with exact cleanup, never concurrent workers;
- per-run HIT count reported separately, followed by the predeclared aggregate.

First-failure survival, offline loss, shadow overlap, fixed-RNG branches, and
accelerated Wine results may reject a candidate but cannot promote it.

## Performance work

Performance changes must preserve the coherent snapshot and fresh publication
boundaries. Validate a new capture or accelerated path with identical action
and delivery streams, dense native-set parity, before-state hashes, Bomb
absence, and exact cleanup before using it for training.

Optimize in this order:

1. remove cross-Wine capture overhead while preserving atomicity;
2. avoid rendering and wall-clock waits only in a validated training mode;
3. reuse fixed-RNG prefixes or full Wine snapshots for branch families;
4. use isolated, CPU-pinned Wine workers only after a sequential/parallel
   differential passes;
5. parallelize offline replay and fitting freely.

## Decision rule for the first generation

Begin with Lunatic / Reimu-A / Stage 6. Use a frozen incumbent and a single
generic action-relative residual family. Stop and report **ineffective** if
disjoint Wine interventions do not support a repeatable alternative, active
first-failure canaries regress, a safety/delivery contract fails, or complete
Stage HIT-count A/B does not improve. Report **effective** only if active
canaries improve first-failure survival and normal-speed complete-Stage A/B
shows a lower aggregate physical HIT count without a safety or latency
regression. A promising offline metric is not an intermediate verdict.
