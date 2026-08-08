# Portable counterfactual learning contract

The counterfactual learner is intentionally split into a game adapter and a
game-independent ranking contract so the same experiment can be reused for
TH08 or another deterministic bullet-hell runtime.

## Game adapter

A game-specific adapter must provide:

1. a coherent physical observation and canonical digest;
2. an explicit scope tuple, including difficulty, character, shot type, and
   stage;
3. a Bomb-free native first-action set with collision authority;
4. exact deterministic reset or replay to a chosen observation;
5. one-action lockstep stepping, with fresh native revalidation on every
   counterfactual continuation tick;
6. factual state and candidate features that exclude RNG/seed identity from
   the deployable policy.

TH06 implements reset with Linux COW replay checkpoints. TH08 may use a
different simulator or snapshot mechanism; it need not copy the TH06 runtime.

## Portable outcome table

Each checkpoint is one ranking group. It records the observation digest,
scope, seed, native legal actions, factual/local-teacher action, and one outcome
per legal first action. The continuation is dynamic: after the substituted
first action, every subsequent action is selected and freshly certified on the
counterfactual state. Static continuation actions from the factual route are
not accepted as authority-preserving evidence.

Outcomes are ordered lexicographically by:

1. successful completion of the requested horizon or stage;
2. no physical HIT;
3. physical ticks survived;
4. minimum future native-legal action count;
5. terminal boundary reserve.

This ordering is independent of TH06 phase names, boss identities, captured
frames, or hand-authored routes. A game may add raw outcome fields, but it may
not weaken collision authority or silently reinterpret an unknown terminal.

## Learner and deployment boundary

The CPU value baseline uses LambdaMART with one query group per checkpoint and
complete-seed train/holdout splits. It ranks only the native action set. A
future fitted-Q, CQL, IQL, or contextual-bandit implementation should consume
the same candidate/outcome table and obey the same grouped split.

Offline rankings are experiment order, not promotion. Promotion requires
unseen-seed full-stage headless rollout and later differential/physical Windows
evidence. Online UCB may adapt a distilled prior inside the safe set; neither
UCB nor a value model may add actions, request Bomb, bypass fresh issue
certification, or weaken fail-close behavior.
