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

For multi-run experiments, `scripts/batch_label_headless_cow.py` enumerates
complete compact runs, samples a bounded terminal neighborhood, and invokes
the same dynamic COW labeler with a fixed worker ceiling. It always includes
the final reconstructable state and resumes only from a complete JSON label
whose recorded input run and checkpoint sequence set match exactly; an
interrupted or mismatched output is recomputed. Besides a terminal window, the
driver can select real HIT rows and the last native-legal state before each
contiguous benchmark authority-release event. Event neighborhoods make
continued-HIT evaluation useful for offline diagnosis without treating forced
or post-HIT rows as factual training data. The batch driver changes scheduling
only, not branch authority.

A game adapter may additionally expose fixed candidate-relative clearance
profiles at declared checkpoints. TH06 computes worst-case clearance across
input delivery delays and intermediate key-transition prefixes at ticks
4/8/12. Profiles are ranker inputs only: they cannot certify an action, enlarge
the native legal set, or replace fresh issue revalidation. A TH08 adapter may
use different source physics while retaining the ordered profile contract.

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
