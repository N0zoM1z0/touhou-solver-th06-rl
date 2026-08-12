# Immutable Wine data plane

## Core separation

Collected data, learning algorithms, and fitted model artifacts are three
different objects:

`original-Wine facts -> versioned learner consumes declared capabilities -> immutable fitted artifact`

A corpus is not owned by the learner generation that first collected it. It is
an immutable physical fact set with provenance and semantic capabilities. A
learner is replaceable code that queries those capabilities. A fit is a derived
artifact bound to exact corpus, source, parameters, and seeds. Replacing an
algorithm must reuse every compatible factual corpus rather than replay the
game merely because the model family changed.

This separation is also the portability boundary. TH06/TH08 adapters emit the
same generic facts and capability declarations; offline RL consumes them
without game-specific routes or phase logic.

## Machine registry

`config/wine_corpus_registry.json` and
`src/th06_rl/wine_corpus_registry.py` bind every currently clean, complete
original-retail Wine Stage by its source inventory, manifest, run metadata,
transition schema, executable hash, Stage, and physical HIT outcome. Selection
is by a required semantic capability set, never by model generation or observed
HIT count.

The initial audit contains 59 clean complete Wine Stages:

| Access | Episodes | Permitted use |
| --- | ---: | --- |
| Training, randomized option semantics | 44 | sequential offline RL, action effect, representation, behavior value |
| Training, deterministic older semantics | 12 | representation and behavior state/value only |
| Infrastructure regression only | 3 | serial/concurrent differential; never learner input |

The 44 sequential episodes comprise 13 transition-v9 Stage-6, 16
transition-v10 Stage-6, and 15 transition-v10 Stage-4 runs. Version 9 has a
complete propensity distribution reconstructible from its frozen randomized
behavior contract; version 10 records the complete vector directly. Both have
factual option successors, native-safe candidate sets, and exact Stage HIT
conservation.

The 12 transition-v6 episodes used deterministic behavior and cannot identify
a randomized treatment effect. They remain valuable for self-supervised hazard
representation, observation normalization, behavior-policy risk/value, and
schema regression. A learner may not pretend missing propensities exist.

The three clean concurrency-differential Stages deliberately remain
`infrastructure-regression-only`: their incompatibility was the test outcome,
and their predeclared contract forbids training admission.

## Admission and exclusion

The registry admits only complete original-Wine Stages with zero dropped rows,
complete Stage trajectory, complete physical outcome, zero capture/corpus/
authority/trace failure, and the exact retail executable hash. First-failure,
short smoke, incomplete, headless, synthetic, authority-failed, and rejected
attempts remain available for their declared diagnostics but are not silently
promoted into the training data plane.

New data appends a new immutable source inventory or intentionally updates an
existing inventory with an audit. Learner code must not scan arbitrary artifact
directories. Registry drift fails closed.

## Partition is a learner-run property

Capability eligibility and statistical role are separate. The corpus registry
says what facts an episode can support. A learner generation separately freezes
which compatible complete episodes are development, qualification, training,
canary, or final evaluation. That prevents both data duplication and holdout
leakage.

Historical qualification can reject an algorithm but cannot authorize new
gameplay. Final efficacy always requires new normal-speed original-Wine,
natural-RNG, complete-Stage HIT-continuation trials.
