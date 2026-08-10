# Stage 6 late-family replay and delivery audit (2026-08-10)

## Decision

The Stage 6 sub31 and sub18 residual hypotheses have zero candidates.  Frozen
UCB remains the only incumbent.  No residual model, Wine shadow policy, active
canary, or broad offline fit is authorized.

One sub31 Wine checkpoint was reconstructed exactly.  Its incumbent action
survived the complete 600-tick branch while the native reactive alternative
reached authority failure after 262 ticks.  A second independent sub31 prefix
and both repeated sub18 prefixes could not reach their target checkpoints in
the reconstructed source runtime.  The old Wine corpus records coherent battle
input but not the actual Ctrl/Shoot sampling edges inside dialogue gaps; those
unobserved edges affect original-retail RNG and later physical state.

This is a data-authority result, not a reason to loosen replay tolerances or to
spend more headless compute.  Insufficient exact independent support returns to
the incumbent by design.

## Source replay change

The reconstructed source checkout now has generic retail dialogue delivery:

- source-GUI dialogue state is the only runtime gate;
- movement is suppressed during dialogue;
- skippable dialogue holds Ctrl;
- unskippable dialogue uses the same bounded Shoot release/held cycle as the
  retail controller;
- Bomb remains unrepresentable.

This is diagnostic delivery infrastructure, not a frame-, seed-, Boss-, or
phase-keyed movement script.  It is pushed on
`headless/th06-rl-headless-spike` at clean commit
`9d39ff31b3e1619eefdb88658ed31e66d625c92b`.  The validated native binary is
SHA-256
`accbd9a70b8bb94dd0dc9451868e745ec1c90c0ba8fb573a24e9c216b5b68e1e`;
the validated MinGW binary is SHA-256
`b81c937a7ac37af16d227943c22fc7ce4f248253594fc2ed5dc6ab9d1e724693`.

Linux and MinGW builds passed.  The dedicated source-Wine prefix was stopped
and audited clean after the comparison.

## Publication is not sampled delivery

The initial replay exporter treated a successful controller publication as if
original retail necessarily sampled that new mask on the next game frame.
Run `20260810T060605Z-421284500` disproved this: at frame 1260 the controller
published `down_right`, but the next coherent retail snapshot still reported
`down_right_fast` at frame 1261 and changed only at frame 1262.  Applying the
publication one frame early caused a false source HIT at tick 1724.

Solver commit `d97f80c` fixes the reconstruction rule:

1. every observed target frame uses that coherent retail snapshot's actual
   sampled input;
2. successful publication/current action may fill only unobserved interior
   frames;
3. dialogue gaps remain explicitly unknown rather than being presented as
   exact battle delivery.

After this correction the same prefix survived beyond tick 2000 and matched
retail input at every common observed snapshot.

## Exact sub31 result

Two independent frozen-UCB sub31 prefixes contained the same narrow physical
hypothesis: incumbent `left` versus generic native-reactive `up_fast`.

The first prefix, run `20260810T054543Z-427979700`, sequence 6724, retail frame
7367, passed both exact checkpoint gates:

- retail/source physical state matched at `1e-6`;
- the native hard-action sets were identical.

The bounded 600-tick COW result was:

| First action | Survival | Terminal | Minimum legal actions | Terminal reserve |
| --- | ---: | --- | ---: | ---: |
| incumbent `left` | 600 | tick limit | 4 | 64.696167 |
| alternative `up_fast` | 262 | authority failure | 1 | 26.2082901 |

The incumbent wins.  Evidence:
`artifacts/wine-first-stage6/retail-replay-cow-sub31-final-v2/20260810T054543Z-427979700-seq6724-pair.json`,
SHA-256
`32df673445fedbb07a7b9dd3a3d7598b34939e80439f7321965f9fa4e650cadc`.

The second prefix, run `20260810T060605Z-421284500`, sequence 5754, retail
frame 6415, could not supply a second exact COW anchor.  The corrected observed
battle inputs matched, but the 246-frame dialogue gap from 4429 to 4675 lacked
the original-retail Ctrl/Shoot sampling edges.  RNG and timeline first diverged
at retail frame 4675.  Both native Linux source and MinGW source-under-Wine
then physically HIT at tick 5170; original retail continued to its authoritative
frame-6481 control dead end.

The one-frame bullet-count discrepancy at retail frame 3827 was transient
cleanup timing and reconverged; it was not treated as the causal divergence.
The durable RNG/timeline divergence begins at the dialogue boundary.

Evidence:

- source-platform report SHA-256
  `4d3924b8c48a9d8883378294d3626affbc20b2b1ab0c40454d5758b7f11723ae`;
- retail/source audit SHA-256
  `a4b1c4e9327e70220e61064487139e3b8b387ac70a256e40e115988a550a15b4`.

One exact incumbent win plus one non-reconstructable independent prefix is not
support for an override.  The sub31 residual count is zero.

## Sub18 result

The two independently repeated sub18 opportunity prefixes shared incumbent
`down_right_fast` versus baseline `down_left`:

| Wine run | Checkpoint | Retail frame | Reconstructed-source result |
| --- | ---: | ---: | --- |
| `20260810T075505Z-923298000` | 4134 | 4955 | physical HIT at tick 4934, before checkpoint |
| `20260810T080212Z-050518400` | 5107 | 5861 | physical HIT at tick 4980, before checkpoint |

The exact labeler rejected both before generating a COW artifact.  It did not
branch from a mismatched state or reinterpret either premature source HIT as a
counterfactual outcome.  Both prefixes cross old dialogue gaps without sampled
delivery evidence, so selecting another nearby frame from the same corpus
would not add independent authority.

The sub18 residual count is zero.

## Bounded future corpus fix

Solver commit `a39738d` advances the authoritative frame schema from v4 to v5.
While dialogue deliberately owns input and battle capture is paused, the
controller now records only a coherent, Bomb-free delivery sample:

- game frame;
- current and previous input masks actually sampled by retail;
- controller-published input mask;
- held-repeat and held-frame counters;
- active/skippable/pulsed-Shoot diagnostics.

The accumulated samples are attached to the next coherent battle
`FrameEvidence`.  They do not create a hazard snapshot, action decision,
transition, reward, or learning row.  Native geometry, fresh issue
certification, fail-close behavior, movement ownership, and Bomb prohibition
are unchanged.  The complete test suite passes with 292 tests.

This does not repair old corpora retroactively.  It makes future Wine prefixes
delivery-complete enough to test the replay hypothesis directly.

## Next gate

Collect a small disjoint original-retail Wine Stage 6 panel with:

1. frozen UCB, exploration zero, and immutable copied state;
2. natural Practice, no life patch, and default first-HIT/authority/Bomb stop;
3. lossless frame-v5 corpus including dialogue delivery;
4. exact PID/input cleanup and no concurrent canonical trials.

Do not fit from the new rows first.  Export each prefix, confirm exact sampled
delivery across dialogue, and check whether repeated failure regions still
exist.  Only a repeated region with independent exact Wine-anchored COW
agreement may create at most one low-activation residual candidate.  Otherwise
retain the incumbent and use the new Wine panel to choose the next diagnostic
hypothesis.
