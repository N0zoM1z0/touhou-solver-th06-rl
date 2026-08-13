# Autonomous Wine generation 1 result

## Verdict

**Ineffective under the predeclared generation-1 evidence budget.**

The grouped learner improved held-out factual-return prediction, but its action
committee did not produce a supported, repeatable action advantage. The
hash-bound shadow gate rejected the candidate, so the state machine correctly
did not authorize an active Wine canary or a complete-Stage A/B. This is a
negative result for this learner generation, not a claim that offline RL in
general is impossible.

No gameplay threshold, reward, feature, exploration distribution, margin, or
promotion gate was changed after observing play.

## Fixed generation

- Code commit: `6c0a07caea14483287a7a80ffcf2933390e6ba3d`
- Environment: original Japanese TH06 1.02h, Wine 11.0, Lunatic Reimu-A Stage 6
- Collection: 10 sequential fixed-RNG episodes in two five-episode rounds
- Exploration: fixed 0.10 uniform mixture inside every non-singleton native-safe
  set; no gameplay eligibility gate
- Return: factual 120-frame discounted return, gamma 0.99
- Fit: clipped-propensity grouped ridge committee, whole-episode holdout
- Shadow authorization: at least 500 rows, 10 unanimous margin-clearing
  proposals, baseline-only publication, p95 at most 4 ms

The complete ignored artifact is
`artifacts/autonomous-wine-generation-1/`. Its `generation.json` SHA-256 is
`1f04b52d220abfd95626a96ab21ff7cf1b5eab5f225b11fae92729395b969876`.

## Physical collection audit

All 10 episodes bound the same executable, native kernel, policy source, and
code commit. In aggregate:

- 40,976 policy decisions;
- 3,841 non-baseline randomized decisions;
- zero physical HITs before authority termination;
- 10 `authority-stop:Hard safe set empty` outcomes;
- zero capture, corpus, trace, or infrastructure failures;
- zero background reactivations and dropped records;
- zero immutable-state violations and leftover prefix processes.

The repeated Hard-empty terminal is a genuine constrained-control outcome. It
is not classified as infrastructure failure: the coherent native authority
found no publishable constant first action and failed closed as designed.

## Learning rounds

| Round | Train rows | Holdout rows | Constant RMSE | Model RMSE | Ratio | Shadow decisions | Committee abstentions | Proposals | p95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,510 | 6,588 | 9.269 | 8.851 | 0.955 | 8,722 | 7,984 | 4 | 1.97 ms |
| 2 | 26,941 | 3,693 | 8.532 | 7.725 | 0.905 | 4,720 | 4,378 | 0 | 2.87 ms |

Every fit gate passed in both rounds. All 18 movement actions exceeded the
minimum sample and clipped-propensity ESS support in round 2. The only failed
shadow gate was the predeclared minimum of 10 proposals. Shadow publication
remained exactly baseline-only and latency stayed within budget.

Artifact bindings:

- round-1 fit report:
  `45968206159ff2906101450a8ce5d79571a8e9e25fd71369ee1738a720abd9db`
- round-1 shadow audit:
  `a69bdf4f9c2c4eb34a3a2103e2ef976e68103416582535994acc3f420928aaac`
- round-2 fit report:
  `fb0d396b05fc8bc1a98d230f3ac00db9e59ddc50ab33026a90fd6154e040ffee`
- round-2 shadow audit:
  `2f4952e278a676d3db485be5984acc472714a6fa437f706bf9bc28fae1496292`

## Interpretation

More data improved prediction of the factual behavior return but did not reduce
the uncertainty that matters for action selection. The second committee
disagreed on nearly every non-baseline best action and emitted no proposal.
Running a complete-Stage "candidate" at that point would compare the baseline
against an identical abstaining baseline, not evaluate a learned policy. The
state machine therefore stopped before active exposure, exactly as declared.

Future work must remain autonomous. A next generation may spend a larger fixed
episode budget with the same algorithm, or predeclare a more expressive general
offline-RL model and validate it across episode groups. It must not target the
observed Hard-empty locations, loosen this failed gate after the fact, or add a
TH06/Stage-6 exception.
