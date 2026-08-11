# Stage 6 episode-grouped Wine failure audit (2026-08-10)

This is the second implementation gate in the Wine-first plan.  It audits the
existing original-retail first-failure prefixes before any new fit or physical
trial.  The machine-readable local report is ignored at
`artifacts/wine-first-stage6/failure-regions-v1/report.json` with SHA-256
`aff4d32fb2d67cefa6b3321add2a7335d41ae4f9290531ba5b43bed331ccb132`.

The audit is hypothesis-generation evidence only.  Its fixed bins are not
movement rules, causal labels, or promotion evidence.

## Input authority

The input set is exactly the 23 physical episodes covered by
`wine-stage6-risk-guard-clean-r12-training-factual-action-audit.json` (SHA-256
`256c9887639c3b33e3c78a74b234f900e2ed337acf6079c40284fc7f84b593dc`).
That replay checked 95,768 policy calls with:

- zero recorded-incumbent mismatches;
- zero recorded-policy mismatches;
- zero shadow action-contract violations;
- fixed scope Lunatic / Reimu-A / Stage 6;
- exact original-retail and native-kernel identities.

The 23 episodes consist of seven direct frozen-UCB runs, twelve four-member
consensus shadow runs, and four later context-reactive shadow runs.  Shadow
episodes are admitted only because the exact replay proves their published
actions equal the frozen incumbent.  Active-canary and HIT-continuation runs
are excluded.

Every prefix is reloaded through the strict first-failure reader.  This checks
complete storage, no dropped records, clean capture/infrastructure history,
exact one-frame terminal attribution, zero Bomb, frozen zero-exploration
metadata, hashes, and incomplete Practice-episode semantics.

## Independent evidence, not row volume

The strict loader retains 91,783 eligible policy rows.  The old 120-frame
proximity label marks 1,290 of them positive, but they come from only 23
physical terminal episodes.  Those 1,290 adjacent rows are not independent
positive experiments.

All 23 terminal events are `control-dead-end`; none is a physical HIT.  The
data therefore supports hypotheses about paths that end with an empty native
safe set.  It does not directly establish which earlier action caused the
dead end, nor does it contain a Wine counterfactual proving that another
action survives.

Eighteen episodes contain at least one frame where the incumbent differed from
the native reactive baseline.  That is 514 rows, but only 114 contiguous
opportunity events after temporal deduplication.  Even these are candidate
branch points, not positive alternative-action labels.

## Repeated versus one-off failures

Automatic source context is used only as a support/partition key:

| Context | Independent episodes | Episodes with a baseline alternative |
| --- | ---: | ---: |
| `boss:0:sub10:life_cb14:timer_cb13:nonspell` | 10 | 9 |
| `boss:0:sub31:life_cb31:timer_cb19:spell` | 4 | 4 |
| `boss:0:sub18:life_cb31:timer_cb31:nonspell` | 3 | 2 |
| `boss:0:sub11:life_cb14:timer_cb13:nonspell` | 2 | 1 |

Nineteen of 23 episodes therefore fall in a context observed independently at
least twice.  Four contexts occur once each and remain RNG/route singletons:
sub17, sub35, the pre-t1240 timeline region, and the pre-t768 timeline region.
They are not eligible for targeted residual work yet.

The generic terminal families, with every episode contributing at most one
vote, are:

| Generic physical family | Episodes | Contexts represented | With any baseline alternative |
| --- | ---: | ---: | ---: |
| boundary + at least 384 bullets + no laser | 13 | 3 | 10 |
| boundary + 128–383 bullets + no laser | 5 | 3 | 3 |
| interior + lasers present | 4 | 1 | 4 |
| boundary + fewer than 128 bullets + no laser | 1 | 1 | 1 |

The strongest repeated specific opportunity is the broad-safe-set,
dense-bullet boundary region where the incumbent chose `right_fast` and the
native baseline proposed `left_fast`: five independent sub10 episodes.  Several
related sub10 pairs have support three or four.  In the sub31 laser family,
the family itself has four-run support, while individual incumbent/baseline
action pairs have only two-run support; it should be branched exhaustively over
the native-safe first-action set rather than converted into a hand rule.

## Why the previous offline fits did not help enough

The broad r12 risk fit learned from 91,783 correlated rows and 1,290 proximity
positives.  Its unit of supervision looked large, but the independent terminal
support was only 23 and every terminal had the same authority-failure type.
The labels mean “near a later dead end,” not “this incumbent action caused the
dead end” or “the baseline action would prevent it.”

The later r13 exact residual filter reduced the set to 983 rows across 21
nonempty run groups and 51 positive rows.  It was still rejected:

- average precision `0.1321720785`;
- ROC AUC `0.6116090213`;
- best nonempty threshold: 2 true positives, 4 false positives, only one
  protected run, one-sided 95% precision lower bound `0.1172760941`;
- best threshold covering two runs: 3 true positives and 12 false positives,
  lower bound `0.0828983018`.

This is not mainly a hyperparameter shortage.  It is a causal-support problem:
row duplication, weak cross-episode support, and no factual Wine outcome for
the proposed alternative.  Another global classifier or sweep would spend
compute without creating that missing evidence.

## Targeted COW queue

The audit admits three bounded hypothesis families for the next headless COW
gate, in this order:

1. sub10 dense-bullet boundary collapse, beginning with the five-run
   `right_fast` versus native `left_fast` opportunity and related supported
   broad-safe-set pairs;
2. sub31 interior laser collapse, four-run family support, with exhaustive
   native-safe first-action branches rather than one preselected direction;
3. sub18 medium-bullet boundary collapse, three-run family support but only two
   episodes with a baseline alternative.

Sub11 and every singleton remain audit-only.  No residual candidate exists yet.
For each queued family, headless matching must use multiple independent seeds,
retain seed-grouped holdouts, and branch every native-safe first action.  A
Wine region cannot be restored as an exact headless snapshot, so matches are
generic feature analogues only.  Disagreement or lack of multi-seed support
returns to the incumbent.

## Reproduction

```bash
PYTHONPATH=src .venv/bin/python scripts/audit_wine_failure_regions.py \
  --factual-action-audit artifacts/wine-stage6-risk-guard-clean-r12-training-factual-action-audit.json \
  --corpus-root artifacts/wine-stage6-firstfailure-corpus \
  --corpus-root artifacts/wine-stage6-risk-consensus-r8-fourfold-shadow-validation-corpus \
  --corpus-root artifacts/wine-stage6-risk-guard-r11-context-reactive-v2-shadow-validation-corpus \
  --retail-sha256 9f76483c46256804792399296619c1274363c31cd8f1775fafb55106fb852245 \
  --native-sha256 d5c79c30b4d46c72f0521d9653d5d99693c0fbc966e241f554732ad3ade3a37e \
  --output artifacts/wine-first-stage6/failure-regions-v1/report.json
```
