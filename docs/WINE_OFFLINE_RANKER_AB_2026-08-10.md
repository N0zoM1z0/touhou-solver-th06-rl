# Immutable offline-ranker Wine A/B, 2026-08-10

## Outcome

The offline-to-original-retail path now exists and has been exercised through
natural Lunatic / Reimu-A / Practice Stage 6 runs. The experiment closes the
old contract gap: a trained offline XGBoost ranker was exported without Linux
runtime dependencies, loaded by 32-bit Windows Python under Wine, and allowed
to rank only the native-gated action set. Fresh issue certification, fail-close
behavior, the validated geometry DLL, and Bomb prohibition were unchanged.

The tested candidate is **not promoted**. In a minimal two-run-per-policy
panel, the frozen UCB control recorded 8 and 9 HITs; the conservative offline
selector recorded 12 and 9. The ranges overlap because retail RNG is not
controlled, but the offline policy has no measured advantage and its mean HIT
count is higher. There is still no Stage 6 NMNB result.

| Frozen policy | Natural HITs | Hard dead-ends | Stale retries | Solve p50 / p95 |
| --- | ---: | ---: | ---: | ---: |
| UCB control r1 | 8 | 34 | 317 | 0.903 / 1.760 ms |
| UCB control r2 | 9 | 41 | 360 | 0.903 / 2.499 ms |
| Offline support32/margin1 r1 | 12 | 50 | 1,616 | 1.614 / 3.901 ms |
| Offline support32/margin1 optimized r2 | 9 | 39 | 859 | 1.267 / 3.197 ms |

The first offline natural run used the same model and selector semantics before
common-feature packing was optimized. The second run is the final adapter and
is the better delivery comparison. Keeping both is intentional; no run was
discarded because of its HIT count.

This panel is evidence that the adapter works and this candidate is not ready
for promotion. It is not a deterministic or statistically powered A/B: the
original retail RNG/action stream cannot yet be replayed identically across
policies. The additional offline solve time and trajectory-dependent stale
retries also remain a delivery covariate, although the optimized native batch
scorer brought worst-case synthetic 18-action scoring under Wine from about
12.0 ms to about 0.44 ms.

## Candidate contract

- Scope: `(3, 0, 0, 6)` only.
- Source model: Stage 6 exact-v5 XGBoost from the immutable CPU policy zoo.
- Source `.joblib` SHA-256:
  `34ab689014429ce1497f4cf7abfb05df852dc7e25128ca880ef49ca3c556882e`.
- Portable model SHA-256:
  `1c21bd595cfd0108e84d88b8446943c248e0faa862093d94450a5b0609d93290`.
- Active immutable state SHA-256:
  `a8a1cff567ca139e33ad9d27949bfd378da9f21f70cc800117b4179bc613f3e4`.
- Final policy plug-in SHA-256:
  `0b9db11f45c0ad3c3409194e015e1a529b750e74fe2507a80826f65c258e92bf`.
- Isolated Win32 scorer DLL SHA-256:
  `6817280f2860c904a11434d825bd79b7dd99edfed63fde7f6305000cf59a0f6a`.
- Validated geometry gate DLL SHA-256 remained:
  `d5c79c30b4d46c72f0521d9653d5d99693c0fbc966e241f554732ad3ade3a37e`.

The selector reproduces the offline zoo's `support32_margin_1_0` rule. An
action other than the generic reactive baseline is eligible only when its
exact `(source_context, action)` support in the model's training runs is at
least 32 and its predicted value is at least 1.0 above the baseline value.
The exported state contains 282 supported pairs. Source context remains a
generic learned feature/support key; it does not select a handwritten route.

The portable state contains the exact encoder, flattened trees, source/model
hashes, and conformance vectors. Windows refuses a missing or hash-mismatched
scorer and refuses a model whose conformance predictions do not match. The
scorer DLL has no game or collision API; it only batch-scores already encoded
feature rows. The original geometry DLL remains the sole collision authority.

## Immutable evaluation contract

`--immutable-policy` requires exploration zero. The Wine runner copies the
declared state into the run artifact, disables policy feedback, failure
feedback, hot reload, checkpoint writes, and the Stage policy transaction, and
asserts equal source/copy SHA-256 values after the controller exits. Every run
below had controller return code 0, equal before/after policy hashes, natural
Practice completion, and zero leftover processes in the dedicated prefix.

HIT continuation is benchmark-only. These runs used `--no-corpus`; their
traces are telemetry and are never training data.

## Reproduction

Build the isolated standard-library/ctypes scorer without rebuilding or
changing the validated geometry DLL:

```bash
cmake -S native -B build/native-win32-fully-static \
  -DCMAKE_SYSTEM_NAME=Windows \
  -DCMAKE_CXX_COMPILER=i686-w64-mingw32-g++ \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build/native-win32-fully-static \
  --target th06_rl_ranker -j4
```

Export the existing model and its exact training-support selector:

```bash
PYTHONPATH=.:src .venv/bin/python \
  scripts/export_offline_ranker_policy.py \
  --manifest artifacts/offline/stage6-exact-policy-zoo/manifest.json \
  --encoder artifacts/offline/stage6-exact-policy-zoo/encoder.json \
  --model artifacts/offline/stage6-exact-policy-zoo/xgboost.joblib \
  --mode active --selection support-margin \
  --score-margin 1.0 --minimum-support 32 \
  --dataset corpus/hf-211374189e2e \
  --native-scorer \
    build/native-win32-fully-static/libth06_rl_ranker.dll \
  --output \
    artifacts/policy/lunatic_reimu_a_stage6_offline_xgboost_exact_support32_margin1_active.json
```

Run a natural immutable Stage 6 trial through ordinary pipes, never a PTY:

```bash
PYTHONPATH=.:src .venv/bin/python scripts/run_wine_retail.py \
  --practice-stage 6 --seconds 0 \
  --exploration-rate 0 --immutable-policy \
  --policy-plugin src/th06_rl/policies/offline_ranker.py \
  --policy-state \
    artifacts/policy/lunatic_reimu_a_stage6_offline_xgboost_exact_support32_margin1_active.json \
  --policy-scorer-library \
    build/native-win32-fully-static/libth06_rl_ranker.dll \
  --artifact-dir artifacts/wine-retail-stage6-offline-example
```

Replace the plug-in/state with `src/th06_rl/policies/adaptive.py` and
`artifacts/policy/lunatic_reimu_a_stage6.json` for the frozen UCB control. Do
not pass the scorer library to the UCB run.

## Evidence hashes

| Run | `report.json` SHA-256 | `trace.jsonl` SHA-256 |
| --- | --- | --- |
| UCB r1 | `99134032c724791ca2608038ebbe8472ffd52ff8b3f89ff82e6fea60605b4be2` | `d640b220d50b12c33ce183e36d1baf2bcd2538bb1bed05b9edda194f9e766846` |
| UCB r2 | `abd3aa8d8454fb711c8fedecc3eb19802c8e8184025c7a2cea84a6b16dfa9817` | `4cf341cbebe0e8c62ac1b6795881305899bbc3f62718f294722e4b9e34b97acc` |
| Offline r1 | `69ab6a0fdabaa8b46306fe1fc58e03d3b0e0a8971d66bc19ae9566fdeadafb62` | `12752695de82be121d86d21057ee5722e45060b623b05578f685377f8b84f871` |
| Offline optimized r2 | `a0e70c32d5297a0db27dd09a9933e6a5b7a360236905586d08ff1b45a0c85eb2` | `95a3352b1041cd8eea25af63c409208f141fa72b5808ad5e9eb62b157b575eaf` |

Artifact directories, in table order:

- `artifacts/wine-retail-stage6-immutable-ucb-natural-r1`
- `artifacts/wine-retail-stage6-immutable-ucb-natural-r2`
- `artifacts/wine-retail-stage6-offline-xgboost-support32-margin1-natural-r1`
- `artifacts/wine-retail-stage6-offline-xgboost-support32-margin1-optimized-natural-r2`

## Decision

Do not promote this XGBoost weight. The next model experiment should retain an
incumbent at the policy level, not merely the generic reactive baseline, and
should use several independently supported models or an exact replay/action-
stream comparison before overriding. More global offline fitting or another
single unrelated Wine route would not resolve the observed variance.
