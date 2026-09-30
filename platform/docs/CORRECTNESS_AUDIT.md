# Model correctness audit — 23 September 2026

## Scope and verdict

This patch corrects concrete defects in the **25-model integration**, not the
scientific validity of 25 research architectures. Original sibling sources stay
pinned and unchanged. Their adapters fix input conversion, fitting, losses,
causal inference and numerical contracts where needed.

**All 25 execute using their required dependencies; this does not mean all 25
are complete implementations of their namesake papers or validated on turbines.**
No field accuracy, calibrated failure probability, certified safety level or
remaining-lifetime guarantee is claimed.

## Training and inference contract

- Supply a dedicated training record to `Orchestrator.fit`. Its default
  `train_fraction` is now **1.0**, not an implicit healthy prefix. Explicit
  fractions remain supported in `(0, 1]`.
- Every adapter validates a nonempty boolean mask of exactly the record length.
  Orchestration excludes invalid-quality rows. Anomaly/forecast/representation/
  compression models additionally exclude labelled faults. Supervised models
  use allowed labelled examples; they no longer silently train outside the mask.
- Sequence training requires **all rows in a window** to be allowed, including
  the target. m05 uses the longest contiguous allowed healthy run, not a
  concatenation of disconnected periods.
- `training_rows` in health is the number of *permitted input rows*, not a
  promise that every row became an optimizer example (windows and m05's
  train/calibration split reduce that number).
- The demo training corpus explicitly injects all seven fault families. The SVM
  cannot invent an electrical-fault class when no positives were observed.
- API bootstrap, the full checker and production CLI fit seed 7 and analyse a
  separate seed 107 record by default. Fault/RUL target metadata is removed
  before inference. Research CLI retains labels only for separate evaluation.
- Features use causal carry-forward and trailing statistics; windows include
  their declared endpoint. No future observation repairs a past gap. Rated
  operating state uses `meta['rated_power_kw']` (demo default 2000 kW), not a
  percentile of the record being scored. Real waveform inputs are preserved.
- Successful per-timestamp output arrays must be aligned, finite and valid;
  probabilities sum to one, RUL/uncertainty cannot be negative. Support outputs
  may use an ordered timestamp subset (e.g. SHAP explains the final sample).
  Fusion requires identical turbine/timestamps, never implicit left-padding.
- Missing classifier classes abstain in probability fusion. Warmup probabilities
  are uniform until the first available waveform, then held from the past.
  Initial anomaly averaging uses available samples, not imaginary zero samples.

## All 25: repairs and remaining limits

| Model | Corrected or verified behavior | Remaining scientific/operational limit |
|---|---|---|
| m01 CNN | Mask-selected waveform fitting, canonical fault classes, causal probability alignment | SCADA-derived demo waveforms are not a high-frequency bearing DAQ |
| m02 ConvLSTM | Complete masked sequences, current available image included, time-based proxy labels, no future backfill | Recurrence images are not spatial wear maps; labels are fault-onset proxies |
| m03 TCN | Correct public forward input/head, complete training windows, eval-mode inference | Short CPU training; residual thresholds need independent site calibration |
| m04 GRU | Complete masked windows, eval-mode forecasting | Not a validated turbine failure detector |
| m05 LSTM | Genuine reference fitting on contiguous permitted healthy data; aligned timestamps and normalized uncertainty | Adequate clean history and reference train/calibration minimums required |
| m06 Informer | Decoder uses preceding observation, **never target row**; eval mode | Upstream simplified attention remains a placeholder, not full ProbSparse Informer |
| m07 SNN | Train-only event threshold; differentiable membrane-logit loss; canonical classes | Event encoding from surrogate waveforms is not a validated spiking sensor pipeline |
| m08 contrastive SSL | Actual symmetric NT-Xent with positives, negatives and self-exclusion; masked waves | Representation quality and transfer need a downstream held-out benchmark |
| m09 DBN | Masked RBM fitting; deterministic probability embeddings at inference | CD-style research pretraining, no demonstrated field utility |
| m10 RF | Only permitted labelled rows fit the estimator; genuine SHAP source | Synthetic class probabilities are not calibrated field probabilities |
| m11 XGBoost | Masked labels/features, explicit minimum two-class requirement | Must train on representative fault examples; no invented missing class |
| m12 SVM | Genuine labelled electrical/non-electrical examples only | Negative class is `not_electrical`, **not proof of healthy operation**; converter proxy is not measured stator diagnosis |
| m13 Deep SVDD | Complete allowed training feature rows and training-only calibration | Training-score calibration is not an independent false-alarm guarantee |
| m14 Isolation Forest | Masked fitting/calibration | Site-specific thresholds and drift evaluation still required |
| m15 HMM | Training scaler fixed; disconnected training lengths preserved; forward filtering instead of future-aware smoothing; ordered states | Ordered latent means do not prove physical degradation stages |
| m16 particle filter | First-observation initialization, actual elapsed hours, finite observation/state checks, isolated reproducible NumPy RNG | Filters upstream RUL proxies; uncertainty is not field-calibrated |
| m17 MLP RUL | Masked fitting; timestamp-based proxy ignores excluded future fault labels | Censored 400 h horizon/time-to-onset is **not measured component lifetime** |
| m18 physics BNN | Predicted power participates in physics loss; kW/scaling consistency; nonzero weight gradient; Gaussian KL regularizer | MC estimates vary between calls; posterior/uncertainty not calibrated; small research BNN |
| m19 twin surrogate | Training-only vibration baseline and row-local fatigue proxy | No FEA validation, causal intervention proof or physical fatigue lifetime |
| m20 GNN | Fit only in training; frozen scaler/weights in prediction; matching vibration feature units; per-step outputs and checked graph indices | Supervision is a vibration-risk proxy on self-loops, **not learned wake/cascade dynamics** |
| m21 SHAP | Genuine upstream TreeExplainer execution, bound only to fitted supported tree; no fake connected success | Explains that tree, not causality or every fused model |
| m22 VAE | Complete masked windows; posterior-mean reconstruction for deterministic scores; latent std separate from score uncertainty | Latent spread is not calibrated output uncertainty |
| m23 compressor | Masked autoencoder training and genuine latent/reconstruction output | Latent-width ratio is not a measured network-bandwidth saving |
| m24 INT8 edge | Masked small-network fitting, actual representative-data INT8 conversion/interpreter inference | This is a tabular edge network, not image MobileNet; target-device accuracy/latency unmeasured |
| m25 TinyML | Masked engineering-rule training; no fabricated positive labels; real C++ export recipe | A learned decision tree is **not a certified safety relay**; deterministic safety limits remain separate |

Software command checks reject NaN, infinity, malformed/negative safety values,
missing recognized parameters and unsafe aliases even when another alias is zero.
The SafetyManager cannot authorize CONTINUE with missing/invalid safety-channel
telemetry. Bounds are **demonstration bounds**, not OEM specifications. Passing
the software check does not itself perform a SCADA write.

## Verification recorded in this workspace

- Full regression suite: **162 passed**, 79.93 seconds, no skipped tests with
  `WTPM_TEST_FULL=1`. Warnings include sklearn convergence (short demo training),
  upstream deprecations and TensorFlow's converter statistics warning; these
  are not evidence of model convergence or field accuracy.
- Independent-record checker: **25/25 ran**, no errors or fallbacks, **28.75 s**,
  peak RSS **1538 MiB**; actual SHAP source `m10-random-forest`, actual INT8
  input/output, **2672-byte** TFLite model.
- Running HTTP service: `/ready`, `/health` and `/analyse` passed
  `deployment/check_local.py --analyse`; all four counts (registered,
  available, fitted, ran) are 25.
- Local logs: `artifacts/audit/tests-final.log`, `full.json`, `full.log`,
  `http.log` (ignored generated evidence, not packaged artifacts).

## Reproduce the checks

With the full dependencies, root/platform packages and pinned sibling sources
installed as described in `docs/deploy-local.md`:

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1
export WTPM_EXTERNAL_DIR="$PWD/external"
WTPM_TEST_FULL=1 python -m pytest tests platform/tests -ra
python deployment/check_full.py --report artifacts/audit/full.json
```

The full test flag is important: without it, heavy regression tests are skipped.
The full-model Docker workflow also runs the regressions against real pinned
sources; its separate `--network none` smoke test verifies offline execution.
Generated reports, sources, model weights and datasets stay ignored by Git.

Regression coverage includes invalid masks for all 25 adapters; poisoned excluded
future rows for RF/XGBoost/IsolationForest/MLP; causal features/window boundaries;
Informer target independence; GNN weight immutability; physics-loss units and
gradients; NT-Xent negatives; particle-filter causality; timestamp alignment;
probability contracts; safety failures; and independent-record all-25 execution
including repeated inference, real SHAP and INT8 I/O.

## What still requires real validation

1. Fit on approved, representative training turbines; use disjoint validation and
   final test turbines/time intervals, grouped to prevent overlapping-window leaks.
2. Supply real high-rate vibration where relevant, measured failure/maintenance
   histories for RUL, FEA/load ground truth for the twin, and fleet topology plus
   wake measurements for cascade claims.
3. Benchmark fault precision/recall, false alarms per turbine-day, detection delay,
   RUL error and uncertainty coverage on held-out real records. Several adapters
   still calibrate residuals on training scores; these thresholds require a
   separate healthy validation set before deployment.
4. Define missing-data/operating-state policies and quality exclusions for each
   evaluation; filled/masked values must not be interpreted as real measurements.
5. Validate pruning/quantization/export on the actual edge device, add persistent
   model/version storage, security and monitoring, and obtain OEM/safety review
   before considering any plant writeback. Keep the app advisory-only meanwhile.
