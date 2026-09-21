# WT-PM repository inspection (performed before any integration code was written)

Date: 2026-09-21. All 25 repositories cloned and read. **No fabrication policy:**
what each repo actually contains is recorded verbatim below; the integration
layer is built around the *currently available* functionality only.

## Global findings

1. **24 of the 25 repositories are single-file research stubs**: each contains
   `model.py` (11–49 lines), `requirements.txt`, `README.md` and a static
   `index.html`. They define *model architectures / training entry points*
   only. None ship trained weights, datasets, tests, packaging, or a CLI.
   They are genuine, runnable architecture definitions — not full products.
2. **`wt-pm-lstm-scada-anomaly` (model 05 in the task numbering) is the sole
   reference-grade repo**: a full package (`wt_pm_lstm`, ~6100 lines) with a
   versioned 12-channel SCADA schema, a physics-based fleet simulator with 7
   labelled fault families, an evaluation protocol, drift monitoring, model
   registry, HTTP API and tests. The platform's data contract extends this
   schema rather than inventing a parallel one.
3. **No repo defines an I/O contract** (except the reference repo). Inputs are
   ad-hoc (`csv_path="scada.csv"`, raw tensors). Every model therefore gets an
   **adapter** that converts the platform `SensorBatch` into the shape its
   `model.py` expects, and converts raw outputs into `ModelOutput`.
4. **Duplicate functionality found**: `wt-pm-vae-reconstruction-loss` and
   `wt-pm-aerozip-autoencoder-compressor` are near-identical autoencoders
   (different purpose: anomaly vs compression — kept separate, roles differ);
   `wt-pm-random-forest-telemetry` / `wt-pm-xgboost-tabular-faults` /
   `wt-pm-svm-rbf-generator-stator` are all tabular classifiers (kept: they
   form the tabular ensemble); `wt-pm-mlp-rul-regression` and
   `wt-pm-convlstm-wear-prognostics` both regress RUL (kept: baseline vs
   spatiotemporal — compared, then fused).
5. **Entry points identified** (exact function/class per repo) — see the table
   in `ARCHITECTURE_REPORT.md`.
6. **Dependency reality in this environment**: numpy/pandas/sklearn/xgboost/
   hmmlearn/torch/tensorflow-cpu/shap/micromlgen/snntorch/torch-geometric all
   install and import. Every adapter still degrades gracefully (reports
   `unavailable` + fallback) if its dependency is absent, because edge targets
   won't have TensorFlow.

## Per-repo audit

| # | repo | contents of model.py | state | integration decision |
|---|------|---------------------|-------|---------------------|
| 01 | 1d-cnn-bearing-vibration | `build_1d_cnn(input_shape, num_classes)` Keras Sequential 3xConv1D | architecture only, no weights/data | adapter trains on surrogate vibration windows derived from `bearing_vib_rms_mm_s`; honestly labelled surrogate |
| 02 | convlstm-wear-prognostics | `build_convlstm(input_shape=(10,64,64,1))` Keras ConvLSTM2D → RUL | architecture only; 64×64 spatial input assumes wear *maps* which no data source provides | integrated with reduced spatial grid built from channel-recurrence images; marked `partial` |
| 03 | tcn-power-curve | full TCN (Chomp1d/TemporalBlock/TCNPowerCurve, 49 lines) | complete architecture | adapter trains power-curve residual model; residual = anomaly evidence |
| 04 | gru-scada-telemetry | `GRUAnomalyDetector(nn.Module)` one-step forecaster | complete architecture | adapter trains one-step forecaster on canonical channels; forecast error = anomaly score |
| 05 | lstm-scada-anomaly | full reference package | **complete** | used directly through its public API (`RunConfig`, `fit_detector`, `Detector.score`); also supplies simulator + drift + eval primitives |
| 06 | informer-long-sequence | `InformerPowerForecaster` with simplified ProbSparse attention (self-declared placeholder) | partial (placeholder attention) | integrated as long-horizon power forecaster; marked `partial` |
| 07 | snn-event-vibration | `EventSNN` (snntorch), ImportError fallback prints warning | architecture only, optional dep | adapter encodes vibration windows to spike trains; falls back to unavailable if snntorch missing |
| 08 | contrastive-ssl-vibration | `ContrastiveEncoder` + simplified NT-Xent (self-declared placeholder) | partial (loss is placeholder) | adapter pretrains encoder on unlabelled vibration surrogates; embeddings feed tabular models |
| 09 | dbn-feature-extraction | `RBM` + `DBN` stacked, classifier head | complete architecture (no CD-k training loop) | adapter adds contrastive-divergence pretraining loop; embeddings feed RUL models |
| 10 | random-forest-telemetry | `train_rf(csv_path, target)` sklearn RF | complete for tabular csv | adapter feeds platform feature table; feature importances kept for XAI |
| 11 | xgboost-tabular-faults | `train_xgboost(csv_path, target)` | complete for tabular csv | primary tabular fault classifier; SHAP-compatible |
| 12 | svm-rbf-generator-stator | `build_svm(csv_path, target)` sklearn pipeline, `probability=True` | complete for tabular csv | adapter restricts to generator-electrical features (stator focus) |
| 13 | deep-svdd-boundary | `DeepSVDDNetwork` + `init_center` + `svdd_loss` | complete architecture | adapter trains one-class boundary on healthy features |
| 14 | isolation-forest-telemetry | `train_isolation_forest(csv_path)` | complete for tabular csv | unsupervised anomaly stream over feature table |
| 15 | hmm-degradation-states | `train_hmm(csv_path, n_states)` GaussianHMM | complete for csv | degradation-state estimator over health-indicator series; states ordered by severity post-hoc |
| 16 | particle-filter-rul | `ParticleFilterRUL` full predict/update/resample | **complete** (pure numpy) | fuses RUL observations from m17/m02 into probabilistic RUL posterior |
| 17 | mlp-rul-regression | `train_mlp(csv_path, target='RUL')` sklearn pipeline | complete for csv | RUL baseline every other RUL model must beat |
| 18 | pg-bnn-wind-turbine | `PhysicsGuidedLoss` (P = τω constraint) + `BayesianLinear` | architecture only (no full BNN net/training) | adapter builds small BNN from `BayesianLinear`, trains with physics loss, MC-samples uncertainty |
| 19 | digital-twin-surrogate | `FEASurrogate` MLP + `residual_loss` (physics term is a stated placeholder) | partial | surrogate trained on simulator load proxies; what-if engine wraps it |
| 20 | gnn-turbines-cascade | `TurbineCascadeGNN` (GCNConv) **with built-in linear fallback if PyG missing** | complete architecture | consumes wake-coupled farm graph from reference simulator |
| 21 | xai-shap-interpretable | `explain_with_shap(model, X)` TreeExplainer/Explainer | complete wrapper | XAIEngine wraps it; explains tabular classifiers + fused decisions |
| 22 | vae-reconstruction-loss | `VAEAnomalyDetector` + `vae_loss` complete | complete architecture | reconstruction-error anomaly stream on feature windows |
| 23 | aerozip-autoencoder-compressor | `AeroZipCompressor` + `compression_ratio` | complete architecture | telemetry compression stage; latent code exposed as features |
| 24 | quantized-mobilenet-edge | `convert_to_tflite(saved_model_dir)` INT8 script | conversion script only (no model) | EdgeInferenceManager uses its INT8 conversion recipe for the platform's small edge net; full MobileNet marked out-of-scope (no image data exists) |
| 25 | tinyml-esp32-safety-relay | `export_esp32_header(X, y)` DecisionTree → micromlgen C++ | complete recipe | SafetyManager trains depth-5 tree on safety-critical channels, exports `model.h`; a pure-python mirror of the tree runs in-loop as the safety gate |

## Duplicate-functionality map

- tabular classification: m10, m11, m12 → routed as an ensemble, fused
- anomaly detection: m05, m04, m13, m14, m22 (+m03 residual) → score fusion
- RUL: m17 (baseline), m02 (spatiotemporal), m16 (filter/fusion) → m16 consumes the others
- representation: m08, m09, m23 → feature providers, not decision makers
- CSV-loading boilerplate (`pd.read_csv`, `select_dtypes`, `fillna(0)`) repeated
  in m10/m11/m12/m14/m15/m17 → replaced by the shared FeaturePipeline; the
  original functions remain untouched in their repos

## Assumptions that had to be corrected in adapters (never in the repos)

- m14 `fillna(0)`: a 0-filled gap is a fake healthy reading → platform passes
  mask-filtered rows instead.
- m15 orders HMM states arbitrarily → adapter re-orders states by mean health
  indicator so state index is monotone in severity.
- m16 `update(observed_rul)` expects an RUL *observation* → fed from m17
  predictions, not ground truth (which production would not have).
- m18's `PhysicsGuidedLoss(power, torque*ω)` compares *measured* power to the
  physics identity — adapter feeds measured channels, so the physics term
  penalises weights only through the shared encoder (as intended).
