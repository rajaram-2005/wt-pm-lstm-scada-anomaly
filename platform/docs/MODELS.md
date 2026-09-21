# All 25 WT-PM models — descriptions, I/O, platform role

This session cannot push to the 24 sibling GitHub repositories (`gh` has
**no write permission** on them). The descriptions below are the canonical
copy for each model. Ready-to-paste README bodies live in
`sibling-readmes/`. Apply on each repo’s `main` with:

```bash
# from a machine that has push access to rajaram-2005/wt-pm-*
for f in platform/docs/sibling-readmes/*.md; do
  repo=$(basename "$f" .md)
  gh repo clone rajaram-2005/$repo /tmp/$repo
  cp "$f" /tmp/$repo/README.md
  (cd /tmp/$repo && git checkout -b docs/platform-description && git commit -am "docs: platform role, I/O contract, Hermes/XAI" && git push -u origin docs/platform-description && gh pr create --fill)
done
```

Model 05 (`wt-pm-lstm-scada-anomaly`, this repo) already has a full README.

| id | repository | adapter | task | description |
|----|------------|---------|------|-------------|
| m01 | wt-pm-1d-cnn-bearing-vibration | `m01-1dcnn-bearing` | fault classification | Keras 1D-CNN on vibration waveforms for gearbox bearing faults. Platform trains on **surrogate** waveforms derived from 10-min RMS (`vibration_is_surrogate=True`). |
| m02 | wt-pm-convlstm-wear-prognostics | `m02-convlstm-wear` | RUL | ConvLSTM2D wear tracker. **Partial**: repo expects 64×64 wear maps that do not exist; adapter uses channel-recurrence images. |
| m03 | wt-pm-tcn-power-curve | `m03-tcn-power-curve` | anomaly | TCN residual of aerodynamic power curve vs wind. |
| m04 | wt-pm-gru-scada-telemetry | `m04-gru-scada-telemetry` | anomaly | GRU one-step SCADA forecaster; forecast error = anomaly score. |
| m05 | wt-pm-lstm-scada-anomaly | `m05-lstm-scada-anomaly` | anomaly + contract hub | Reference LSTM/GRU autoencoder, 12-channel schema, simulator, drift, API. |
| m06 | wt-pm-informer-long-sequence | `m06-informer-forecast` | forecasting / anomaly | Informer long-horizon power forecast. **Partial**: attention is a placeholder in upstream `model.py`. |
| m07 | wt-pm-snn-event-vibration | `m07-snn-vibration` | fault classification | snnTorch SNN on event-encoded vibration (edge). |
| m08 | wt-pm-contrastive-ssl-vibration | `m08-contrastive-ssl` | feature extraction | SimCLR-style encoder; **partial** NT-Xent. Embeddings feed tabular models. |
| m09 | wt-pm-dbn-feature-extraction | `m09-dbn-features` | feature extraction | RBM/DBN stack; platform adds CD-1 pretraining the repo omits. |
| m10 | wt-pm-random-forest-telemetry | `m10-random-forest` | fault classification | sklearn RF; feature importances feed XAI. |
| m11 | wt-pm-xgboost-tabular-faults | `m11-xgboost-tabular` | fault classification | Primary tabular voter; SHAP TreeExplainer source (Hermes `explain` tool). |
| m12 | wt-pm-svm-rbf-generator-stator | `m12-svm-generator` | fault classification | SVM-RBF on electrical channels (generator/converter specialist). |
| m13 | wt-pm-deep-svdd-boundary | `m13-deep-svdd` | anomaly | One-class Deep SVDD hypersphere on healthy features. |
| m14 | wt-pm-isolation-forest-telemetry | `m14-isolation-forest` | anomaly | Cheapest unsupervised stream; universal fallback. |
| m15 | wt-pm-hmm-degradation-states | `m15-hmm-degradation` | degradation | GaussianHMM; states re-ordered by severity (HOW SEVERE). |
| m16 | wt-pm-particle-filter-rul | `m16-particle-filter-rul` | RUL | Particle filter fusing m17/m02 into a probabilistic RUL posterior. |
| m17 | wt-pm-mlp-rul-regression | `m17-mlp-rul` | RUL | sklearn MLP baseline every other RUL model must beat (proxy target). |
| m18 | wt-pm-pg-bnn-wind-turbine | `m18-pg-bnn` | anomaly + uncertainty | Physics-guided BNN, P=τω loss, MC uncertainty. |
| m19 | wt-pm-digital-twin-surrogate | `m19-digital-twin` | surrogate | FEA-surrogate MLP; Hermes `what_if` tool. |
| m20 | wt-pm-gnn-turbines-cascade | `m20-gnn-cascade` | graph | GNN cascade risk on wake-coupled farm graph. |
| m21 | wt-pm-xai-shap-interpretable | `m21-xai-shap` | explainability | SHAP wrapper; Hermes `explain` + contrastive + counterfactual. |
| m22 | wt-pm-vae-reconstruction-loss | `m22-vae-reconstruction` | anomaly | VAE reconstruction-error stream. |
| m23 | wt-pm-aerozip-autoencoder-compressor | `m23-aerozip` | compression | 8:1 telemetry compression; latent as features. |
| m24 | wt-pm-quantized-mobilenet-edge | `m24-quantized-edge` | edge | INT8 TFLite recipe (full MobileNet out of scope — no image data). |
| m25 | wt-pm-tinyml-esp32-safety-relay | `m25-tinyml-safety` | safety | Depth-5 tree → ESP32 `model.h`; Hermes `safety` tool. |

**Hermes tools** (Thought → Action → Observation): `sensor_quality`, `anomaly`, `diagnose`, `explain`, `rul`, `safety`, `what_if`, `finish`.

**XAI stack:** m21 SHAP + healthy-band contrastive z + feature counterfactual + Hermes narrative (observations never invented).
