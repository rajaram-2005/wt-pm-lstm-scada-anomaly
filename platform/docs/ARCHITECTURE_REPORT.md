# WT-PM Unified Platform — Architecture Report

**GitHub map (this repo vs 24 siblings, mermaid, commands):**
[ARCHITECTURE.md](../../ARCHITECTURE.md)

**Deliverable:** 25 specialized models + 1 common data layer + 1 orchestrator +
1 fusion engine + 1 diagnosis engine + 1 RUL engine + 1 XAI layer +
1 digital-twin layer + 1 edge/safety layer + 1 Hermes agent +
1 monitoring/evaluation layer.

The integration layer lives in `platform/wtpm_platform/` inside the
`wt-pm-lstm-scada-anomaly` repository (the reference repo of the collection).
The 24 sibling repositories are **cloned read-only into `external/`** and are
called through adapters — none of them was modified, and each remains
independently usable exactly as before.

```text
Sensors/SCADA ──> data.ingest_* ──> data.clean/operating_state ──> FeaturePipeline
                                                                    │ tabular features
                                                                    │ sequence windows
                                                                    │ vibration surrogates
                                                                    ▼
      ModelRouter (per condition) ──> InferenceEngine (parallel/timeout/retry/fallback)
                                                                    ▼
                                   25-Model Engine (adapters over external repos)
                                                                    ▼
   FusionEngine (weighted / confidence / Bayesian-PoE / temporal consensus / conflicts)
        ▼                 ▼                    ▼                 ▼
  AnomalyEngine    DiagnosisEngine      HMM degradation      RULManager (fusion + m16)
  (WHAT)           (WHAT fault/WHERE)   (SEVERITY)           (HOW LONG)
        └──────────────┬────────────────────┴───────────────────┘
                       ▼
        risk score ──> XAIEngine (WHY, SHAP m21) ──> recommendation (WHAT NEXT)
                       ▼
        SafetyManager (m25 gate + hard limits) ──> CONTINUE / INSPECT / DERATE / TRIP
                       ▼
        DigitalTwinInterface (m19 what-if) · GNN fleet cascade (m20)
                       ▼
        HTTP API + dashboard (`wtpm-platform serve`) · ExperimentTracker · drift monitor
```

## Repository → Model → Purpose → Input → Output → Integration Role → Deployment → Dependencies → Status

| # | Repository | Model | Purpose | Input (adapter view) | Output (contract) | Integration role | Deployment | Deps | Status |
|---|-----------|-------|---------|---------------------|-------------------|-----------------|-----------|------|--------|
| m01 | wt-pm-1d-cnn-bearing-vibration | Keras 1D-CNN | bearing fault classification | vibration waveforms (surrogate, flagged) | prediction, probability | drivetrain fault voter | cloud, edge-GPU | tensorflow | integrated (surrogate data) |
| m02 | wt-pm-convlstm-wear-prognostics | Keras ConvLSTM2D | spatiotemporal wear RUL | 16×16 channel-recurrence image sequences | rul_hours | RUL point estimator | cloud | tensorflow | **partial** (repo expects wear maps that don't exist; reduced grid) |
| m03 | wt-pm-tcn-power-curve | TCN (torch) | power-curve degradation | sequence windows | anomaly_score (power residual) | anomaly stream | cloud, edge-GPU | torch | integrated |
| m04 | wt-pm-gru-scada-telemetry | GRU forecaster (torch) | SCADA stream anomaly | sequence windows | anomaly_score | anomaly stream | cloud, edge | torch | integrated |
| m05 | wt-pm-lstm-scada-anomaly | full reference package | SCADA sequence anomaly + contract | canonical 12-ch timeline | anomaly_score, uncertainty, threshold | reference detector; schema/eval/drift authority | cloud, edge-CPU | numpy | **complete** (used via public API) |
| m06 | wt-pm-informer-long-sequence | Informer (torch) | long-horizon power forecast | sequence windows | anomaly_score (residual) | forecast residual stream | cloud | torch | **partial** (repo's attention is a placeholder) |
| m07 | wt-pm-snn-event-vibration | snnTorch SNN | event vibration classes | delta-encoded spike bins | prediction, probability | low-power vibration voter | edge | snntorch | integrated (surrogate events) |
| m08 | wt-pm-contrastive-ssl-vibration | SimCLR-style encoder | SSL vibration features | unlabelled waveforms | embeddings | feature provider | cloud, edge-GPU | torch | **partial** (repo's NT-Xent simplified) |
| m09 | wt-pm-dbn-feature-extraction | RBM stack | deep features | tabular features | embeddings | feature provider | cloud | torch | integrated (platform added CD-1 loop) |
| m10 | wt-pm-random-forest-telemetry | sklearn RF | telemetry fault classes | tabular features + labels | prediction, probability, importances | tabular voter + XAI evidence | cloud, edge-CPU | sklearn | integrated |
| m11 | wt-pm-xgboost-tabular-faults | XGBoost | tabular fault classes | tabular features + labels | prediction, probability | primary tabular voter; SHAP source | cloud, edge-CPU | xgboost | integrated |
| m12 | wt-pm-svm-rbf-generator-stator | SVM-RBF | electrical faults | electrical features | prediction, probability | generator/converter specialist | cloud | sklearn | integrated (skips records without electrical-fault examples) |
| m13 | wt-pm-deep-svdd-boundary | Deep SVDD (torch) | one-class boundary | tabular features | anomaly_score | anomaly stream | cloud, edge | torch | integrated (uses repo's init_center + svdd_loss) |
| m14 | wt-pm-isolation-forest-telemetry | IsolationForest | unsupervised outliers | tabular features | anomaly_score | cheapest stream; universal fallback | cloud, edge-CPU | sklearn | integrated (adapter refuses fillna(0)) |
| m15 | wt-pm-hmm-degradation-states | GaussianHMM | degradation states | health-indicator series | degradation_state, posterior | SEVERITY answer; feeds m16 decay rate | cloud | hmmlearn | integrated (severity-ordered states) |
| m16 | wt-pm-particle-filter-rul | particle filter (numpy) | probabilistic RUL | upstream RUL observations + m15 states | rul_hours, uncertainty | RUL fusion core | cloud, edge-CPU | numpy | **complete** (+ underflow guard in adapter) |
| m17 | wt-pm-mlp-rul-regression | sklearn MLP | baseline RUL | tabular features + proxy target | rul_hours | RUL baseline all others must beat | cloud, edge-CPU | sklearn | integrated (proxy target, stated) |
| m18 | wt-pm-pg-bnn-wind-turbine | BayesianLinear + physics loss | physics-guided power model | context channels, P=τω | anomaly_score, uncertainty | physics-aware stream + uncertainty | cloud | torch | integrated (platform assembled BNN from repo blocks) |
| m19 | wt-pm-digital-twin-surrogate | FEA surrogate MLP | load proxies + what-if | operating-condition channels | stress/deflection/fatigue proxies | digital-twin layer | cloud, edge-GPU | torch | integrated (physics-proxy targets, stated) |
| m20 | wt-pm-gnn-turbines-cascade | GCN (PyG or repo fallback) | fleet cascade risk | farm node features + wake edges | risk class per turbine | fleet layer | cloud | torch(+pyg) | integrated (wake graph from m05's simulate_fleet) |
| m21 | wt-pm-xai-shap-interpretable | SHAP wrapper | explanations | fitted tree model + features | top-8 attributions | XAI layer | cloud | shap | integrated |
| m22 | wt-pm-vae-reconstruction-loss | VAE (torch) | reconstruction anomaly | tabular features | anomaly_score, uncertainty | anomaly stream | cloud, edge-GPU | torch | integrated (uses repo's vae_loss) |
| m23 | wt-pm-aerozip-autoencoder-compressor | autoencoder (torch) | telemetry compression | tabular features | latent code, recon error, ratio | backhaul compression + features | edge | torch | integrated |
| m24 | wt-pm-quantized-mobilenet-edge | TFLite INT8 recipe | edge quantization | any Keras model + rep. data | .tflite artifact | edge conversion service | edge | tensorflow | integrated (recipe applied to platform edge net; full MobileNet out of scope — no image data) |
| m25 | wt-pm-tinyml-esp32-safety-relay | DecisionTree→C++ | MCU safety trips | 4 safety channels | TRIP/continue + model.h | safety gate + ESP32 export | MCU | sklearn, micromlgen | integrated |

**Status legend** — *complete*: repo functionality used as-is; *integrated*: repo architecture + platform-supplied training/adaptation; *partial*: repo itself is incomplete (placeholder internals) and is integrated around what exists, honestly labelled.

## The seven questions and who answers them

| Question | Engine | Models |
|----------|--------|--------|
| What is happening? | AnomalyEngine + DiagnosisEngine | m03–m06, m13, m14, m18, m22 (fused) → m01, m07, m10–m12 (voted) |
| Where is it happening? | DiagnosisEngine subsystem map | fault→subsystem table + m12 (generator), m01/m07 (drivetrain) |
| Why is it happening? | XAIEngine | m21 SHAP on m11/m10 + physics residual feature + anomaly evidence bundle |
| How severe is it? | m15 severity-ordered HMM states | m15 (+ health indicator) |
| How long until failure? | RULManager | m17 baseline + m02 → inverse-variance fusion → m16 particle filter |
| What should happen next? | Orchestrator recommendation | risk score over all of the above |
| Can it keep operating? | SafetyManager | m25 gate + hard limits + RUL derating |

## Honest limitations (no fabricated capabilities)

1. **All 24 sibling repos ship untrained architectures.** The platform trains
   them per-record on simulated data from model 05's physics simulator (rung 1
   of the fidelity ladder). Reported metrics are held-out-record metrics on
   that simulator, not field performance.
2. **Vibration waveforms are surrogates** synthesised from the 10-min RMS
   channel (flagged `vibration_is_surrogate=True`). m01/m07/m08 would accept
   real DAQ waveforms unchanged.
3. **RUL targets are proxies** (time-to-next-fault-onset). No repo has
   run-to-failure data; every RUL output says so in its explanation.
4. **m02, m06, m08 are partial by upstream design** (placeholder attention,
   simplified loss, missing wear-map data source) and are labelled as such in
   the registry and in every run report.
5. **m24's full MobileNet path is out of scope** — there is no image data
   anywhere in the collection; its INT8 conversion recipe is applied to the
   platform's small telemetry edge net instead (3.4 KB artifact, MCU-sized).
6. Cross-model comparison tables group only comparable tasks (anomaly vs
   anomaly, RUL vs RUL); learned fusion weights come from held-out ROC-AUC,
   never from the training band.

## Verification performed

- `pytest platform/tests` — 23/23 pass (contract, cleaning, fusion incl.
  conflict detection and confidence weighting, router deployment/label gating,
  8 adapter behaviour tests, fallback resolution, metrics, drift).
- `pytest tests` (the reference repo's own 43 tests) — all pass after
  integration; the reference project remains fully functional.
- `git status` in all 24 external clones — pristine, zero modifications.
- End-to-end: research mode (fit 23 models, held-out cross-model comparison,
  learned fusion weights, run record), production mode (unlabelled record,
  drift report, what-if scenario, JSON output), fleet mode (6 wake-coupled
  turbines through m20), edge mode (model.h + INT8 tflite artifacts).
