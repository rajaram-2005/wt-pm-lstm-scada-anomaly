# WT-PM Unified Platform

One production-shaped, modular predictive-maintenance intelligence system over
the 25 `rajaram-2005/wt-pm-*` research repositories — **without replacing any
of them**. Each repo keeps its research purpose and stays independently
usable; the platform calls the original `model.py` code through adapters.

- Inspection audit (done before any code): [`docs/INSPECTION.md`](docs/INSPECTION.md)
- Architecture report + repo/model table: [`docs/ARCHITECTURE_REPORT.md`](docs/ARCHITECTURE_REPORT.md)

## Layout

```
platform/
├── wtpm_platform/
│   ├── contracts.py      # WTDataSchema, SensorBatch, ModelOutput, OperatingContext
│   ├── data.py           # ingestion, cleaning/sync, FeaturePipeline
│   ├── base.py           # BaseWTModel, ModelSpec, ModelRegistry, repo loader
│   ├── adapters/
│   │   ├── anomaly.py            # m03 m04 m05 m13 m14 m22
│   │   ├── classification.py     # m01 m07 m08 m09 m10 m11 m12 m23
│   │   ├── prognostics.py        # m02 m06 m15 m16 m17
│   │   └── physics_graph_edge.py # m18 m19 m20 m21 m24 m25
│   ├── fusion.py         # FusionEngine (weighted/Bayesian/temporal/conflicts)
│   ├── engines.py        # Anomaly/Diagnosis/Uncertainty/RUL/XAI/Safety/Twin/Edge
│   ├── orchestrator.py   # ModelRouter, InferenceEngine, Orchestrator
│   ├── evaluation.py     # Evaluator, drift, ExperimentTracker
│   ├── cli.py            # research | production | fleet | edge | serve | inspect
│   └── api.py            # HTTP API + dashboard (stdlib only)
└── tests/                # 23 integration tests
```

## Exact commands to run the complete system

```bash
# 0) one-time setup (from the repo root)
git clone https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly
cd wt-pm-lstm-scada-anomaly
for r in wt-pm-1d-cnn-bearing-vibration wt-pm-convlstm-wear-prognostics \
         wt-pm-tcn-power-curve wt-pm-gru-scada-telemetry wt-pm-informer-long-sequence \
         wt-pm-snn-event-vibration wt-pm-contrastive-ssl-vibration \
         wt-pm-dbn-feature-extraction wt-pm-random-forest-telemetry \
         wt-pm-xgboost-tabular-faults wt-pm-svm-rbf-generator-stator \
         wt-pm-deep-svdd-boundary wt-pm-isolation-forest-telemetry \
         wt-pm-hmm-degradation-states wt-pm-particle-filter-rul \
         wt-pm-mlp-rul-regression wt-pm-pg-bnn-wind-turbine \
         wt-pm-digital-twin-surrogate wt-pm-gnn-turbines-cascade \
         wt-pm-xai-shap-interpretable wt-pm-vae-reconstruction-loss \
         wt-pm-aerozip-autoencoder-compressor wt-pm-quantized-mobilenet-edge \
         wt-pm-tinyml-esp32-safety-relay; do
  git clone --depth 1 https://github.com/rajaram-2005/$r external/$r
done
pip install -e .                    # the reference package (model 05 + schema)
pip install -e platform             # the platform (numpy/pandas/sklearn core)
pip install -e "platform[full]"     # optional: torch, tf, xgboost, shap, ... (all 25)

# 1) what is available in this environment?
wtpm-platform inspect

# 2) RESEARCH MODE: train, evaluate on a held-out record, compare, track
wtpm-platform research --days 20

# 3) PRODUCTION MODE: unlabelled inference + drift + what-if + alerts JSON
wtpm-platform production --days 15 --out production_result.json

# 4) fleet-level cascade analysis (m20 over wake-coupled farm)
wtpm-platform fleet --turbines 6 --days 8

# 5) edge/MCU artifacts: ESP32 model.h + INT8 .tflite
wtpm-platform edge --out edge_artifacts

# 6) HTTP API + live dashboard on :8100
wtpm-platform serve --port 8100
#   GET  /health  /registry  /plan       POST /analyse

# 7) tests
pytest platform/tests -q     # platform integration tests
pytest tests -q              # reference repo's own suite (must stay green)
```

Missing frameworks are never fatal: an adapter whose dependency is absent
reports itself `unavailable`, the router skips it, and its fallback covers the
task (e.g. every deep anomaly model falls back to the isolation forest).
