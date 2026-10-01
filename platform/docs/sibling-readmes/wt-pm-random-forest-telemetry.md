# Random Forest Telemetry

> **Category:** Component Fault Classification  
> **Platform adapter:** `m10-random-forest`  
> **Status:** integrated

## Description

Ensemble Random Forest classifier providing feature importance rankings for SCADA sensor diagnostics.

This is model **m10** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m10-random-forest` — it does not replace or fork this code.

## Model

`train_rf(csv_path="scada.csv", target="state")` in [`model.py`](model.py) fits a
`RandomForestClassifier` (200 trees, unlimited depth, fixed seed 42, `n_jobs=-1`) on a CSV
feature table and prints the ten most important features.

## Structure

```
├── model.py           # Architecture / training entry point (unchanged by the platform)
├── requirements.txt   # Dependencies
├── .gitignore         # Environment, data and model-artifact exclusions
├── README.md          # Project overview (this file)
├── index.html         # Landing page (GitHub Pages)
└── docs/
    └── index.html     # Copy of the landing page
```

## Setup (standalone)

```bash
pip install -r requirements.txt
python model.py
```

`python model.py` defines `train_rf`; it loads no data until a CSV path is supplied. `requirements.txt` lists `scikit-learn pandas numpy`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features + labels (research mode) |
| Output (WTDataSchema) | prediction, probability, feature_importances |
| Integration role | Tabular fault voter (WHAT/WHERE); feature importances feed the XAI evidence bundle. It is also the fallback target for heavier classifiers. |
| Deployment / fallback | cloud, edge-CPU · scikit-learn · is itself a fallback target |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction`, `probability`, `subsystem`, `inference_time_ms` and `explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The platform's Hermes loop uses classifier votes in its `diagnose` tool; the forest's importances appear in the XAI evidence bundle. Observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m10-random-forest should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Labels come from the simulator's fault families (research mode); production runs vote without labels.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
