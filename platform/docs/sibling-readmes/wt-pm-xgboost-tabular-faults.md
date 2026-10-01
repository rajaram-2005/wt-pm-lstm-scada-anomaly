# XGBoost Tabular Faults

> **Category:** Component Fault Classification  
> **Platform adapter:** `m11-xgboost-tabular`  
> **Status:** integrated (primary tabular voter)

## Description

Gradient boosted decision trees for real-time multi-class component state classification using structured sensor features.

This is model **m11** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m11-xgboost-tabular` — it does not replace or fork this code.

## Model

`train_xgboost(csv_path="features.csv", target="fault_label")` in [`model.py`](model.py)
fits an `XGBClassifier` (300 trees, depth 6, learning rate 0.05, subsample 0.8, `mlogloss`)
on a stratified 80/20 split of a CSV feature table and prints the classification report.

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

`python model.py` defines `train_xgboost`; it loads no data until a CSV path is supplied. `requirements.txt` lists `xgboost scikit-learn pandas numpy`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features + labels (research mode) |
| Output (WTDataSchema) | prediction, probability |
| Integration role | Primary tabular fault voter (WHAT/WHERE) and the SHAP TreeExplainer source for the XAI engine. Hermes `diagnose` tool. |
| Deployment / fallback | cloud, edge-CPU · falls back to `m10-random-forest` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction`, `probability`, `subsystem`, `inference_time_ms` and `explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The platform's Hermes loop uses this classifier's vote in `diagnose` and its fitted booster in `explain` (via m21 SHAP). Observations are the adapter's real outputs — never invented; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m11-xgboost-tabular should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Labels come from the simulator's fault families (research mode); production runs vote without labels.
- `use_label_encoder=False` pins the legacy XGBoost API; the adapter tolerates both old and new xgboost releases.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
