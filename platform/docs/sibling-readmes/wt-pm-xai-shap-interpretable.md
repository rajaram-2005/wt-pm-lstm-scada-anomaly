# XAI SHAP Interpretable

> **Category:** Physics-Informed Models  
> **Platform adapter:** `m21-xai-shap`  
> **Status:** integrated

## Description

SHAP and LIME model interpretation wrapper providing root-cause diagnostic explanations for black-box alerts.

This is model **m21** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m21-xai-shap` — it does not replace or fork this code.

## Model

`explain_with_shap(model, X)` in [`model.py`](model.py) dispatches to
`shap.TreeExplainer` when the model exposes `get_booster` (XGBoost) and to the generic
`shap.Explainer` otherwise, then renders summary + waterfall plots and returns the SHAP values.
A commented stub shows the LIME alternative.

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

`python model.py` defines `explain_with_shap`; nothing is explained until a fitted model and feature frame are supplied. `requirements.txt` lists `shap lime xgboost scikit-learn`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | a fitted tree model + feature matrix (the platform passes m11/m10 and the batch features) |
| Output (WTDataSchema) | explanation — top-8 feature attributions; extra['shap_values'] |
| Integration role | XAI layer (WHY): powers Hermes `explain`, the contrastive healthy-band comparison and feature counterfactuals. |
| Deployment / fallback | cloud · shap · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `explanation` (top attributions) and `inference_time_ms`. The remaining optional fields (`prediction`, `probability`, `anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Hermes `explain` calls this wrapper; the reported attributions are the real SHAP values of the fitted tree classifiers — observations are never invented. The Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m21-xai-shap should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Wrapper only — it explains the fitted tree classifiers; there are no explanations of their own to ship.
- SHAP values are computed on the tabular classifiers (m10/m11), not on the deep models.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
