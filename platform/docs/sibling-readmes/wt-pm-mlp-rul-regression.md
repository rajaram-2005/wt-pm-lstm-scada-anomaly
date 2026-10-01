# MLP RUL Regression

> **Category:** RUL Prognostics  
> **Platform adapter:** `m17-mlp-rul`  
> **Status:** integrated (proxy target)

## Description

Multi-Layer Perceptron baseline regressor mapping accumulated operating hours to component failure limits.

This is model **m17** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m17-mlp-rul` — it does not replace or fork this code.

## Model

`train_mlp(csv_path="rul_data.csv", target="RUL")` in [`model.py`](model.py) builds a
scikit-learn pipeline — `StandardScaler` → `MLPRegressor(hidden_layer_sizes=(128, 64),
activation='relu', max_iter=500, seed 42)` — fits it on a CSV feature table and prints R².

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

`python model.py` defines `train_mlp`; it loads no data until a CSV path is supplied. `requirements.txt` lists `scikit-learn pandas numpy`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features + RUL **proxy** target (research mode only) |
| Output (WTDataSchema) | rul_hours |
| Integration role | The deliberately simple RUL baseline every other RUL model must beat; its predictions are fed to the particle filter (m16) as observations. |
| Deployment / fallback | cloud, edge-CPU · scikit-learn · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `rul_hours`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `anomaly_score`, `degradation_state`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Hermes quotes the baseline's RUL estimate alongside the fancier estimators in its `rul` observations — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m17-mlp-rul should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The RUL target is a **proxy** (time-to-next-fault-onset in the simulator); no run-to-failure field data exists, and every explanation says so.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
