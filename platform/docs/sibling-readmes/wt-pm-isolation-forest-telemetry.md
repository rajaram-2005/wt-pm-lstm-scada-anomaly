# Isolation Forest

> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m14-isolation-forest`  
> **Status:** integrated (universal fallback)

## Description

Unsupervised Isolation Forest for high-speed multi-dimensional SCADA telemetry outlier detection.

This is model **m14** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m14-isolation-forest` — it does not replace or fork this code.

## Model

`train_isolation_forest(csv_path="scada.csv", contamination=0.02)` in [`model.py`](model.py)
fits an `IsolationForest` (200 trees, fixed seed 42, `n_jobs=-1`) on the numeric columns of a
CSV and annotates it with `anomaly` (-1 = anomaly) and `score` (`decision_function`) columns.

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

`python model.py` defines `train_isolation_forest`; the `__main__` block is a no-op placeholder until a CSV path is supplied. `requirements.txt` lists `scikit-learn pandas numpy`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | tabular features — mask-filtered rows (the adapter never uses the repo's `fillna(0)` default) |
| Output (WTDataSchema) | anomaly_score — inverted decision function |
| Integration role | Cheapest unsupervised anomaly stream; the **universal fallback** for the heavier detectors. |
| Deployment / fallback | cloud, edge-CPU · scikit-learn · is itself the fallback target |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `anomaly_score`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The forest's outlier score is one vote in the fused anomaly evidence Hermes reasons over; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m14-isolation-forest should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The repo's `fillna(0)` would turn gaps into fake healthy readings — the platform passes mask-filtered rows instead; the repo itself is untouched.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
