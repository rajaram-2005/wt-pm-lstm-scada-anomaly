# SVM-RBF Generator Stator

> **Category:** Component Fault Classification  
> **Platform adapter:** `m12-svm-generator`  
> **Status:** integrated (electrical specialist)

## Description

Support Vector Machine with Radial Basis Function kernel for generator stator/rotor insulation fault separation.

This is model **m12** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m12-svm-generator` — it does not replace or fork this code.

## Model

`build_svm(csv_path="generator.csv", target="stator_fault")` in [`model.py`](model.py) builds a
scikit-learn pipeline — `StandardScaler` → `SVC(kernel='rbf', C=1.0, gamma='scale',
probability=True, class_weight='balanced')` — and fits it on a CSV feature table.

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

`python model.py` defines `build_svm`; it loads no data until a CSV path is supplied. `requirements.txt` lists `scikit-learn pandas numpy`.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | electrical features — generator/converter channels + labels (research mode) |
| Output (WTDataSchema) | prediction, probability |
| Integration role | Generator/converter electrical-fault specialist (WHAT/WHERE). Hermes `diagnose` tool. |
| Deployment / fallback | cloud · falls back to `m11-xgboost-tabular` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction`, `probability`, `subsystem` (generator/converter), `inference_time_ms` and `explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The SVM's binary electrical vote is one input to Hermes `diagnose`; records without electrical-fault examples are skipped, not guessed. Observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m12-svm-generator should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The platform restricts the adapter to generator/converter electrical features (stator focus); other channels stay with the tabular ensemble.
- `class_weight='balanced'` is upstream; the adapter keeps it — rare electrical faults are deliberately not majority-voted away.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
