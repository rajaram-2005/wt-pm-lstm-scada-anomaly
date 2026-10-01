# GNN Turbine Cascade

> **Category:** Component Fault Classification  
> **Platform adapter:** `m20-gnn-cascade`  
> **Status:** integrated (wake graph from simulator)

## Description

Graph Neural Network capturing spatial dependency graphs between components to track cascading failure propagation.

This is model **m20** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m20-gnn-cascade` — it does not replace or fork this code.

## Model

`TurbineCascadeGNN(in_channels=16, hidden=32, out_classes=4)` in [`model.py`](model.py):

| Block | Layers |
|---|---|
| Message passing | 2 × (`GCNConv` → ReLU) over the turbine/component graph |
| Readout | `Linear(hidden, out_classes)` per node |
| Fallback | if `torch-geometric` is not installed, the repo itself defines a 2-layer MLP with the same signature (per-turbine, no graph) |

`python model.py` defines the architecture; it loads no data and trains nothing.

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

`python model.py` defines `TurbineCascadeGNN`; it loads no data and trains nothing. If `torch-geometric` is absent the file prints a notice and defines the linear fallback instead.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | fleet batches + wake graph — per-turbine node features, edges from wake coupling (built by model 05's `simulate_fleet`) |
| Output (WTDataSchema) | prediction — cascade-risk class per turbine |
| Integration role | Fleet layer: lifts per-turbine anomaly records to fleet-level cascade risk (`wt-pm fleet`). |
| Deployment / fallback | cloud · torch-geometric optional (repo has its own linear fallback) |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction` (risk class), `probability`, `inference_time_ms` and `explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Fleet-mode outputs appear in the platform report and the Hermes trace as observations of which turbines are wake-coupled into a developing cascade; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m20-gnn-cascade should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The wake graph comes from the **reference simulator**, not a SCADA farm topology feed.
- Training supervision for cascade risk is synthetic (self-loop risk labels); real cascade labels do not exist in the collection.
- Without torch-geometric the repo silently degrades to a per-turbine linear model — the adapter reports which path ran.
