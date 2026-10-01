# Digital Twin Surrogate

> **Category:** Physics-Informed Models  
> **Platform adapter:** `m19-digital-twin`  
> **Status:** integrated (physics-proxy targets)

## Description

Deep neural surrogate replacing heavy Finite Element Analysis (FEA) for real-time digital twin residual tracking.

This is model **m19** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m19-digital-twin` — it does not replace or fork this code.

## Model

`model.py` ships two building blocks:

| Symbol | What it is |
|---|---|
| `FEASurrogate(input_dim=8, hidden=64, output_dim=3)` | 3-layer MLP (Linear + ReLU twice, then Linear) mapping operating-condition vectors `[wind_speed, rpm, pitch, yaw, temp, ...]` to `[von_mises, deflection, fatigue]` load proxies |
| `residual_loss(pred, fea_target, physics_weight=0.1)` | MSE data loss plus a stated-placeholder physics term |

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

`python model.py` defines `FEASurrogate` and `residual_loss`; it loads no data and trains nothing. `requirements.txt` lists `torch numpy scipy`; `model.py` itself only imports PyTorch.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | operating-condition channels (wind speed, rotor rpm, pitch, temperatures) |
| Output (WTDataSchema) | prediction — stress / deflection / fatigue load proxies |
| Integration role | Digital-twin layer. Hermes `what_if` queries are answered by the surrogate; physics residuals feed back to the anomaly detectors. |
| Deployment / fallback | cloud, edge-GPU · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction` (the three load proxies), `inference_time_ms` and `explanation`. The remaining optional fields (`probability`, `anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The platform's Hermes loop (Thought → Action → Observation) uses this model through its `what_if` tool: counterfactual operating points are pushed through the surrogate and the predicted load change is reported as an observation — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m19-digital-twin should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The surrogate is trained against **physics-proxy load targets from the reference simulator**, not FEA runs or strain gauges.
- `residual_loss`'s physics term is a self-declared placeholder upstream; the platform documents this rather than completing it.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
