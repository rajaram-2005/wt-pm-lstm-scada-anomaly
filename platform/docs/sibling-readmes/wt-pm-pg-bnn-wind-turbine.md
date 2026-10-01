# PG-BNN Physics-Guided

> **Category:** Physics-Informed Models  
> **Platform adapter:** `m18-pg-bnn`  
> **Status:** integrated (BNN assembled from repo blocks)

## Description

Physics-Guided Bayesian Neural Network embedding thermodynamic/mechanical loss constraints with predictive uncertainty estimates.

This is model **m18** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m18-pg-bnn` — it does not replace or fork this code.

## Model

[`model.py`](model.py) ships two blocks:

| Symbol | What it is |
|---|---|
| `PhysicsGuidedLoss(lambda_physics=0.1)` | MSE data loss + the mechanical-power constraint `P = τ · ω` with `ω = 2π·RPM/60` |
| `BayesianLinear(in_features, out_features)` | reparameterised linear layer with per-weight `mu`/`logvar` |

The repo ships the blocks, not a full network: the platform assembles a small BNN from
`BayesianLinear` and trains it with the physics loss. `python model.py` defines the blocks only.

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

`python model.py` defines `PhysicsGuidedLoss` and `BayesianLinear`; it loads no data and trains nothing. `requirements.txt` lists `torch pyro-ppl numpy pandas matplotlib`; `model.py` itself only imports PyTorch.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | torque / rpm / power context channels + tabular features |
| Output (WTDataSchema) | anomaly_score, uncertainty — Monte-Carlo sampled predictive spread |
| Integration role | Physics-aware anomaly stream: predicts power from context under the P=τω loss; large physics-consistent residuals flag anomalies with calibrated uncertainty. |
| Deployment / fallback | cloud · torch · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `anomaly_score`, `uncertainty`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `degradation_state`, `rul_hours`) are left to other models.

## Hermes agent + explainable AI

The BNN's uncertainty is quoted in Hermes observations whenever it is routed; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m18-pg-bnn should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The repo ships loss + layer blocks only; the platform assembles and trains the network (documented in the registry notes).
- The corrected adapter feeds **predicted** power into the physics term and normalises both sides in kW, so the constraint actually gradients the weights.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
