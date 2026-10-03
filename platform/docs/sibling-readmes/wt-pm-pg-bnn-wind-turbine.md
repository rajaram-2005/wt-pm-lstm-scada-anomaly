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
| `PhysicsGuidedLoss(lambda_physics=0.1)` | MSE data loss + soft mechanical-power penalty; the adapter scales the relation to `P_electric ≈ 0.94 × τ · ω` in consistent kW units |
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
| Input (adapter view) | wind speed, ambient temperature, rotor speed, pitch, torque; measured power is the prediction target |
| Output (WTDataSchema) | anomaly_score + uncertainty; `/analyse` adds predicted/measured power and model/physics residuals under `physics.pg_bnn` |
| Integration role | Physics-aware anomaly voter with held-out healthy residual calibration and repeatable Monte-Carlo epistemic spread; the spread is not calibrated uncertainty. |
| Deployment / fallback | cloud · torch · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `anomaly_score`, `uncertainty`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `degradation_state`, `rul_hours`) are left to other models.

## Hermes agent + explainable AI

When m18 runs, Hermes carries its measured/predicted power, model and physics residuals, and MC epistemic spread as tool observations; it does not invent them. The adapter's output is also visible in the `/analyse` response and dashboard. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals.

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
- The held-out healthy tail calibrates model residuals only; posterior spread and the approximate `mean ± 1.96σ` band are not calibrated coverage guarantees.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
