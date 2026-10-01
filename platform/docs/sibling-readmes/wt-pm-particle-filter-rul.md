# Particle Filter RUL

> **Category:** RUL Prognostics  
> **Platform adapter:** `m16-particle-filter-rul`  
> **Status:** complete

## Description

Stochastic Particle Filtering fused with Deep Neural Networks to update probabilistic RUL decay distributions continuously.

This is model **m16** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m16-particle-filter-rul` — it does not replace or fork this code.

## Model

`ParticleFilterRUL(num_particles=1000, init_rul=1000)` in [`model.py`](model.py) is a
**complete, pure-numpy** sequential Monte Carlo filter:

| Method | What it does |
|---|---|
| `predict(degradation_rate, noise)` | propagates every particle down by the decay rate + Gaussian noise, floored at 0 |
| `update(observed_rul, obs_noise)` | Gaussian likelihood reweighting + normalisation |
| `resample()` | multinomial resampling, weights reset uniform |
| `estimate()` | weighted mean RUL |

`python model.py` defines the class; it loads no data.

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

`python model.py` defines `ParticleFilterRUL`; it loads no data. Despite `requirements.txt` listing `scipy`/`torch`, `model.py` itself only imports numpy.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | RUL observations from sibling estimators (m17 MLP, m02 ConvLSTM) + degradation rate from m15 |
| Output (WTDataSchema) | rul_hours, uncertainty — the full particle posterior |
| Integration role | RUL fusion core (HOW LONG): consumes sibling RUL estimates as observations and reports a distribution, not just a point. |
| Deployment / fallback | cloud, edge-CPU · numpy only · no fallback |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `rul_hours`, `uncertainty`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `anomaly_score`, `degradation_state`) are left to other models.

## Hermes agent + explainable AI

Hermes reports the filter's posterior mean and spread in its `rul` observations — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m16-particle-filter-rul should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The filter is fed **sibling RUL estimates**, never ground truth (which production would not have).
- The platform adds an underflow guard around the likelihood weighting; the repo's math is otherwise used as-is.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
