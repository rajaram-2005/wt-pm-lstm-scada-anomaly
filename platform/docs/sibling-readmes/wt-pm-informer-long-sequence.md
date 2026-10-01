# Informer Long Sequence

> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m06-informer-forecast`  
> **Status:** partial (placeholder attention)

## Description

ProbSparse self-attention Informer architecture for ultra-long multi-step wind farm power forecasting.

This is model **m06** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m06-informer-forecast` — it does not replace or fork this code.

## Model

[`model.py`](model.py) defines:

| Symbol | What it is |
|---|---|
| `ProbSparseAttention(d_model=512, n_heads=8, factor=5)` | attention block — **a self-declared placeholder**: dense QKV attention with a comment to replace it with full Informer ProbSparse attention |
| `InformerPowerForecaster(enc_in=12, dec_in=12, c_out=1, d_model=512)` | encoder embedding → attention over encoder output → linear projection to power |

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

`python model.py` defines the two classes; it loads no data and trains nothing. `requirements.txt` lists `torch pandas numpy`; `model.py` itself only imports PyTorch.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | sequence windows over the canonical 12 SCADA channels |
| Output (WTDataSchema) | prediction — long-horizon power forecast; anomaly_score — forecast residual |
| Integration role | Forecast-residual anomaly stream. |
| Deployment / fallback | cloud · falls back to `m03-tcn-power-curve` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction` (power forecast), `anomaly_score` (residual), `inference_time_ms` and `explanation`. The remaining optional fields (`probability`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

Forecast residuals join the anomaly evidence bundle Hermes reasons over; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m06-informer-forecast should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- **Partial by upstream design:** the repo's ProbSparse attention is a self-declared placeholder; the platform integrates around what exists and labels the model `partial` in the registry and in every run report.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
