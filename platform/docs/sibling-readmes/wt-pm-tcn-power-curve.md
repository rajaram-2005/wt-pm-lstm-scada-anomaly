# TCN Power Curve

> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m03-tcn-power-curve`  
> **Status:** integrated

## Description

Temporal Convolutional Network for parallel sequence processing and aerodynamic power curve degradation tracking.

This is model **m03** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m03-tcn-power-curve` — it does not replace or fork this code.

## Model

[`model.py`](model.py) is a **complete 49-line causal TCN**:

| Symbol | What it is |
|---|---|
| `Chomp1d` | trims right-padding so convolutions stay causal |
| `TemporalBlock` | 2 × (weight-normed `Conv1d` → chomp → ReLU → `Dropout(0.2)`) + residual 1×1 downsample |
| `TCNPowerCurve(input_size, num_channels=[32,64,64], kernel_size=3, dropout=0.2)` | stacked dilated blocks (`dilation = 2**i`) → `Linear(num_channels[-1], 1)` power prediction from the last time step |

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

`python model.py` defines `Chomp1d`, `TemporalBlock` and `TCNPowerCurve`; it loads no data and trains nothing. `requirements.txt` lists `torch pandas numpy`; `model.py` itself only imports PyTorch.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | sequence windows + `power_kw` channel |
| Output (WTDataSchema) | anomaly_score — power-curve residual; prediction — power forecast |
| Integration role | Aerodynamic power-curve degradation tracking (rotor blades / converter). Hermes `anomaly` tool. |
| Deployment / fallback | cloud, edge-GPU · falls back to `m04-gru-scada-telemetry` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `anomaly_score`, `prediction` (power forecast), `inference_time_ms` and `explanation`. The remaining optional fields (`probability`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The TCN's residual score is one anomaly vote in Hermes `anomaly`; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m03-tcn-power-curve should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- The platform trains the residual model briefly on simulator windows; no field power-curve reference ships here.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
