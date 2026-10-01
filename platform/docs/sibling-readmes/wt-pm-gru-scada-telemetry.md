# GRU SCADA Telemetry

> **Category:** SCADA Anomaly Detection  
> **Platform adapter:** `m04-gru-scada-telemetry`  
> **Status:** integrated

## Description

Gated Recurrent Unit model for lightweight, high-throughput SCADA continuous stream processing.

This is model **m04** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m04-gru-scada-telemetry` — it does not replace or fork this code.

## Model

`GRUAnomalyDetector(input_size, hidden_size, num_layers)` in [`model.py`](model.py):

| Block | Layers |
|---|---|
| Sequence model | `nn.GRU(input_size, hidden_size, num_layers, batch_first=True)` |
| Head | `nn.Linear(hidden_size, input_size)` — predicts the **next step** of every channel from the last hidden state |

One-step-ahead forecast error is the anomaly score. `python model.py` defines the class; it loads no data and trains nothing.

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

`python model.py` defines `GRUAnomalyDetector`; it loads no data and trains nothing. `requirements.txt` lists `torch pandas numpy scikit-learn`; `model.py` itself only imports PyTorch.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | sequence windows over the canonical 12 SCADA channels |
| Output (WTDataSchema) | anomaly_score — standardised one-step forecast error |
| Integration role | Anomaly stream (WHAT). Hermes `anomaly` tool. |
| Deployment / fallback | cloud, edge-GPU, edge-CPU · falls back to `m14-isolation-forest` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `anomaly_score`, `inference_time_ms` and `explanation`. The remaining optional fields (`prediction`, `probability`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The platform's Hermes loop (Thought → Action → Observation) uses this model through its `anomaly` tool: the GRU's forecast error counts as one anomaly vote alongside the other detectors. Observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m04-gru-scada-telemetry should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Architecture only upstream — the platform supplies the training loop (short fits on simulator windows).
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
