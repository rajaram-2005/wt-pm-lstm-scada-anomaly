# SNN Event Vibration

> **Category:** Edge AI  
> **Platform adapter:** `m07-snn-vibration`  
> **Status:** integrated (surrogate events)

## Description

Spiking Neural Network processing asynchronous event-based threshold spikes for ultra-low power vibration monitoring.

This is model **m07** of the WT-PM wind-turbine predictive-maintenance suite. The repository stays
independently usable (`python model.py`). The unified **wt-pm** platform wraps the original
`model.py` through adapter `m07-snn-vibration` — it does not replace or fork this code.

## Model

`EventSNN(input_size=64, hidden=128, output_size=4)` in [`model.py`](model.py):

| Block | Layers |
|---|---|
| Input stage | `nn.Linear(input_size, hidden)` → `snn.Leaky(beta=0.9)` LIF membrane |
| Output stage | `nn.Linear(hidden, output_size)` → `snn.Leaky(beta=0.9)` |
| Forward | returns output spikes + both membrane states (recurrent stepping) |

If `snntorch` is not installed the repo prints an install hint and defines no class — the
platform adapter then reports itself unavailable.

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

`python model.py` defines `EventSNN` if snntorch is installed, otherwise prints an install hint. It loads no data and trains nothing.

To use the model from your own code, import from `model.py` directly — the entry points above
are the exact symbols the platform adapter calls.

## Platform contract

| | |
|---|---|
| Input (adapter view) | event-encoded vibration windows — delta/threshold spike bins from surrogate waveforms |
| Output (WTDataSchema) | prediction, probability — spike-rate logits as class evidence |
| Integration role | Low-power vibration voter (drivetrain). Hermes `diagnose` tool. |
| Deployment / fallback | cloud, edge-CPU, edge-GPU · falls back to `m01-1dcnn-bearing` |

Records follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` are required;
this adapter also fills `prediction`, `probability`, `subsystem` (drivetrain), `inference_time_ms` and `explanation`. The remaining optional fields (`anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to other models.

## Hermes agent + explainable AI

The SNN's class vote counts as one drivetrain vote in Hermes `diagnose`; observations are the adapter's real outputs — never invented. XAI is m21 SHAP + contrastive healthy-band z + counterfactuals (SHAP is computed on the tree classifiers, not on this SNN).

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m07-snn-vibration should report "available": true
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md)
and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Event encodings are built from **surrogate** vibration waveforms synthesised from the 10-minute RMS channel (flagged `vibration_is_surrogate=True`); real DAQ recordings would slot in unchanged.
- Optional dependency: without `snntorch` the model reports unavailable and the router falls back to the 1D-CNN.
- Metrics on the platform are held-out **simulator** records (fidelity rung 1), not certified asset performance.
