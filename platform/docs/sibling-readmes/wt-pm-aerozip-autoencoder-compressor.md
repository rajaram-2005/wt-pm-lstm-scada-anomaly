# AeroZip Autoencoder Compressor

> **Category:** Edge AI  
> **Platform adapter:** `m23-aerozip`  
> **Status:** integrated

## Description

Deep Autoencoder compressing high-dimensional telemetry into latent space for low-bandwidth cellular/satellite transfer.

This is model **m23** of the WT-PM wind-turbine predictive-maintenance suite (second repository alphabetically in the collection). The repository stays independently usable (`python model.py`). The unified **wt-pm** platform wraps the original `model.py` through adapter `m23-aerozip` — it does not replace or fork this code.

## Model

[`model.py`](model.py) defines `AeroZipCompressor(input_dim=64, latent_dim=8)` (a PyTorch `nn.Module` bottleneck autoencoder) and `compression_ratio(input_dim=64, latent_dim=8)`:

| Block | Layers |
|---|---|
| `encoder` | `Linear(input_dim, 32)` → `ReLU` → `Linear(32, latent_dim)` |
| `decoder` | `Linear(latent_dim, 32)` → `ReLU` → `Linear(32, input_dim)` |
| `forward(x)` | Returns `(latent, reconstructed)` with shapes `(..., latent_dim)` and `(..., input_dim)` |
| `compression_ratio` | Returns `input_dim / latent_dim` (`8.0` for the default `64 → 8` configuration) |

- **Input:** tabular feature tensor `x` of shape `(batch_size, input_dim)` (`float32`).
- **Loss (caller-supplied):** mean-squared reconstruction error, e.g. `nn.MSELoss()(reconstructed, x)`.
- **Size:** 4,744 parameters for the default `(input_dim=64, latent_dim=8)` configuration (2,794 parameters when the platform instantiates it on its 38-column feature table with `latent_dim=4`).

## Structure

```
├── model.py           # Autoencoder architecture (AeroZipCompressor, compression_ratio); "python model.py" prints its summary
├── requirements.txt   # Dependencies
├── .gitignore         # Environment, data and model-artifact exclusions
├── README.md          # Project overview
├── index.html         # Landing page (GitHub Pages)
└── docs/
    └── index.html     # Copy of the landing page
```

## Setup (standalone)

```bash
pip install -r requirements.txt
python model.py
```

`python model.py` builds the reference configuration (`input_dim=64, latent_dim=8`) and prints the PyTorch module structure, parameter count (`4,744`), and nominal compression ratio (`8.0:1`). It loads no data and trains nothing. Verified with Python 3.11 and PyTorch 2.14.

`requirements.txt` lists `torch numpy`; `model.py` itself only imports `torch` (`torch.nn`).

To use the model from your own code:

```python
import torch
import torch.nn as nn
from model import AeroZipCompressor, compression_ratio

model = AeroZipCompressor(input_dim=64, latent_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

# x: (batch_size, 64) float32 tensor of standardised telemetry features
latent, reconstructed = model(x)
loss = criterion(reconstructed, x)
ratio = compression_ratio(64, 8)  # 8.0
```

## Platform contract

| | |
|---|---|
| Input (adapter view) | 38-column tabular feature matrix `batch.features` from `FeaturePipeline` (12 SCADA channels + 12-step rolling mean/std + `physics_power_residual_kw` + `power_per_v3`), z-score standardised on permitted healthy training rows |
| Output (`ModelOutput` / `WTDataSchema`) | `explanation` (`"AeroZip latent code, compression ratio 9.5:1"`), plus `extra={"latent": (T, 4), "recon_error": (T,), "compression_ratio": 9.5}` |
| Integration role | Backhaul compression + latent feature stage (`TaskType.COMPRESSION`, routed in `support`). Not a fault or anomaly voter. |
| Deployment / fallback | `cloud`, `edge_cpu`, `edge_gpu` (requires `torch`) · no fallback model configured |

Records emitted via `ModelOutput.to_records()` follow `WTDataSchema` (`wt-pm.platform.v1`): `timestamp`, `turbine_id` and `model_id` (`m23-aerozip`) are required; this adapter also fills `subsystem` (`unknown`), `inference_time_ms` and `explanation`, while `latent`, `recon_error` and `compression_ratio` live on `ModelOutput.extra` (and `compression_ratio` is surfaced under `result["support"]["m23-aerozip"]` in `Orchestrator.analyse()`). Decision fields (`prediction`, `probability`, `anomaly_score`, `degradation_state`, `rul_hours`, `uncertainty`) are left to the anomaly, classification and prognostics models.

## Hermes agent + explainable AI

In the platform's `Orchestrator.analyse()` pipeline, `ModelRouter` places `m23-aerozip` in the `support` bucket (`routed["support"]`) so it executes alongside the other representation/edge/twin adapters and reports its status, latency and `compression_ratio` in `model_health` and `support["m23-aerozip"]`. The Hermes agent loop (Thought → Action → Observation) reads the diagnostic, anomaly, RUL, twin, and safety outputs to answer the seven operator questions; `m23-aerozip` is a telemetry-compression support stage rather than a diagnostic voter. Observations in Hermes are always real adapter/engine outputs — never invented. XAI is `m21` SHAP (`TreeExplainer` on the tree classifiers) + contrastive healthy-band z + counterfactuals; the Hermes trace is the operator-readable explanation.

To run this model inside the platform (Python ≥ 3.10):

```bash
pip install "wt-pm[torch] @ git+https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly.git#subdirectory=platform"
wt-pm fetch-models --dir ./external      # clones this repo and its siblings into ./external
export WTPM_EXTERNAL_DIR=$PWD/external   # adapters import external/<repo>/model.py
wt-pm inspect                            # m23-aerozip should report "available": true
wt-pm agent --days 8                     # fits m23-aerozip + runs the Hermes trace
wt-pm serve --port 8100                  # HTTP API, e.g. GET /health and GET /registry
```

If `torch` is already installed in your environment, you can drop the `[torch]` extra when installing `wt-pm`.

More on how the 25 models fit together: [platform architecture](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/ARCHITECTURE.md) and the [per-model table](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/platform/docs/MODELS.md).

## Honest limits

- Architecture stub: `model.py` defines `AeroZipCompressor` and `compression_ratio` only — no data loader, no quantizer/packet serializer and no training loop. No trained weights or field datasets ship here.
- `compression_ratio` (`input_dim / latent_dim`, `8.0:1` for `64 → 8` or `9.5:1` for the platform's `38 → 4` feature table) is the ratio of feature dimensions, **not** a measured network-bandwidth or byte-level backhaul saving.
- The platform trains `m23-aerozip` briefly (60 Adam epochs, `lr=1e-3`, MSE loss) on healthy training rows of each record and exposes `latent`, `recon_error` and `compression_ratio` in `ModelOutput.extra`; those arrays are not currently fed into anomaly score fusion or downstream classifiers.
- Metrics on the platform come from held-out **simulator** records (fidelity rung 1, see the [fidelity ladder](https://github.com/rajaram-2005/wt-pm-lstm-scada-anomaly/blob/main/docs/fidelity_ladder.md)), not certified asset performance.
